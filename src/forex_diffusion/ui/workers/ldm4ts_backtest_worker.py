"""
LDM4TS Backtesting Worker - Background worker for LDM4TS backtesting.

Handles:
- Historical data loading
- Rolling forecast generation
- Metrics calculation (MAE, RMSE, coverage, etc.)
- Results streaming to UI
"""
from __future__ import annotations

from typing import Dict, Any, List

import pandas as pd
import numpy as np
from PySide6.QtCore import QRunnable, QObject, Signal
from loguru import logger


class LDM4TSBacktestSignals(QObject):
    """Signals for LDM4TS backtesting worker"""
    progress = Signal(int)  # Progress percentage (0-100)
    status = Signal(str)  # Status message
    result_row = Signal(dict)  # Single result row for table
    metrics_update = Signal(dict)  # Overall metrics update
    backtest_complete = Signal(dict)  # Final metrics
    error = Signal(str)  # Error message


class LDM4TSBacktestWorker(QRunnable):
    """
    Background worker for LDM4TS backtesting.
    Runs backtest in separate thread and streams results.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self.signals = LDM4TSBacktestSignals()
        self._stop_requested = False

    def stop(self):
        """Request worker to stop"""
        self._stop_requested = True
        logger.info("LDM4TS backtest stop requested")

    def run(self):
        """Execute backtest"""
        try:
            self.signals.status.emit("Initializing backtest...")
            logger.info(f"Starting LDM4TS backtest with params: {self.params}")
            
            # Extract parameters
            checkpoint = self.params['checkpoint']
            symbol = self.params['symbol']
            timeframe = self.params['timeframe']
            start_date = self.params['start_date']
            end_date = self.params['end_date']
            window_size = self.params['window_size']
            horizons_str = self.params['horizons']
            num_samples = self.params['num_samples']
            step_size = self.params['step_size']
            
            # Parse horizons
            horizons = [int(h.strip()) for h in horizons_str.split(',')]
            
            # Step 1: Load LDM4TS model
            self.signals.status.emit("Loading LDM4TS model...")
            self.signals.progress.emit(5)
            
            from ...services.ldm4ts_inference_service import LDM4TSInferenceService
            
            service = LDM4TSInferenceService.get_instance(checkpoint_path=checkpoint)
            
            if not service._initialized:
                raise RuntimeError("Failed to initialize LDM4TS service")
            
            logger.info(f"LDM4TS model loaded from {checkpoint}")
            self.signals.status.emit("Model loaded")
            self.signals.progress.emit(10)
            
            # Step 2: Load historical data
            self.signals.status.emit("Loading historical data...")
            
            df_hist = self._load_historical_data(symbol, timeframe, start_date, end_date)
            
            if df_hist is None or len(df_hist) < window_size:
                raise RuntimeError(f"Insufficient historical data: {len(df_hist) if df_hist is not None else 0} candles")
            
            logger.info(f"Loaded {len(df_hist)} historical candles")
            self.signals.status.emit(f"Data loaded: {len(df_hist)} candles")
            self.signals.progress.emit(15)
            
            # Step 3: Run rolling forecasts
            self.signals.status.emit("Running rolling forecasts...")
            
            results = []
            num_forecasts = (len(df_hist) - window_size) // step_size
            
            for i in range(0, len(df_hist) - window_size, step_size):
                if self._stop_requested:
                    logger.info("Backtest stopped by user")
                    self.signals.status.emit("Backtest stopped")
                    return
                
                # Extract window
                window_start = i
                window_end = i + window_size
                window_df = df_hist.iloc[window_start:window_end]
                
                # Run forecast
                try:
                    # Convert window to numpy array
                    window_array = window_df[['open', 'high', 'low', 'close', 'volume']].values
                    
                    # Run prediction
                    prediction = service.predict(
                        ohlcv=window_array,
                        horizons=horizons,
                        num_samples=num_samples,
                        symbol=symbol
                    )
                    
                    # Extract predictions for each horizon
                    for horizon_idx, horizon_min in enumerate(horizons):
                        # Get true value at horizon
                        future_idx = window_end + horizon_min - 1
                        if future_idx >= len(df_hist):
                            continue
                        
                        true_value = df_hist.iloc[future_idx]['close']
                        
                        # Get predicted value and uncertainty from prediction object
                        predicted_value = prediction.mean[horizon_min]
                        uncertainty = prediction.std[horizon_min]
                        q05_value = prediction.q05[horizon_min]
                        q95_value = prediction.q95[horizon_min]
                        
                        # Calculate error
                        error = abs(predicted_value - true_value)
                        
                        # Calculate coverage (95% interval: q05 to q95)
                        coverage = 1.0 if q05_value <= true_value <= q95_value else 0.0
                        
                        # Create result row
                        result = {
                            'timestamp': df_hist.index[window_end - 1].strftime('%Y-%m-%d %H:%M:%S'),
                            'horizon': f"{horizon_min}m",
                            'true': f"{true_value:.5f}",
                            'predicted': f"{predicted_value:.5f}",
                            'uncertainty': f"{uncertainty:.5f}",
                            'error': f"{error:.5f}",
                            'coverage': f"{coverage:.1f}",
                        }
                        
                        results.append(result)
                        
                        # Emit result to UI
                        self.signals.result_row.emit(result)
                    
                except Exception as e:
                    logger.warning(f"Forecast failed at index {i}: {e}")
                    continue
                
                # Update progress
                progress = 15 + int((i / (len(df_hist) - window_size)) * 80)
                self.signals.progress.emit(progress)
                
                # Update status
                forecast_num = (i // step_size) + 1
                self.signals.status.emit(f"Processing forecast {forecast_num}/{num_forecasts}...")
                
                # Update interim metrics
                if len(results) > 0 and len(results) % 10 == 0:
                    interim_metrics = self._calculate_metrics(results)
                    self.signals.metrics_update.emit(interim_metrics)
            
            # Step 4: Calculate final metrics
            self.signals.status.emit("Calculating final metrics...")
            self.signals.progress.emit(95)
            
            final_metrics = self._calculate_metrics(results)
            
            logger.info(f"Backtest complete. {len(results)} predictions generated.")
            logger.info(f"Metrics: MAE={final_metrics['mae']:.5f}, RMSE={final_metrics['rmse']:.5f}, Coverage={final_metrics['coverage']:.2%}")
            
            self.signals.progress.emit(100)
            self.signals.status.emit(f"Backtest complete: {len(results)} predictions")
            self.signals.backtest_complete.emit(final_metrics)
            
        except Exception as e:
            logger.exception(f"LDM4TS backtest failed: {e}")
            self.signals.error.emit(str(e))
            self.signals.status.emit(f"Backtest failed: {e}")

    def _load_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtest period"""
        try:
            from ...services.marketdata import MarketDataService
            from sqlalchemy import text
            from dateutil.parser import parse
            
            ms = MarketDataService()
            
            # Parse dates
            start_ts = int(parse(start_date).timestamp() * 1000)
            end_ts = int(parse(end_date).timestamp() * 1000)
            
            # Fetch data
            with ms.engine.connect() as conn:
                query = text("""
                    SELECT ts_utc, open, high, low, close, volume
                    FROM market_data_candles
                    WHERE symbol = :symbol 
                      AND timeframe = :timeframe
                      AND ts_utc BETWEEN :start_ts AND :end_ts
                    ORDER BY ts_utc ASC
                """)
                rows = conn.execute(query, {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_ts": start_ts,
                    "end_ts": end_ts
                }).fetchall()
            
            if not rows:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': pd.to_datetime(r[0], unit='ms', utc=True),
                    'open': float(r[1]),
                    'high': float(r[2]),
                    'low': float(r[3]),
                    'close': float(r[4]),
                    'volume': float(r[5]) if r[5] is not None else 0.0
                }
                for r in rows
            ])
            df = df.set_index('timestamp')
            
            return df
            
        except Exception as e:
            logger.exception(f"Failed to load historical data: {e}")
            return None

    def _calculate_metrics(self, results: List[dict]) -> dict:
        """Calculate metrics from results"""
        if not results:
            return {}
        
        # Extract values
        errors = [float(r['error']) for r in results]
        coverages = [float(r['coverage']) for r in results]
        
        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        coverage = np.mean(coverages)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'num_predictions': len(results),
        }
