"""
LDM4TS Backtest Integration

Extends the integrated backtest engine to support LDM4TS forecasting signals.
Provides historical OHLCV windows for vision encoding and uncertainty-based
position sizing.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from .integrated_backtest import IntegratedBacktester, BacktestConfig, BacktestResult, Trade
from ..inference.ldm4ts_inference import LDM4TSInferenceService, LDM4TSPrediction
from ..intelligence.unified_signal_fusion import UnifiedSignalFusion, FusedSignal
from ..intelligence.signal_quality_scorer import SignalQualityScorer, SignalSource


@dataclass
class LDM4TSBacktestConfig(BacktestConfig):
    """
    Extended backtest config with LDM4TS settings.
    
    Inherits from BacktestConfig and adds LDM4TS-specific parameters.
    """
    # LDM4TS settings
    use_ldm4ts: bool = False
    ldm4ts_checkpoint_path: Optional[str] = None
    ldm4ts_horizons: List[int] = None
    ldm4ts_uncertainty_threshold: float = 0.5
    ldm4ts_min_strength: float = 0.3
    ldm4ts_position_scaling: bool = True
    ldm4ts_num_samples: int = 50
    ldm4ts_window_size: int = 100  # Candles for vision encoding
    
    def __post_init__(self):
        """Set default horizons."""
        if self.ldm4ts_horizons is None:
            self.ldm4ts_horizons = [15, 60, 240]


class LDM4TSBacktester(IntegratedBacktester):
    """
    Extended backtester with LDM4TS support.
    
    Extends IntegratedBacktester to add:
    - LDM4TS prediction generation from historical OHLCV
    - Signal fusion integration
    - Uncertainty-based position sizing
    - Performance comparison vs baseline
    """
    
    def __init__(self, config: LDM4TSBacktestConfig):
        """
        Initialize LDM4TS backtester.
        
        Args:
            config: Extended configuration with LDM4TS settings
        """
        super().__init__(config)
        
        self.config: LDM4TSBacktestConfig = config
        
        # LDM4TS components
        self.ldm4ts_service: Optional[LDM4TSInferenceService] = None
        self.signal_fusion: Optional[UnifiedSignalFusion] = None
        
        # Initialize LDM4TS if enabled
        if config.use_ldm4ts:
            self._initialize_ldm4ts()
        
        # Tracking
        self.ldm4ts_predictions: List[Dict[str, Any]] = []
        self.ldm4ts_signals: List[FusedSignal] = []
        
        logger.info(f"LDM4TS Backtester initialized (use_ldm4ts={config.use_ldm4ts})")
    
    def _initialize_ldm4ts(self):
        """Initialize LDM4TS inference service and signal fusion."""
        try:
            # Load LDM4TS service
            self.ldm4ts_service = LDM4TSInferenceService.get_instance()
            
            if self.config.ldm4ts_checkpoint_path:
                self.ldm4ts_service.load_model(
                    checkpoint_path=self.config.ldm4ts_checkpoint_path,
                    horizons=self.config.ldm4ts_horizons
                )
                logger.info(f"✅ Loaded LDM4TS model: {self.config.ldm4ts_checkpoint_path}")
            else:
                logger.warning("⚠️  No LDM4TS checkpoint path provided - predictions will fail")
            
            # Initialize signal fusion
            quality_scorer = SignalQualityScorer()
            self.signal_fusion = UnifiedSignalFusion(quality_scorer=quality_scorer)
            
            logger.info("✅ LDM4TS components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LDM4TS: {e}", exc_info=True)
            self.config.use_ldm4ts = False
    
    def _get_historical_ohlcv_window(
        self,
        data: pd.DataFrame,
        bar_index: int,
        window_size: int = 100
    ) -> Optional[np.ndarray]:
        """
        Extract historical OHLCV window for LDM4TS.
        
        Args:
            data: Full OHLCV DataFrame
            bar_index: Current bar index
            window_size: Number of candles to fetch
            
        Returns:
            OHLCV numpy array [window_size, 5] or None if insufficient data
        """
        start_idx = max(0, bar_index - window_size + 1)
        end_idx = bar_index + 1
        
        if end_idx - start_idx < window_size:
            # Insufficient historical data
            return None
        
        window_df = data.iloc[start_idx:end_idx]
        
        # Extract OHLCV columns
        ohlcv = window_df[['open', 'high', 'low', 'close', 'volume']].values
        
        return ohlcv
    
    def _generate_ldm4ts_signals(
        self,
        data: pd.DataFrame,
        bar_index: int,
        current_time: datetime,
        current_price: float
    ) -> List[FusedSignal]:
        """
        Generate LDM4TS forecast signals for current bar.
        
        Args:
            data: OHLCV DataFrame
            bar_index: Current bar index
            current_time: Current timestamp
            current_price: Current price
            
        Returns:
            List of FusedSignal objects from LDM4TS
        """
        if not self.config.use_ldm4ts or not self.ldm4ts_service or not self.signal_fusion:
            return []
        
        # Get historical OHLCV window
        ohlcv = self._get_historical_ohlcv_window(
            data,
            bar_index,
            window_size=self.config.ldm4ts_window_size
        )
        
        if ohlcv is None:
            return []
        
        try:
            # Generate LDM4TS forecast
            prediction = self.ldm4ts_service.predict(
                ohlcv=ohlcv,
                num_samples=self.config.ldm4ts_num_samples,
                symbol=self.config.symbol
            )
            
            # Store prediction for analysis
            self.ldm4ts_predictions.append({
                'timestamp': current_time,
                'bar_index': bar_index,
                'prediction': prediction.to_dict()
            })
            
            # Generate signals via signal fusion
            signals = self.signal_fusion._collect_ldm4ts_signals(
                symbol=self.config.symbol,
                timeframe='1m',  # Adjust based on actual timeframe
                current_ohlcv=ohlcv,
                horizons=self.config.ldm4ts_horizons
            )
            
            # Filter by strength threshold
            filtered_signals = [
                s for s in signals
                if s.strength >= self.config.ldm4ts_min_strength
            ]
            
            if filtered_signals:
                logger.debug(
                    f"Generated {len(filtered_signals)} LDM4TS signals "
                    f"at bar {bar_index} (time={current_time})"
                )
                self.ldm4ts_signals.extend(filtered_signals)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Failed to generate LDM4TS signals at bar {bar_index}: {e}", exc_info=True)
            return []
    
    def _calculate_position_size_with_uncertainty(
        self,
        signal: FusedSignal,
        base_size: float
    ) -> float:
        """
        Adjust position size based on LDM4TS uncertainty.
        
        Args:
            signal: FusedSignal with uncertainty metadata
            base_size: Base position size from risk management
            
        Returns:
            Adjusted position size (may be 0 if uncertainty too high)
        """
        if not self.config.ldm4ts_position_scaling or signal.source != SignalSource.LDM4TS_FORECAST:
            return base_size
        
        if not signal.metadata:
            return base_size
        
        uncertainty_pct = signal.metadata.get('uncertainty_pct', 0)
        threshold = self.config.ldm4ts_uncertainty_threshold
        
        # Reject if uncertainty too high
        if uncertainty_pct >= threshold:
            logger.debug(
                f"Signal rejected: uncertainty {uncertainty_pct:.3f}% >= threshold {threshold:.3f}%"
            )
            return 0.0
        
        # Calculate adjustment factor
        uncertainty_factor = 1.0 - (uncertainty_pct / threshold)
        adjusted_size = base_size * uncertainty_factor
        
        logger.debug(
            f"Position sizing: base={base_size:.2f}, uncertainty={uncertainty_pct:.3f}%, "
            f"factor={uncertainty_factor:.2f}, adjusted={adjusted_size:.2f}"
        )
        
        return adjusted_size
    
    def _check_entry(
        self,
        bar_index: int,
        data: pd.DataFrame,
        features: pd.DataFrame,
        current_time: datetime,
        current_price: float
    ):
        """
        Override entry check to include LDM4TS signals.
        
        Extends parent method to:
        1. Generate LDM4TS signals
        2. Combine with ML ensemble signals
        3. Apply uncertainty-based position sizing
        
        Args:
            bar_index: Current bar index
            data: OHLCV data
            features: Features DataFrame
            current_time: Current timestamp
            current_price: Current price
        """
        # Original ML ensemble logic (parent class)
        ml_signal = None
        ml_confidence = 0.0
        
        if self.ml_ensemble:
            X = features.iloc[bar_index:bar_index+1]
            try:
                prediction = self.ml_ensemble.predict(X)[0]
                probabilities = self.ml_ensemble.predict_proba(X)[0]
                ml_confidence = np.max(probabilities)
                
                if prediction != 0 and ml_confidence >= self.config.min_signal_confidence:
                    ml_signal = prediction
            except:
                pass
        
        # NEW: LDM4TS signals
        ldm4ts_signals = []
        if self.config.use_ldm4ts:
            ldm4ts_signals = self._generate_ldm4ts_signals(
                data,
                bar_index,
                current_time,
                current_price
            )
        
        # Combine signals (prioritize LDM4TS if available, otherwise ML)
        selected_signal = None
        selected_confidence = 0.0
        signal_source = 'ml'
        
        if ldm4ts_signals:
            # Use best LDM4TS signal
            best_ldm4ts = max(ldm4ts_signals, key=lambda s: s.quality_score.composite_score)
            selected_signal = 1 if best_ldm4ts.direction == 'bull' else -1
            selected_confidence = best_ldm4ts.quality_score.composite_score
            signal_source = 'ldm4ts'
            
            # Store full signal for position sizing
            selected_fused_signal = best_ldm4ts
            
        elif ml_signal is not None:
            selected_signal = ml_signal
            selected_confidence = ml_confidence
            signal_source = 'ml'
            selected_fused_signal = None
        else:
            # No signal
            return
        
        # Detect regime (from parent class logic)
        regime = None
        if self.regime_detector and self.config.use_regime_detection:
            regime_data = data.iloc[max(0, bar_index-100):bar_index+1]
            if len(regime_data) >= 50:
                try:
                    regime_state = self.regime_detector.predict_current(regime_data)
                    regime = regime_state.regime.value if regime_state else None
                except:
                    pass
        
        # Calculate position size
        direction = 'long' if selected_signal > 0 else 'short'
        
        # Get stop/target from LDM4TS signal if available
        if selected_fused_signal:
            stop_price = selected_fused_signal.stop_price or current_price * 0.98
            take_profit = selected_fused_signal.target_price or current_price * 1.02
        else:
            stop_distance = current_price * 0.02
            stop_price = current_price - stop_distance if direction == 'long' else current_price + stop_distance
            take_profit = current_price + stop_distance * 2 if direction == 'long' else current_price - stop_distance * 2
        
        # Base position size (from parent class)
        base_size = self._calculate_position_size(
            current_price,
            stop_price,
            regime,
            selected_confidence
        )
        
        # NEW: Apply uncertainty adjustment for LDM4TS
        if signal_source == 'ldm4ts' and selected_fused_signal:
            final_size = self._calculate_position_size_with_uncertainty(
                selected_fused_signal,
                base_size
            )
            
            if final_size == 0.0:
                # Signal rejected due to high uncertainty
                return
        else:
            final_size = base_size
        
        # Calculate costs
        entry_cost = self._calculate_cost(current_price, final_size)
        
        # Open trade
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=self.config.symbol,
            direction=direction,
            entry_time=current_time,
            entry_price=current_price,
            size=final_size,
            stop_loss=stop_price,
            take_profit=take_profit,
            entry_cost=entry_cost,
            entry_signal_confidence=selected_confidence,
            entry_regime=regime
        )
        
        self.open_positions.append(trade)
        self.trade_counter += 1
        self.current_capital -= entry_cost
        
        logger.debug(
            f"Opened {direction} position: entry={current_price:.5f}, "
            f"stop={stop_price:.5f}, target={take_profit:.5f}, "
            f"size={final_size:.2f}, source={signal_source}, confidence={selected_confidence:.2f}"
        )
    
    def run(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        labels: pd.Series,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run backtest with LDM4TS support.
        
        Extends parent run() method to include LDM4TS tracking.
        
        Args:
            data: OHLCV data
            features: Calculated features
            labels: Target labels
            verbose: Print progress
            
        Returns:
            BacktestResult with LDM4TS metrics
        """
        # Reset LDM4TS tracking
        self.ldm4ts_predictions = []
        self.ldm4ts_signals = []
        
        # Run parent backtest
        result = super().run(data, features, labels, verbose)
        
        # Add LDM4TS metrics to result
        if self.config.use_ldm4ts and self.ldm4ts_predictions:
            result.metadata = result.metadata or {}
            result.metadata['ldm4ts'] = {
                'total_predictions': len(self.ldm4ts_predictions),
                'total_signals': len(self.ldm4ts_signals),
                'avg_uncertainty': np.mean([
                    p['prediction'].get('uncertainty_pct', 0)
                    for p in self.ldm4ts_predictions
                    if 'prediction' in p
                ]),
                'signals_by_direction': {
                    'bull': sum(1 for s in self.ldm4ts_signals if s.direction == 'bull'),
                    'bear': sum(1 for s in self.ldm4ts_signals if s.direction == 'bear'),
                    'neutral': sum(1 for s in self.ldm4ts_signals if s.direction == 'neutral')
                }
            }
            
            if verbose:
                logger.info("\n" + "=" * 80)
                logger.info("LDM4TS METRICS")
                logger.info("=" * 80)
                logger.info(f"Total predictions: {result.metadata['ldm4ts']['total_predictions']}")
                logger.info(f"Total signals: {result.metadata['ldm4ts']['total_signals']}")
                logger.info(f"Avg uncertainty: {result.metadata['ldm4ts']['avg_uncertainty']:.3f}%")
                logger.info(f"Bull signals: {result.metadata['ldm4ts']['signals_by_direction']['bull']}")
                logger.info(f"Bear signals: {result.metadata['ldm4ts']['signals_by_direction']['bear']}")
                logger.info("=" * 80)
        
        return result
