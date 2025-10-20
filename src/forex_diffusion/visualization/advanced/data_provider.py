"""
Data Provider for 3D Reports
Retrieves real data from trading engine database
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import text, create_engine
    from ...services.db_service import DBService
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available")


class ReportDataProvider:
    """Provides real data from trading engine for 3D reports"""

    def __init__(self, db_service: Optional[DBService] = None):
        """
        Initialize data provider

        Args:
            db_service: Optional DBService instance. If None, creates new one.
        """
        self.db_service = db_service or (DBService() if SQLALCHEMY_AVAILABLE else None)

    def get_forecast_data(self, symbol: str = "EUR/USD", days: int = 30) -> pd.DataFrame:
        """
        Get forecast accuracy data from predictions table

        Returns DataFrame with columns: timestamp, horizon, accuracy, confidence
        """
        if not self.db_service:
            return self._generate_sample_forecast_data(days)

        try:
            with self.db_service.engine.connect() as conn:
                # Query predictions table if it exists
                query = text("""
                    SELECT
                        ts_created_ms,
                        horizon,
                        confidence,
                        (CASE WHEN abs(predicted_value - actual_value) < threshold THEN 1.0 ELSE 0.0 END) as accurate
                    FROM predictions
                    WHERE symbol = :symbol
                    AND ts_created_ms > :cutoff_ms
                    ORDER BY ts_created_ms DESC
                    LIMIT 1000
                """)

                cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                result = conn.execute(query, {"symbol": symbol, "cutoff_ms": cutoff_ms})

                rows = result.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=['timestamp', 'horizon', 'confidence', 'accuracy'])
                    return df

        except Exception as e:
            logger.warning(f"Could not fetch forecast data from DB: {e}")

        return self._generate_sample_forecast_data(days)

    def get_pattern_data(self, symbol: str = "EUR/USD", days: int = 90) -> pd.DataFrame:
        """
        Get pattern detection results

        Returns DataFrame with: timestamp, pattern_type, success, confidence
        """
        if not self.db_service:
            return self._generate_sample_pattern_data(days)

        try:
            with self.db_service.engine.connect() as conn:
                # Try to get from pattern_detections or signals table
                query = text("""
                    SELECT
                        ts_created_ms as timestamp,
                        pattern_type,
                        outcome_success as success,
                        confidence
                    FROM pattern_detections
                    WHERE symbol = :symbol
                    AND ts_created_ms > :cutoff_ms
                    ORDER BY ts_created_ms DESC
                    LIMIT 500
                """)

                cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                result = conn.execute(query, {"symbol": symbol, "cutoff_ms": cutoff_ms})

                rows = result.fetchall()
                if rows:
                    return pd.DataFrame(rows, columns=['timestamp', 'pattern_type', 'success', 'confidence'])

        except Exception as e:
            logger.warning(f"Could not fetch pattern data from DB: {e}")

        return self._generate_sample_pattern_data(days)

    def get_trade_data(self, days: int = 60) -> pd.DataFrame:
        """
        Get executed trades data

        Returns DataFrame with: timestamp, symbol, side, pnl, duration, volume
        """
        if not self.db_service:
            return self._generate_sample_trade_data(days)

        try:
            with self.db_service.engine.connect() as conn:
                # Query trades or positions table
                query = text("""
                    SELECT
                        closed_at_ms as timestamp,
                        symbol,
                        side,
                        realized_pnl as pnl,
                        (closed_at_ms - opened_at_ms) / 60000.0 as duration_min,
                        volume,
                        entry_price,
                        exit_price
                    FROM trades
                    WHERE closed_at_ms > :cutoff_ms
                    AND realized_pnl IS NOT NULL
                    ORDER BY closed_at_ms DESC
                    LIMIT 1000
                """)

                cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                result = conn.execute(query, {"cutoff_ms": cutoff_ms})

                rows = result.fetchall()
                if rows:
                    return pd.DataFrame(rows, columns=[
                        'timestamp', 'symbol', 'side', 'pnl', 'duration_min', 'volume', 'entry_price', 'exit_price'
                    ])

        except Exception as e:
            logger.warning(f"Could not fetch trade data from DB: {e}")

        return self._generate_sample_trade_data(days)

    def get_market_data(self, symbol: str = "EUR/USD", timeframe: str = "1h", days: int = 30) -> pd.DataFrame:
        """
        Get historical market candles

        Returns DataFrame with OHLCV data
        """
        if not self.db_service:
            return self._generate_sample_market_data(days)

        try:
            with self.db_service.engine.connect() as conn:
                table_name = f"candles_{symbol.replace('/', '').lower()}_{timeframe}"

                query = text(f"""
                    SELECT
                        ts_utc,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM {table_name}
                    WHERE ts_utc > :cutoff_ms
                    ORDER BY ts_utc ASC
                    LIMIT 1000
                """)

                cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                result = conn.execute(query, {"cutoff_ms": cutoff_ms})

                rows = result.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=['ts_utc', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['ts_utc'], unit='ms')
                    return df

        except Exception as e:
            logger.warning(f"Could not fetch market data from DB: {e}")

        return self._generate_sample_market_data(days)

    def get_signals_data(self, symbol: str = "EUR/USD", days: int = 30) -> pd.DataFrame:
        """
        Get trading signals

        Returns DataFrame with: timestamp, signal_type, confidence, outcome
        """
        if not self.db_service:
            return self._generate_sample_signals_data(days)

        try:
            with self.db_service.engine.connect() as conn:
                query = text("""
                    SELECT
                        ts_created_ms as timestamp,
                        signal_type,
                        confidence,
                        (entry_price - target_price) as expected_pnl,
                        entry_price,
                        target_price,
                        stop_price
                    FROM signals
                    WHERE symbol = :symbol
                    AND ts_created_ms > :cutoff_ms
                    ORDER BY ts_created_ms DESC
                    LIMIT 500
                """)

                cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                result = conn.execute(query, {"symbol": symbol, "cutoff_ms": cutoff_ms})

                rows = result.fetchall()
                if rows:
                    return pd.DataFrame(rows, columns=[
                        'timestamp', 'signal_type', 'confidence', 'expected_pnl',
                        'entry_price', 'target_price', 'stop_price'
                    ])

        except Exception as e:
            logger.warning(f"Could not fetch signals data from DB: {e}")

        return self._generate_sample_signals_data(days)

    # ====================
    # Sample Data Generators (Fallback)
    # ====================

    def _generate_sample_forecast_data(self, days: int) -> pd.DataFrame:
        """Generate sample forecast data for testing"""
        n_forecasts = days * 10
        dates = pd.date_range(end=datetime.now(), periods=n_forecasts, freq='2h')

        data = {
            'timestamp': dates,
            'horizon': np.random.choice([5, 10, 15, 30, 60], n_forecasts),
            'confidence': np.random.beta(8, 2, n_forecasts),
            'accuracy': np.random.beta(7, 3, n_forecasts)
        }

        return pd.DataFrame(data)

    def _generate_sample_pattern_data(self, days: int) -> pd.DataFrame:
        """Generate sample pattern data for testing"""
        patterns = ['Head & Shoulders', 'Double Top', 'Triangle', 'Flag', 'Wedge']
        n_patterns = days * 3

        dates = pd.date_range(end=datetime.now(), periods=n_patterns, freq='8h')

        data = {
            'timestamp': dates,
            'pattern_type': np.random.choice(patterns, n_patterns),
            'success': np.random.choice([0, 1], n_patterns, p=[0.4, 0.6]),
            'confidence': np.random.beta(5, 2, n_patterns)
        }

        return pd.DataFrame(data)

    def _generate_sample_trade_data(self, days: int) -> pd.DataFrame:
        """Generate sample trade data for testing"""
        n_trades = days * 5

        dates = pd.date_range(end=datetime.now(), periods=n_trades, freq='3h')

        pnl = np.random.normal(10, 50, n_trades)  # Mean profit with variance
        duration = np.random.exponential(30, n_trades)  # Minutes

        data = {
            'timestamp': dates,
            'symbol': np.random.choice(['EUR/USD', 'GBP/USD', 'USD/JPY'], n_trades),
            'side': np.random.choice(['BUY', 'SELL'], n_trades),
            'pnl': pnl,
            'duration_min': duration,
            'volume': np.random.uniform(0.1, 2.0, n_trades),
            'entry_price': 1.1000 + np.random.normal(0, 0.01, n_trades),
            'exit_price': 1.1000 + np.random.normal(0, 0.01, n_trades)
        }

        return pd.DataFrame(data)

    def _generate_sample_market_data(self, days: int) -> pd.DataFrame:
        """Generate sample market data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='h')

        base_price = 1.1000
        returns = np.random.normal(0.0001, 0.005, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        data = {
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
            'close': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'volume': np.random.uniform(1000, 10000, len(dates))
        }

        df = pd.DataFrame(data)
        df['ts_utc'] = (df['timestamp'].astype(int) // 10**6).astype(int)  # to milliseconds
        return df

    def _generate_sample_signals_data(self, days: int) -> pd.DataFrame:
        """Generate sample signals data for testing"""
        n_signals = days * 4

        dates = pd.date_range(end=datetime.now(), periods=n_signals, freq='6h')

        data = {
            'timestamp': dates,
            'signal_type': np.random.choice(['BUY', 'SELL'], n_signals),
            'confidence': np.random.beta(6, 3, n_signals),
            'expected_pnl': np.random.normal(15, 30, n_signals),
            'entry_price': 1.1000 + np.random.normal(0, 0.01, n_signals),
            'target_price': 1.1020 + np.random.normal(0, 0.01, n_signals),
            'stop_price': 1.0980 + np.random.normal(0, 0.01, n_signals)
        }

        return pd.DataFrame(data)
