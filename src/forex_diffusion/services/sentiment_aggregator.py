"""
Sentiment aggregator service.

Processes raw sentiment data and calculates:
- Moving averages (5min, 15min, 1h)
- Sentiment changes
- Contrarian signals

Refactored to use ThreadedBackgroundService base class.
"""
from __future__ import annotations

from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
from collections import deque

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .base_service import ThreadedBackgroundService


class SentimentAggregatorService(ThreadedBackgroundService):
    """
    Background service that processes sentiment data and calculates derived metrics.
    
    Inherits from ThreadedBackgroundService for lifecycle management and error recovery.
    """

    def __init__(self, engine: Engine, symbols: List[str] | None = None, interval_seconds: int = 30):
        """
        Initialize sentiment aggregator service.
        
        Args:
            engine: SQLAlchemy engine for database access
            symbols: List of symbols to process (None = load from config)
            interval_seconds: Interval between sentiment processing runs (default: 30s)
        """
        # Initialize base class with circuit breaker enabled
        super().__init__(
            engine=engine,
            symbols=symbols,
            interval_seconds=interval_seconds,
            enable_circuit_breaker=True
        )
        
        # Sentiment-specific state: cache for sentiment history
        self._sentiment_history: Dict[str, deque] = {}
        self._history_window = 3600  # Keep 1 hour of history (in seconds)
    
    @property
    def service_name(self) -> str:
        """Service name for logging."""
        return "SentimentAggregatorService"
    
    def _process_iteration(self):
        """
        Process one sentiment aggregation iteration.
        
        Called by base class in background thread. Processes sentiment data
        and calculates metrics for all configured symbols.
        """
        symbols = self.get_symbols()  # Use base class method
        for sym in symbols:
            self._process_sentiment_for_symbol(sym)

    def _process_sentiment_for_symbol(self, symbol: str):
        """Process sentiment data for a symbol."""
        try:
            # Get recent sentiment data (last hour)
            ts_start = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp() * 1000)

            with self.engine.connect() as conn:
                query = text(
                    "SELECT ts_utc, long_pct, short_pct, total_traders, confidence "
                    "FROM sentiment_data "
                    "WHERE symbol = :symbol AND ts_utc >= :ts_start "
                    "ORDER BY ts_utc ASC"
                )
                rows = conn.execute(query, {"symbol": symbol, "ts_start": ts_start}).fetchall()

            if len(rows) < 5:  # Need at least 5 data points
                return

            df = pd.DataFrame(rows, columns=["ts_utc", "long_pct", "short_pct", "total_traders", "confidence"])
            df["ts_dt"] = pd.to_datetime(df["ts_utc"], unit="ms", utc=True)
            df = df.set_index("ts_dt").sort_index()

            # Calculate moving averages
            df["long_pct_ma_5m"] = df["long_pct"].rolling(window=10, min_periods=5).mean()  # Assuming 30s updates
            df["long_pct_ma_15m"] = df["long_pct"].rolling(window=30, min_periods=15).mean()
            df["long_pct_ma_1h"] = df["long_pct"].rolling(window=120, min_periods=60).mean()

            # Calculate sentiment change (delta from 1h ago)
            df["sentiment_change"] = df["long_pct"] - df["long_pct"].shift(120)  # 1h ago

            # Contrarian signal (if >70% long, bearish signal; if <30% long, bullish signal)
            df["contrarian_signal"] = 0.0
            df.loc[df["long_pct"] > 70, "contrarian_signal"] = -1.0  # Bearish
            df.loc[df["long_pct"] < 30, "contrarian_signal"] = 1.0   # Bullish

            # Cache latest metrics
            latest = df.iloc[-1]
            metrics = {
                "timestamp": int(latest.name.timestamp() * 1000),
                "long_pct": latest["long_pct"],
                "short_pct": latest["short_pct"],
                "long_pct_ma_5m": latest["long_pct_ma_5m"] if pd.notna(latest["long_pct_ma_5m"]) else None,
                "long_pct_ma_15m": latest["long_pct_ma_15m"] if pd.notna(latest["long_pct_ma_15m"]) else None,
                "long_pct_ma_1h": latest["long_pct_ma_1h"] if pd.notna(latest["long_pct_ma_1h"]) else None,
                "sentiment_change": latest["sentiment_change"] if pd.notna(latest["sentiment_change"]) else None,
                "contrarian_signal": latest["contrarian_signal"],
            }

            # Update cache
            if symbol not in self._sentiment_history:
                self._sentiment_history[symbol] = deque(maxlen=120)  # Keep last 1 hour at 30s intervals
            self._sentiment_history[symbol].append(metrics)

            logger.debug(
                f"Sentiment metrics for {symbol}: long={metrics['long_pct']:.1f}%, "
                f"change={metrics.get('sentiment_change', 'N/A'):.1f}%, "
                f"contrarian={metrics['contrarian_signal']}"
            )

        except Exception as e:
            logger.error(f"Failed to process sentiment for {symbol}: {e}")



    def get_latest_sentiment_metrics(self, symbol: str) -> Optional[Dict]:
        """Get latest sentiment metrics for a symbol."""
        if symbol in self._sentiment_history and self._sentiment_history[symbol]:
            return self._sentiment_history[symbol][-1]
        return None

    def get_sentiment_signal(self, symbol: str) -> Optional[str]:
        """
        Get sentiment-based trading signal.

        Returns:
            "bullish", "bearish", or "neutral"
        """
        metrics = self.get_latest_sentiment_metrics(symbol)
        if not metrics:
            return None

        contrarian = metrics.get("contrarian_signal", 0.0)
        if contrarian > 0:
            return "bullish"
        elif contrarian < 0:
            return "bearish"
        else:
            return "neutral"

    def get_sentiment_history(self, symbol: str, minutes: int = 60) -> Optional[pd.DataFrame]:
        """Get sentiment history as DataFrame."""
        if symbol not in self._sentiment_history:
            return None

        history = list(self._sentiment_history[symbol])
        if not history:
            return None

        # Filter by time window
        now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        cutoff_ts = now_ts - (minutes * 60 * 1000)
        filtered = [h for h in history if h["timestamp"] >= cutoff_ts]

        if not filtered:
            return None

        df = pd.DataFrame(filtered)
        df["ts_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("ts_dt").sort_index()

        return df
