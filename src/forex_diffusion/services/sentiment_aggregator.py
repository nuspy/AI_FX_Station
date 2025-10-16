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

    def __init__(self, engine: Engine, symbols: List[str] | None = None, interval_seconds: int = 60):
        """
        Initialize sentiment aggregator service.
        
        Args:
            engine: SQLAlchemy engine for database access
            symbols: List of symbols to process (None = load from config)
            interval_seconds: Interval between sentiment processing runs (default: 60s = 1 min)
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
                "total_traders": int(latest["total_traders"]) if pd.notna(latest["total_traders"]) else 0,
                "long_pct_ma_5m": latest["long_pct_ma_5m"] if pd.notna(latest["long_pct_ma_5m"]) else None,
                "long_pct_ma_15m": latest["long_pct_ma_15m"] if pd.notna(latest["long_pct_ma_15m"]) else None,
                "long_pct_ma_1h": latest["long_pct_ma_1h"] if pd.notna(latest["long_pct_ma_1h"]) else None,
                "sentiment_change": latest["sentiment_change"] if pd.notna(latest["sentiment_change"]) else None,
                "contrarian_signal": latest["contrarian_signal"],
            }

            # Update cache
            if symbol not in self._sentiment_history:
                self._sentiment_history[symbol] = deque(maxlen=12)  # Keep last 1 hour at 5min intervals (12 * 5min = 60min)
            self._sentiment_history[symbol].append(metrics)

            # Format sentiment change safely
            sentiment_change = metrics.get('sentiment_change')
            change_str = f"{sentiment_change:.1f}%" if sentiment_change is not None else "N/A"
            
            logger.debug(
                f"Sentiment metrics for {symbol}: long={metrics['long_pct']:.1f}%, "
                f"change={change_str}, "
                f"contrarian={metrics['contrarian_signal']}"
            )

        except Exception as e:
            logger.error(f"Failed to process sentiment for {symbol}: {e}")



    def get_latest_sentiment_metrics(self, symbol: str) -> Optional[Dict]:
        """
        Get latest sentiment metrics for a symbol in UI-compatible format.
        
        Returns:
            Dictionary with sentiment metrics for SentimentPanel:
            - sentiment: "bullish", "bearish", or "neutral"
            - confidence: 0.0-1.0
            - ratio: -1.0 to +1.0 sentiment ratio
            - total_traders: Total volume count
            - long_pct: Percentage long (0-100)
            - short_pct: Percentage short (0-100)
            - contrarian_signal: -1.0 to +1.0 contrarian indicator
        """
        # Try memory cache first
        if symbol in self._sentiment_history and self._sentiment_history[symbol]:
            cached = self._sentiment_history[symbol][-1]
            # Add contrarian signal calculation
            long_pct = cached.get("long_pct", 50.0)
            if long_pct > 70:
                contrarian = -(long_pct - 50) / 50  # Negative when crowd is long
            elif long_pct < 30:
                contrarian = (50 - long_pct) / 50  # Positive when crowd is short
            else:
                contrarian = 0.0
            
            return {
                "sentiment": self._classify_sentiment(cached.get("long_pct", 50.0)),
                "confidence": abs(cached.get("long_pct", 50.0) - 50.0) / 50.0,  # 0-1 based on distance from 50%
                "ratio": (cached.get("long_pct", 50.0) - 50.0) / 50.0,  # -1 to +1
                "total_traders": int(cached.get("total_traders", 0)),
                "long_pct": cached.get("long_pct", 50.0),
                "short_pct": cached.get("short_pct", 50.0),
                "contrarian_signal": contrarian,
            }
        
        # Fallback to database
        try:
            with self.engine.connect() as conn:
                query = text(
                    "SELECT sentiment, ratio, buy_volume, sell_volume, confidence, "
                    "long_pct, short_pct "
                    "FROM sentiment_data "
                    "WHERE symbol = :symbol "
                    "ORDER BY ts_utc DESC LIMIT 1"
                )
                row = conn.execute(query, {"symbol": symbol}).fetchone()
                
                if row:
                    long_pct = row[5] if row[5] is not None else 50.0
                    short_pct = row[6] if row[6] is not None else 50.0
                    
                    # Calculate contrarian signal
                    if long_pct > 70:
                        contrarian = -(long_pct - 50) / 50
                    elif long_pct < 30:
                        contrarian = (50 - long_pct) / 50
                    else:
                        contrarian = 0.0
                    
                    return {
                        "sentiment": row[0] or "neutral",
                        "confidence": row[4] if row[4] is not None else 0.0,
                        "ratio": row[1] if row[1] is not None else 0.0,
                        "total_traders": int((row[2] or 0) + (row[3] or 0)),
                        "long_pct": long_pct,
                        "short_pct": short_pct,
                        "contrarian_signal": contrarian,
                    }
        except Exception as e:
            logger.error(f"Failed to fetch sentiment from database for {symbol}: {e}")
        
        return None
    
    def _classify_sentiment(self, long_pct: float) -> str:
        """Classify sentiment based on long percentage."""
        if long_pct > 60:
            return "bullish"
        elif long_pct < 40:
            return "bearish"
        else:
            return "neutral"

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
