"""
Aggregator service: scheduled aggregation of market_data_ticks -> market_data_candles.

This service implements a stateful aggregation strategy to ensure data integrity and
prevent data loss, even in cases of processing delays or restarts.

Refactored to use ThreadedBackgroundService base class for better maintainability.
"""
from __future__ import annotations

import threading
from typing import List, Dict, Optional
from datetime import datetime, timezone

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .base_service import ThreadedBackgroundService
from ..data import io as data_io

# Standard timeframes to be aggregated
TF_RULES: Dict[str, Dict] = {
    "1m": {"rule": "1min", "minutes": 1},
    "5m": {"rule": "5min", "minutes": 5},
    "15m": {"rule": "15min", "minutes": 15},
    "30m": {"rule": "30min", "minutes": 30},
    "1h": {"rule": "60min", "minutes": 60},
    "4h": {"rule": "240min", "minutes": 240},
    "1d": {"rule": "1D", "minutes": 1440},
}

class AggregatorService(ThreadedBackgroundService):
    """
    A stateful background aggregator that periodically converts ticks into candles.
    
    Inherits from ThreadedBackgroundService for lifecycle management, error recovery,
    and metrics collection.
    """
    
    def __init__(self, engine: Engine, symbols: List[str] | None = None, interval_seconds: float = 60.0):
        """
        Initialize aggregator service.
        
        Args:
            engine: SQLAlchemy engine for database access
            symbols: List of symbols to process (None = load from config)
            interval_seconds: Interval between aggregation runs (default: 60s = 1 minute)
        """
        # Initialize base class with circuit breaker enabled
        super().__init__(
            engine=engine,
            symbols=symbols,
            interval_seconds=interval_seconds,
            enable_circuit_breaker=True
        )
        
        # Aggregator-specific state
        self._last_processed_ts: Dict[tuple[str, str], int] = {}
        self._state_lock = threading.Lock()  # Thread safety for state dictionary
    
    @property
    def service_name(self) -> str:
        """Service name for logging."""
        return "AggregatorService"

    def _get_last_candle_ts(self, symbol: str, timeframe: str) -> Optional[int]:
        """Retrieves the timestamp of the last saved candle from the database."""
        try:
            with self.engine.connect() as conn:
                query = text(
                    "SELECT MAX(ts_utc) FROM market_data_candles "
                    "WHERE symbol = :symbol AND timeframe = :timeframe"
                )
                result = conn.execute(query, {"symbol": symbol, "timeframe": timeframe}).scalar_one_or_none()
                return int(result) if result else None
        except Exception as e:
            logger.exception(f"Failed to get last candle timestamp for {symbol} {timeframe}: {e}")
            return None

    def _process_iteration(self):
        """
        Process one aggregation iteration.
        
        Called by base class in background thread. Aggregates ticks to candles
        for all configured symbols and timeframes at appropriate intervals.
        """
        # Wait until next minute boundary for time-aligned aggregation
        self._sleep_until_next_minute()
        
        ts_now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        minute_idx = int(ts_now.timestamp() / 60)

        symbols = self.get_symbols()  # Use base class method
        for sym in symbols:
            for tf, info in TF_RULES.items():
                # Only aggregate at appropriate intervals (e.g., 5m at 0,5,10,...)
                if minute_idx % info["minutes"] == 0:
                    self._aggregate_for_symbol(sym, tf, ts_now)

    def _aggregate_for_symbol(self, symbol: str, timeframe: str, ts_now: datetime):
        state_key = (symbol, timeframe)
        
        # Thread-safe access to last processed timestamp
        with self._state_lock:
            last_ts = self._last_processed_ts.get(state_key)
        
        if last_ts is None:
            last_ts = self._get_last_candle_ts(symbol, timeframe)
        if not last_ts:
            last_ts = int(ts_now.timestamp() * 1000) - (24 * 60 * 60 * 1000)

        start_ms = last_ts
        end_ms = int(ts_now.timestamp() * 1000)

        if start_ms >= end_ms:
            return

        # Fetch ticks (read-only, no transaction needed)
        with self.engine.connect() as conn:
            # Corrected query to fetch ticks with the right timeframe
            # Extended to support tick_volume and real_volume from multi-provider data
            query = text(
                "SELECT ts_utc, price, bid, ask, volume, tick_volume, real_volume, provider_source "
                "FROM market_data_ticks "
                "WHERE symbol = :symbol AND timeframe = 'tick' AND ts_utc > :start AND ts_utc <= :end "
                "ORDER BY ts_utc ASC"
            )
            rows = conn.execute(query, {"symbol": symbol, "start": start_ms, "end": end_ms}).fetchall()

        if not rows:
            return

        # logger.info(f"Found {len(rows)} new ticks for {symbol} to aggregate into {timeframe}.")
        df_ticks = pd.DataFrame(rows, columns=["ts_utc", "price", "bid", "ask", "volume", "tick_volume", "real_volume", "provider_source"])
        df_ticks['price'] = df_ticks['price'].fillna((df_ticks['bid'] + df_ticks['ask']) / 2).ffill()
        if df_ticks.empty or df_ticks['price'].isnull().all():
            return

        df_ticks["ts_dt"] = pd.to_datetime(df_ticks["ts_utc"], unit="ms", utc=True)
        df_ticks = df_ticks.set_index("ts_dt").sort_index()

        rule = TF_RULES[timeframe]["rule"]
        ohlc = df_ticks["price"].resample(rule, label="right", closed="right").agg(["first", "max", "min", "last"])
        if ohlc.empty:
            return

        # Aggregate volumes (sum for all volume types)
        vol = df_ticks["volume"].resample(rule, label="right", closed="right").sum() if "volume" in df_ticks.columns else None
        tick_vol = df_ticks["tick_volume"].resample(rule, label="right", closed="right").sum() if "tick_volume" in df_ticks.columns else None
        real_vol = df_ticks["real_volume"].resample(rule, label="right", closed="right").sum() if "real_volume" in df_ticks.columns else None

        # Track provider source (use most recent)
        provider = df_ticks["provider_source"].resample(rule, label="right", closed="right").last() if "provider_source" in df_ticks.columns else None

        candles = []
        for idx, row in ohlc.iterrows():
            if row.isnull().all(): continue
            candle = {
                "ts_utc": int(idx.timestamp() * 1000),
                "open": row["first"],
                "high": row["max"],
                "low": row["min"],
                "close": row["last"],
                "volume": vol.get(idx) if vol is not None else None,
                "tick_volume": tick_vol.get(idx) if tick_vol is not None else None,
                "real_volume": real_vol.get(idx) if real_vol is not None else None,
                "provider_source": provider.get(idx) if provider is not None else "tiingo",
            }
            candles.append(candle)

        if not candles:
            return

        df_candles = pd.DataFrame(candles)
        
        # Use transaction to ensure atomicity of upsert + state update
        try:
            report = data_io.upsert_candles(self.engine, df_candles, symbol, timeframe, resampled=(timeframe != '1m'))
            # logger.info(f"AggregatorService: Upserted {len(df_candles)} candles for {symbol} {timeframe} (report={report})")
            
            # Only update state if upsert succeeded
            # Thread-safe update of last processed timestamp
            with self._state_lock:
                self._last_processed_ts[state_key] = end_ms
        except Exception as e:
            logger.error(f"Failed to upsert candles for {symbol} {timeframe}: {e}")
            # Don't update state if upsert failed - will retry next iteration
            raise

    def _sleep_until_next_minute(self):
        """Sleep until the next minute boundary for time-aligned aggregation."""
        import time
        now = datetime.now(timezone.utc)
        to_sleep = 60 - now.second - (now.microsecond / 1_000_000)
        time.sleep(max(0, to_sleep))
