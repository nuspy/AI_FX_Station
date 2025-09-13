"""
Aggregator service: scheduled aggregation of market_data_ticks -> market_data_candles.

This service implements a stateful aggregation strategy to ensure data integrity and
prevent data loss, even in cases of processing delays or restarts.
"""
from __future__ import annotations

import threading
import time
from typing import List, Dict, Optional
from datetime import datetime, timezone

import pandas as pd
from loguru import logger
from sqlalchemy import text

from .db_service import DBService
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

class AggregatorService:
    """
    A stateful background aggregator that periodically converts ticks into candles.
    """
    def __init__(self, engine, symbols: List[str] | None = None):
        self.engine = engine
        self.db = DBService(engine=self.engine)
        self._symbols = symbols or []
        self._stop_event = threading.Event()
        self._thread = None
        self._last_processed_ts: Dict[tuple[str, str], int] = {}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("AggregatorService started (symbols=%s)", self._symbols or "<all>")

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("AggregatorService stopped")

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

    def _run_loop(self):
        self._sleep_until_next_minute()
        while not self._stop_event.is_set():
            try:
                ts_now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                minute_idx = int(ts_now.timestamp() / 60)

                symbols = self._symbols or self._get_symbols_from_config()
                for sym in symbols:
                    for tf, info in TF_RULES.items():
                        if minute_idx % info["minutes"] == 0:
                            self._aggregate_for_symbol(sym, tf, ts_now)

            except Exception as e:
                logger.exception(f"AggregatorService loop error: {e}")
            
            self._sleep_until_next_minute()

    def _aggregate_for_symbol(self, symbol: str, timeframe: str, ts_now: datetime):
        state_key = (symbol, timeframe)
        
        last_ts = self._last_processed_ts.get(state_key) or self._get_last_candle_ts(symbol, timeframe)
        if not last_ts:
            last_ts = int(ts_now.timestamp() * 1000) - (24 * 60 * 60 * 1000)

        start_ms = last_ts
        end_ms = int(ts_now.timestamp() * 1000)

        if start_ms >= end_ms:
            return

        with self.engine.connect() as conn:
            # Corrected query to fetch ticks with the right timeframe
            query = text(
                "SELECT ts_utc, price, bid, ask, volume FROM market_data_ticks "
                "WHERE symbol = :symbol AND timeframe = 'tick' AND ts_utc > :start AND ts_utc <= :end ORDER BY ts_utc ASC"
            )
            rows = conn.execute(query, {"symbol": symbol, "start": start_ms, "end": end_ms}).fetchall()

        if not rows:
            return

        logger.info(f"Found {len(rows)} new ticks for {symbol} to aggregate into {timeframe}.")
        df_ticks = pd.DataFrame(rows, columns=["ts_utc", "price", "bid", "ask", "volume"])
        df_ticks['price'] = df_ticks['price'].fillna((df_ticks['bid'] + df_ticks['ask']) / 2).ffill()
        if df_ticks.empty or df_ticks['price'].isnull().all():
            return

        df_ticks["ts_dt"] = pd.to_datetime(df_ticks["ts_utc"], unit="ms", utc=True)
        df_ticks = df_ticks.set_index("ts_dt").sort_index()

        rule = TF_RULES[timeframe]["rule"]
        ohlc = df_ticks["price"].resample(rule, label="right", closed="right").agg(["first", "max", "min", "last"])
        if ohlc.empty:
            return

        vol = df_ticks["volume"].resample(rule, label="right", closed="right").sum() if "volume" in df_ticks.columns else None
        candles = []
        for idx, row in ohlc.iterrows():
            if row.isnull().all(): continue
            candles.append({
                "ts_utc": int(idx.timestamp() * 1000),
                "open": row["first"], "high": row["max"], "low": row["min"], "close": row["last"],
                "volume": vol.get(idx) if vol is not None else None
            })

        if not candles:
            return

        df_candles = pd.DataFrame(candles)
        report = data_io.upsert_candles(self.engine, df_candles, symbol, timeframe, resampled=(timeframe != '1m'))
        logger.info(f"AggregatorService: Upserted {len(df_candles)} candles for {symbol} {timeframe} (report={report})")

        self._last_processed_ts[state_key] = end_ms

    def _get_symbols_from_config(self) -> List[str]:
        from ..utils.config import get_config
        cfg = get_config()
        return getattr(cfg.data, "symbols", [])

    def _sleep_until_next_minute(self):
        now = datetime.now(timezone.utc)
        to_sleep = 60 - now.second - (now.microsecond / 1_000_000)
        time.sleep(max(0, to_sleep))
