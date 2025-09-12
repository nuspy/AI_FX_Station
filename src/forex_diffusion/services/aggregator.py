"""
Aggregator service: scheduled aggregation of market_data_ticks -> market_data_candles.

- Runs a background thread aligned to minute boundaries.
- Every minute aggregates ticks into 1m candles.
- On minute boundaries that are multiples of other TFs, also aggregate 5m,15m,30m,1h,4h,1d.
- Uses DBService to access engine and data_io.upsert_candles to persist candles.
"""
from __future__ import annotations

import threading
import time
from typing import List, Dict
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from .db_service import DBService
from ..data import io as data_io


# mapping of logical TF label -> pandas resample rule and minutes
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
    Background aggregator that periodically converts ticks into candles.

    Usage:
      ag = AggregatorService(engine=DBService().engine, symbols=["EUR/USD"])
      ag.start()
      ...
      ag.stop()
    """

    def __init__(self, engine, symbols: List[str] | None = None, run_1m: bool = True):
        self.engine = engine
        self.db = DBService(engine=self.engine)
        self._symbols = symbols or []
        self._stop_event = threading.Event()
        self._thread = None
        self.run_1m = run_1m

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("AggregatorService started (symbols=%s)", self._symbols or "<all>")

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("AggregatorService stopped")

    def set_symbols(self, symbols: List[str]):
        self._symbols = symbols

    def _sleep_until_next_minute(self):
        # Sleep until next minute boundary (aligned to UTC)
        now = datetime.now(timezone.utc)
        next_min = (now.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)).timestamp()
        to_sleep = max(0.0, next_min - now.timestamp())
        time.sleep(to_sleep)

    def _run_loop(self):
        # initial alignment: wait until next minute boundary
        self._sleep_until_next_minute()
        while not self._stop_event.is_set():
            try:
                ts_now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                # compute minute index (minutes since epoch) for alignment
                minute_idx = int(pd.Timestamp(ts_now).value // 1_000_000 // 60)
                # decide which TFs to compute now
                tfs_to_compute = []
                if self.run_1m:
                    tfs_to_compute.append("1m")
                for tf, info in TF_RULES.items():
                    if tf == "1m": continue
                    mins = info["minutes"]
                    if (minute_idx % mins) == 0:
                        tfs_to_compute.append(tf)

                if not tfs_to_compute:
                    continue

                symbols = self._symbols or self._get_symbols_from_config()
                for sym in symbols:
                    try:
                        self._aggregate_for_symbol(sym, tfs_to_compute, ts_now)
                    except Exception as e:
                        logger.exception("AggregatorService: aggregation failed for %s: {}", sym, e)
            except Exception as e:
                logger.exception("AggregatorService loop error: {}", e)
            # sleep until next minute boundary
            self._sleep_until_next_minute()

    def _get_symbols_from_config(self) -> List[str]:
        try:
            from ..utils.config import get_config
            cfg = get_config()
            data_cfg = getattr(cfg, "data", None) or {}
            if isinstance(data_cfg, dict):
                syms = data_cfg.get("symbols", []) or []
            else:
                syms = getattr(data_cfg, "symbols", []) or []
            return syms or []
        except Exception:
            return []

    def _aggregate_for_symbol(self, symbol: str, tfs: List[str], ts_now: datetime):
        """
        For given symbol and list of timeframes, collect ticks for each TF window and persist candles.
        ts_now is aligned to minute boundary (UTC).
        """
        end_ms = int(pd.Timestamp(ts_now).value // 1_000_000)
        max_mins = max(TF_RULES[tf]["minutes"] for tf in tfs)
        start_ms = end_ms - (max_mins * 60 * 1000)

        try:
            from sqlalchemy import text as _text
            with self.engine.connect() as conn:
                q = _text("SELECT ts_utc, price, bid, ask, volume FROM market_data_ticks WHERE symbol = :s AND ts_utc >= :a AND ts_utc < :b ORDER BY ts_utc ASC")
                rows = conn.execute(q, {"s": symbol, "a": int(start_ms), "b": int(end_ms)}).fetchall()
        except Exception as e:
            logger.debug("AggregatorService: DB query for ticks failed: {}", e)
            rows = []

        if not rows:
            return

        df_ticks = pd.DataFrame(rows, columns=["ts_utc", "price", "bid", "ask", "volume"])
        df_ticks['price'] = df_ticks['price'].fillna((df_ticks['bid'] + df_ticks['ask']) / 2).ffill()
        if df_ticks.empty or df_ticks['price'].isnull().all():
            return

        df_ticks["ts_dt"] = pd.to_datetime(df_ticks["ts_utc"].astype("int64"), unit="ms", utc=True)
        df_ticks = df_ticks.set_index("ts_dt").sort_index()

        for tf in tfs:
            try:
                rule = TF_RULES[tf]["rule"]
                ohlc = df_ticks["price"].resample(rule, label="right", closed="right").agg(["first", "max", "min", "last"])
                if ohlc.empty:
                    continue
                
                vol = df_ticks["volume"].resample(rule, label="right", closed="right").sum() if "volume" in df_ticks.columns and not df_ticks["volume"].isnull().all() else None
                
                candles = []
                for idx, row in ohlc.iterrows():
                    if row.isnull().all():
                        continue
                    ts_ms = int(idx.tz_convert("UTC").timestamp() * 1000)
                    o, h, l, c = row["first"], row["max"], row["min"], row["last"]
                    v = float(vol.loc[idx]) if (vol is not None and idx in vol.index and not pd.isna(vol.loc[idx])) else None
                    candles.append({"ts_utc": ts_ms, "open": o, "high": h, "low": l, "close": c, "volume": v})
                
                if not candles:
                    continue

                df_candles = pd.DataFrame(candles)
                resampled_flag = False if tf == "1m" else True
                rep = data_io.upsert_candles(self.engine, df_candles, symbol, tf, resampled=resampled_flag)
                logger.info("AggregatorService: upserted %d candles for %s %s (report=%s)", len(df_candles), symbol, tf, rep)
            except Exception as e:
                logger.exception("AggregatorService: failed to aggregate ticks->%s for %s: {}", tf, symbol, e)
                continue
