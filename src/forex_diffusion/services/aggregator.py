"""
Aggregator service: scheduled aggregation of market_data_ticks -> market_data_candles.

- Runs a background thread aligned to minute boundaries.
- Every minute aggregates ticks into 1m candles.
- On minute boundaries that are multiples of other TFs, also aggregate 5m,15m,30m,1h,5h,12h,1d,5d,10d,15d,30d.
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
    "5h": {"rule": "300min", "minutes": 300},
    "12h": {"rule": "720min", "minutes": 720},
    "1d": {"rule": "1D", "minutes": 1440},
    "5d": {"rule": "5D", "minutes": 1440 * 5},
    "10d": {"rule": "10D", "minutes": 1440 * 10},
    "15d": {"rule": "15D", "minutes": 1440 * 15},
    "30d": {"rule": "30D", "minutes": 1440 * 30},
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
                for tf, info in TF_RULES.items():
                    mins = info["minutes"]
                    if mins == 1 and not self.run_1m:
                        continue
                    if (minute_idx % (mins // 1)) == 0:
                        tfs_to_compute.append(tf)
                # default: always compute 1m
                if "1m" not in tfs_to_compute:
                    tfs_to_compute.insert(0, "1m")

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
        # end timestamp (exclusive) = ts_now in ms
        end_ms = int(pd.Timestamp(ts_now).value // 1_000_000)
        for tf in tfs:
            info = TF_RULES.get(tf)
            if info is None:
                continue
            mins = info["minutes"]
            # start is end - mins minutes
            start_ms = end_ms - (mins * 60 * 1000)
            # Fetch ticks in window [start_ms, end_ms)
            try:
                with self.engine.connect() as conn:
                    q = "SELECT ts_utc, price, volume FROM market_data_ticks WHERE symbol = :s AND ts_utc >= :a AND ts_utc < :b ORDER BY ts_utc ASC"
                    rows = conn.execute(pd.io.sql.text(q), {"s": symbol, "a": int(start_ms), "b": int(end_ms)}).fetchall()
            except Exception:
                # fallback using SQLAlchemy text import
                try:
                    from sqlalchemy import text as _text
                    with self.engine.connect() as conn:
                        rows = conn.execute(_text("SELECT ts_utc, price, volume FROM market_data_ticks WHERE symbol = :s AND ts_utc >= :a AND ts_utc < :b ORDER BY ts_utc ASC"), {"s": symbol, "a": int(start_ms), "b": int(end_ms)}).fetchall()
                except Exception as e:
                    logger.debug("AggregatorService: DB query for ticks failed: {}", e)
                    rows = []

            if not rows:
                # nothing to aggregate for this TF
                continue

            # Build DataFrame
            try:
                recs = []
                for r in rows:
                    # support Row._mapping or tuple
                    if hasattr(r, "_mapping"):
                        m = r._mapping
                        recs.append({"ts_utc": int(m.get("ts_utc")), "price": float(m.get("price")) if m.get("price") is not None else None, "volume": float(m.get("volume")) if m.get("volume") is not None else None})
                    else:
                        # tuple (ts,price,volume)
                        tsv = int(r[0])
                        pricev = r[1]
                        volv = r[2] if len(r) > 2 else None
                        recs.append({"ts_utc": int(tsv), "price": float(pricev) if pricev is not None else None, "volume": float(volv) if volv is not None else None})
                df_ticks = pd.DataFrame(recs)
                if df_ticks.empty:
                    continue
                df_ticks["ts_dt"] = pd.to_datetime(df_ticks["ts_utc"].astype("int64"), unit="ms", utc=True)
                df_ticks = df_ticks.set_index("ts_dt").sort_index()
                # resample by rule
                rule = info["rule"]
                # produce OHLCV from price series
                ohlc = df_ticks["price"].resample(rule, label="right", closed="right").agg(["first", "max", "min", "last"])
                if ohlc.empty:
                    continue
                vol = df_ticks["volume"].resample(rule, label="right", closed="right").sum() if "volume" in df_ticks.columns else None
                candles = []
                for idx, row in ohlc.iterrows():
                    if row.isnull().all():
                        continue
                    ts_ms = int(idx.tz_convert("UTC").timestamp() * 1000)
                    o = float(row["first"]) if not pd.isna(row["first"]) else 0.0
                    h = float(row["max"]) if not pd.isna(row["max"]) else o
                    l = float(row["min"]) if not pd.isna(row["min"]) else o
                    c = float(row["last"]) if not pd.isna(row["last"]) else o
                    v = float(vol.loc[idx]) if (vol is not None and idx in vol.index and not pd.isna(vol.loc[idx])) else None
                    candles.append({"ts_utc": ts_ms, "open": o, "high": h, "low": l, "close": c, "volume": v})
                if not candles:
                    continue
                df_candles = pd.DataFrame(candles)
                # resampled flag: True for aggregated from ticks (except for 1m maybe mark resampled False)
                resampled_flag = False if tf == "1m" else True
                rep = data_io.upsert_candles(self.engine, df_candles, symbol, tf, resampled=resampled_flag)
                logger.info("AggregatorService: upserted %d candles for %s %s (report=%s)", len(df_candles), symbol, tf, rep)
            except Exception as e:
                logger.exception("AggregatorService: failed to aggregate ticks->%s for %s: {}", tf, symbol, e)
                continue
