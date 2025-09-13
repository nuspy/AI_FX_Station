"""
Market data providers and MarketDataService.

- TiingoClient: client bridge to Tiingo REST endpoints (candles & intraday).
- MarketDataService: new backfill strategy:
  - download ticks (1m fallback) first, then 1m..24h
  - avoid weekend (Fri 22:00 -> Sun 22:00 UTC)
  - request only missing intervals per timeframe (each timeframe autonomous)
  - upsert candles into market_data_candles
"""

from __future__ import annotations

import os
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict

import httpx
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

from ..utils.config import get_config
from ..utils import time_utils
from ..data import io as data_io

def parse_symbol(symbol: str) -> Tuple[str, str]:
    """
    Parse a symbol like "EUR/USD" into ("EUR", "USD").
    """
    if "/" in symbol:
        a, b = symbol.split("/")
        return a.strip(), b.strip()
    raise ValueError(f"Unsupported symbol format: {symbol}")

class TiingoClient:
    """
    Minimal Tiingo client for FX candles usage.
    Implements robust retry with Fibonacci backoff up to a large number of attempts.
    """
    BASE_URL = "https://api.tiingo.com/tiingo/fx"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        cfg = get_config()
        ti_cfg = getattr(getattr(cfg, "providers", None), "tiingo", None)
        self.api_key = api_key or (getattr(ti_cfg, "key", None) if ti_cfg else None) or os.environ.get("TIINGO_API_KEY")
        if not self.api_key:
            logger.warning("Tiingo API key not configured; requests may fail.")
        self._client = httpx.Client(timeout=timeout)

    def _fib_waits(self, max_attempts: int):
        a, b = 1, 1
        for _ in range(max_attempts):
            yield a
            a, b = b, a + b

    def _get_json_with_retry(self, url: str, params: dict, max_attempts: int = 50) -> Optional[dict]:
        attempt = 0
        for wait_s in self._fib_waits(max_attempts):
            attempt += 1
            try:
                headers = {"Authorization": f"Token {self.api_key}"} if self.api_key else {}
                r = self._client.get(url, params=params, headers=headers)
                r.raise_for_status()
                # Tiingo returns JSON array for prices endpoint
                return r.json()
            except Exception as e:
                logger.warning("Tiingo request attempt %d failed: %s", attempt, e)
                if attempt >= max_attempts:
                    logger.error("Tiingo request failed after %d attempts: %s %s", attempt, url, params)
                    return None
                # fibonacci backoff in seconds (cap at 1800s to avoid extremely long sleeps)
                sleep_s = min(wait_s, 1800)
                time.sleep(sleep_s)
        return None

    def get_candles(self, symbol: str, start_date: str, end_date: str, resample_freq: str = "1min", fmt: str = "json") -> pd.DataFrame:
        """
        Query Tiingo FX prices for symbol in date range [start_date, end_date] (YYYY-MM-DD).
        resample_freq examples: '1min','5min','15min','1hour','1day'
        Returns DataFrame with ts_utc (ms), open, high, low, close, (maybe volume)
        """
        # symbol format expected by tiingo e.g. "EURUSD"
        from_sym, to_sym = parse_symbol(symbol)
        ticker = f"{from_sym}{to_sym}"
        url = f"{self.BASE_URL}/{ticker}/prices"
        params = {"startDate": start_date, "endDate": end_date, "resampleFreq": resample_freq, "format": fmt}
        data = self._get_json_with_retry(url, params=params, max_attempts=50)
        if not data:
            return pd.DataFrame()
        # Data expected as list of dicts with 'date','open','high','low','close'
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame()
            # date -> datetime (tiingo returns ISO with timezone)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["ts_utc"] = (df["date"].view("int64") // 1_000_000).astype("int64")
            # ensure columns present
            for c in ["open", "high", "low", "close"]:
                if c not in df.columns:
                    df[c] = None
            cols = ["ts_utc", "open", "high", "low", "close"]
            if "volume" in df.columns:
                cols.append("volume")
            return df[cols].copy()
        except Exception as e:
            logger.exception("Failed to parse Tiingo response: %s", e)
            return pd.DataFrame()

    def get_ticks(self, symbol: str, start_date: str, end_date: str, fmt: str = "json") -> pd.DataFrame:
        """
        Fallback tick fetch: Tiingo's FX endpoint doesn't expose per-tick in same API.
        We fallback to requesting 1min data and treat as ticks for the backfill strategy.
        """
        return self.get_candles(symbol, start_date=start_date, end_date=end_date, resample_freq="1min", fmt=fmt)


class MarketDataService:
    """
    High-level service to orchestrate data acquisition and DB ingest following new strategy.
    """
    def __init__(self, database_url: Optional[str] = None):
        self.cfg = get_config()
        db_url = database_url or getattr(self.cfg.db, "database_url", None)
        if not db_url:
            raise ValueError("Database URL not configured")
        self.engine = create_engine(db_url, future=True)
        self.provider = TiingoClient()  # Default provider: Tiingo
        # Default timeframes to fetch after ticks: 1m up to 1d (24h)
        self.timeframes_priority = ["tick", "1m", "5m", "15m", "30m", "60m", "1h", "4h", "1d"]

    def backfill_symbol_timeframe(self, symbol: str, timeframe: str, force_full: bool = False, progress_cb: Optional[callable] = None):
        """
        Backfill implementation replaced:
        - For the requested timeframe we will detect missing intervals since last candle (or from epoch if force_full)
        - We download ticks first (1min fallback), then progressively higher timeframes up to 1 day.
        - Each timeframe detects its own missing gaps and requests only those (avoiding weekend).
        - If progress_cb provided, emits determinate progress 0..100 based on number of subranges processed.
        """
        logger.info("Starting backfill for %s %s", symbol, timeframe)
        # overall period
        last_ts, now_ms = self._get_last_candle_ts(symbol, timeframe)
        if force_full or last_ts is None:
            start_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=365 * 30)).timestamp() * 1000)
        else:
            start_ms = int(last_ts) + 1

        if start_ms >= now_ms:
            logger.info("No backfill required: start >= now.")
            if progress_cb: 
                try: progress_cb(100)
                except Exception: pass
            return

        # Determine timeframes to process (skip 'tick')
        if timeframe in self.timeframes_priority:
            idx = self.timeframes_priority.index(timeframe)
            tfs_to_process = self.timeframes_priority[1: idx + 1]
        else:
            tfs_to_process = ["1m", "5m", "15m", "30m", "60m", "1h", "4h", "1d"]

        # Pre-compute total subranges for determinate progress
        total_subranges = 0
        try:
            # 1m ranges (ticks fallback)
            r_1m = self._find_missing_intervals(symbol, "1m", start_ms, now_ms)
            for (s_ms, e_ms) in r_1m:
                total_subranges += max(1, len(time_utils.split_range_avoid_weekend(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))))
            # other TFs
            for tf in tfs_to_process:
                if tf == "1m":
                    continue
                r_tf = self._find_missing_intervals(symbol, tf, start_ms, now_ms)
                for (s_ms, e_ms) in r_tf:
                    total_subranges += max(1, len(time_utils.split_range_avoid_weekend(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))))
        except Exception:
            total_subranges = 0
        processed = 0
        def _emit_progress():
            if progress_cb and total_subranges > 0:
                try:
                    pct = int(min(100, max(0, (processed / total_subranges) * 100)))
                    progress_cb(pct)
                except Exception:
                    pass

        # 1) Ensure ticks/1m
        try:
            tf = "1m"
            missing_ranges = self._find_missing_intervals(symbol, tf, start_ms, now_ms)
            for (s_ms, e_ms) in missing_ranges:
                subranges = time_utils.split_range_avoid_weekend(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))
                for (sub_s, sub_e) in subranges:
                    start_date = sub_s.date().isoformat()
                    end_date = sub_e.date().isoformat()
                    logger.info("Requesting ticks(1m) %s - %s for %s", start_date, end_date, symbol)
                    df = self.provider.get_ticks(symbol, start_date=start_date, end_date=end_date)
                    if df is not None and not df.empty:
                        report = data_io.upsert_candles(self.engine, df, symbol, "1m")
                        logger.info("Upsert 1m report: %s", report)
                    processed += 1
                    _emit_progress()
        except Exception as e:
            logger.exception("Tick fetch/aggregation failed: %s", e)

        # 2) Process other TFs autonomously
        for tf in tfs_to_process:
            if tf == "1m":
                continue
            try:
                missing_ranges = self._find_missing_intervals(symbol, tf, start_ms, now_ms)
                for (s_ms, e_ms) in missing_ranges:
                    subranges = time_utils.split_range_avoid_weekend(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))
                    for (sub_s, sub_e) in subranges:
                        start_date = sub_s.date().isoformat()
                        end_date = sub_e.date().isoformat()
                        resample = self._tf_to_tiingo_resample(tf)
                        logger.info("Requesting %s candles %s - %s for %s", tf, start_date, end_date, symbol)
                        df = self.provider.get_candles(symbol, start_date=start_date, end_date=end_date, resample_freq=resample)
                        if df is not None and not df.empty:
                            report = data_io.upsert_candles(self.engine, df, symbol, tf)
                            logger.info("Upsert report for %s %s: %s", symbol, tf, report)
                        processed += 1
                        _emit_progress()
            except Exception as e:
                logger.exception("Backfill failed for %s %s: %s", symbol, tf, e)

        if progress_cb:
            try: progress_cb(100)
            except Exception: pass

    def _ensure_ticks_then_aggregate(self, symbol: str, start_ms: int, end_ms: int):
        """
        Download ticks (fallback to 1min) for date ranges missing and upsert them as 1m candles.
        The aggregator (existing elsewhere) may produce other TFs from ticks in realtime, but for historic backfill
        we upsert the 1m candles directly so higher TFs can be resampled or fetched separately.
        """
        # Determine missing 1m candle ranges
        tf = "1m"
        missing_ranges = self._find_missing_intervals(symbol, tf, start_ms, end_ms)
        if not missing_ranges:
            logger.info("No missing 1m ranges detected for %s", symbol)
            return
        for (s_ms, e_ms) in missing_ranges:
            # split avoiding weekend
            s_dt = time_utils.ms_to_utc_dt(s_ms)
            e_dt = time_utils.ms_to_utc_dt(e_ms)
            subranges = time_utils.split_range_avoid_weekend(s_dt, e_dt)
            for (sub_s, sub_e) in subranges:
                start_date = sub_s.date().isoformat()
                end_date = sub_e.date().isoformat()
                logger.info("Requesting ticks(1m) %s - %s for %s", start_date, end_date, symbol)
                df = self.provider.get_ticks(symbol, start_date=start_date, end_date=end_date)
                if df is None or df.empty:
                    logger.info("No tick/1m data returned for %s %s-%s", symbol, start_date, end_date)
                    continue
                # Upsert as 1m timeframe
                report = data_io.upsert_candles(self.engine, df, symbol, "1m")
                logger.info("Upsert 1m report: %s", report)

    def _backfill_timeframe_autonomous(self, symbol: str, timeframe: str, global_start_ms: int, global_end_ms: int):
        """
        For a given timeframe compute missing expected period ends between global_start_ms and global_end_ms,
        split into ranges avoiding weekend and request only those ranges from Tiingo.
        """
        logger.info("Backfilling autonomous timeframe %s for %s", timeframe, symbol)
        # find missing intervals (list of (s_ms,e_ms)) where series lacks bars
        missing_ranges = self._find_missing_intervals(symbol, timeframe, global_start_ms, global_end_ms)
        if not missing_ranges:
            logger.info("No missing ranges for %s %s", symbol, timeframe)
            return
        # For each missing range, split by weekend and request Tiingo candles per subrange
        for (s_ms, e_ms) in missing_ranges:
            s_dt = time_utils.ms_to_utc_dt(s_ms)
            e_dt = time_utils.ms_to_utc_dt(e_ms)
            subranges = time_utils.split_range_avoid_weekend(s_dt, e_dt)
            for (sub_s, sub_e) in subranges:
                start_date = sub_s.date().isoformat()
                end_date = sub_e.date().isoformat()
                # map timeframe to tiingo resample string
                resample = self._tf_to_tiingo_resample(timeframe)
                logger.info("Requesting %s candles %s - %s for %s", timeframe, start_date, end_date, symbol)
                df = self.provider.get_candles(symbol, start_date=start_date, end_date=end_date, resample_freq=resample)
                if df is None or df.empty:
                    logger.info("Tiingo returned no data for %s %s %s", symbol, timeframe, (start_date, end_date))
                    continue
                # Upsert into DB
                report = data_io.upsert_candles(self.engine, df, symbol, timeframe)
                logger.info("Upsert report for %s %s: %s", symbol, timeframe, report)

    def _tf_to_tiingo_resample(self, tf: str) -> str:
        """
        Convert timeframe like '1m','5m','1h','1d' into Tiingo resampleFreq string: '1min','5min','1hour','1day'
        """
        tf = tf.strip().lower()
        if tf == "tick":
            return "1min"
        if tf.endswith("m"):
            return f"{int(tf[:-1])}min"
        if tf.endswith("h"):
            return f"{int(tf[:-1])}hour"
        if tf.endswith("d"):
            return f"{int(tf[:-1])}day"
        # fallback
        return "1min"

    def _find_missing_intervals(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> List[Tuple[int, int]]:
        """
        Determine contiguous ranges [s_ms,e_ms] where expected period-end timestamps for timeframe are missing in DB.
        Returns list of start_ms,end_ms pairs in milliseconds (UTC). Each pair should be inclusive bounds for requesting.
        """
        if timeframe == "tick":
            # ticks handled via 1m fallback aggregator; return a single range
            return [(start_ms, end_ms)]
        # generate expected period ends for given timeframe between start and end
        expected = time_utils.generate_expected_period_ends(start_ms, end_ms, timeframe)
        if not expected:
            return []
        # fetch existing timestamps from DB within [start_ms, end_ms]
        try:
            with self.engine.connect() as conn:
                from sqlalchemy import text
                q = text("SELECT ts_utc FROM market_data_candles WHERE symbol = :symbol AND timeframe = :timeframe AND ts_utc BETWEEN :s AND :e")
                rows = conn.execute(q, {"symbol": symbol, "timeframe": timeframe, "s": start_ms, "e": end_ms}).fetchall()
                existing = set(int(r[0]) for r in rows if r and r[0] is not None)
        except Exception as e:
            logger.exception("Failed to query existing candles for gap detection: %s", e)
            existing = set()

        # compute missing expected points (ints)
        expected_set = set(expected)
        missing_points = sorted(list(expected_set - existing))
        if not missing_points:
            return []

        # group consecutive expected timestamps into contiguous runs
        runs: List[List[int]] = []
        run = [missing_points[0]]
        for t in missing_points[1:]:
            # previous expected step: use pandas freq to compute delta
            prev = run[-1]
            # compute expected delta based on timeframe
            # convert both to pandas datetimes to compare
            prev_dt = pd.to_datetime(prev, unit="ms", utc=True)
            t_dt = pd.to_datetime(t, unit="ms", utc=True)
            delta = t_dt - prev_dt
            # if delta equals freq then continue run
            freq = time_utils.tf_to_pandas_freq(timeframe)
            try:
                expected_delta = pd.Timedelta(freq)
            except Exception:
                expected_delta = delta
            if delta <= expected_delta + pd.Timedelta(seconds=1):
                run.append(t)
            else:
                runs.append(run)
                run = [t]
        if run:
            runs.append(run)

        # convert runs to [start_ms,end_ms] inclusive for requesting (add a small margin)
        out_ranges: List[Tuple[int, int]] = []
        for r in runs:
            s = r[0]
            e = r[-1]
            # expand e by one period to ensure provider returns inclusive last bar
            try:
                freq = time_utils.tf_to_pandas_freq(timeframe)
                e_dt = pd.to_datetime(e, unit="ms", utc=True)
                e_dt_plus = e_dt + pd.Timedelta(freq)
            except Exception:
                e_dt_plus = pd.to_datetime(e, unit="ms", utc=True)
            out_ranges.append((int(s), int(e_dt_plus.view("int64") // 1_000_000)))
        return out_ranges

    def _get_last_candle_ts(self, symbol: str, timeframe: str) -> Tuple[Optional[int], int]:
        now_ms = int(time.time() * 1000)
        try:
            with self.engine.connect() as conn:
                from sqlalchemy import text
                query = text("SELECT MAX(ts_utc) FROM market_data_candles WHERE symbol = :symbol AND timeframe = :timeframe")
                last_ts = conn.execute(query, {"symbol": symbol, "timeframe": timeframe}).scalar_one_or_none()
                return (int(last_ts) if last_ts else None, now_ms)
        except Exception as e:
            logger.exception("Failed to get last candle timestamp: %s", e)
            return (None, now_ms)
