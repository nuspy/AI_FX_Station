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
        # provider symbol aliases (only for REST requests)
        #self._aliases: Dict[str, str] = {
        #    "AUX/USD": "XAU/USD",  # internal alias -> provider known pair
        #}

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
                # log request (do not print token)
                logger.info("HTTP GET {} params={} (attempt {}/{})", url, params, attempt, max_attempts)
                r = self._client.get(url, params=params, headers=headers)
                r.raise_for_status()
                # success log
                try:
                    size = len(r.content) if r.content is not None else 0
                    logger.info("HTTP GET {} -> {} ({} bytes)", url, r.status_code, size)
                except Exception:
                    pass
                # Tiingo returns JSON array for prices endpoint
                return r.json()
            except Exception as e:
                # enrich error with status/text if available
                status = None
                body_snip = None
                try:
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        status = getattr(resp, "status_code", None)
                        txt = getattr(resp, "text", "")
                        body_snip = (txt[:200] + ("..." if len(txt) > 200 else "")) if isinstance(txt, str) else None
                except Exception:
                    pass
                if status is not None or body_snip is not None:
                    logger.warning("HTTP GET {} failed (status={}): {}", url, status, body_snip)
                logger.warning("Tiingo request attempt {} failed: {}", attempt, e)
                if attempt >= max_attempts:
                    logger.error("Tiingo request failed after {} attempts: {} {}", attempt, url, params)
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
        # provider alias mapping (keep DB symbol unchanged)
        provider_symbol = self._aliases.get(symbol, symbol)
        if provider_symbol != symbol:
            logger.info("Provider symbol alias: '{}' -> '{}'", symbol, provider_symbol)
        # symbol format expected by tiingo e.g. "EURUSD"
        from_sym, to_sym = parse_symbol(provider_symbol)
        ticker = f"{from_sym}{to_sym}"
        url = f"{self.BASE_URL}/{ticker}/prices"
        params = {"startDate": start_date, "endDate": end_date, "resampleFreq": resample_freq, "format": fmt}
        data = self._get_json_with_retry(url, params=params, max_attempts=50)
        if not data:
            logger.warning("Tiingo returned empty for {} {}-{} (resample={})", provider_symbol, start_date, end_date, resample_freq)
            return pd.DataFrame()
        # Data expected as list of dicts with 'date','open','high','low','close'
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame()
            # date -> datetime (tiingo returns ISO with timezone)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            # pandas 2.x: cast a int64 (ns), poi ms
            df["ts_utc"] = (df["date"].astype("int64") // 1_000_000).astype("int64")
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
        # Note: "60m" is an alias of "1h" -> keep only "1h" to avoid duplicates
        self.timeframes_priority = ["tick", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            # REST backfill guard: disabled by default (only enabled explicitly by UI backfill)
        self.rest_enabled: bool = False

    def backfill_symbol_timeframe(self, symbol: str, timeframe: str, force_full: bool = False, progress_cb: Optional[callable] = None, start_ms_override: Optional[int] = None):
        """
        Backfill implementation replaced:
        - For the requested timeframe we will detect missing intervals since last candle (or from epoch if force_full)
        - We download ticks first (1min fallback), then progressively higher timeframes up to 1 day.
        - Each timeframe detects its own missing gaps and requests only those (avoiding weekend).
        - If progress_cb provided, emits determinate progress 0..100 based on number of subranges processed.
        - If start_ms_override provided, compute gaps in [start_ms_override, now] (range mode).
        """
        logger.info("Starting backfill for {} {} (override={}, force_full={})", symbol, timeframe, start_ms_override, force_full)
        # Guard: block any REST backfill unless explicitly enabled (UI backfill button)
        if not getattr(self, "rest_enabled", False):
            logger.info("REST backfill disabled; skipping backfill for {} {}.", symbol, timeframe)
            if progress_cb:
                try: progress_cb(100)
                except Exception: pass
            return
        # overall period
        last_ts, now_ms = self._get_last_candle_ts(symbol, timeframe)
        try:
            if start_ms_override is not None:
                start_iso = time_utils.ms_to_utc_dt(int(start_ms_override)).isoformat()
                now_iso = time_utils.ms_to_utc_dt(int(now_ms)).isoformat()
                logger.info("Backfill effective range (override): {} -> {}", start_iso, now_iso)
        except Exception:
            pass
        if start_ms_override is not None:
            start_ms = int(start_ms_override)
        elif force_full or last_ts is None:
            start_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=365 * 30)).timestamp() * 1000)
        else:
            start_ms = int(last_ts) + 1

        if start_ms >= now_ms:
            logger.info("No backfill required: start >= now.")
            if progress_cb: 
                try: progress_cb(100)
                except Exception: pass
            return

        # Normalize alias (e.g., "60m" -> "1h") and determine timeframes to process (skip 'tick')
        tf_req = (timeframe or "").strip().lower()
        alias_map = {"60m": "1h"}
        tf_norm = alias_map.get(tf_req, tf_req)
        if tf_norm in self.timeframes_priority:
            idx = self.timeframes_priority.index(tf_norm)
            tfs_to_process = self.timeframes_priority[1: idx + 1]
        else:
            # fallback default chain (no "60m" to avoid duplicate of "1h")
            tfs_to_process = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        # ensure uniqueness while preserving order
        _seen = set()
        tfs_to_process = [t for t in tfs_to_process if not (t in _seen or _seen.add(t))]
        # if request was an alias, log normalization
        if tf_req != tf_norm:
            logger.info("Normalized timeframe alias: '{}' -> '{}'", tf_req, tf_norm)

        # Pre-compute total subranges for determinate progress (guard against huge ranges)
        total_subranges = 0
        try:
            range_days = (now_ms - start_ms) / (24 * 3600 * 1000)
            if range_days <= 370:  # avoid huge O(N) precompute for multi-year 1m ranges
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
            else:
                logger.warning("Backfill range is large (%.0f days); skipping progress precompute for speed.", range_days)
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
                        try:
                            # drop NaN and duplicates on ts_utc before upsert
                            df = df.dropna(subset=["ts_utc"]).copy()
                            before = len(df)
                            df = df.drop_duplicates(subset=["ts_utc"], keep="last")
                            if len(df) != before:
                                logger.info("Dedup 1m: {} -> {} rows for {} {}-{}", before, len(df), symbol, start_date, end_date)
                        except Exception:
                            pass
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
                            try:
                                df = df.dropna(subset=["ts_utc"]).copy()
                                before = len(df)
                                df = df.drop_duplicates(subset=["ts_utc"], keep="last")
                                if len(df) != before:
                                    logger.info("Dedup {}: {} -> {} rows for {} {}-{}", tf, before, len(df), symbol, start_date, end_date)
                            except Exception:
                                pass
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
            # pandas 2.x: usa timestamp() per ottenere secondi, poi ms
            try:
                e_ms = int(e_dt_plus.timestamp() * 1000)
            except Exception:
                # fallback robusto
                e_ms = int(pd.to_datetime(e_dt_plus).value // 1_000_000)
            out_ranges.append((int(s), e_ms))
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

    def _get_first_candle_ts(self, symbol: str) -> Optional[int]:
        """
        Return the earliest ts_utc across all timeframes for given symbol, or None if no candles exist.
        """
        try:
            with self.engine.connect() as conn:
                from sqlalchemy import text
                query = text("SELECT MIN(ts_utc) FROM market_data_candles WHERE symbol = :symbol")
                first_ts = conn.execute(query, {"symbol": symbol}).scalar_one_or_none()
                return int(first_ts) if first_ts is not None else None
        except Exception as e:
            logger.exception("Failed to get first candle timestamp: %s", e)
            return None
