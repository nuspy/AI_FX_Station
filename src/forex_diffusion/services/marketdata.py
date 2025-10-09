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
        # always initialize to avoid AttributeError in get_candles
        self._aliases: Dict[str, str] = {
            "AUX/USD": "XAU/USD",  # internal alias -> provider known pair
        }

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
                # logger.info("HTTP GET {} params={} (attempt {}/{})", url, params, attempt, max_attempts)
                r = self._client.get(url, params=params, headers=headers)
                r.raise_for_status()
                # success log
                try:
                    size = len(r.content) if r.content is not None else 0
                    # logger.info("HTTP GET {} -> {} ({} bytes)", url, r.status_code, size)
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
        aliases = getattr(self, "_aliases", {}) or {}
        provider_symbol = aliases.get(symbol, symbol)
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
            logger.exception("Failed to parse Tiingo response: {}", e)
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
    Now respects primary_data_provider setting with automatic fallback.
    """
    def __init__(self, database_url: Optional[str] = None):
        self.cfg = get_config()
        db_url = database_url or getattr(self.cfg.db, "database_url", None)
        if not db_url:
            raise ValueError("Database URL not configured")
        self.engine = create_engine(db_url, future=True)

        # Initialize provider based on settings (with fallback to Tiingo)
        self._init_provider()

        # Default timeframes to fetch after ticks: 1m up to 1w (week)
        # Note: "60m" is an alias of "1h" -> keep only "1h" to avoid duplicates
        self.timeframes_priority = ["tick", "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        # REST backfill guard: disabled by default (only enabled explicitly by UI backfill)
        self.rest_enabled: bool = True

    def _init_provider(self):
        """Initialize provider based on primary_data_provider setting with fallback."""
        self.fallback_occurred = False
        self.fallback_reason = None
        self.requested_provider = None

        try:
            from ..utils.user_settings import get_setting
            primary_provider = get_setting("primary_data_provider", "tiingo").lower()
            self.requested_provider = primary_provider
        except Exception as e:
            logger.warning(f"Could not load provider settings: {e}, using Tiingo")
            primary_provider = "tiingo"
            self.requested_provider = "tiingo"

        # Try to create the primary provider
        try:
            # TIINGO DISABLED FOR TESTING - ONLY CTRADER ALLOWED
            # if primary_provider == "tiingo":
            #     self.provider = TiingoClient()
            #     self.provider_name = "Tiingo"
            #     logger.info("MarketDataService using Tiingo provider")

            if primary_provider == "ctrader":
                # Initialize cTrader client (NO FALLBACK - will raise error if fails)
                from .ctrader_client import CTraderClient
                logger.info("Initializing cTrader client (NO FALLBACK)...")
                client = CTraderClient()
                # Force initialization to catch errors immediately (lazy init)
                client._ensure_initialized()
                self.provider = client
                self.provider_name = "cTrader"
                logger.info("MarketDataService using cTrader provider (NO FALLBACK)")
                # NO try/except - errors will propagate and crash the app
                # This is intentional for testing cTrader standalone

            elif primary_provider == "tiingo":
                raise ValueError(
                    "Tiingo provider is DISABLED for testing. "
                    "Only 'ctrader' is supported. "
                    "Set primary_data_provider='ctrader' in Settings."
                )

            else:
                raise ValueError(
                    f"Unknown provider '{primary_provider}'. "
                    f"Only 'ctrader' is supported (Tiingo disabled for testing). "
                    f"Set primary_data_provider='ctrader' in Settings."
                )

        except Exception as e:
            # NO FALLBACK - let errors propagate
            logger.error(f"FATAL: Failed to initialize provider '{primary_provider}': {e}")
            logger.error("NO FALLBACK AVAILABLE - Tiingo is disabled for testing")
            self.fallback_occurred = True
            self.fallback_reason = f"Provider initialization error: {str(e)}. NO FALLBACK (Tiingo disabled)."
            raise  # Re-raise to crash the app

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
        if not getattr(self, "rest_enabled", True):
            logger.info("REST backfill disabled; skipping backfill for {} {}.", symbol, timeframe)
            if progress_cb:
                try: progress_cb(100)
                except Exception: pass
            return
        # overall period
        last_ts, now_ms = self._get_last_candle_ts(symbol, timeframe)

        # BOOTSTRAP: If timeframe is completely empty (last_ts is None), download first month automatically
        if last_ts is None and start_ms_override is None and not force_full:
            logger.info("Timeframe {} {} is empty - bootstrapping with 1 month of historical data", symbol, timeframe)
            start_ms_override = int((datetime.now(tz=timezone.utc) - timedelta(days=30)).timestamp() * 1000)

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

        # Normalize alias (e.g., "60m" -> "1h")
        tf_req = (timeframe or "").strip().lower()
        alias_map = {"60m": "1h"}
        tf_norm = alias_map.get(tf_req, tf_req)

        # NEW LOGIC: Only process the requested timeframe (no cascading)
        # Each timeframe is independent and fetched directly from API
        tfs_to_process = [tf_norm]

        # if request was an alias, log normalization
        if tf_req != tf_norm:
            logger.info("Normalized timeframe alias: '{}' -> '{}'", tf_req, tf_norm)

        # Pre-compute total subranges for determinate progress (using monthly splits for large ranges)
        total_subranges = 0
        try:
            range_days = (now_ms - start_ms) / (24 * 3600 * 1000)
            # For large ranges (>90 days), use monthly splits; otherwise single request per gap
            use_monthly_split = range_days > 90

            if range_days <= 370:  # avoid huge O(N) precompute for multi-year 1m ranges
                # Only process the requested timeframe (no cascading)
                for tf in tfs_to_process:
                    r_tf = self._find_missing_intervals(symbol, tf, start_ms, now_ms)
                    for (s_ms, e_ms) in r_tf:
                        if use_monthly_split:
                            total_subranges += len(time_utils.split_range_by_month(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms)))
                        else:
                            total_subranges += 1
            else:
                logger.warning("Backfill range is large ({:.0f} days); skipping progress precompute for speed.", range_days)
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

        # Determine if we need monthly splits (only for large historical backfills >90 days)
        range_days = (now_ms - start_ms) / (24 * 3600 * 1000)
        use_monthly_split = range_days > 90

        # Process ONLY the requested timeframe (no cascading to other TFs)
        for tf in tfs_to_process:
            try:
                # Special handling for 1m: use get_ticks API
                if tf == "1m":
                    missing_ranges = self._find_missing_intervals(symbol, tf, start_ms, now_ms)
                    for (s_ms, e_ms) in missing_ranges:
                        if use_monthly_split:
                            subranges = time_utils.split_range_by_month(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))
                        else:
                            subranges = [(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))]

                        for (sub_s, sub_e) in subranges:
                            start_date = sub_s.date().isoformat()
                            end_date = sub_e.date().isoformat()
                            logger.info("Requesting 1m candles {} - {} for {}", start_date, end_date, symbol)
                            df = self.provider.get_ticks(symbol, start_date=start_date, end_date=end_date)
                            if df is not None and not df.empty:
                                try:
                                    df = df.dropna(subset=["ts_utc"]).copy()
                                    before = len(df)
                                    df = df.drop_duplicates(subset=["ts_utc"], keep="last")
                                    if len(df) != before:
                                        logger.info("Dedup 1m: {} -> {} rows for {} {}-{}", before, len(df), symbol, start_date, end_date)
                                except Exception:
                                    pass
                                report = data_io.upsert_candles(self.engine, df, symbol, "1m")
                            processed += 1
                            _emit_progress()
                    continue
                # Special handling for 1w: Tiingo doesn't support "week", download 1d and resample locally
                if tf == "1w":
                    self._backfill_weekly(symbol, start_ms, now_ms, use_monthly_split)
                    processed += 1
                    _emit_progress()
                    continue

                missing_ranges = self._find_missing_intervals(symbol, tf, start_ms, now_ms)
                for (s_ms, e_ms) in missing_ranges:
                    # Split by month only for large backfills, otherwise single request
                    if use_monthly_split:
                        subranges = time_utils.split_range_by_month(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))
                    else:
                        subranges = [(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))]

                    for (sub_s, sub_e) in subranges:
                        start_date = sub_s.date().isoformat()
                        end_date = sub_e.date().isoformat()
                        resample = self._tf_to_tiingo_resample(tf)
                        logger.info("Requesting {} candles {} - {} for {}", tf, start_date, end_date, symbol)
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
                        processed += 1
                        _emit_progress()
            except Exception as e:
                logger.exception("Backfill failed for {} {}: {}", symbol, tf, e)

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
            logger.info("No missing 1m ranges detected for {}", symbol)
            return

        # Determine if we need monthly splits (only for large ranges >90 days)
        range_days = (end_ms - start_ms) / (24 * 3600 * 1000)
        use_monthly_split = range_days > 90

        for (s_ms, e_ms) in missing_ranges:
            s_dt = time_utils.ms_to_utc_dt(s_ms)
            e_dt = time_utils.ms_to_utc_dt(e_ms)

            # Split by month only for large backfills, otherwise single request
            if use_monthly_split:
                subranges = time_utils.split_range_by_month(s_dt, e_dt)
            else:
                subranges = [(s_dt, e_dt)]

            for (sub_s, sub_e) in subranges:
                start_date = sub_s.date().isoformat()
                end_date = sub_e.date().isoformat()
                logger.info("Requesting 1m candles {} - {} for {}", start_date, end_date, symbol)
                df = self.provider.get_ticks(symbol, start_date=start_date, end_date=end_date)
                if df is None or df.empty:
                    logger.info("No tick/1m data returned for {} {}-{}", symbol, start_date, end_date)
                    continue
                # Upsert as 1m timeframe
                report = data_io.upsert_candles(self.engine, df, symbol, "1m")
                # logger.info("Upsert 1m report: %s", report)

    def _backfill_timeframe_autonomous(self, symbol: str, timeframe: str, global_start_ms: int, global_end_ms: int):
        """
        For a given timeframe compute missing expected period ends between global_start_ms and global_end_ms,
        split into ranges avoiding weekend and request only those ranges from Tiingo.
        """
        logger.info("Backfilling autonomous timeframe {} for {}", timeframe, symbol)
        # find missing intervals (list of (s_ms,e_ms)) where series lacks bars
        missing_ranges = self._find_missing_intervals(symbol, timeframe, global_start_ms, global_end_ms)
        if not missing_ranges:
            logger.info("No missing ranges for {} {}", symbol, timeframe)
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
                logger.info("Requesting {} candles {} - {} for {}", timeframe, start_date, end_date, symbol)
                df = self.provider.get_candles(symbol, start_date=start_date, end_date=end_date, resample_freq=resample)
                if df is None or df.empty:
                    logger.info("Tiingo returned no data for {} {} {}-{}", symbol, timeframe, start_date, end_date)
                    continue
                # Upsert into DB
                report = data_io.upsert_candles(self.engine, df, symbol, timeframe)
                # logger.info("Upsert report for %s %s: %s", symbol, timeframe, report)

    def _backfill_weekly(self, symbol: str, start_ms: int, end_ms: int, use_monthly_split: bool):
        """
        Backfill 1w timeframe by downloading 1d data and resampling to weekly.
        Tiingo API doesn't support 'week' resampleFreq, so we must resample locally.
        """
        import pandas as pd

        # Find missing 1w intervals
        missing_ranges = self._find_missing_intervals(symbol, "1w", start_ms, end_ms)
        if not missing_ranges:
            logger.info("No missing ranges for {} 1w", symbol)
            return

        for (s_ms, e_ms) in missing_ranges:
            # Download 1d data for the range
            if use_monthly_split:
                subranges = time_utils.split_range_by_month(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))
            else:
                subranges = [(time_utils.ms_to_utc_dt(s_ms), time_utils.ms_to_utc_dt(e_ms))]

            all_daily_data = []
            for (sub_s, sub_e) in subranges:
                start_date = sub_s.date().isoformat()
                end_date = sub_e.date().isoformat()
                logger.info("Requesting 1d candles {} - {} for {} (to resample to 1w)", start_date, end_date, symbol)
                df = self.provider.get_candles(symbol, start_date=start_date, end_date=end_date, resample_freq="1day")
                if df is not None and not df.empty:
                    all_daily_data.append(df)

            if not all_daily_data:
                logger.info("No 1d data available for 1w resample for {} {}-{}", symbol, s_ms, e_ms)
                continue

            # Concatenate all daily data
            df_daily = pd.concat(all_daily_data, ignore_index=True)
            df_daily = df_daily.dropna(subset=["ts_utc"]).copy()
            df_daily = df_daily.drop_duplicates(subset=["ts_utc"], keep="last")

            # Resample to weekly
            df_daily["ts_dt"] = pd.to_datetime(df_daily["ts_utc"], unit="ms", utc=True)
            df_daily = df_daily.set_index("ts_dt").sort_index()

            # Resample OHLC to weekly (W = week ending Sunday)
            resampler = df_daily.resample("W", label="right", closed="right")

            weekly_ohlc = pd.DataFrame({
                "open": resampler["open"].first(),
                "high": resampler["high"].max(),
                "low": resampler["low"].min(),
                "close": resampler["close"].last()
            })

            if "volume" in df_daily.columns:
                weekly_ohlc["volume"] = resampler["volume"].sum()

            weekly_ohlc = weekly_ohlc.dropna(subset=["close"])
            if weekly_ohlc.empty:
                continue

            # Convert back to format expected by upsert_candles
            weekly_candles = []
            for idx, row in weekly_ohlc.iterrows():
                weekly_candles.append({
                    "ts_utc": int(idx.timestamp() * 1000),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row.get("volume", None) if "volume" in weekly_ohlc.columns else None
                })

            if weekly_candles:
                df_weekly = pd.DataFrame(weekly_candles)
                report = data_io.upsert_candles(self.engine, df_weekly, symbol, "1w")
                logger.info("Upserted {} weekly candles for {}", len(df_weekly), symbol)

    def _tf_to_tiingo_resample(self, tf: str) -> str:
        """
        Convert timeframe like '1m','5m','1h','1d' into Tiingo resampleFreq string: '1min','5min','1hour','1day'
        Note: 1w is NOT supported by Tiingo API, handled separately via _backfill_weekly()
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
            logger.exception("Failed to query existing candles for gap detection: {}", e)
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

        # AGGREGATE nearby gaps: merge ranges separated by less than 1 hour to reduce API calls
        if len(out_ranges) > 1:
            aggregation_threshold_ms = 60 * 60 * 1000  # 1 hour
            merged: List[Tuple[int, int]] = []
            current_start, current_end = out_ranges[0]

            for next_start, next_end in out_ranges[1:]:
                # If gap between current_end and next_start is small, merge them
                if next_start - current_end < aggregation_threshold_ms:
                    current_end = next_end  # Extend current range
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = next_start, next_end

            merged.append((current_start, current_end))
            out_ranges = merged

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
            logger.exception("Failed to get last candle timestamp: {}", e)
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
            logger.exception("Failed to get first candle timestamp: {}", e)
            return None
