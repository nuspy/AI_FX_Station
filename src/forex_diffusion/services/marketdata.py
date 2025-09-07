"""
Market data providers and MarketDataService.

- AlphaVantageClient: client bridge to the official 'alpha_vantage' library (preferred) with httpx fallback for historical endpoints.
- MarketDataService: orchestrates missing-interval detection, download, normalization and upsert into DB
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..utils.config import get_config
from ..data import io as data_io
from sqlalchemy import create_engine

# try to import official alpha_vantage library; if not available we'll fallback to httpx
try:
    from alpha_vantage.foreignexchange import ForeignExchange  # type: ignore
    from alpha_vantage.timeseries import TimeSeries  # type: ignore
    _HAS_ALPHA_VANTAGE = True
except Exception:
    _HAS_ALPHA_VANTAGE = False


def parse_symbol(symbol: str) -> Tuple[str, str]:
    """
    Parse a symbol like "EUR/USD" into ("EUR", "USD").
    """
    if "/" in symbol:
        a, b = symbol.split("/")
        return a.strip(), b.strip()
    elif symbol.upper().startswith("FX_"):
        # fallback to splitting underscore
        parts = symbol.split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
    raise ValueError(f"Unsupported symbol format: {symbol}")


class AlphaVantageClient:
    """
    Alpha Vantage client bridge: uses official alpha_vantage library for realtime where possible,
    and falls back to httpx for historical FX_DAILY / FX_INTRADAY endpoints.
    """
    def __init__(self, key: Optional[str] = None, base_url: str = "https://www.alphavantage.co/query", rate_limit_per_minute: int = 5):
        cfg = get_config()
        av_cfg = getattr(cfg, "providers", {}).get("alpha_vantage", {}) if hasattr(cfg, "providers") else {}
        from ..utils.user_settings import get_setting
        user_key = get_setting("alpha_vantage_api_key", None)
        self.api_key = key or av_cfg.get("key") or av_cfg.get("api_key") or user_key or os.environ.get("ALPHAVANTAGE_KEY")
        self.base_url = av_cfg.get("base_url", base_url)
        self.rate_limit = rate_limit_per_minute or av_cfg.get("rate_limit_per_minute", 5)
        self._client = httpx.Client(timeout=30.0)
        self._last_req_ts = 0.0
        self._min_period = 60.0 / float(self.rate_limit) if self.rate_limit > 0 else 0.0

        # init official client objects if library available
        self._fx = None
        self._ts = None
        if _HAS_ALPHA_VANTAGE and self.api_key:
            try:
                self._fx = ForeignExchange(key=self.api_key, output_format="pandas")  # type: ignore
                self._ts = TimeSeries(key=self.api_key, output_format="pandas")  # type: ignore
            except Exception as e:
                logger.warning("AlphaVantage library init failed, will fallback to HTTP client: {}", e)
                self._fx = None
                self._ts = None

    def _throttle(self):
        # Simple throttle to respect rate limit
        elapsed = time.time() - self._last_req_ts
        if elapsed < self._min_period:
            to_sleep = self._min_period - elapsed
            time.sleep(to_sleep)

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=60), retry=retry_if_exception_type(Exception))
    def _get(self, params: dict) -> dict:
        self._throttle()
        params = params.copy()
        params["apikey"] = self.api_key
        r = self._client.get(self.base_url, params=params)
        self._last_req_ts = time.time()
        r.raise_for_status()
        data = r.json()
        if "Error Message" in data:
            raise RuntimeError(f"AlphaVantage error: {data.get('Error Message')}")
        if "Note" in data:
            # rate limit message, raise to trigger retry/backoff
            logger.warning("AlphaVantage rate limit note: {}", data.get("Note"))
            raise RuntimeError(f"AlphaVantage rate limit: {data.get('Note')}")
        return data

    def _parse_time_series(self, ts_dict: dict, tz_utc: bool = True) -> pd.DataFrame:
        """
        Convert Alpha Vantage time series JSON to DataFrame with ts_utc(ms), open, high, low, close, volume (if present)
        """
        records = []
        for stamp_str, vals in ts_dict.items():
            try:
                dt = pd.to_datetime(stamp_str, utc=True)
                ts_ms = int(dt.view("int64") // 1_000_000)
            except Exception:
                dt = pd.to_datetime(stamp_str)
                dt = dt.tz_localize(timezone.utc)
                ts_ms = int(dt.view("int64") // 1_000_000)
            record = {
                "ts_utc": ts_ms,
                "open": float(vals.get("1. open") or vals.get("open") or 0.0),
                "high": float(vals.get("2. high") or vals.get("high") or 0.0),
                "low": float(vals.get("3. low") or vals.get("low") or 0.0),
                "close": float(vals.get("4. close") or vals.get("close") or 0.0),
            }
            vol = vals.get("5. volume") or vals.get("volume")
            if vol is not None:
                try:
                    record["volume"] = float(vol)
                except Exception:
                    record["volume"] = None
            records.append(record)
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df = df.sort_values("ts_utc").reset_index(drop=True)
        return df

    def get_historical(self, symbol: str, timeframe: str, start_ts_ms: Optional[int] = None, end_ts_ms: Optional[int] = None) -> pd.DataFrame:
        """
        Download historical data for symbol and timeframe.
        Uses HTTP fallback for FX_DAILY and FX_INTRADAY endpoints.
        """
        from_sym, to_sym = parse_symbol(symbol)
        if timeframe in ["1d", "1D", "daily"]:
            params = {"function": "FX_DAILY", "from_symbol": from_sym, "to_symbol": to_sym, "outputsize": "full", "datatype": "json"}
            data = self._get(params)
            ts_key = "Time Series FX (Daily)"
            ts = data.get(ts_key) or data.get("Time Series FX (Daily)") or data.get("Time Series (Daily)") or {}
            df = self._parse_time_series(ts)
        else:
            interval_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min", "1h": "60min"}
            interval = interval_map.get(timeframe, "1min")
            params = {"function": "FX_INTRADAY", "from_symbol": from_sym, "to_symbol": to_sym, "interval": interval, "outputsize": "full", "datatype": "json"}
            data = self._get(params)
            ts_key = None
            for k in data.keys():
                if "Time Series FX" in k or "Time Series" in k:
                    ts_key = k
                    break
            ts = data.get(ts_key, {})
            df = self._parse_time_series(ts)
        if df.empty:
            logger.warning("AlphaVantage returned empty data for {} {}", symbol, timeframe)
            return df
        if start_ts_ms is not None:
            df = df[df["ts_utc"] >= int(start_ts_ms)].reset_index(drop=True)
        if end_ts_ms is not None:
            df = df[df["ts_utc"] <= int(end_ts_ms)].reset_index(drop=True)
        return df

    def get_current_price(self, symbol: str) -> dict:
        """
        Try official library first for realtime exchange rate; fallback to HTTP query.
        """
        from_sym, to_sym = parse_symbol(symbol)
        # try official library ForeignExchange.get_currency_exchange_rate
        if self._fx is not None:
            try:
                data, _ = self._fx.get_currency_exchange_rate(from_symbol=from_sym, to_symbol=to_sym)  # type: ignore
                return data
            except Exception as e:
                logger.debug("alpha_vantage lib get_currency_exchange_rate failed: {}", e)
        # fallback HTTP
        params = {"function": "CURRENCY_EXCHANGE_RATE", "from_currency": from_sym, "to_currency": to_sym}
        data = self._get(params)
        rate_info = data.get("Realtime Currency Exchange Rate", {}) or {}
        return rate_info


class MarketDataService:
    """
    High-level service to orchestrate data acquisition and DB ingest.
    - select provider based on config
    - for each (symbol, timeframe) compute missing interval and fetch missing data
    - normalize and upsert to DB via data_io functions
    """
    def __init__(self, database_url: Optional[str] = None):
        self.cfg = get_config()
        self.db_url = database_url or getattr(self.cfg.db, "database_url", None)
        if not self.db_url:
            raise ValueError("Database URL not configured")
        self.engine = create_engine(self.db_url, future=True)
        # instantiate default provider
        providers_cfg = getattr(self.cfg, "providers", {}) if hasattr(self.cfg, "providers") else {}
        default = providers_cfg.get("default", "alpha_vantage")
        if default == "alpha_vantage":
            self.provider = AlphaVantageClient()
        else:
            # fallback to alpha
            logger.warning("Unknown provider '{}', falling back to AlphaVantage", default)
            self.provider = AlphaVantageClient()

    def compute_missing_interval(self, symbol: str, timeframe: str) -> Tuple[Optional[int], int]:
        """
        Returns (start_ts_ms, end_ts_ms) to fetch.
        If no data in DB, start_ts_ms = None (caller may request full history)
        end_ts_ms is current time in ms.
        """
        t_last = data_io.get_last_ts_for_symbol_tf(self.engine, symbol, timeframe)
        now_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
        return (t_last, now_ms)

    def backfill_symbol_timeframe(self, symbol: str, timeframe: str, force_full: bool = False) -> dict:
        """
        Backfill symbol/timeframe from last DB timestamp to now.
        If force_full True, download full history per provider mapping (e.g., 20 years daily etc.)
        Returns QA report dict.
        """
        # decide timeframe mapping: for long-run daily + intraday recent segment per spec
        cfg = self.cfg
        df_report = {"symbol": symbol, "timeframe": timeframe, "actions": []}
        # compute last timestamp
        t_last = data_io.get_last_ts_for_symbol_tf(self.engine, symbol, timeframe)
        now_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
        # If no data present or force_full and timeframe daily -> request full 20y daily
        if t_last is None or force_full:
            # For daily use long-run history years
            if timeframe in ["1d", "1D", "daily"]:
                years = int(cfg.data.backfill.history_years if hasattr(cfg, "data") and hasattr(cfg.data, "backfill") else cfg.data.get("backfill", {}).get("history_years", 20))
                # In AlphaVantage we request FX_DAILY with outputsize full and rely on truncation by start_ts here
                start_ts_ms = None
                end_ts_ms = now_ms
                df = self.provider.get_historical(symbol=symbol, timeframe="1d", start_ts_ms=start_ts_ms, end_ts_ms=end_ts_ms)
                report = data_io.upsert_candles(self.engine, df, symbol, "1d", resampled=False)
                df_report["actions"].append({"type": "full_daily_backfill", "report": report})
            else:
                # for intraday, fallback to recent segment
                recent_days = int(getattr(cfg.data.backfill, "intraday_recent_days", 90))
                start_ts = pd.Timestamp.utcnow() - pd.Timedelta(days=recent_days)
                start_ts_ms = int(start_ts.value // 1_000_000)
                df = self.provider.get_historical(symbol=symbol, timeframe="1m", start_ts_ms=start_ts_ms, end_ts_ms=now_ms)
                report = data_io.upsert_candles(self.engine, df, symbol, "1m", resampled=False)
                df_report["actions"].append({"type": "intraday_recent_backfill", "report": report})
            return df_report

        # Otherwise fetch from t_last + delta to now
        # compute small sliding windows to avoid huge requests (provider limits)
        chunk_seconds = 24 * 3600 * 7  # 7 days chunk for intraday by default
        start_ms = t_last + 1
        end_ms = now_ms
        segments = []
        cur_start = start_ms
        while cur_start <= end_ms:
            cur_end = cur_start + chunk_seconds * 1000 - 1
            if cur_end > end_ms:
                cur_end = end_ms
            segments.append((cur_start, cur_end))
            cur_start = cur_end + 1

        for seg_start, seg_end in segments:
            try:
                df_seg = self.provider.get_historical(symbol=symbol, timeframe=timeframe, start_ts_ms=seg_start, end_ts_ms=seg_end)
                if df_seg is None or df_seg.empty:
                    df_report["actions"].append({"segment": (seg_start, seg_end), "note": "no_data"})
                    continue
                # If provider granularity differs, mark resampled accordingly (handled by resampling later)
                # Validate, normalize and upsert via data_io
                report = data_io.upsert_candles(self.engine, df_seg, symbol, timeframe, resampled=False)
                df_report["actions"].append({"segment": (seg_start, seg_end), "report": report})
            except Exception as e:
                logger.exception("Failed to fetch segment {}-{} for {}/{}: {}", seg_start, seg_end, symbol, timeframe, e)
                df_report["actions"].append({"segment": (seg_start, seg_end), "error": str(e)})
        return df_report

    def ensure_startup_backfill(self):
        """
        Run alembic upgrade head externally; then for each configured symbol/timeframe,
        compute and fill missing data. This function is sync and may be invoked in a worker pool.
        """
        cfg = self.cfg
        symbols = getattr(cfg.data, "symbols", None) or (cfg.data.get("symbols", []) if isinstance(cfg.data, dict) else [])
        timeframes = getattr(cfg, "timeframes", None) or (cfg.timeframes.get("native", []) if isinstance(cfg.timeframes, dict) else [])
        reports = []
        for sym in symbols:
            for tf in timeframes:
                try:
                    r = self.backfill_symbol_timeframe(sym, tf, force_full=False)
                    reports.append(r)
                except Exception as e:
                    logger.exception("Backfill failed for {}/{}: {}", sym, tf, e)
                    reports.append({"symbol": sym, "timeframe": tf, "error": str(e)})
        return reports
