"""
Market data providers and MarketDataService.

- AlphaVantageClient: client bridge to the official 'alpha_vantage' library (preferred) with httpx fallback for historical endpoints.
- MarketDataService: orchestrates missing-interval detection, download, normalization and upsert into DB
"""

from __future__ import annotations

import os
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
    raise ValueError(f"Unsupported symbol format: {symbol}")

class AlphaVantageClient:
    """
    Alpha Vantage client bridge.
    """
    def __init__(self, key: Optional[str] = None, **kwargs):
        cfg = get_config()
        av_cfg = getattr(getattr(cfg, "providers", None), "alpha_vantage", None)
        self.api_key = key or (getattr(av_cfg, "key", None) if av_cfg else None) or os.environ.get("ALPHAVANTAGE_KEY")
        self.base_url = (getattr(av_cfg, "base_url", "https://www.alphavantage.co/query") if av_cfg else "https://www.alphavantage.co/query")
        rate_limit = (getattr(av_cfg, "rate_limit_per_minute", 5) if av_cfg else 5)
        self._client = httpx.Client(timeout=30.0)
        self._min_period = 60.0 / float(rate_limit) if rate_limit > 0 else 0.0
        self._last_req_ts = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_req_ts
        if elapsed < self._min_period:
            time.sleep(self._min_period - elapsed)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get(self, params: dict) -> dict:
        self._throttle()
        params["apikey"] = self.api_key
        r = self._client.get(self.base_url, params=params)
        self._last_req_ts = time.time()
        r.raise_for_status()
        data = r.json()
        if "Error Message" in data or "Information" in data:
            raise RuntimeError(f"AlphaVantage error: {data.get('Error Message') or data.get('Information')}")
        return data

    def get_historical(self, symbol: str, timeframe: str, **kwargs) -> pd.DataFrame:
        from_sym, to_sym = parse_symbol(symbol)
        function = "FX_DAILY" if timeframe in ["1d", "daily"] else "FX_INTRADAY"
        interval = "60min" if timeframe == "1h" else timeframe # Basic mapping
        
        params = {
            "function": function,
            "from_symbol": from_sym,
            "to_symbol": to_sym,
            "outputsize": "full",
            "datatype": "json"
        }
        if function == "FX_INTRADAY":
            params["interval"] = interval

        data = self._get(params)
        key = next((k for k in data if "Time Series" in k), None)
        if not key:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.rename(columns={
            "1. open": "open", "2. high": "high",
            "3. low": "low", "4. close": "close", "5. volume": "volume"
        }, inplace=True)
        df.index = pd.to_datetime(df.index)
        df["ts_utc"] = df.index.astype('int64') // 1_000_000
        return df[["ts_utc", "open", "high", "low", "close", "volume"]].astype(float)

class MarketDataService:
    """
    High-level service to orchestrate data acquisition and DB ingest.
    """
    def __init__(self, database_url: Optional[str] = None):
        self.cfg = get_config()
        db_url = database_url or getattr(self.cfg.db, "database_url", None)
        if not db_url:
            raise ValueError("Database URL not configured")
        self.engine = create_engine(db_url, future=True)
        self.provider = AlphaVantageClient() # Default provider

    def backfill_symbol_timeframe(self, symbol: str, timeframe: str, force_full: bool = False):
        logger.info(f"Starting backfill for {symbol} {timeframe}...")
        t_last, now_ms = self._get_last_candle_ts(symbol, timeframe)

        start_ts = None
        if not force_full and t_last:
            start_ts = t_last + 1
        
        df_candles = self.provider.get_historical(symbol, timeframe, start_ts_ms=start_ts)
        
        if not df_candles.empty:
            logger.info(f"Fetched {len(df_candles)} new candles for {symbol} {timeframe}.")
            report = data_io.upsert_candles(self.engine, df_candles, symbol, timeframe)
            logger.info(f"Upsert report: {report}")
        else:
            logger.info(f"No new data found for {symbol} {timeframe}.")

    def _get_last_candle_ts(self, symbol: str, timeframe: str) -> Tuple[Optional[int], int]:
        now_ms = int(time.time() * 1000)
        try:
            with self.engine.connect() as conn:
                from sqlalchemy import text
                query = text("SELECT MAX(ts_utc) FROM market_data_candles WHERE symbol = :symbol AND timeframe = :timeframe")
                last_ts = conn.execute(query, {"symbol": symbol, "timeframe": timeframe}).scalar_one_or_none()
                return (int(last_ts) if last_ts else None, now_ms)
        except Exception as e:
            logger.exception(f"Failed to get last candle timestamp: {e}")
            return (None, now_ms)
