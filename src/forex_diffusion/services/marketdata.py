"""
Market data providers and MarketDataService.

- AlphaVantageClient: client bridge to the official 'alpha_vantage' library (preferred) with httpx fallback for historical endpoints.
- MarketDataService: orchestrates missing-interval detection, download, normalization and upsert into DB
"""

from __future__ import annotations

import os
import threading
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
from .db_service import DBService
from .aggregator import AggregatorService

# try to import official alpha_vantage library; if not available we'll fallback to httpx
try:
    from alpha_vantage.foreignexchange import ForeignExchange  # type: ignore
    from alpha_vantage.timeseries import TimeSeries  # type: ignore
    _HAS_ALPHA_VANTAGE = True
except Exception:
    _HAS_ALPHA_VANTAGE = False

# optional websocket support (best-effort)
try:
    import websocket  # type: ignore
    _HAS_WEBSOCKET_CLIENT = True
except Exception:
    _HAS_WEBSOCKET_CLIENT = False

# --- Tiingo client (REST + optional WS configured externally)
class TiingoClient:
    """
    Minimal Tiingo REST client for FX via Tiingo FX endpoint.
    Fallbacks: uses httpx to query configured base_url and api key.
    Exposes get_current_price(symbol) and get_historical(symbol,timeframe,...)
    """
    def __init__(self, key: Optional[str] = None, base_url: str = "https://api.tiingo.com"):
        cfg = get_config()
        providers_cfg = getattr(cfg, "providers", {}) if hasattr(cfg, "providers") else {}
        ti_cfg = providers_cfg.get("tiingo", {}) if isinstance(providers_cfg, dict) else {}
        from ..utils.user_settings import get_setting
        user_key = get_setting("tiingo_api_key", None)
        self.api_key = key or ti_cfg.get("key") or user_key or os.environ.get("TIINGO_API_KEY")
        self.base_url = ti_cfg.get("base_url", base_url)
        self._client = httpx.Client(timeout=30.0)

    def _get(self, path: str, params: dict = None):
        if params is None:
            params = {}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        r = self._client.get(url, params=params, headers=headers)

        #logger.debug(params)

        try:
            r.raise_for_status()
        except Exception as e:
            # log response body for debugging and re-raise
            try:
                logger.debug("Tiingo HTTP error: status={} url={} body={}", r.status_code, url, r.text)
            except Exception:
                pass
            raise
        # try decode JSON and log if empty/invalid
        try:
            data = r.json()
        except Exception:
            try:
                logger.debug("Tiingo response not JSON: status={} url={} body={}", r.status_code, url, r.text)
            except Exception:
                pass
            raise
        # if data empty, log the raw body for debugging
        try:
            if not data:
                logger.debug("Tiingo returned empty data for url={} params={} body={}", url, params, r.text)
        except Exception:
            pass

        return data

    def parse_symbol_pair(self, symbol: str):
        a, b = parse_symbol(symbol)
        return a.lower(), b.lower()

    def get_current_price(self, symbol: str) -> dict:
        """
        Query Tiingo for a current price. Primary try endpoint; fallback to a short historical query
        and return the last close if available.
        """
        a, b = self.parse_symbol_pair(symbol)
        path = f"tiingo/fx/{a}{b}/prices"
        try:
            # best-effort direct call (may be restricted); try a small params set
            data = self._get(path, params={"resampleFreq": "1min", "startDate": None})
            if isinstance(data, list) and len(data) > 0:
                last = data[-1]
                try:
                    ts = int(pd.to_datetime(last.get("date")).value // 1_000_000)
                    price = float(last.get("close"))
                    return {"ts_utc": ts, "price": price}
                except Exception:
                    # fallthrough to historical fallback
                    pass
        except Exception:
            # log and fall back to historical query
            logger.debug("Tiingo direct current price failed for %s, falling back to historical", symbol)

        # Historical fallback: request recent 5 minutes (1m bars) and take last close
        try:
            now_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
            start_ms = now_ms - 5 * 60 * 1000
            df = self.get_historical(symbol=symbol, timeframe="1m", start_ts_ms=start_ms, end_ts_ms=now_ms)
            if isinstance(df, pd.DataFrame) and not df.empty:
                last = df.iloc[-1]
                try:
                    return {"ts_utc": int(last["ts_utc"]), "price": float(last["close"])}
                except Exception:
                    return {}
        except Exception as e:
            logger.debug("Tiingo historical fallback failed for %s: {}", symbol, e)
        return {}

    def get_historical(self, symbol: str, timeframe: str, start_ts_ms: Optional[int] = None, end_ts_ms: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve historical data from Tiingo for FX pair.
        Maps timeframe to daily/intraday where possible.
        """
        a, b = self.parse_symbol_pair(symbol)
        path = f"tiingo/fx/{a}{b}/prices"
        params = {}
        # Tiingo supports startDate/endDate as YYYY-MM-DD, so convert if needed
        if start_ts_ms is not None:
            params["startDate"] = pd.to_datetime(start_ts_ms, unit="ms").strftime("%Y-%m-%d")
        if end_ts_ms is not None:
            params["endDate"] = pd.to_datetime(end_ts_ms, unit="ms").strftime("%Y-%m-%d")
        try:
            data = self._get(path, params=params)
            # convert to DataFrame with ts_utc/open/high/low/close/volume
            recs = []
            for it in data:
                dt = pd.to_datetime(it.get("date"))
                ts_ms = int(dt.value // 1_000_000)
                recs.append({
                    "ts_utc": ts_ms,
                    "open": float(it.get("open", it.get("close", 0.0))),
                    "high": float(it.get("high", it.get("close", 0.0))),
                    "low": float(it.get("low", it.get("close", 0.0))),
                    "close": float(it.get("close", 0.0)),
                    "volume": it.get("volume", None),
                })
            df = pd.DataFrame(recs)
            return df
        except Exception:
            return pd.DataFrame([])

# ---- Provider factory in MarketDataService


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
        av_cfg = getattr(getattr(cfg, "providers", None), "alpha_vantage", None)

        from ..utils.user_settings import get_setting
        user_key = get_setting("alpha_vantage_api_key", None)

        self.api_key = key or (getattr(av_cfg, "key", None) or getattr(av_cfg, "api_key", None) if av_cfg else None) or user_key or os.environ.get("ALPHAVANTAGE_KEY")
        self.base_url = (getattr(av_cfg, "base_url", None) if av_cfg else None) or base_url

        rate_limit_from_cfg = getattr(av_cfg, "rate_limit_per_minute", None) if av_cfg else None
        self.rate_limit = rate_limit_from_cfg if rate_limit_from_cfg is not None else rate_limit_per_minute

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
                # Prefer pandas.Timestamp.value (ns since epoch) if available, fallback to timestamp()
                try:
                    ts_ms = int(dt.value // 1_000_000)
                except Exception:
                    ts_ms = int(pd.Timestamp(dt).timestamp() * 1000)
            except Exception:
                dt = pd.to_datetime(stamp_str)
                dt = dt.tz_localize(timezone.utc)
                try:
                    ts_ms = int(dt.value // 1_000_000)
                except Exception:
                    ts_ms = int(pd.Timestamp(dt).timestamp() * 1000)
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
                # correct parameter names for alpha_vantage ForeignExchange API
                data, _ = self._fx.get_currency_exchange_rate(from_currency=from_sym, to_currency=to_sym)  # type: ignore
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
    - select provider based on config (default: tiingo)
    - for each (symbol, timeframe) compute missing interval and fetch missing data
    - normalize and upsert to DB via data_io functions
    """
    def __init__(self, database_url: Optional[str] = None, provider_name: Optional[str] = None, poll_interval: float = 1.0):
        self.cfg = get_config()
        self.db_url = database_url or getattr(self.cfg.db, "database_url", None)
        if not self.db_url:
            raise ValueError("Database URL not configured")
        self.engine = create_engine(self.db_url, future=True)

        # provider selection
        provs_cfg = getattr(self.cfg, "providers", {}) if hasattr(self.cfg, "providers") else {}
        default = provider_name or getattr(provs_cfg, "default", None) or (provs_cfg.get("default") if isinstance(provs_cfg, dict) else None) or "tiingo"
        self._poll_interval = float(poll_interval)
        self._provider_name = None
        self.provider = None
        self.set_provider(default)

    def available_providers(self):
        # list known providers
        return ["tiingo", "alpha_vantage"]

    def set_provider(self, name: str):
        logger.debug(f"TRACE: MarketDataService.set_provider called with '{name}'")
        name = (name or "").lower()
        if name == self._provider_name:
            return
        if name == "alpha_vantage":
            self.provider = AlphaVantageClient()
        elif name == "tiingo":
            self.provider = TiingoClient()
        else:
            logger.warning("Unknown provider '%s', falling back to tiingo", name)
            self.provider = TiingoClient()
            name = "tiingo"
        self._provider_name = name
        logger.info("MarketDataService provider set to %s", name)

    def start_ws_streaming(self, uri: str | None = None, tickers: Optional[list[str]] = None, threshold: str = "5") -> None:
        logger.critical("CRITICAL_TRACE: MarketDataService.start_ws_streaming has been called.")
        """
        Start a background thread that connects to Tiingo WebSocket and subscribes to tickers.
        Received quotes are published via event_bus and upserted into DB as 1-row candles.

        Also start AggregatorService to periodically aggregate persisted ticks into candles.
        """
        if getattr(self, "_ws_thread", None) and getattr(self, "_ws_thread").is_alive():
            return
        self._ws_stop = False
        ws_uri = uri or "wss://api.tiingo.com/fx"
        syms = tickers or getattr(self, "_realtime_symbols", None) or ["eurusd"]

        # prepare aggregator for these symbols (normalize to uppercase strings)
        try:
            symbols_for_agg = [s.upper() if isinstance(s, str) else s for s in (syms if isinstance(syms, (list, tuple)) else [syms])]
        except Exception:
            symbols_for_agg = [str(syms).upper()]

        try:
            # instantiate & start aggregator (background scheduler)
            self._aggregator = AggregatorService(engine=self.engine, symbols=symbols_for_agg)
            self._aggregator.start()
        except Exception as e:
            try:
                logger.exception("Failed to start AggregatorService: {}", e)
            except Exception:
                pass

        try:
            api_key = None
            # try config / env via TiingoClient logic
            if hasattr(self, "provider") and hasattr(self.provider, "api_key"):
                api_key = getattr(self.provider, "api_key", None)
            # fallback to environment variables if not present
            if not api_key:
                import os
                api_key = os.environ.get("TIINGO_APIKEY") or os.environ.get("TIINGO_API_KEY") or os.environ.get("TIINGO_API")
        except Exception:
            api_key = None

        try:
            _logger = None
            from loguru import logger as _logger
            _logger.debug("WS streamer: starting with api_key_present=%s tickers=%s threshold=%s", bool(api_key), syms, threshold)
        except Exception:
            pass

        # create DBService instance for the worker to persist ticks
        try:
            dbs = DBService(engine=self.engine)
        except Exception:
            dbs = None

        def _ws_worker():
            import json as _json
            import time as _time
            from loguru import logger as _logger
            try:
                from ..utils.event_bus import publish
            except Exception:
                publish = lambda *a, **k: None

            while not getattr(self, "_ws_stop", False):
                try:
                    try:
                        # lazy import websocket-client
                        from websocket import create_connection, WebSocketException
                    except Exception as e:
                        _logger.warning("WS streamer: websocket-client not installed: {}", e)
                        return
                    ws = None
                    try:
                        ws = create_connection(ws_uri, timeout=10)
                        _logger.info("WS streamer: connected to {}", ws_uri)
                        # prepare subscribe payload per Tiingo example
                        sub = {
                            "eventName": "subscribe",
                            "authorization": api_key or "",
                            "eventData": {
                                "thresholdLevel": str(threshold),
                                "tickers": [str(s).lower() for s in syms] if isinstance(syms, (list, tuple)) else [str(syms).lower()]
                            }
                        }
                        try:
                            ws.send(_json.dumps(sub))
                            _logger.debug("WS streamer: sent subscribe payload")
                        except Exception as e:
                            _logger.debug("WS streamer: failed to send subscribe: {}", e)

                        # receive loop
                        while not getattr(self, "_ws_stop", False):
                            try:
                                logger.debug("TRACE: _ws_worker: Waiting to receive message from WebSocket...")
                                raw = ws.recv()
                                if not raw:
                                    continue
                                try:
                                    msg = _json.loads(raw)
                                except Exception:
                                    continue
                                logger.debug(f"TRACE: _ws_worker: Received raw message: {raw}")
                                # Tiingo uses messageType "A" with data array for quotes
                                try:
                                    mtype = msg.get("messageType")
                                    if mtype == "A":
                                        data = msg.get("data", [])
                                        # expected layout: ["Q", "pair", "iso_ts", size, bid, ask, ...]
                                        if isinstance(data, list) and len(data) >= 6:
                                            pair = data[1]  # e.g., 'eurusd'
                                            iso_ts = data[2]
                                            try:
                                                # convert iso ts to ms
                                                import pandas as _pd
                                                ts_ms = int(_pd.to_datetime(iso_ts).value // 1_000_000)
                                            except Exception:
                                                ts_ms = int(time.time() * 1000)
                                            bid = data[4]
                                            ask = data[5]
                                            price = bid if bid is not None else ask
                                            payload = {
                                                "symbol": str(pair).upper(),
                                                "timeframe": "tick",  # CRITICAL: Ensure timeframe is 'tick' for raw data
                                                "ts_utc": int(ts_ms),
                                                "price": float(price) if price is not None else None,
                                                "bid": bid,
                                                "ask": ask,
                                            }
                                            logger.debug(f"TRACE: _ws_worker: Constructed tick payload: {payload}")
                                            # publish to UI (thread-safe queue)
                                            try:
                                                publish("tick", payload)
                                            except Exception:
                                                pass
                                            # persist tick into market_data_ticks (do not compute candles here)
                                            try:
                                                if dbs is not None:
                                                    payload_tick = payload.copy()
                                                    payload_tick["volume"] = None
                                                    payload_tick["ts_created_ms"] = int(time.time() * 1000)
                                                    logger.debug(f"TRACE: _ws_worker: Calling dbs.write_tick for {payload_tick.get('symbol')}")
                                                    ok = dbs.write_tick(payload_tick)
                                                    _logger.debug(f"TRACE: _ws_worker: Returned from dbs.write_tick (status: {ok}) for {payload_tick.get('symbol')}")

                                            except Exception as e:
                                                try:
                                                    _logger.exception("WS writer: write_tick failed: {}", e)
                                                except Exception:
                                                    pass
                                except Exception:
                                    # ignore non-quote messages
                                    pass
                            except WebSocketException:
                                break
                            except Exception:
                                break
                    finally:
                        try:
                            if ws is not None:
                                ws.close()
                        except Exception:
                            pass
                except Exception as e:
                    # wait before reconnect
                    try:
                        _logger.debug("WS streamer exception, will retry in 3s: {}", e)
                    except Exception:
                        pass
                    _time.sleep(3.0)
            # exit
            try:
                _logger.info("WS streamer stopped")
            except Exception:
                pass

        th = threading.Thread(target=_ws_worker, daemon=True)
        th.start()
        self._ws_thread = th

    def stop_ws_streaming(self) -> None:
        try:
            self._ws_stop = True
            if getattr(self, "_ws_thread", None) is not None:
                try:
                    self._ws_thread.join(timeout=1.0)
                except Exception:
                    pass
            # stop aggregator if running
            try:
                if getattr(self, "_aggregator", None) is not None:
                    try:
                        self._aggregator.stop()
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def provider_name(self) -> str:
        return self._provider_name or "tiingo"

    def poll_interval(self) -> float:
        return self._poll_interval

    def set_poll_interval(self, seconds: float):
        self._poll_interval = float(seconds)

    def compute_missing_interval(self, symbol: str, timeframe: str) -> Tuple[Optional[int], int]:
        """
        Returns (start_ts_ms, end_ts_ms) to fetch.
        If no data in DB, start_ts_ms = None (caller may request full history)
        end_ts_ms is current time in ms.
        """
        t_last = data_io.get_last_ts_for_symbol_tf(self.engine, symbol, timeframe)
        now_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
        return (t_last, now_ms)

    def backfill_symbol_timeframe(self, symbol: str, timeframe: str, force_full: bool = False, years: Optional[int] = None, months: Optional[int] = None) -> dict:
        """
        Tick-first backfill:
          - Compute candidate time window (respecting 'years' override for full history).
          - Exclude weekend windows Fri22:00UTC -> Sun22:00UTC.
          - Detect missing minute buckets in market_data_ticks (ts_utc//60000).
          - For contiguous missing minute ranges, fetch ticks (provider.get_historical with 'tick' if available,
            otherwise fetch candles and synthesize one tick per candle price=close).
          - Persist ticks via DBService.write_tick.
          - After ticks persisted, aggregate ticks into candles for multiple target timeframes and upsert via data_io.upsert_candles.
        Returns a report dict with counts and per-segment info.
        """
        # helpers
        from sqlalchemy import text as _text
        from ..services.db_service import DBService  # local import for clarity (DBService already imported at module top)
        dbs = DBService(engine=self.engine)

        def week_exclusion_segments(a_ms: int, b_ms: int):
            """
            Yield segments within [a_ms,b_ms) that exclude Fri22->Sun22 windows.
            Returns list of {'start':..., 'end':...}
            """
            import datetime
            out = []
            if a_ms >= b_ms:
                return out
            a_dt = datetime.datetime.utcfromtimestamp(a_ms / 1000.0)
            b_dt = datetime.datetime.utcfromtimestamp(b_ms / 1000.0)
            cur = a_dt
            while cur < b_dt:
                # next Friday 22:00 UTC
                days_to_friday = (4 - cur.weekday()) % 7
                friday = (cur + datetime.timedelta(days=days_to_friday)).replace(hour=22, minute=0, second=0, microsecond=0)
                if friday < cur:
                    friday += datetime.timedelta(weeks=1)
                seg_end_dt = min(friday, b_dt)
                if cur < seg_end_dt:
                    out.append({"start": int(cur.timestamp() * 1000), "end": int(seg_end_dt.timestamp() * 1000)})
                # skip weekend window
                weekend_end = friday + datetime.timedelta(days=2)  # sunday 22:00
                cur = weekend_end
                if cur < a_dt:
                    cur = a_dt
            return out

        def existing_tick_minutes(a_ms: int, b_ms: int) -> set:
            """
            Return set of minute buckets (int) for which ticks exist in market_data_ticks for symbol/timeframe in [a_ms,b_ms).
            """
            try:
                q = _text("SELECT DISTINCT (ts_utc/60000) AS m FROM market_data_ticks WHERE symbol = :s AND timeframe = :tf AND ts_utc >= :a AND ts_utc < :b")
                with self.engine.connect() as conn:
                    rows = conn.execute(q, {"s": symbol, "tf": timeframe, "a": int(a_ms), "b": int(b_ms)}).fetchall()
                    return set(int(r[0]) for r in rows if r[0] is not None)
            except Exception as e:
                logger.exception("Failed to query existing tick minutes: {}", e)
                return set()

        def missing_minute_ranges(a_ms: int, b_ms: int, existing_minutes: set):
            """
            Compute contiguous missing minute ranges (ms) within [a_ms,b_ms) given existing minute buckets.
            Returns list of {'start':ms,'end':ms}
            """
            start_min = int(a_ms // 60000)
            end_min = int((b_ms - 1) // 60000) + 1
            missing = []
            cur_missing_start = None
            for m in range(start_min, end_min):
                if m not in existing_minutes:
                    if cur_missing_start is None:
                        cur_missing_start = m
                else:
                    if cur_missing_start is not None:
                        missing.append({"start": cur_missing_start * 60000, "end": m * 60000})
                        cur_missing_start = None
            if cur_missing_start is not None:
                missing.append({"start": cur_missing_start * 60000, "end": end_min * 60000})
            return missing

        def fetch_and_persist_ticks(a_ms: int, b_ms: int) -> dict:
            """
            Fetch ticks for [a_ms,b_ms) using provider. If provider returns candles, synthesize ticks.
            Persist ticks via DBService.write_tick and return counts.
            """
            inserted = 0
            failed = 0
            provider_rows = 0
            try:
                # prefer explicit tick endpoint
                if hasattr(self.provider, "get_historical_ticks"):
                    df_src = self.provider.get_historical_ticks(symbol=symbol, timeframe=timeframe, start_ts_ms=a_ms, end_ts_ms=b_ms)
                else:
                    # try asking for tick granularity; fall back to timeframe if provider doesn't support
                    try:
                        df_src = self.provider.get_historical(symbol=symbol, timeframe="tick", start_ts_ms=a_ms, end_ts_ms=b_ms)
                    except Exception:
                        df_src = self.provider.get_historical(symbol=symbol, timeframe=timeframe, start_ts_ms=a_ms, end_ts_ms=b_ms)
            except Exception as e:
                logger.exception("Provider fetch failed for %s %s [%s,%s): {}", symbol, timeframe, a_ms, b_ms, e)
                return {"provider_rows": 0, "inserted": 0, "failed": 0, "error": str(e)}

            if df_src is None:
                return {"provider_rows": 0, "inserted": 0, "failed": 0}

            if not isinstance(df_src, pd.DataFrame):
                try:
                    df_src = pd.DataFrame(df_src)
                except Exception:
                    logger.warning("Provider returned non-tabular data for [%s,%s); skipping", a_ms, b_ms)
                    return {"provider_rows": 0, "inserted": 0, "failed": 0}

            if df_src.empty:
                return {"provider_rows": 0, "inserted": 0, "failed": 0}

            provider_rows = len(df_src)
            # Detect tick-like vs candle-like
            is_tick_like = "ts_utc" in df_src.columns and any(c in df_src.columns for c in ("price", "bid", "ask"))
            is_candle_like = "ts_utc" in df_src.columns and all(c in df_src.columns for c in ("open", "high", "low", "close"))


            if is_tick_like:
                logger.debug(" ### Tick in tick-like format: %s", df_src)

                for _, r in df_src.iterrows():
                    try:
                        ts_val = int(r.get("ts_utc"))
                    except Exception:
                        failed += 1
                        continue
                    payload_tick = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "ts_utc": ts_val,
                        "price": r.get("price", None),
                        "bid": r.get("bid", None),
                        "ask": r.get("ask", None),
                        "volume": r.get("volume", None),
                        "ts_created_ms": int(time.time() * 1000),
                    }
                    try:
                        ok = dbs.write_tick(payload_tick)
                        inserted += 1 if ok else 0
                        failed += 0 if ok else 1
                    except Exception:
                        failed += 1
                return {"provider_rows": provider_rows, "inserted": inserted, "failed": failed}

            if is_candle_like:
                logger.debug(" ### Tick in tick-like format: %s", df_src)
                # synthesize one tick per candle (ts_utc -> price=close)
                for _, r in df_src.iterrows():
                    try:
                        ts_val = int(r.get("ts_utc"))
                    except Exception:
                        failed += 1
                        continue
                    payload_tick = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "ts_utc": ts_val,
                        "price": r.get("close", None),
                        "bid": None,
                        "ask": None,
                        "volume": r.get("volume", None) if "volume" in r else None,
                        "ts_created_ms": int(time.time() * 1000),
                    }
                    try:
                        ok = dbs.write_tick(payload_tick)
                        inserted += 1 if ok else 0
                        failed += 0 if ok else 1
                    except Exception:
                        failed += 1
                return {"provider_rows": provider_rows, "inserted": inserted, "failed": failed}

            # unknown structure -> skip
            logger.warning("Provider returned unexpected columns for [%s,%s): cols=%s", a_ms, b_ms, list(df_src.columns))
            return {"provider_rows": provider_rows, "inserted": 0, "failed": provider_rows}

        # Start main flow
        df_report = {"symbol": symbol, "timeframe": timeframe, "provider_rows": 0, "ticks_inserted": 0, "candles_upserted": 0, "missing_ranges": []}

        # Determine window to fill: if force_full, use years setting, otherwise use last timestamp to now
        t_last, now_ms = self.compute_missing_interval(symbol, timeframe)
        try:
            # Get data config for defaults
            data_cfg = getattr(self.cfg, "data", {}) if hasattr(self.cfg, "data") else {}
            if isinstance(data_cfg, dict):
                data_cfg = data_cfg
            else:
                data_cfg = getattr(data_cfg, "__dict__", {})

            if force_full or t_last is None:
                # limit by years param: start = now - years*365
                years_cfg = int(years) if (years is not None) else int(
                    data_cfg.get("backfill", {}).get("history_years", 0))
                months_cfg = int(months) if (months is not None) else int(
                    data_cfg.get("backfill", {}).get("history_months", 0) + years_cfg * 12)
                end_ts_ms = now_ms
                start_ts_ms = max(0, end_ts_ms - int(months_cfg * 30 * 24 * 3600 * 1000))
            else:
                start_ts_ms = int(t_last) + 1
                end_ts_ms = now_ms
        except Exception as e:
            logger.debug("Error computing time window: {}", e)
            start_ts_ms = int(t_last) + 1 if t_last is not None else None
            end_ts_ms = now_ms

        if start_ts_ms is None:
            # fallback: request recent_days intraday
            recent_days = int(data_cfg.get("backfill", {}).get("intraday_recent_days", 90))
            start_ts_ms = int(pd.Timestamp.utcnow().value // 1_000_000) - recent_days * 86400 * 1000
            end_ts_ms = now_ms

        # Build segments excluding weekend windows
        segments = week_exclusion_segments(start_ts_ms, end_ts_ms) or [{"start": start_ts_ms, "end": end_ts_ms}]

        for seg in segments:
            a = int(seg["start"]); b = int(seg["end"])
            if b <= a:
                continue
            # compute existing minute buckets and missing minute ranges
            existing = existing_tick_minutes(a, b)
            missing_ranges = missing_minute_ranges(a, b, existing)
            for mr in missing_ranges:
                df_report["missing_ranges"].append(mr)
                rep = fetch_and_persist_ticks(mr["start"], mr["end"])
                df_report["provider_rows"] += rep.get("provider_rows", 0)
                df_report["ticks_inserted"] += rep.get("inserted", 0)

        # After ticks persisted, aggregate ticks into candles for requested timeframes
        try:
            # Collect ticks for the full segments
            all_ticks_frames = []
            for seg in segments:
                a = int(seg["start"]); b = int(seg["end"])
                if b <= a:
                    continue
                q = _text("SELECT ts_utc, price, volume FROM market_data_ticks WHERE symbol=:s AND timeframe=:tf AND ts_utc >= :a AND ts_utc < :b ORDER BY ts_utc ASC")
                with self.engine.connect() as conn:
                    rows = conn.execute(q, {"s": symbol, "tf": timeframe, "a": a, "b": b}).fetchall()
                    if not rows:
                        continue
                    df_ticks = pd.DataFrame([dict(r._mapping) if hasattr(r, "_mapping") else {"ts_utc": r[0], "price": r[1], "volume": r[2] if len(r)>2 else None} for r in rows])
                    all_ticks_frames.append(df_ticks)
            if all_ticks_frames:
                all_ticks = pd.concat(all_ticks_frames, ignore_index=True)
                all_ticks["ts_dt"] = pd.to_datetime(all_ticks["ts_utc"].astype("int64"), unit="ms", utc=True)
                all_ticks = all_ticks.set_index("ts_dt").sort_index()

                # Define target timeframes and pandas rules
                tf_rules = {
                    "1m": "1min",
                    "5m": "5min",
                    "15m": "15min",
                    "30m": "30min",
                    "1h": "60min",
                    "5h": "300min",
                    "12h": "720min",
                    "1d": "1D",
                    "5d": "5D",
                    "10d": "10D",
                    "15d": "15D",
                    "30d": "30D",
                }

                for tgt_tf, rule in tf_rules.items():
                    ohlc = all_ticks["price"].resample(rule, label="right", closed="right").agg(["first", "max", "min", "last"])
                    if ohlc.empty:
                        continue
                    vol = all_ticks["volume"].resample(rule, label="right", closed="right").sum() if "volume" in all_ticks.columns else None
                    candles_rows = []
                    for idx, row in ohlc.iterrows():
                        if row.isnull().all():
                            continue
                        ts_ms = int(idx.tz_convert("UTC").timestamp() * 1000)
                        o = float(row["first"]) if not pd.isna(row["first"]) else 0.0
                        h = float(row["max"]) if not pd.isna(row["max"]) else o
                        l = float(row["min"]) if not pd.isna(row["min"]) else o
                        c = float(row["last"]) if not pd.isna(row["last"]) else o
                        v = float(vol.loc[idx]) if (vol is not None and idx in vol.index and not pd.isna(vol.loc[idx])) else None
                        candles_rows.append({"ts_utc": ts_ms, "open": o, "high": h, "low": l, "close": c, "volume": v})
                    if candles_rows:
                        df_candles = pd.DataFrame(candles_rows)
                        rep = data_io.upsert_candles(self.engine, df_candles, symbol, tgt_tf, resampled=True)
                        df_report["candles_upserted"] += len(df_candles)
                        df_report.setdefault("upsert_reports", []).append({"timeframe": tgt_tf, "report": rep})
        except Exception as e:
            logger.exception("Aggregation/upsert ticks->candles failed: {}", e)
            df_report["aggregate_error"] = str(e)

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

        This implementation is robust to different config shapes:
          - pydantic Settings with extra fields
          - plain dict loaded from YAML
          - missing data/timeframes -> sensible defaults
        """
        cfg = self.cfg

        # Try various ways to obtain the 'data' section
        data_cfg = None
        try:
            data_cfg = getattr(cfg, "data", None)
        except Exception:
            data_cfg = None

        if data_cfg is None:
            try:
                # pydantic v2: model_dump returns dict
                if hasattr(cfg, "model_dump"):
                    data_cfg = cfg.model_dump().get("data", None)
                elif isinstance(cfg, dict):
                    data_cfg = cfg.get("data", None)
            except Exception:
                data_cfg = None

        # Extract symbols
        symbols = []
        try:
            if data_cfg is None:
                symbols = []
            elif isinstance(data_cfg, dict):
                symbols = data_cfg.get("symbols", []) or []
            else:
                # pydantic container-like object
                symbols = getattr(data_cfg, "symbols", []) or []
        except Exception:
            symbols = []

        # Fallback default if no symbols configured
        if not symbols:
            symbols = ["EUR/USD"]

        # Extract timeframes (prefer data.timeframes.native, then top-level timeframes.native)
        timeframes = []
        try:
            if data_cfg is None:
                timeframes = []
            elif isinstance(data_cfg, dict):
                tf = data_cfg.get("timeframes", {})
                if isinstance(tf, dict):
                    timeframes = tf.get("native", []) or []
                else:
                    # unexpected shape
                    timeframes = []
            else:
                tf = getattr(data_cfg, "timeframes", None)
                if tf is None:
                    timeframes = []
                else:
                    # try attribute 'native' or dict-like get
                    try:
                        timeframes = getattr(tf, "native", None) or (tf.get("native") if hasattr(tf, "get") else [])
                    except Exception:
                        timeframes = []
        except Exception:
            timeframes = []

        # If still empty, try top-level timeframes in config
        if not timeframes:
            try:
                tf_cfg = getattr(cfg, "timeframes", None)
                if tf_cfg is None and hasattr(cfg, "model_dump"):
                    tf_cfg = cfg.model_dump().get("timeframes", None)
                if isinstance(tf_cfg, dict):
                    timeframes = tf_cfg.get("native", []) or []
                else:
                    timeframes = getattr(tf_cfg, "native", None) or []
            except Exception:
                timeframes = []

        # final fallback defaults
        if not timeframes:
            timeframes = ["1m", "1d"]

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
