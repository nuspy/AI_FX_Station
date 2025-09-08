"""
Real-time ingest service.

- Polls provider current price endpoint at configured interval (per-symbol)
- Converts tick => 1-minute candle (open=high=low=close=price) at tick timestamp
- Validates and upserts candles via data.io.upsert_candles (runs in worker thread)
- On first tick per symbol records rt_start_ts and launches backfill for interval
  (last_db_ts_before + 1 .. rt_start_ts - 1) in a separate thread to avoid blocking real-time ingestion.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from ..data import io as data_io
from ..services.marketdata import AlphaVantageClient, MarketDataService
from ..utils.config import get_config


class RealTimeIngestService:
    """
    Real-time ingestion and backfill coordinator.

    Args:
      engine: SQLAlchemy engine
      market_service: MarketDataService instance (provides provider client)
      symbols: list of symbols to subscribe
      timeframe: target candle timeframe for upsert (e.g., '1m')
      poll_interval: seconds between provider polls per symbol
      db_writer: optional DBWriter for async writes (used for logging only; candles upsert executed in thread)
    """
    def __init__(
        self,
        engine,
        market_service: Optional[MarketDataService] = None,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1m",
        poll_interval: float = 1.0,
        db_writer: Optional[Any] = None,
    ):
        self.engine = engine
        self.cfg = get_config()
        self.market_service = market_service or MarketDataService()
        self.provider = self.market_service.provider

        # Resolve symbols robustly: prefer explicit parameter, then config.data.symbols (supports pydantic Settings or plain dict),
        # finally fallback to a sensible default.
        resolved_symbols: List[str] = []
        if symbols:
            resolved_symbols = list(symbols)
        else:
            try:
                data_section = getattr(self.cfg, "data", None)
            except Exception:
                data_section = None
            if data_section is None:
                try:
                    if hasattr(self.cfg, "model_dump"):
                        data_section = self.cfg.model_dump().get("data", None)
                    elif isinstance(self.cfg, dict):
                        data_section = self.cfg.get("data", None)
                except Exception:
                    data_section = None
            # normalize to dict-like
            if isinstance(data_section, dict):
                resolved_symbols = data_section.get("symbols", []) or []
            else:
                try:
                    resolved_symbols = list(getattr(data_section, "symbols", []) or [])
                except Exception:
                    resolved_symbols = []
        # final fallback
        if not resolved_symbols:
            resolved_symbols = ["EUR/USD"]

        self.symbols = resolved_symbols
        self.timeframe = timeframe
        self.poll_interval = float(poll_interval)
        self._thread = None
        self._stop_event = threading.Event()
        # per-symbol metadata
        self._last_db_ts_before: Dict[str, Optional[int]] = {}
        self._rt_start_ts: Dict[str, Optional[int]] = {}
        self._first_tick_seen: Dict[str, bool] = {}
        # lock for thread-safe access
        self._lock = threading.Lock()
        # optional background writer to persist features asynchronously
        self.db_writer = db_writer

    def start(self):
        """Start background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="RealTimeIngest", daemon=True)
        # initialize per-symbol last_db timestamps
        for sym in self.symbols:
            try:
                ts = data_io.get_last_ts_for_symbol_tf(self.engine, sym, self.timeframe)
                self._last_db_ts_before[sym] = ts
                self._first_tick_seen[sym] = False
                self._rt_start_ts[sym] = None
                logger.info("RealTime: last_db_ts for {} {} = {}", sym, self.timeframe, ts)
            except Exception as e:
                logger.warning("RealTime: could not get last_db_ts for {} {}: {}", sym, self.timeframe, e)
                self._last_db_ts_before[sym] = None
                self._first_tick_seen[sym] = False
                self._rt_start_ts[sym] = None
        self._thread.start()
        logger.info("RealTimeIngestService started for symbols: {}", self.symbols)

    def stop(self):
        """Signal stop and wait for thread to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("RealTimeIngestService stopped")

    def _run_loop(self):
        """Main loop: poll provider for each symbol in round-robin."""
        while not self._stop_event.is_set():
            for sym in list(self.symbols):
                if self._stop_event.is_set():
                    break
                try:
                    self._poll_symbol(sym)
                except Exception as e:
                    logger.exception("RealTime: polling error for {}: {}", sym, e)
                time.sleep(self.poll_interval)
        logger.debug("RealTime polling loop exited")

    def _poll_symbol(self, symbol: str):
        """Poll provider for current price and process it."""
        # provider may expose get_current_price or currentPrices endpoint
        try:
            # provider client interface: get_current_price(symbol) -> dict
            data = None
            if hasattr(self.provider, "get_current_price"):
                data = self.provider.get_current_price(symbol)
            elif hasattr(self.provider, "get_historical"):
                # fallback: request very recent historical 1m bar
                now_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
                data_df = self.provider.get_historical(symbol=symbol, timeframe=self.timeframe, start_ts_ms=now_ms - 60_000, end_ts_ms=now_ms)
                if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                    # take last row as tick
                    last = data_df.iloc[-1]
                    data = {"ts_utc": int(last["ts_utc"]), "price": float(last["close"])}
            if data is None:
                return
            # normalize price and timestamp
            ts_ms = None
            price = None
            # data dictionaries may vary by provider
            if isinstance(data, dict):
                # AlphaVantage get_current_price returns nested dict keys; try to parse common fields
                if "Realtime Currency Exchange Rate" in data:
                    r = data["Realtime Currency Exchange Rate"]
                    price = float(r.get("5. Exchange Rate") or r.get("5. Exchange Rate", 0.0))
                    # timestamp field may exist
                    ts_ms = None
                elif "price" in data and "ts_utc" in data:
                    price = float(data["price"])
                    ts_ms = int(data["ts_utc"])
                elif "price" in data:
                    price = float(data["price"])
                else:
                    # attempt to find any numeric value in dict
                    for k, v in data.items():
                        try:
                            price = float(v)
                            break
                        except Exception:
                            continue
            # fallback timestamp to now
            if ts_ms is None:
                ts_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
            if price is None:
                # Log raw provider response to debug parsing issues
                try:
                    logger.debug("RealTime: no price found for {}. raw data: {}", symbol, data)
                except Exception:
                    logger.debug("RealTime: no price found for {}", symbol)
                return

            # Build a 1-row candle (open/high/low/close = price)
            candle = {
                "ts_utc": int(ts_ms),
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
                "volume": None,
            }
            df = pd.DataFrame([candle])
            # Validate and upsert (runs in this background thread)
            dfv, vreport = data_io.validate_candles_df(df, symbol=symbol, timeframe=self.timeframe)
            try:
                # upsert into DB synchronously (but we're in background thread)
                upsert_report = data_io.upsert_candles(self.engine, dfv, symbol, self.timeframe, resampled=False)
                logger.debug("RealTime: upsert report for {} {}: {}", symbol, self.timeframe, upsert_report)
            except Exception as e:
                logger.exception("RealTime: failed upsert for {}: {}", symbol, e)

            # Tick counting per-minute: increment counter for minute bucket (ts minute end)
            minute_ts = (int(ts_ms) // 60000) * 60000 + 60000  # period end
            with self._lock:
                prev = getattr(self, "_tick_counts", None)
                if prev is None:
                    self._tick_counts = {}
                key = (symbol, minute_ts)
                self._tick_counts[key] = self._tick_counts.get(key, 0) + 1
                # if minute bucket complete (current ts_ms beyond minute end), persist and clear
                # approximate: if current time > minute_ts + small slack
                now_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
                slack = 2000  # 2 sec slack
                if now_ms >= minute_ts + slack:
                    count = self._tick_counts.pop(key, 0)
                    try:
                        if getattr(self, "db_writer", None) is not None:
                            self.db_writer.write_tick_async(symbol=symbol, timeframe=self.timeframe, ts_utc=minute_ts, tick_count=count)
                        else:
                            from ..services.db_service import DBService
                            dbs = DBService(engine=self.engine)
                            dbs.write_tick_aggregate(symbol=symbol, timeframe=self.timeframe, ts_utc=minute_ts, tick_count=count)
                    except Exception as e:
                        logger.exception("RealTime: failed to persist tick aggregate for {}: {}", symbol, e)

            # On first tick, set rt_start and trigger historical backfill in separate thread
            with self._lock:
                if not self._first_tick_seen.get(symbol, False):
                    self._first_tick_seen[symbol] = True
                    self._rt_start_ts[symbol] = int(ts_ms)
                    last_db_ts = self._last_db_ts_before.get(symbol, None)
                    # Launch backfill to fill gap (last_db_ts+1 .. rt_start-1)
                    backfill_thread = threading.Thread(
                        target=self._run_backfill_for_symbol,
                        args=(symbol, last_db_ts, int(ts_ms)),
                        name=f"Backfill-{symbol}",
                        daemon=True,
                    )
                    backfill_thread.start()

        except Exception as e:
            logger.exception("RealTime: processing error for {}: {}", symbol, e)

    def _run_backfill_for_symbol(self, symbol: str, last_db_ts_before: Optional[int], rt_start_ts_ms: int):
        """
        Perform a historical backfill for the symbol from last_db_ts_before+1 to rt_start_ts_ms-1.
        If last_db_ts_before is None, delegate to market_service.backfill_symbol_timeframe to get full history.
        """
        try:
            if last_db_ts_before is None:
                logger.info("RealTime backfill: no last_db_ts for {}, triggering full backfill", symbol)
                # force full backfill for daily/intraday split as per MarketDataService logic
                self.market_service.backfill_symbol_timeframe(symbol, self.timeframe, force_full=True)
                return
            start_ms = int(last_db_ts_before) + 1
            end_ms = int(rt_start_ts_ms) - 1
            if end_ms <= start_ms:
                logger.info("RealTime backfill: no gap to fill for {} (start={}, end={})", symbol, start_ms, end_ms)
                return
            logger.info("RealTime backfill: fetching {} {} from {} to {}", symbol, self.timeframe, start_ms, end_ms)
            # call data_io.backfill_from_provider which downloads, validates and upserts
            report = data_io.backfill_from_provider(self.engine, self.provider, symbol, self.timeframe, start_ts_ms=start_ms, end_ts_ms=end_ms)
            logger.info("RealTime backfill report for {}: {}", symbol, report)
        except Exception as e:
            logger.exception("RealTime backfill failed for {}: {}", symbol, e)
