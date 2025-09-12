"""
Real-time ingest service.

 - Polls a provider's current price endpoint at a configured interval.
 - Persists the retrieved price data as a raw tick in the `market_data_ticks` table.
 - An `AggregatorService` runs in the background to periodically build candles from these raw ticks.
 - On the first tick for a symbol, it triggers a historical backfill to ensure data continuity.
 """

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from ..data import io as data_io
from ..services.marketdata import MarketDataService
from ..services.db_service import DBService
from ..services.aggregator import AggregatorService
from ..utils.config import get_config


class RealTimeIngestService:
    """
    Real-time ingestion and backfill coordinator.

    Args:
      engine: SQLAlchemy engine.
       market_service: An instance of MarketDataService to interact with data providers.
       symbols: A list of symbols to poll.
       timeframe: The base timeframe, typically '1m'.
       poll_interval: Seconds between polling attempts for each symbol.
     """
    def __init__(
        self,
        engine,
        market_service: Optional[MarketDataService] = None,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1m",
        poll_interval: float = 1.0,
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
        # Service to aggregate ticks into candles
        self.aggregator_service = AggregatorService(engine=self.engine, symbols=self.symbols)


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
        self.aggregator_service.start()
        self._thread.start()
        logger.info("RealTimeIngestService started for symbols: {}", self.symbols)

    def stop(self):
        """Signal stop and wait for thread to finish."""
        self.aggregator_service.stop()
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
                    self._poll_and_write_tick(sym)
                except Exception as e:
                    logger.exception("RealTime: polling error for {}: {}", sym, e)
                time.sleep(self.poll_interval)
        logger.debug("RealTime polling loop exited")

    def _poll_and_write_tick(self, symbol: str):
        """Poll provider for current price and persist it as a raw tick."""
        try:
            data = None
            if hasattr(self.provider, "get_current_price"):
                data = self.provider.get_current_price(symbol)
            elif hasattr(self.provider, "get_historical"):
                logger.debug(f"Provider for {symbol} has no get_current_price, falling back to get_historical.")
                # fallback: request very recent historical 1m bar
                now_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
                data_df = self.provider.get_historical(symbol=symbol, timeframe=self.timeframe, start_ts_ms=now_ms - 60_000, end_ts_ms=now_ms)
                if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                    # take last row as tick
                    last = data_df.iloc[-1]
                    data = {"ts_utc": int(last["ts_utc"]), "price": float(last["close"])}
            if data is None:
                logger.debug(f"Provider returned no data for {symbol}.")
                return

            logger.debug(f"Provider data for {symbol}: {data}")
            ts_ms = None
            price = None

            bid = None
            ask = None

            if isinstance(data, pd.DataFrame):
                try:
                    last = data.iloc[-1]
                    # common column names produced by AlphaVantage
                    cand_cols = [
                        "5. Exchange Rate",
                        "Exchange Rate",
                        "8. Bid Price",
                        "Bid Price",
                        "9. Ask Price",
                        "Ask Price",
                        "5. exchange_rate",  # fallback lowercase variants
                    ]
                    for c in cand_cols:
                        if c in last.index:
                            try:
                                price = float(last[c])
                                break
                            except Exception:
                                continue
                    # fallback: take first numeric value in the row
                    if price is None:
                        for v in last:
                            try:
                                price = float(v)
                                break
                            except Exception:
                                continue
                    # try to parse timestamp field if present
                    if "6. Last Refreshed" in last.index:
                        try:
                            dt = pd.to_datetime(last["6. Last Refreshed"], utc=True)
                            try:
                                ts_ms = int(dt.value // 1_000_000)
                            except Exception:
                                ts_ms = int(pd.Timestamp(dt).timestamp() * 1000)
                        except Exception:
                            ts_ms = None
                except Exception:
                    # fall back to other handlers below
                    price = None
                    ts_ms = None

            # data dictionaries may vary by provider
            elif isinstance(data, dict):
                # AlphaVantage get_current_price returns nested dict keys; try to parse common fields
                if "Realtime Currency Exchange Rate" in data:
                    r = data["Realtime Currency Exchange Rate"]
                    try:
                        price = float(r.get("5. Exchange Rate") or r.get("Exchange Rate") or 0.0)
                        bid = float(r.get("8. Bid Price")) if r.get("8. Bid Price") else None
                        ask = float(r.get("9. Ask Price")) if r.get("9. Ask Price") else None
                        ts_str = r.get("6. Last Refreshed")
                        if ts_str:
                            ts_ms = int(pd.to_datetime(ts_str, utc=True).value // 1_000_000)
                    except Exception:
                        price = None
                elif "price" in data and "ts_utc" in data:
                    price = float(data["price"])
                    ts_ms = int(data["ts_utc"])
                    bid = float(data["bid"]) if data.get("bid") else None
                    ask = float(data["ask"]) if data.get("ask") else None
                elif "price" in data:
                    price = float(data["price"])

            if ts_ms is None:
                ts_ms = int(time.time() * 1000)

            if price is None:
                logger.debug("RealTime: no price found for {}. raw data: {}", symbol, data)
                return

            tick_payload = {
                "symbol": symbol,
                "timeframe": "tick",  # Mark as a raw tick
                "ts_utc": int(ts_ms),
                "price": float(price),
                "bid": bid,
                "ask": ask,
                "volume": None,
            }

            logger.debug(f"Constructed tick payload: {tick_payload}")
            dbs = DBService(engine=self.engine)
            dbs.write_tick(tick_payload)

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
        Perform a historical backfill for the symbol using the main MarketDataService.
         """
        try:
            logger.info(f"RealTime backfill triggered for {symbol}. Delegating to MarketDataService.")
            # The main backfill function is smart enough to find all gaps in a given window.
            # We trigger it to ensure the history is complete up to the point real-time started.
            # Let's use a reasonable lookback, e.g., a few months, to catch any recent gaps.
            report = self.market_service.backfill_symbol_timeframe(
                symbol=symbol, timeframe=self.timeframe, months=3
            )
            logger.info(f"RealTime backfill for {symbol} completed. Report: {report}")

        except Exception as e:
            logger.exception("RealTime backfill failed for {}: {}", symbol, e)
