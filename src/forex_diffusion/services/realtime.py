"""
Real-time ingestion service via polling.

This service provides a fallback mechanism for real-time data by polling a provider's
current price endpoint at a configured interval. It's simpler than a WebSocket
connection and serves as a robust backup.

- Polls a provider's `get_current_price` endpoint.
- Persists the retrieved price data as a raw tick by directly calling `DBService.write_tick`.
- The `AggregatorService` then processes these ticks to build candles.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

from loguru import logger

from ..services.marketdata import MarketDataService
from ..services.db_service import DBService
from ..utils.config import get_config

class RealTimeIngestionService:
    """
    A service that polls for real-time prices and writes them as ticks to the DB.
    """
    def __init__(
        self,
        db_service: DBService,
        market_service: MarketDataService,
        symbols: Optional[List[str]] = None,
        poll_interval: float = 2.0, # Polling is less frequent than streaming
    ):
        self.db_service = db_service
        self.market_service = market_service
        self.provider = self.market_service.provider
        self.symbols = symbols or self._get_symbols_from_config()
        self.poll_interval = float(poll_interval)
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="RealTimeIngest", daemon=True)
        self._thread.start()
        logger.info(f"RealTimeIngestionService started for symbols: {self.symbols}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("RealTimeIngestionService stopped")

    def _run_loop(self):
        """Main loop: polls the provider for each symbol in a round-robin fashion."""
        while not self._stop_event.is_set():
            for symbol in self.symbols:
                if self._stop_event.is_set():
                    break
                try:
                    self._poll_and_write_tick(symbol)
                except Exception as e:
                    logger.exception(f"RealTime: polling error for {symbol}: {e}")
                time.sleep(self.poll_interval)
        logger.debug("RealTime polling loop exited")

    def _poll_and_write_tick(self, symbol: str):
        """Polls the provider for the current price and persists it as a raw tick."""
        try:
            data = self.provider.get_current_price(symbol)
            if not data or not isinstance(data, dict):
                logger.debug(f"Polling for {symbol} returned no data.")
                return

            # The provider should return a dict with standard keys.
            # We ensure basic validation before creating the tick payload.
            price = data.get("price")
            ts_utc = data.get("ts_utc")

            if price is None or ts_utc is None:
                logger.warning(f"Polling data for {symbol} is missing 'price' or 'ts_utc'. Data: {data}")
                return

            tick_payload = {
                "symbol": symbol,
                "ts_utc": int(ts_utc),
                "price": float(price),
                "bid": float(data["bid"]) if data.get("bid") is not None else None,
                "ask": float(data["ask"]) if data.get("ask") is not None else None,
                "volume": None, # Polling usually doesn't provide volume
            }

            self.db_service.write_tick(tick_payload)
            logger.debug(f"Persisted polled tick for {symbol} at {ts_utc}")

        except Exception as e:
            logger.exception(f"Error during polling and writing tick for {symbol}: {e}")

    def _get_symbols_from_config(self) -> List[str]:
        try:
            cfg = get_config()
            data_cfg = getattr(cfg, "data", {})
            return data_cfg.get("symbols", ["EUR/USD"])
        except Exception:
            return ["EUR/USD"]
