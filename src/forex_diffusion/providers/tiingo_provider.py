"""
Tiingo provider implementation.

Wraps existing TiingoClient and TiingoWSConnector to conform to BaseProvider interface.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Any

import pandas as pd
from loguru import logger

from .base import BaseProvider, ProviderCapability
from ..services.marketdata import TiingoClient
from ..services.tiingo_ws_connector import TiingoWSConnector


class TiingoProvider(BaseProvider):
    """
    Tiingo market data provider.

    Capabilities:
    - QUOTES: Real-time via WebSocket
    - BARS: Historical OHLCV via REST
    - HISTORICAL_BARS: Historical candlestick data
    - WEBSOCKET: Real-time streaming
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="tiingo", config=config)

        self.api_key = config.get("api_key") if config else None
        self.ws_uri = config.get("ws_uri", "wss://api.tiingo.com/fx") if config else "wss://api.tiingo.com/fx"
        self.tickers = config.get("tickers", ["eurusd"]) if config else ["eurusd"]

        # REST client for historical data
        self.rest_client = TiingoClient(api_key=self.api_key)

        # WebSocket connector for real-time data
        self.ws_connector: Optional[TiingoWSConnector] = None
        self._ws_queue: Optional[asyncio.Queue] = None
        self._ws_running = False

    @property
    def capabilities(self) -> List[ProviderCapability]:
        """Tiingo supports quotes, bars, and WebSocket streaming."""
        return [
            ProviderCapability.QUOTES,
            ProviderCapability.BARS,
            ProviderCapability.HISTORICAL_BARS,
            ProviderCapability.WEBSOCKET,
        ]

    async def connect(self) -> bool:
        """Connect to Tiingo (WebSocket for real-time)."""
        try:
            if self.ws_connector is None:
                # Create queue for async communication
                self._ws_queue = asyncio.Queue(maxsize=10000)

                # Create WS connector with handlers
                self.ws_connector = TiingoWSConnector(
                    uri=self.ws_uri,
                    api_key=self.api_key,
                    tickers=self.tickers,
                    chart_handler=self._on_ws_message,
                    db_handler=None,  # DB writing handled separately
                    status_handler=self._on_ws_status,
                )

            # Start WebSocket in thread
            self.ws_connector.start()
            self._ws_running = True
            self.health.is_connected = True
            self._start_time = datetime.now()

            logger.info(f"[{self.name}] Connected to Tiingo WebSocket")
            return True

        except Exception as e:
            logger.error(f"[{self.name}] Failed to connect: {e}")
            self.health.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Tiingo WebSocket."""
        try:
            if self.ws_connector:
                self.ws_connector.stop()
                self.ws_connector = None

            self._ws_running = False
            self.health.is_connected = False
            logger.info(f"[{self.name}] Disconnected from Tiingo")

        except Exception as e:
            logger.error(f"[{self.name}] Error during disconnect: {e}")

    def _on_ws_message(self, payload: Dict[str, Any]) -> None:
        """Handle WebSocket message (callback from WS thread)."""
        try:
            # Update health metrics
            self.health.last_message_ts = datetime.now()

            # Push to async queue (non-blocking)
            if self._ws_queue and not self._ws_queue.full():
                # Use sync queue put since we're in sync callback
                try:
                    self._ws_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    logger.warning(f"[{self.name}] WS queue full, dropping message")

        except Exception as e:
            logger.error(f"[{self.name}] Error in WS message handler: {e}")
            self.health.errors.append(str(e))

    def _on_ws_status(self, status: str) -> None:
        """Handle WebSocket status changes."""
        logger.debug(f"[{self.name}] WebSocket status: {status}")

        if status == "ws_down":
            self.health.is_connected = False
        elif status == "ws_restored":
            self.health.is_connected = True

    # Market data methods

    async def _get_current_price_impl(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price from WebSocket stream (last message)."""
        # Not directly supported - use REST or stream
        logger.warning(f"[{self.name}] get_current_price not directly supported, use stream_quotes")
        return None

    async def _get_historical_bars_impl(
        self, symbol: str, timeframe: str, start_ts_ms: int, end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """Get historical bars from Tiingo REST API."""
        try:
            # Convert timestamps to dates
            start_dt = datetime.fromtimestamp(start_ts_ms / 1000)
            end_dt = datetime.fromtimestamp(end_ts_ms / 1000)

            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")

            # Map timeframe to Tiingo resampleFreq
            resample_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1hour",
                "4h": "4hour",
                "1d": "1day",
            }
            resample_freq = resample_map.get(timeframe, "1min")

            # Use existing TiingoClient
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self.rest_client.get_candles,
                symbol,
                start_date,
                end_date,
                resample_freq,
            )

            logger.info(f"[{self.name}] Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get historical bars: {e}")
            self.health.errors.append(str(e))
            return None

    async def _stream_quotes_impl(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Stream real-time quotes from WebSocket."""
        try:
            # Update subscribed tickers
            self.tickers = [s.lower().replace("/", "") for s in symbols]

            # Ensure connected
            if not self.health.is_connected:
                await self.connect()

            # Stream from queue
            while self._ws_running:
                try:
                    # Wait for message with timeout
                    msg = await asyncio.wait_for(self._ws_queue.get(), timeout=1.0)
                    yield msg

                except asyncio.TimeoutError:
                    # No message, continue
                    continue

                except Exception as e:
                    logger.error(f"[{self.name}] Error in stream: {e}")
                    self.health.errors.append(str(e))
                    break

        except Exception as e:
            logger.error(f"[{self.name}] Stream failed: {e}")
            self.health.errors.append(str(e))
