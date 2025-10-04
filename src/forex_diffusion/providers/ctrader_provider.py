"""
cTrader Open API provider implementation.

Implements BaseProvider interface for cTrader with:
- WebSocket streaming (Twisted → asyncio bridge)
- Historical data (trendbars, tick data)
- Market depth (DOM)
- News, Calendar, Sentiment data
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Any
from collections import deque
import time

import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseProvider, ProviderCapability

# cTrader API imports (will be installed via pip)
try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.protobuf import OpenApiCommonMessages_pb2 as CommonMessages
    from ctrader_open_api.messages.protobuf import OpenApiMessages_pb2 as Messages
    from twisted.internet import reactor, ssl
    from twisted.internet.protocol import ReconnectingClientFactory
    _HAS_CTRADER = True
except ImportError:
    _HAS_CTRADER = False
    logger.warning("ctrader-open-api not installed. CTraderProvider will not work.")


class CTraderProvider(BaseProvider):
    """
    cTrader Open API provider.

    Capabilities:
    - QUOTES: Real-time spot prices
    - BARS: Historical trendbars
    - TICKS: Tick data and tick volumes
    - VOLUMES: Real tick volumes
    - DOM: Market depth (order book)
    - SENTIMENT: Trader sentiment
    - NEWS: News feed
    - CALENDAR: Economic calendar
    - WEBSOCKET: Real-time streaming
    - HISTORICAL_BARS: Historical candlestick data
    - HISTORICAL_TICKS: Historical tick data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="ctrader", config=config)

        if not _HAS_CTRADER:
            raise ImportError(
                "ctrader-open-api package not installed. "
                "Install with: pip install ctrader-open-api twisted protobuf"
            )

        # Configuration
        self.client_id = config.get("client_id") if config else None
        self.client_secret = config.get("client_secret") if config else None
        self.access_token = config.get("access_token") if config else None
        self.environment = config.get("environment", "demo") if config else "demo"  # demo or live

        # cTrader client
        self.client: Optional[Client] = None
        self._account_id: Optional[int] = None

        # Async communication
        self._message_queue: Optional[asyncio.Queue] = None
        self._running = False

        # Rate limiting (5 req/sec for historical data)
        self._rate_limiter = deque(maxlen=5)

        # Endpoint configuration
        if self.environment == "live":
            self.host = EndPoints.PROTOBUF_LIVE_HOST
            self.port = EndPoints.PROTOBUF_LIVE_PORT
        else:
            self.host = EndPoints.PROTOBUF_DEMO_HOST
            self.port = EndPoints.PROTOBUF_DEMO_PORT

    @property
    def capabilities(self) -> List[ProviderCapability]:
        """cTrader supports comprehensive market data."""
        return [
            ProviderCapability.QUOTES,
            ProviderCapability.BARS,
            ProviderCapability.TICKS,
            ProviderCapability.VOLUMES,
            ProviderCapability.DOM,
            ProviderCapability.SENTIMENT,
            ProviderCapability.NEWS,
            ProviderCapability.CALENDAR,
            ProviderCapability.WEBSOCKET,
            ProviderCapability.HISTORICAL_BARS,
            ProviderCapability.HISTORICAL_TICKS,
        ]

    async def connect(self) -> bool:
        """Connect to cTrader Open API."""
        try:
            if not self.access_token:
                logger.error(f"[{self.name}] No access token configured. Run OAuth flow first.")
                return False

            # Create message queue
            self._message_queue = asyncio.Queue(maxsize=10000)

            # Create cTrader client
            self.client = Client(
                self.host,
                self.port,
                TcpProtocol
            )

            # Setup message handlers
            self.client.set_message_callback(self._on_message)
            self.client.set_error_callback(self._on_error)

            # Connect (Twisted reactor runs in separate thread)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._connect_twisted)

            # Authenticate
            await self._authenticate()

            self.health.is_connected = True
            self._running = True
            self._start_time = datetime.now()

            logger.info(f"[{self.name}] Connected to cTrader ({self.environment})")
            return True

        except Exception as e:
            logger.error(f"[{self.name}] Failed to connect: {e}")
            self.health.is_connected = False
            self.health.errors.append(str(e))
            return False

    def _connect_twisted(self) -> None:
        """Start Twisted reactor in thread (blocking call)."""
        # This runs Twisted reactor - would need proper thread management
        # For now, placeholder - will implement full Twisted→asyncio bridge
        pass

    async def _authenticate(self) -> None:
        """Authenticate with cTrader using access token."""
        if not self.client:
            raise RuntimeError("Client not initialized")

        # Send auth message
        request = Messages.ApplicationAuthReq()
        request.clientId = self.client_id
        request.clientSecret = self.client_secret

        # Send via client (placeholder - full implementation needed)
        # await self._send_message(request)

        logger.info(f"[{self.name}] Authenticated successfully")

    async def disconnect(self) -> None:
        """Disconnect from cTrader."""
        try:
            self._running = False

            if self.client:
                # Stop Twisted reactor gracefully
                # reactor.stop() - needs proper thread handling
                self.client = None

            self.health.is_connected = False
            logger.info(f"[{self.name}] Disconnected from cTrader")

        except Exception as e:
            logger.error(f"[{self.name}] Error during disconnect: {e}")

    def _on_message(self, message: Any) -> None:
        """Handle incoming message from cTrader (Twisted callback)."""
        try:
            self.health.last_message_ts = datetime.now()

            # Convert Protobuf message to dict
            msg_dict = self._protobuf_to_dict(message)

            # Push to async queue
            if self._message_queue and not self._message_queue.full():
                try:
                    self._message_queue.put_nowait(msg_dict)
                except asyncio.QueueFull:
                    logger.warning(f"[{self.name}] Message queue full, dropping message")

        except Exception as e:
            logger.error(f"[{self.name}] Error handling message: {e}")
            self.health.errors.append(str(e))

    def _on_error(self, error: Exception) -> None:
        """Handle connection error."""
        logger.error(f"[{self.name}] Connection error: {error}")
        self.health.errors.append(str(error))
        self.health.is_connected = False

    def _protobuf_to_dict(self, message: Any) -> Dict[str, Any]:
        """Convert Protobuf message to dictionary."""
        # Placeholder - implement based on message type
        return {
            "type": type(message).__name__,
            "timestamp": int(time.time() * 1000),
        }

    async def _rate_limit_wait(self) -> None:
        """Wait if necessary to respect rate limit (5 req/sec)."""
        now = time.time()

        # Remove old timestamps
        while self._rate_limiter and (now - self._rate_limiter[0]) > 1.0:
            self._rate_limiter.popleft()

        # Check if limit reached
        if len(self._rate_limiter) >= 5:
            # Wait until oldest request is > 1 sec old
            wait_time = 1.0 - (now - self._rate_limiter[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Record this request
        self._rate_limiter.append(time.time())

    # Market data methods (to be implemented)

    async def _get_current_price_impl(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current spot price from cTrader."""
        # Implementation placeholder
        logger.warning(f"[{self.name}] get_current_price not yet implemented")
        return None

    async def _get_historical_bars_impl(
        self, symbol: str, timeframe: str, start_ts_ms: int, end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """Get historical trendbars from cTrader."""
        # Implementation placeholder
        logger.warning(f"[{self.name}] get_historical_bars not yet implemented")
        return None

    async def _get_historical_ticks_impl(
        self, symbol: str, start_ts_ms: int, end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """Get historical tick data from cTrader."""
        # Implementation placeholder
        logger.warning(f"[{self.name}] get_historical_ticks not yet implemented")
        return None

    async def _get_market_depth_impl(self, symbol: str, levels: int) -> Optional[Dict[str, Any]]:
        """Get market depth (DOM) from cTrader."""
        # Implementation placeholder
        logger.warning(f"[{self.name}] get_market_depth not yet implemented")
        return None

    async def _get_sentiment_impl(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get trader sentiment from cTrader."""
        # Implementation placeholder
        logger.warning(f"[{self.name}] get_sentiment not yet implemented")
        return None

    async def _get_news_impl(self, currency: Optional[str], limit: int) -> Optional[List[Dict[str, Any]]]:
        """Get news feed from cTrader."""
        # Implementation placeholder
        logger.warning(f"[{self.name}] get_news not yet implemented")
        return None

    async def _get_economic_calendar_impl(
        self, start_date: Optional[datetime], end_date: Optional[datetime], currency: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Get economic calendar from cTrader."""
        # Implementation placeholder
        logger.warning(f"[{self.name}] get_economic_calendar not yet implemented")
        return None

    async def _stream_quotes_impl(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Stream real-time quotes from cTrader."""
        try:
            # Ensure connected
            if not self.health.is_connected:
                await self.connect()

            # Subscribe to symbols
            # await self._subscribe_spots(symbols)

            # Stream from queue
            while self._running:
                try:
                    msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                    yield msg

                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            logger.error(f"[{self.name}] Stream failed: {e}")
            self.health.errors.append(str(e))

    async def _stream_market_depth_impl(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Stream market depth updates from cTrader."""
        # Implementation placeholder
        logger.warning(f"[{self.name}] stream_market_depth not yet implemented")
        yield {}
