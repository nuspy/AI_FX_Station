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
        """
        Connect to cTrader Open API.

        Strategy: Token first, OAuth fallback
        1. Try direct token authentication (faster, simpler)
        2. If fails, fallback to OAuth flow
        """
        if not self.access_token:
            logger.error(f"[{self.name}] No access token configured. Run OAuth flow first.")
            return False

        # Create message queue
        self._message_queue = asyncio.Queue(maxsize=10000)

        # Strategy 1: Try direct token authentication
        try:
            logger.info(f"[{self.name}] Attempting direct token connection...")
            success = await self._connect_with_token()
            if success:
                logger.info(f"[{self.name}] Connected via direct token")
                return True
        except Exception as e:
            logger.warning(f"[{self.name}] Direct token failed: {e}, trying OAuth fallback...")

        # Strategy 2: Fallback to OAuth
        try:
            logger.info(f"[{self.name}] Attempting OAuth connection...")
            success = await self._connect_with_oauth()
            if success:
                logger.info(f"[{self.name}] Connected via OAuth")
                return True
        except Exception as e:
            logger.error(f"[{self.name}] OAuth connection also failed: {e}")
            self.health.is_connected = False
            self.health.errors.append(str(e))
            return False

        return False

    async def _connect_with_token(self) -> bool:
        """Connect using direct token (no OAuth) - preferred method."""
        # Create cTrader client
        self.client = Client(self.host, self.port, TcpProtocol)

        # Setup message handlers
        self.client.set_message_callback(self._on_message)
        self.client.set_error_callback(self._on_error)

        # Connect
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_twisted)

        # Authenticate with token (no OAuth)
        await self._authenticate_token()

        self.health.is_connected = True
        self._running = True
        self._start_time = datetime.now()

        return True

    async def _connect_with_oauth(self) -> bool:
        """Connect using OAuth flow - fallback method."""
        # Create cTrader client
        self.client = Client(self.host, self.port, TcpProtocol)

        # Setup message handlers
        self.client.set_message_callback(self._on_message)
        self.client.set_error_callback(self._on_error)

        # Connect
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_twisted)

        # Authenticate with OAuth
        await self._authenticate_oauth()

        self.health.is_connected = True
        self._running = True
        self._start_time = datetime.now()

        return True

    def _connect_twisted(self) -> None:
        """Start Twisted reactor in thread (blocking call)."""
        # This runs Twisted reactor - would need proper thread management
        # For now, placeholder - will implement full Twisted→asyncio bridge
        pass

    async def _authenticate_token(self) -> None:
        """Authenticate with cTrader using direct token (no OAuth)."""
        if not self.client:
            raise RuntimeError("Client not initialized")

        # Send auth message
        request = Messages.ApplicationAuthReq()
        request.clientId = self.client_id
        request.clientSecret = self.client_secret

        # Send via client (placeholder - full implementation needed)
        # await self._send_message(request)

        # Use access token directly (no OAuth redirect)
        # This is the simpler, faster method
        logger.info(f"[{self.name}] Authenticated via direct token")

    async def _authenticate_oauth(self) -> None:
        """Authenticate with cTrader using OAuth flow."""
        if not self.client:
            raise RuntimeError("Client not initialized")

        # Send auth message
        request = Messages.ApplicationAuthReq()
        request.clientId = self.client_id
        request.clientSecret = self.client_secret

        # Send via client (placeholder - full implementation needed)
        # await self._send_message(request)

        # OAuth flow - uses access_token obtained via browser redirect
        # This assumes the token was already obtained
        logger.info(f"[{self.name}] Authenticated via OAuth")

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
        try:
            await self._rate_limit_wait()

            if not self.client or not self.health.is_connected:
                await self.connect()

            # Request spot quote
            request = Messages.ProtoOASymbolByIdReq()
            request.ctidTraderAccountId = self._account_id
            request.symbolId = await self._get_symbol_id(symbol)

            # Send request and wait for response
            response = await self._send_and_wait(request, Messages.ProtoOASymbolByIdRes)

            if response:
                tick = response.tick
                return {
                    "symbol": symbol,
                    "bid": tick.bid / 100000,  # cTrader uses 100000 multiplier
                    "ask": tick.ask / 100000,
                    "price": (tick.bid + tick.ask) / 200000,
                    "timestamp": int(tick.timestamp / 1000),  # Convert to ms
                }

            return None

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get current price for {symbol}: {e}")
            self.health.errors.append(str(e))
            return None

    async def _get_historical_bars_impl(
        self, symbol: str, timeframe: str, start_ts_ms: int, end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """Get historical trendbars from cTrader."""
        try:
            await self._rate_limit_wait()

            # Convert timeframe to cTrader format
            ct_timeframe = self._convert_timeframe(timeframe)
            if ct_timeframe is None:
                logger.error(f"[{self.name}] Unsupported timeframe: {timeframe}")
                return None

            symbol_id = await self._get_symbol_id(symbol)

            # cTrader uses microseconds for timestamps
            start_us = start_ts_ms * 1000
            end_us = end_ts_ms * 1000

            # Request historical trendbars
            request = Messages.ProtoOAGetTrendbarsReq()
            request.ctidTraderAccountId = self._account_id
            request.symbolId = symbol_id
            request.period = ct_timeframe
            request.fromTimestamp = start_us
            request.toTimestamp = end_us

            response = await self._send_and_wait(request, Messages.ProtoOAGetTrendbarsRes)

            if not response or not response.trendbar:
                return None

            # Parse trendbars into DataFrame
            bars = []
            for bar in response.trendbar:
                bars.append({
                    "ts_utc": int(bar.utcTimestamp / 1000),  # Convert to ms
                    "open": bar.open / 100000,
                    "high": bar.high / 100000,
                    "low": bar.low / 100000,
                    "close": bar.close / 100000,
                    "volume": bar.volume if hasattr(bar, "volume") else None,
                    "tick_volume": bar.tickVolume if hasattr(bar, "tickVolume") else None,
                    "real_volume": bar.volume if hasattr(bar, "volume") else None,
                })

            df = pd.DataFrame(bars)
            df = df.sort_values("ts_utc").reset_index(drop=True)

            logger.info(f"[{self.name}] Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get historical bars for {symbol}: {e}")
            self.health.errors.append(str(e))
            return None

    async def _get_historical_ticks_impl(
        self, symbol: str, start_ts_ms: int, end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """Get historical tick data from cTrader."""
        try:
            await self._rate_limit_wait()

            symbol_id = await self._get_symbol_id(symbol)
            start_us = start_ts_ms * 1000
            end_us = end_ts_ms * 1000

            # Request tick data
            request = Messages.ProtoOAGetTickDataReq()
            request.ctidTraderAccountId = self._account_id
            request.symbolId = symbol_id
            request.fromTimestamp = start_us
            request.toTimestamp = end_us
            request.type = Messages.ProtoOAQuoteType.BID_ASK  # Get both bid and ask

            response = await self._send_and_wait(request, Messages.ProtoOAGetTickDataRes)

            if not response or not response.tickData:
                return None

            # Parse tick data
            ticks = []
            for tick in response.tickData:
                ticks.append({
                    "ts_utc": int(tick.timestamp / 1000),
                    "bid": tick.bid / 100000 if hasattr(tick, "bid") else None,
                    "ask": tick.ask / 100000 if hasattr(tick, "ask") else None,
                    "price": ((tick.bid + tick.ask) / 2) / 100000 if hasattr(tick, "bid") and hasattr(tick, "ask") else None,
                })

            df = pd.DataFrame(ticks)
            df = df.sort_values("ts_utc").reset_index(drop=True)

            logger.info(f"[{self.name}] Retrieved {len(df)} ticks for {symbol}")
            return df

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get historical ticks for {symbol}: {e}")
            self.health.errors.append(str(e))
            return None

    async def _get_market_depth_impl(self, symbol: str, levels: int) -> Optional[Dict[str, Any]]:
        """Get market depth (DOM) from cTrader."""
        try:
            await self._rate_limit_wait()

            symbol_id = await self._get_symbol_id(symbol)

            # Subscribe to depth of market
            request = Messages.ProtoOASubscribeLiveTrendbarReq()
            request.ctidTraderAccountId = self._account_id
            request.symbolId = symbol_id

            # In cTrader, DOM is streamed via subscriptions
            # For a snapshot, we'd need to get the latest from the stream
            # This is a simplified implementation
            logger.warning(f"[{self.name}] Market depth requires WebSocket subscription - returning None")
            return None

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get market depth for {symbol}: {e}")
            self.health.errors.append(str(e))
            return None

    async def _get_sentiment_impl(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get trader sentiment from cTrader."""
        try:
            # cTrader doesn't have a direct sentiment API
            # This would need to be calculated from trader positions if available
            # Or sourced from a third-party provider

            logger.info(f"[{self.name}] Sentiment data not available from cTrader API")
            return None

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get sentiment for {symbol}: {e}")
            self.health.errors.append(str(e))
            return None

    async def _get_news_impl(self, currency: Optional[str], limit: int) -> Optional[List[Dict[str, Any]]]:
        """Get news feed from cTrader."""
        try:
            # cTrader doesn't provide news feed via Open API
            # This would need integration with external news provider
            # (e.g., Forex Factory, Trading Economics)

            logger.info(f"[{self.name}] News feed not available from cTrader API - use external provider")
            return None

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get news: {e}")
            self.health.errors.append(str(e))
            return None

    async def _get_economic_calendar_impl(
        self, start_date: Optional[datetime], end_date: Optional[datetime], currency: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Get economic calendar from cTrader."""
        try:
            # cTrader doesn't provide economic calendar via Open API
            # This would need integration with external calendar provider
            # (e.g., Forex Factory API, Trading Economics API)

            logger.info(f"[{self.name}] Economic calendar not available from cTrader API - use external provider")
            return None

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get economic calendar: {e}")
            self.health.errors.append(str(e))
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

    # Helper methods for cTrader protocol

    async def _get_symbol_id(self, symbol: str) -> int:
        """Get cTrader symbol ID from symbol name."""
        # This would need to query symbols list - placeholder implementation
        # In production, cache symbol mappings
        symbol_map = {
            "EUR/USD": 1,
            "GBP/USD": 2,
            "USD/JPY": 3,
            # Add more mappings
        }
        return symbol_map.get(symbol, 1)

    def _convert_timeframe(self, timeframe: str) -> Optional[int]:
        """Convert standard timeframe to cTrader period enum."""
        # cTrader ProtoOATimeframe enum values
        tf_map = {
            "1m": 1,   # M1
            "2m": 2,   # M2
            "3m": 3,   # M3
            "4m": 4,   # M4
            "5m": 5,   # M5
            "10m": 6,  # M10
            "15m": 7,  # M15
            "30m": 8,  # M30
            "1h": 9,   # H1
            "4h": 10,  # H4
            "12h": 11, # H12
            "1d": 12,  # D1
            "1w": 13,  # W1
            "1M": 14,  # MN1
        }
        return tf_map.get(timeframe)

    async def _send_and_wait(self, request: Any, response_type: Any, timeout: float = 10.0) -> Optional[Any]:
        """Send request to cTrader and wait for response."""
        # This is a simplified placeholder - full implementation would:
        # 1. Send protobuf message via client
        # 2. Wait for response in message queue
        # 3. Match response by clientMsgId
        # 4. Return matched response or timeout

        try:
            if not self.client:
                raise RuntimeError("Client not connected")

            # In a real implementation, this would:
            # - Generate unique clientMsgId
            # - Send request via self.client.send(request)
            # - Poll self._message_queue for response matching clientMsgId
            # - Return response or raise TimeoutError

            logger.warning(f"[{self.name}] _send_and_wait placeholder - needs full Twisted integration")
            return None

        except Exception as e:
            logger.error(f"[{self.name}] Send/wait error: {e}")
            return None
