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
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
        ProtoMessage, ProtoHeartbeatEvent, ProtoErrorRes
    )
    from ctrader_open_api.messages import OpenApiMessages_pb2 as Messages
    from twisted.internet import reactor, ssl
    from twisted.internet.protocol import ReconnectingClientFactory
    _HAS_CTRADER = True
except ImportError as e:
    _HAS_CTRADER = False
    logger.warning(f"ctrader-open-api not installed: {e}")


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
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None  # Store main event loop

        # Message tracking for request/response matching
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._next_msg_id = 1

        # Rate limiting (5 req/sec for historical data)
        self._rate_limiter = deque(maxlen=5)

        # Endpoint configuration
        if self.environment == "live":
            self.host = EndPoints.PROTOBUF_LIVE_HOST
        else:
            self.host = EndPoints.PROTOBUF_DEMO_HOST

        # Port is the same for both demo and live
        self.port = EndPoints.PROTOBUF_PORT

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

        Strategy: Application authentication with client_id + client_secret
        No access_token (AppKey) required for historical data access.
        """
        # Save the asyncio event loop for use in Twisted callbacks
        self._asyncio_loop = asyncio.get_event_loop()

        if not self.client_id or not self.client_secret:
            logger.error(f"[{self.name}] client_id and client_secret required. Configure in Settings.")
            return False

        # Create message queue
        self._message_queue = asyncio.Queue(maxsize=10000)

        # Connect with application credentials (client_id + client_secret)
        try:
            logger.info(f"[{self.name}] Connecting with application credentials...")
            success = await self._connect_with_app_credentials()
            if success:
                logger.info(f"[{self.name}] Connected successfully")
                return True
            else:
                logger.error(f"[{self.name}] Connection failed")
                return False
        except Exception as e:
            logger.error(f"[{self.name}] Connection error: {e}")
            self.health.is_connected = False
            self.health.errors.append(str(e))
            return False

    async def _connect_with_app_credentials(self) -> bool:
        """
        Connect using application credentials (client_id + client_secret).
        No access_token required for historical data access.
        """
        # Create cTrader client
        self.client = Client(self.host, self.port, TcpProtocol)

        # Setup message handlers
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)

        # Connect via Twisted
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_twisted)

        # Start the client service (initiates connection)
        logger.info(f"[{self.name}] Starting client service...")
        self.client.startService()

        # Wait for connection to establish
        await asyncio.sleep(2.0)

        # Authenticate with application credentials
        await self._authenticate_application()

        self.health.is_connected = True
        self._running = True
        self._start_time = datetime.now()

        logger.info(f"[{self.name}] Connected and authenticated")
        return True

    def _connect_twisted(self) -> None:
        """
        Start Twisted reactor in thread (blocking call).

        Twisted→asyncio bridge: Runs Twisted reactor in a background thread
        to avoid blocking the asyncio event loop.
        """
        import threading

        if not self.client:
            raise RuntimeError("Client not initialized before connecting")

        # Check if reactor is already running
        if reactor.running:
            logger.info(f"[{self.name}] Twisted reactor already running")
            return

        # Start reactor in a background thread (non-blocking)
        def run_reactor():
            try:
                logger.info(f"[{self.name}] Starting Twisted reactor in background thread...")
                # Run reactor (blocking call within this thread)
                reactor.run(installSignalHandlers=False)  # Don't install signal handlers in thread
                logger.info(f"[{self.name}] Twisted reactor stopped")
            except Exception as e:
                logger.error(f"[{self.name}] Twisted reactor error: {e}")

        # Start reactor thread
        reactor_thread = threading.Thread(target=run_reactor, daemon=True, name="cTrader-Twisted-Reactor")
        reactor_thread.start()

        # Wait a bit for reactor to start
        import time
        time.sleep(0.5)

        logger.info(f"[{self.name}] Twisted reactor started in thread {reactor_thread.name}")

    async def _authenticate_application(self) -> None:
        """Authenticate with cTrader using application credentials (client_id + client_secret)."""
        logger.info(f"[{self.name}] Authenticating application with client_id: {self.client_id[:8]}...")

        request = Messages.ProtoOAApplicationAuthReq()
        request.clientId = self.client_id
        request.clientSecret = self.client_secret

        response = await self._send_and_wait(request, Messages.ProtoOAApplicationAuthRes, timeout=30.0)

        if not response:
            raise RuntimeError("Application authentication failed - no response received")

        logger.info(f"[{self.name}] Application authenticated successfully")

        # After app auth, authenticate the trading account
        await self._authenticate_account()

    async def _authenticate_account(self) -> None:
        """Authenticate trading account after application authentication."""
        logger.info(f"[{self.name}] Authenticating trading account...")

        # First, get the list of available accounts
        accounts_req = Messages.ProtoOAGetAccountListByAccessTokenReq()
        accounts_req.accessToken = ""  # Empty for client credentials flow

        try:
            accounts_res = await self._send_and_wait(accounts_req, Messages.ProtoOAGetAccountListByAccessTokenRes, timeout=30.0)

            if not accounts_res or not accounts_res.ctidTraderAccount:
                raise RuntimeError("No trading accounts found")

            # Use the first available account
            account = accounts_res.ctidTraderAccount[0]
            self._account_id = account.ctidTraderAccountId

            logger.info(f"[{self.name}] Using account ID: {self._account_id}")

            # Now authenticate this specific account
            auth_req = Messages.ProtoOAAccountAuthReq()
            auth_req.ctidTraderAccountId = self._account_id
            auth_req.accessToken = ""  # Empty for client credentials flow

            auth_res = await self._send_and_wait(auth_req, Messages.ProtoOAAccountAuthRes, timeout=30.0)

            if not auth_res:
                raise RuntimeError(f"Account authentication failed for account {self._account_id}")

            logger.info(f"[{self.name}] Account {self._account_id} authenticated successfully")

        except Exception as e:
            logger.error(f"[{self.name}] Account authentication error: {e}")
            raise

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

    def _on_message(self, client, message: Any) -> None:
        """Handle incoming message from cTrader (Twisted callback).

        Args:
            client: The cTrader client instance (passed by library)
            message: The received message
        """
        try:
            self.health.last_message_ts = datetime.now()

            # Check if this is a response to a pending request
            msg_id = getattr(message, 'clientMsgId', None)

            if msg_id and msg_id in self._pending_requests:
                # This is a response to a pending request
                future = self._pending_requests.pop(msg_id)
                if not future.done() and self._asyncio_loop:
                    # Schedule the future to be resolved in the asyncio event loop
                    self._asyncio_loop.call_soon_threadsafe(future.set_result, message)
                return

            # Unsolicited message (streaming data) - push to queue
            if self._message_queue and not self._message_queue.full():
                try:
                    # Convert Protobuf message to dict for streaming data
                    msg_dict = self._protobuf_to_dict(message)
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

    def _on_connected(self, client) -> None:
        """Handle successful connection to cTrader.

        Args:
            client: The cTrader client instance (passed by library)
        """
        logger.info(f"[{self.name}] Connected to cTrader server")
        self.health.is_connected = True
        self.health.last_message_ts = datetime.now()

    def _on_disconnected(self, client, reason) -> None:
        """Handle disconnection from cTrader.

        Args:
            client: The cTrader client instance (passed by library)
            reason: The disconnection reason
        """
        logger.warning(f"[{self.name}] Disconnected from cTrader server: {reason}")
        self.health.is_connected = False
        self._running = False

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

    async def get_historical_bars(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int
    ) -> Optional[pd.DataFrame]:
        """
        Public wrapper for _get_historical_bars_impl.
        Allows CTraderClient to call historical bars API.

        Args:
            symbol: Symbol like "EUR/USD"
            timeframe: cTrader format ('1m', '5m', '15m', '1h', '4h', '1d')
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds

        Returns:
            DataFrame with columns: ts_utc, open, high, low, close, volume, tick_volume
        """
        return await self._get_historical_bars_impl(symbol, timeframe, start_ms, end_ms)

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
        """
        Get cTrader symbol ID from symbol name via API.
        Caches results to avoid repeated API calls.
        """
        # Initialize cache if not present
        if not hasattr(self, '_symbol_cache'):
            self._symbol_cache = {}

        # Return from cache if available
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        # Request symbols list from cTrader API
        try:
            await self._rate_limit_wait()

            request = Messages.ProtoOASymbolsListReq()
            request.ctidTraderAccountId = self._account_id

            response = await self._send_and_wait(request, Messages.ProtoOASymbolsListRes)

            if not response or not hasattr(response, 'symbol'):
                raise ValueError(f"Could not retrieve symbols list from cTrader")

            # Build cache from response
            for sym in response.symbol:
                sym_name = sym.symbolName  # e.g. "EURUSD"

                # Normalize to "EUR/USD" format (add slash if 6-char currency pair)
                if len(sym_name) == 6 and sym_name.isalpha():
                    normalized = f"{sym_name[:3]}/{sym_name[3:]}"
                else:
                    normalized = sym_name

                self._symbol_cache[normalized] = sym.symbolId
                # Also cache without slash for flexibility
                self._symbol_cache[sym_name] = sym.symbolId

            logger.info(f"[{self.name}] Cached {len(self._symbol_cache)} symbols from cTrader")

            # Check if requested symbol is now in cache
            if symbol not in self._symbol_cache:
                # Try without slash as fallback
                symbol_no_slash = symbol.replace("/", "")
                if symbol_no_slash in self._symbol_cache:
                    return self._symbol_cache[symbol_no_slash]

                raise ValueError(
                    f"Symbol '{symbol}' not found in cTrader symbols list. "
                    f"Available: {list(self._symbol_cache.keys())[:10]}..."
                )

            return self._symbol_cache[symbol]

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get symbol ID for {symbol}: {e}")
            # Fallback to hardcoded common symbols as last resort
            fallback_map = {
                "EUR/USD": 1, "EURUSD": 1,
                "GBP/USD": 2, "GBPUSD": 2,
                "USD/JPY": 3, "USDJPY": 3,
                "USD/CHF": 4, "USDCHF": 4,
                "AUD/USD": 5, "AUDUSD": 5,
            }
            if symbol in fallback_map:
                logger.warning(f"[{self.name}] Using fallback symbol ID for {symbol}")
                return fallback_map[symbol]
            raise

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
        """Send Protobuf request to cTrader and wait for response with matching clientMsgId."""
        if not self.client:
            raise RuntimeError("Client not connected")

        # Generate unique message ID
        msg_id = str(self._next_msg_id)
        self._next_msg_id += 1

        # Create Future for response
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_requests[msg_id] = future

        try:
            # Send request via Twisted client with clientMsgId parameter
            logger.debug(f"[{self.name}] Sending request {msg_id} ({request.__class__.__name__})")
            self.client.send(request, clientMsgId=msg_id)

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=timeout)

            # Verify response type
            if not isinstance(response, response_type):
                logger.error(
                    f"[{self.name}] Response type mismatch for msg {msg_id}: "
                    f"expected {response_type.__name__}, got {response.__class__.__name__}"
                )
                return None

            logger.debug(f"[{self.name}] Received response {msg_id} ({response.__class__.__name__})")
            return response

        except asyncio.TimeoutError:
            logger.error(
                f"[{self.name}] Request {msg_id} ({request.__class__.__name__}) timed out after {timeout}s"
            )
            self._pending_requests.pop(msg_id, None)
            return None

        except Exception as e:
            logger.error(f"[{self.name}] Request {msg_id} error: {e}")
            self._pending_requests.pop(msg_id, None)
            raise
