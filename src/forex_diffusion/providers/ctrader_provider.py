"""
cTrader Open API provider implementation.

Implements BaseProvider interface for cTrader with:
- WebSocket streaming (Twisted → asyncio bridge)
- Historical data (trendbars, tick data)
- Market depth (DOM)
- News, Calendar, Sentiment data
"""

from __future__ import annotations

# CRITICAL: Disable Twisted's default error logging for Deferred
# This prevents "Unhandled error in Deferred" spam from ctrader-open-api library
import sys
import warnings

# Monkey-patch Deferred.__del__ to NOT log errors (we handle them in callbacks)
def _silent_deferred_del(self):
    """Silent version of Deferred.__del__ that doesn't spam logs."""
    # Just clear the failure without logging
    if hasattr(self, 'failResult') and self.failResult is not None:
        self.failResult = None

# Apply monkey-patch BEFORE importing defer
import twisted.internet.defer as _defer_module
_defer_module.Deferred.__del__ = _silent_deferred_del

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Any
from collections import deque
import time

import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseProvider, ProviderCapability


class CTraderAuthorizationError(Exception):
    """Raised when cTrader trading account is not authorized."""
    pass


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
        self.refresh_token = config.get("refresh_token") if config else None
        self.access_token = config.get("access_token") if config else None
        self.environment = config.get("environment", "demo") if config else "demo"  # demo or live

        # cTrader client
        self.client: Optional[Client] = None
        self._account_id: Optional[int] = None

        # Async communication
        self._message_queue: Optional[asyncio.Queue] = None
        self._running = False
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None  # Store main event loop
        
        # Callback for real-time tick updates (spot events)
        self.on_tick_callback: Optional[callable] = None

        # Message tracking for request/response matching
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._next_msg_id = 1

        # Rate limiting (5 req/sec for historical data)
        self._rate_limiter = deque(maxlen=5)

        # DOM (market depth) data buffers
        self._dom_buffer: Dict[str, Dict] = {}  # symbol -> latest DOM snapshot
        self._depth_quotes: Dict[str, Dict[int, Dict]] = {}  # symbol -> {quote_id -> quote_data}
        
        # Last valid spot prices (cTrader sends 0 when value doesn't change)
        self._last_spot_prices: Dict[str, Dict[str, float]] = {}  # symbol -> {bid, ask}

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
        try:
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

            # Check if we actually connected
            if not self.health.is_connected:
                error_msg = (
                    f"Failed to connect to cTrader server at {self.host}:{self.port}. "
                    "Possible causes:\n"
                    "1. No internet connection\n"
                    "2. DNS lookup failed (cannot resolve hostname)\n"
                    "3. Firewall blocking connection\n"
                    "4. cTrader server is down"
                )
                logger.error(f"[{self.name}] {error_msg}")
                raise RuntimeError(error_msg)

            # Authenticate with application credentials
            await self._authenticate_application()

            self.health.is_connected = True
            self._running = True
            self._start_time = datetime.now()

            logger.info(f"[{self.name}] Connected and authenticated")
            
            # Auto-subscribe to spot quotes and DOM for configured symbols
            try:
                default_symbols = ["EUR/USD", "GBP/USD", "USD/JPY"]
                logger.info(f"[{self.name}] Auto-subscribing to real-time data for {default_symbols}")
                await self._subscribe_spots(default_symbols)
                await self._subscribe_depth_quotes(default_symbols)
                logger.info(f"[{self.name}] ✓ Auto-subscription completed")
                logger.info(f"[{self.name}] 📊 Listening for: Spot quotes + DOM/Level2 data")
            except Exception as e:
                logger.warning(f"[{self.name}] Auto-subscription failed (will retry on stream): {e}")
            
            return True

        except Exception as e:
            # Check for DNS-specific errors
            error_str = str(e).lower()
            if 'dns' in error_str or 'lookup' in error_str or 'resolve' in error_str:
                logger.error(
                    f"[{self.name}] DNS lookup failed for {self.host}. "
                    "Please check your internet connection and DNS settings."
                )
            self.health.is_connected = False
            raise

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

        # Install global error handler for unhandled Twisted errors
        from twisted.python import log as twisted_log
        def twisted_error_handler(event):
            """Handle unhandled Twisted errors."""
            if event.get('isError'):
                failure = event.get('failure')
                if failure:
                    error_msg = str(failure.value) if hasattr(failure, 'value') else str(failure)
                    error_type = type(failure.value).__name__ if hasattr(failure, 'value') else 'Unknown'

                    # Check for DNS errors
                    if 'DNS' in error_type or 'DNSLookupError' in error_type:
                        logger.error(
                            f"[{self.name}] DNS lookup failed for {self.host}. "
                            "Cannot resolve hostname. Check internet connection and DNS settings."
                        )
                    # Check for connection errors
                    elif 'Connection' in error_type:
                        logger.error(
                            f"[{self.name}] Connection failed to {self.host}:{self.port}. "
                            f"Error: {error_msg}"
                        )
                    # Suppress TimeoutError - these are handled gracefully by callbacks
                    elif 'TimeoutError' in error_type or 'Deferred' in str(error_msg):
                        # Log with traceback to identify WHERE this timeout is coming from
                        import traceback
                        stack = ''.join(traceback.format_stack())
                        logger.debug(
                            f"[{self.name}] Deferred timeout (already handled by callback) ({error_type}): {error_msg}.\n"
                            f"Stack trace:\n{stack}"
                        )
                    else:
                        logger.error(
                            f"[{self.name}] Unhandled Twisted error ({error_type}): {error_msg}"
                        )

                    # Mark as not connected only for real connection errors
                    if 'Connection' in error_type:
                        self.health.is_connected = False

        twisted_log.addObserver(twisted_error_handler)

        # Start reactor in a background thread (non-blocking)
        def run_reactor():
            try:
                # Redirect stderr in THIS thread to suppress Twisted's Deferred error spam
                import sys
                from io import StringIO
                
                class SilentDeferredStderr:
                    def __init__(self, original):
                        self.original = original
                        self.buffer = []
                        
                    def write(self, text):
                        # Suppress "Unhandled error in Deferred" and its traceback
                        if "Unhandled error in Deferred:" in text:
                            self.buffer = [text]  # Start buffering
                            return
                        
                        # Continue buffering traceback lines
                        if self.buffer:
                            self.buffer.append(text)
                            # Stop buffering when we hit a blank line or non-traceback
                            if not text.strip() or (text.strip() and not any(x in text for x in 
                                ["File ", "Traceback", "twisted", "defer", "TimeoutError", "---", "Error"])):
                                self.buffer = []  # Clear buffer, drop the error
                            return
                        
                        # Normal output
                        self.original.write(text)
                    
                    def flush(self):
                        self.original.flush()
                    
                    def __getattr__(self, name):
                        return getattr(self.original, name)
                
                # Replace stderr ONLY in reactor thread
                original_stderr = sys.stderr
                sys.stderr = SilentDeferredStderr(original_stderr)
                
                logger.info(f"[{self.name}] Starting Twisted reactor in background thread...")
                # Run reactor (blocking call within this thread)
                reactor.run(installSignalHandlers=False)  # Don't install signal handlers in thread
                logger.info(f"[{self.name}] Twisted reactor stopped")
                
                # Restore stderr
                sys.stderr = original_stderr
            except Exception as e:
                logger.error(f"[{self.name}] Twisted reactor error: {e}")
                self.health.is_connected = False

        # Start reactor thread
        reactor_thread = threading.Thread(target=run_reactor, daemon=True, name="cTrader-Twisted-Reactor")
        reactor_thread.start()

        # No need to wait - reactor will handle async operations when ready
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

    async def _refresh_access_token(self) -> Optional[str]:
        """Refresh access token using refresh token."""
        if not self.refresh_token:
            return None

        try:
            from ctrader_open_api import Auth

            # We need a redirect_uri for Auth class, but it's not used in refresh
            auth = Auth(self.client_id, self.client_secret, "http://localhost:8080/callback")

            logger.info(f"[{self.name}] Refreshing access token...")
            token_response = auth.refreshToken(self.refresh_token)

            if 'accessToken' in token_response:
                new_token = token_response['accessToken']
                self.access_token = new_token  # Update instance variable
                logger.info(f"[{self.name}] Access token refreshed successfully")

                # Save both tokens to settings
                try:
                    from ..utils.user_settings import set_setting
                    set_setting('ctrader_access_token', new_token)
                    logger.debug(f"[{self.name}] New access token saved to settings")

                    # Update refresh token if provided
                    if 'refreshToken' in token_response:
                        self.refresh_token = token_response['refreshToken']
                        set_setting('ctrader_refresh_token', self.refresh_token)
                        logger.debug(f"[{self.name}] New refresh token saved to settings")
                except Exception as save_error:
                    logger.warning(f"[{self.name}] Failed to save tokens to settings: {save_error}")

                return new_token
            else:
                logger.error(f"[{self.name}] Token refresh failed: {token_response}")
                return None

        except Exception as e:
            logger.error(f"[{self.name}] Failed to refresh token: {e}")
            return None

    async def _authenticate_account(self) -> None:
        """Authenticate trading account after application authentication."""
        logger.info(f"[{self.name}] Authenticating trading account...")

        # Try to refresh access token if we have refresh token but no access token
        if not self.access_token and self.refresh_token:
            self.access_token = await self._refresh_access_token()

        if not self.access_token:
            raise CTraderAuthorizationError(
                "No access token available. Please provide credentials:\n\n"
                "Option 1 - Use existing tokens (recommended if you have them):\n"
                "1. Open provider configuration dialog\n"
                "2. Enter Access Token and Refresh Token\n"
                "3. Save settings\n\n"
                "Option 2 - OAuth2 authorization (generates new tokens):\n"
                "1. Open provider configuration dialog\n"
                "2. Enter Client ID and Client Secret\n"
                "3. Click 'Authorize with cTrader' button\n"
                "4. Complete authorization in browser"
            )

        # First, get the list of available accounts
        accounts_req = Messages.ProtoOAGetAccountListByAccessTokenReq()
        accounts_req.accessToken = self.access_token

        # Retry logic with token refresh
        max_retries = 2
        for attempt in range(max_retries):
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
                auth_req.accessToken = self.access_token

                auth_res = await self._send_and_wait(auth_req, Messages.ProtoOAAccountAuthRes, timeout=30.0)

                if not auth_res:
                    raise RuntimeError(f"Account authentication failed for account {self._account_id}")

                logger.info(f"[{self.name}] Account {self._account_id} authenticated successfully")
                return  # Success, exit retry loop

            except Exception as e:
                error_msg = str(e).lower()
                # Check if error is due to expired/invalid token
                if ('unauthorized' in error_msg or 'invalid' in error_msg or
                    'expired' in error_msg or 'token' in error_msg) and attempt < max_retries - 1:

                    logger.warning(f"[{self.name}] Token appears invalid/expired, attempting refresh (attempt {attempt + 1}/{max_retries})...")

                    if self.refresh_token:
                        new_token = await self._refresh_access_token()
                        if new_token:
                            # Update the request with new token and retry
                            accounts_req.accessToken = new_token
                            logger.info(f"[{self.name}] Token refreshed, retrying authentication...")
                            continue
                        else:
                            logger.error(f"[{self.name}] Token refresh failed, cannot retry")
                    else:
                        logger.error(f"[{self.name}] No refresh token available for retry")

                # If not a token error or last attempt, re-raise
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

            # Log message type for debugging (include payloadType for better visibility)
            msg_type = type(message).__name__
            payload_type = getattr(message, 'payloadType', None)
            #PLUTOTOUCHlogger.debug(
            #    f"[{self.name}] Received message: type={msg_type}, "
            #   f"payloadType={payload_type}, "
            #    f"has_clientMsgId={hasattr(message, 'clientMsgId')}"
            #)

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
                    # Decode ProtoMessage wrapper if needed
                    decoded_message = message
                    if hasattr(message, 'payloadType') and hasattr(message, 'payload'):
                        # This is a ProtoMessage wrapper - decode the payload
                        payload_type = message.payloadType
                        
                        # Type 2155 = ProtoOADepthEvent (DOM data)
                        if payload_type == 2155:
                            decoded_message = Messages.ProtoOADepthEvent()
                            decoded_message.ParseFromString(message.payload)
                            logger.debug(f"[{self.name}] ✓ Decoded ProtoOADepthEvent (type 2155)")
                        
                        # Type 2131 = ProtoOASpotEvent (spot quotes - real-time prices)
                        elif payload_type == 2131:
                            decoded_message = Messages.ProtoOASpotEvent()
                            decoded_message.ParseFromString(message.payload)
                            logger.debug(f"[{self.name}] ✓ Decoded ProtoOASpotEvent (type 2131)")
                        
                        # Type 2128 = ProtoOATrailingSLChangedEvent (trailing stop loss changed)
                        elif payload_type == 2128:
                            logger.debug(f"[{self.name}] ⊘ Skipping ProtoOATrailingSLChangedEvent (type 2128)")
                            return  # Skip processing
                        
                        # Type 2157 = ProtoOAOrderErrorEvent (order error notification)
                        elif payload_type == 2157:
                            logger.debug(f"[{self.name}] ⊘ Skipping ProtoOAOrderErrorEvent (type 2157)")
                            return  # Skip processing
                        
                        # Type 2127 = ProtoOAExecutionEvent (order execution/cancellation/fill)
                        elif payload_type == 2127:
                            logger.debug(f"[{self.name}] ⊘ Skipping ProtoOAExecutionEvent (type 2127) - order event")
                            return  # Skip processing
                        
                        else:
                            # Unknown payload type - log it with message details
                            msg_type = type(message).__name__
                            logger.warning(f"[{self.name}] ⚠️ Unknown payloadType={payload_type}, msgType={msg_type}, attempting spot decode...")
                            try:
                                decoded_message = Messages.ProtoOASpotEvent()
                                decoded_message.ParseFromString(message.payload)
                                logger.info(f"[{self.name}] ✓ Successfully decoded unknown type {payload_type} (msgType={msg_type}) as ProtoOASpotEvent")
                            except Exception as decode_err:
                                logger.warning(f"[{self.name}] ⚠️ Failed spot decode for type {payload_type} (msgType={msg_type}): {decode_err}")
                    
                    # Convert Protobuf message to dict for streaming data
                    msg_dict = self._protobuf_to_dict(decoded_message)
                    self._message_queue.put_nowait(msg_dict)
                except asyncio.QueueFull:
                    logger.warning(f"[{self.name}] Message queue full, dropping message")

        except Exception as e:
            logger.error(f"[{self.name}] Error handling message: {e}")
            self.health.errors.append(str(e))

    def _on_error(self, error: Exception) -> None:
        """Handle connection error."""
        error_msg = str(error)
        logger.error(f"[{self.name}] Connection error: {error_msg}")
        
        # Provide helpful message for common errors
        if "503" in error_msg and "No accounts" in error_msg:
            logger.warning(
                f"[{self.name}] ⚠️ cTrader account not available. "
                "This usually means:\n"
                "  1. Demo account expired (renew at https://ctrader.com)\n"
                "  2. OAuth token needs refresh\n"
                "  3. cTrader server maintenance\n"
                "Solution: Go to Settings → Providers → cTrader → Re-authenticate"
            )
        
        self.health.errors.append(error_msg)
        self.health.is_connected = False

    def _on_connected(self, client) -> None:
        """Handle successful connection to cTrader.

        Args:
            client: The cTrader client instance (passed by library)
        """
        logger.info(f"[{self.name}] 🔌 Connected to cTrader server")
        self.health.is_connected = True
        self.health.last_message_ts = datetime.now()
        logger.info(f"[{self.name}] 📡 WebSocket connection active, waiting for authentication...")

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
        msg_type = type(message).__name__

        # Handle ProtoOASpotEvent (real-time spot quotes)
        if msg_type == "ProtoOASpotEvent":
            try:
                # Get symbol name from symbolId (reverse lookup)
                symbol_id = message.symbolId
                symbol_name = None

                # Find symbol name from cache
                if hasattr(self, '_symbol_cache'):
                    for name, sid in self._symbol_cache.items():
                        if sid == symbol_id:
                            symbol_name = name
                            break
                
                # Log if symbol not found
                if not symbol_name:
                    logger.warning(f"[{self.name}] ⚠️ Symbol ID {symbol_id} not found in cache. Cache has {len(self._symbol_cache) if hasattr(self, '_symbol_cache') else 0} symbols. First few: {list(self._symbol_cache.items())[:5] if hasattr(self, '_symbol_cache') else []}")
                    # Skip this spot event if we can't resolve symbol
                    return {"type": "error", "error": f"Unknown symbol_id {symbol_id}"}

                # cTrader uses 100000 multiplier for prices
                # NOTE: cTrader sends 0 for bid/ask when value hasn't changed
                raw_bid = message.bid / 100000 if hasattr(message, 'bid') and message.bid else 0
                raw_ask = message.ask / 100000 if hasattr(message, 'ask') and message.ask else 0
                
                # Get last valid prices for this symbol
                if symbol_name not in self._last_spot_prices:
                    self._last_spot_prices[symbol_name] = {"bid": raw_bid, "ask": raw_ask}
                
                # Use raw value if non-zero, otherwise use last valid value
                final_bid = raw_bid if raw_bid > 0 else self._last_spot_prices[symbol_name].get("bid", 0)
                final_ask = raw_ask if raw_ask > 0 else self._last_spot_prices[symbol_name].get("ask", 0)
                
                # Update last valid prices
                if raw_bid > 0:
                    self._last_spot_prices[symbol_name]["bid"] = raw_bid
                if raw_ask > 0:
                    self._last_spot_prices[symbol_name]["ask"] = raw_ask
                
                spot_data = {
                    "type": "spot",
                    "symbol": symbol_name or f"ID:{symbol_id}",
                    "symbol_id": symbol_id,
                    "bid": final_bid,
                    "ask": final_ask,
                    "timestamp": message.timestamp if hasattr(message, 'timestamp') else int(time.time() * 1000),
                }
                logger.debug(f"[{self.name}] 💹 SPOT: {symbol_name} bid={final_bid:.5f} ask={final_ask:.5f} (raw: {raw_bid:.5f}/{raw_ask:.5f})")
                
                # Send to chart via callback if registered
                # NOTE: Called from Twisted thread, callback must handle thread safety
                if self.on_tick_callback:
                    try:
                        self.on_tick_callback(spot_data)
                    except Exception as e:
                        logger.error(f"[{self.name}] Error in tick callback: {e}")
                
                return spot_data
            except Exception as e:
                logger.error(f"[{self.name}] Error converting ProtoOASpotEvent: {e}")
                return {"type": "error", "error": str(e)}

        # Handle ProtoOADepthEvent (market depth / DOM updates)
        if msg_type == "ProtoOADepthEvent":
            try:
                # Get symbol name from symbolId (reverse lookup)
                symbol_id = message.symbolId
                symbol_name = None

                # Find symbol name from cache
                if hasattr(self, '_symbol_cache'):
                    for name, sid in self._symbol_cache.items():
                        if sid == symbol_id:
                            symbol_name = name
                            break

                if not symbol_name:
                    symbol_name = f"ID:{symbol_id}"

                # Initialize depth quotes buffer for this symbol if not exists
                if symbol_name not in self._depth_quotes:
                    self._depth_quotes[symbol_name] = {}

                # Process deleted quotes (remove from buffer)
                if hasattr(message, 'deletedQuotes'):
                    for quote_id in message.deletedQuotes:
                        self._depth_quotes[symbol_name].pop(quote_id, None)

                # Process new/updated quotes
                if hasattr(message, 'newQuotes'):
                    for quote in message.newQuotes:
                        quote_id = quote.id
                        # cTrader uses 100000 multiplier for prices
                        quote_data = {
                            'id': quote_id,
                            'size': quote.size if hasattr(quote, 'size') else 0,
                        }

                        # Quote can be either bid or ask
                        if hasattr(quote, 'bid') and quote.bid > 0:
                            quote_data['bid'] = quote.bid / 100000
                            quote_data['side'] = 'bid'
                        elif hasattr(quote, 'ask') and quote.ask > 0:
                            quote_data['ask'] = quote.ask / 100000
                            quote_data['side'] = 'ask'

                        self._depth_quotes[symbol_name][quote_id] = quote_data

                # Build order book from current quotes
                bids = []
                asks = []

                for quote_data in self._depth_quotes[symbol_name].values():
                    if quote_data.get('side') == 'bid' and 'bid' in quote_data:
                        bids.append({
                            'price': quote_data['bid'],
                            'size': quote_data['size']
                        })
                    elif quote_data.get('side') == 'ask' and 'ask' in quote_data:
                        asks.append({
                            'price': quote_data['ask'],
                            'size': quote_data['size']
                        })

                # Sort bids (descending) and asks (ascending)
                bids.sort(key=lambda x: x['price'], reverse=True)
                asks.sort(key=lambda x: x['price'])

                # Calculate DOM metrics
                total_bid_volume = sum(b['size'] for b in bids)
                total_ask_volume = sum(a['size'] for a in asks)

                # Calculate spread if we have both bids and asks
                spread = None
                if bids and asks:
                    spread = asks[0]['price'] - bids[0]['price']

                # Calculate mid price and imbalance
                mid_price = None
                imbalance = None
                if bids and asks:
                    mid_price = (bids[0]['price'] + asks[0]['price']) / 2
                    if total_bid_volume + total_ask_volume > 0:
                        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                
                # Update DOM buffer (compatible with database schema for OrderFlowPanel)
                dom_data = {
                    "type": "depth",
                    "symbol": symbol_name,
                    "symbol_id": symbol_id,
                    "bids": bids[:10],  # Top 10 levels
                    "asks": asks[:10],  # Top 10 levels
                    "best_bid": bids[0]['price'] if bids else None,
                    "best_ask": asks[0]['price'] if asks else None,
                    "total_bid_volume": total_bid_volume,
                    "total_ask_volume": total_ask_volume,
                    "bid_depth": total_bid_volume,  # Alias for OrderFlowPanel compatibility
                    "ask_depth": total_ask_volume,  # Alias for OrderFlowPanel compatibility
                    "mid_price": mid_price,
                    "spread": spread,
                    "imbalance": imbalance,
                    "timestamp": int(time.time() * 1000),
                }

                self._dom_buffer[symbol_name] = dom_data
                logger.info(f"[{self.name}] 📈 DOM updated for {symbol_name}: {len(bids)} bids, {len(asks)} asks, buffer keys now: {list(self._dom_buffer.keys())}")
                
                # DOM stored in RAM buffer only (not database)
                # All consumers (OrderFlowPanel, AutomatedTradingEngine, etc.) 
                # use LIMIT 1 queries, so historical data not needed
                # This prevents unnecessary database writes that slow down the app

                return dom_data

            except Exception as e:
                logger.error(f"[{self.name}] Error converting ProtoOADepthEvent: {e}")
                return {"type": "error", "error": str(e)}

        # Handle other message types (placeholder)
        return {
            "type": msg_type,
            "timestamp": int(time.time() * 1000),
        }
    
    def _store_dom_to_db(self, dom_data: Dict[str, Any]) -> None:
        """Store DOM snapshot to market_depth table."""
        try:
            import json
            from sqlalchemy import text
            
            # Prepare data for storage
            symbol = dom_data['symbol']
            timestamp = dom_data['timestamp']
            
            # Format bids/asks as JSON arrays [[price, size], ...]
            bids_json = json.dumps([[b['price'], b['size']] for b in dom_data['bids']])
            asks_json = json.dumps([[a['price'], a['size']] for a in dom_data['asks']])
            
            # Calculate mid price and imbalance
            best_bid = dom_data['bids'][0]['price'] if dom_data['bids'] else None
            best_ask = dom_data['asks'][0]['price'] if dom_data['asks'] else None
            mid_price = (best_bid + best_ask) / 2.0 if best_bid and best_ask else None
            spread = dom_data.get('spread')
            
            total_bid = dom_data.get('total_bid_volume', 0)
            total_ask = dom_data.get('total_ask_volume', 0)
            total = total_bid + total_ask
            imbalance = (total_bid - total_ask) / total if total > 0 else 0.0
            
            # Get database engine (lazy initialization)
            if not hasattr(self, '_db_engine'):
                from ..services.db_service import DBService
                db_service = DBService()
                self._db_engine = db_service.engine
            
            # Insert into database (UPSERT - update if duplicate timestamp)
            with self._db_engine.begin() as conn:
                query = text(
                    "INSERT INTO market_depth (symbol, ts_utc, bids, asks, mid_price, spread, imbalance, provider) "
                    "VALUES (:symbol, :ts_utc, :bids, :asks, :mid_price, :spread, :imbalance, :provider) "
                    "ON CONFLICT(symbol, ts_utc, provider) DO UPDATE SET "
                    "bids=excluded.bids, asks=excluded.asks, mid_price=excluded.mid_price, "
                    "spread=excluded.spread, imbalance=excluded.imbalance"
                )
                conn.execute(query, {
                    'symbol': symbol,
                    'ts_utc': timestamp,
                    'bids': bids_json,
                    'asks': asks_json,
                    'mid_price': mid_price,
                    'spread': spread,
                    'imbalance': imbalance,
                    'provider': 'ctrader'
                })
            
            logger.debug(f"[{self.name}] Stored DOM for {symbol} to database")
            
        except Exception as e:
            logger.error(f"[{self.name}] Error storing DOM to database: {e}", exc_info=True)

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

            # cTrader ProtoOAGetTrendbarsReq uses MILLISECONDS for timestamps
            # (confirmed from cTrader API docs - NOT microseconds!)
            start_ms = start_ts_ms
            end_ms = end_ts_ms

            # Request historical trendbars
            request = Messages.ProtoOAGetTrendbarsReq()
            request.ctidTraderAccountId = self._account_id
            request.symbolId = symbol_id
            request.period = ct_timeframe
            request.fromTimestamp = start_ms
            request.toTimestamp = end_ms

            response = await self._send_and_wait(request, Messages.ProtoOAGetTrendbarsRes)

            # Debug: check response content
            if not response:
                logger.warning(f"[{self.name}] No response received for {symbol} {timeframe} {start_ts_ms}-{end_ts_ms}")
                return None

            if not hasattr(response, 'trendbar'):
                logger.warning(f"[{self.name}] Response has no 'trendbar' field for {symbol} {timeframe}")
                return None

            if not response.trendbar:
                logger.warning(f"[{self.name}] Empty trendbar list for {symbol} {timeframe} {start_ts_ms}-{end_ts_ms} (range: {start_ts_ms / 1000} to {end_ts_ms / 1000} UTC)")
                return None

            # Parse trendbars into DataFrame
            # ProtoOATrendbar structure uses delta encoding for prices:
            # - utcTimestampInMinutes: timestamp in minutes (not ms!)
            # - low: direct low price
            # - deltaOpen, deltaHigh, deltaClose: deltas from low price
            # All prices divided by 100000
            bars = []
            for bar in response.trendbar:
                # Timestamp: convert from minutes to milliseconds
                ts_minutes = getattr(bar, 'utcTimestampInMinutes', 0)
                ts_utc_ms = int(ts_minutes * 60 * 1000)

                # Low price (direct)
                low_price = getattr(bar, 'low', 0) / 100000

                # Calculate open, high, close from deltas
                delta_open = getattr(bar, 'deltaOpen', 0)
                delta_high = getattr(bar, 'deltaHigh', 0)
                delta_close = getattr(bar, 'deltaClose', 0)

                open_price = (low_price * 100000 + delta_open) / 100000
                high_price = (low_price * 100000 + delta_high) / 100000
                close_price = (low_price * 100000 + delta_close) / 100000

                # Volume (tick volume)
                volume = getattr(bar, 'volume', 0)

                bars.append({
                    "ts_utc": ts_utc_ms,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "tick_volume": volume,  # cTrader trendbars use tick volume
                    "real_volume": volume,
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
        """Get historical tick data from cTrader (both BID and ASK)."""
        try:
            await self._rate_limit_wait()

            symbol_id = await self._get_symbol_id(symbol)
            start_us = start_ts_ms * 1000
            end_us = end_ts_ms * 1000

            # Request BID ticks (type=1)
            logger.debug(f"[{self.name}] Requesting BID ticks for {symbol}")
            bid_request = Messages.ProtoOAGetTickDataReq()
            bid_request.ctidTraderAccountId = self._account_id
            bid_request.symbolId = symbol_id
            bid_request.fromTimestamp = start_us
            bid_request.toTimestamp = end_us
            bid_request.type = 1  # BID

            bid_response = await self._send_and_wait(bid_request, Messages.ProtoOAGetTickDataRes)

            # Request ASK ticks (type=2)
            await self._rate_limit_wait()  # Respect rate limit between requests
            logger.debug(f"[{self.name}] Requesting ASK ticks for {symbol}")
            ask_request = Messages.ProtoOAGetTickDataReq()
            ask_request.ctidTraderAccountId = self._account_id
            ask_request.symbolId = symbol_id
            ask_request.fromTimestamp = start_us
            ask_request.toTimestamp = end_us
            ask_request.type = 2  # ASK

            ask_response = await self._send_and_wait(ask_request, Messages.ProtoOAGetTickDataRes)

            # Parse BID ticks
            bid_data = {}
            if bid_response and bid_response.tickData:
                for tick_data in bid_response.tickData:
                    ts = int(tick_data.timestamp / 1000)  # Convert from microseconds
                    bid_data[ts] = tick_data.tick / 100000  # cTrader uses 100000 multiplier

            # Parse ASK ticks
            ask_data = {}
            if ask_response and ask_response.tickData:
                for tick_data in ask_response.tickData:
                    ts = int(tick_data.timestamp / 1000)
                    ask_data[ts] = tick_data.tick / 100000

            if not bid_data and not ask_data:
                logger.warning(f"[{self.name}] No tick data received for {symbol}")
                return None

            # Merge BID and ASK data by timestamp
            # Use all unique timestamps from both datasets
            all_timestamps = sorted(set(bid_data.keys()) | set(ask_data.keys()))

            ticks = []
            for ts in all_timestamps:
                bid_value = bid_data.get(ts)
                ask_value = ask_data.get(ts)

                # Calculate mid price if both available, otherwise use available one
                if bid_value is not None and ask_value is not None:
                    price = (bid_value + ask_value) / 2
                elif bid_value is not None:
                    price = bid_value
                elif ask_value is not None:
                    price = ask_value
                else:
                    continue  # Skip if neither available

                ticks.append({
                    "ts_utc": ts,
                    "bid": bid_value,
                    "ask": ask_value,
                    "price": price,  # Mid price for analysis/sampling
                })

            df = pd.DataFrame(ticks)
            df = df.sort_values("ts_utc").reset_index(drop=True)

            logger.info(f"[{self.name}] Retrieved {len(df)} ticks for {symbol} (BID: {len(bid_data)}, ASK: {len(ask_data)})")
            return df

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get historical ticks for {symbol}: {e}")
            self.health.errors.append(str(e))
            return None

    async def get_historical_ticks(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> Optional[pd.DataFrame]:
        """
        Public wrapper for _get_historical_ticks_impl.
        Allows CTraderClient to call historical ticks API.

        Args:
            symbol: Symbol like "EUR/USD"
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds

        Returns:
            DataFrame with columns: ts_utc, bid, ask, price
            - bid: BID price (may be None for some timestamps)
            - ask: ASK price (may be None for some timestamps)
            - price: Mid price (bid+ask)/2 when both available, otherwise bid or ask
        """
        return await self._get_historical_ticks_impl(symbol, start_ms, end_ms)

    async def _get_market_depth_impl(self, symbol: str, levels: int) -> Optional[Dict[str, Any]]:
        """
        Get market depth (DOM) from cTrader.

        Note: DOM data is streamed via subscriptions. Use get_current_dom() to access
        the latest DOM snapshot from the buffer, or stream_market_depth() for real-time updates.
        """
        try:
            # Check if we already have DOM data for this symbol
            if symbol in self._dom_buffer:
                return self._dom_buffer[symbol]

            logger.warning(
                f"[{self.name}] No DOM data available for {symbol}. "
                "Subscribe to DOM updates using stream_market_depth() or check CTraderWebSocketService."
            )
            return None

        except Exception as e:
            logger.error(f"[{self.name}] Failed to get market depth for {symbol}: {e}")
            self.health.errors.append(str(e))
            return None

    def get_current_dom(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest DOM (market depth) snapshot from buffer.

        This is a synchronous method that returns the most recent DOM data
        received via the streaming subscription.

        Args:
            symbol: Symbol name (e.g., "EUR/USD")

        Returns:
            Dict with DOM data:
            - symbol: Symbol name
            - bids: List of bid levels [{price, size}, ...]
            - asks: List of ask levels [{price, size}, ...]
            - total_bid_volume: Total volume on bid side
            - total_ask_volume: Total volume on ask side
            - spread: Current spread (ask - bid)
            - timestamp: Timestamp in milliseconds

            None if no data available (not subscribed or no updates yet)
        """
        return self._dom_buffer.get(symbol)

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
            await self._subscribe_spots(symbols)

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
        """Stream market depth (DOM) updates from cTrader."""
        try:
            # Ensure connected
            if not self.health.is_connected:
                await self.connect()

            # Subscribe to depth quotes for symbols
            await self._subscribe_depth_quotes(symbols)

            # Stream DOM updates from queue
            while self._running:
                try:
                    msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)

                    # Only yield depth-related messages
                    if msg.get('type') == 'depth':
                        yield msg

                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            logger.error(f"[{self.name}] DOM stream failed: {e}")
            self.health.errors.append(str(e))

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

    async def _subscribe_spots(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time spot quotes for symbols via WebSocket.

        Args:
            symbols: List of symbol names (e.g., ["EUR/USD", "GBP/USD"])

        Returns:
            True if subscription successful, False otherwise
        """
        try:
            # Get symbol IDs for all requested symbols
            symbol_ids = []
            for symbol in symbols:
                try:
                    symbol_id = await self._get_symbol_id(symbol)
                    symbol_ids.append(symbol_id)
                except Exception as e:
                    logger.warning(f"[{self.name}] Could not get symbol ID for {symbol}: {e}")
                    continue

            if not symbol_ids:
                logger.error(f"[{self.name}] No valid symbol IDs found for subscription")
                return False

            # Send subscription request
            logger.info(f"[{self.name}] Subscribing to spot quotes for {len(symbol_ids)} symbols")
            request = Messages.ProtoOASubscribeSpotsReq()
            request.ctidTraderAccountId = self._account_id
            request.symbolId.extend(symbol_ids)  # Add all symbol IDs to repeated field

            # Send subscription request (fire-and-forget, no response expected)
            # Spot events will arrive asynchronously via message stream
            logger.debug(f"[{self.name}] Sending spot quotes subscription request (fire-and-forget)")
            
            # Use very long timeout (3600s = 1 hour) since we don't expect a response
            # The library always creates a Deferred, but we don't wait for it
            deferred = self.client.send(request, responseTimeoutInSeconds=3600)
            
            # Add callbacks to consume the Deferred and prevent garbage collection errors
            if deferred:
                deferred.addCallback(lambda _: None)  # Ignore success
                deferred.addErrback(lambda err: logger.debug(f"[{self.name}] Spot subscription deferred error (ignored): {err}"))
            
            logger.info(f"[{self.name}] 📡 Subscribed to spot quotes for {len(symbols)} symbols")
            return True  # Subscription sent successfully

        except Exception as e:
            logger.error(f"[{self.name}] Error subscribing to spot quotes: {e}")
            return False

    async def _subscribe_depth_quotes(self, symbols: List[str]) -> bool:
        """
        Subscribe to depth quotes (order book / DOM) for symbols.

        Args:
            symbols: List of symbol names (e.g., ["EUR/USD", "GBP/USD"])

        Returns:
            True if subscription successful, False otherwise
        """
        try:
            # Get symbol IDs for all requested symbols
            symbol_ids = []
            for symbol in symbols:
                try:
                    symbol_id = await self._get_symbol_id(symbol)
                    symbol_ids.append(symbol_id)
                    # Initialize depth quotes buffer for this symbol
                    self._depth_quotes[symbol] = {}
                except Exception as e:
                    logger.warning(f"[{self.name}] Could not get symbol ID for {symbol}: {e}")
                    continue

            if not symbol_ids:
                logger.error(f"[{self.name}] No valid symbol IDs found for depth subscription")
                return False

            # Send depth quotes subscription request
            logger.info(f"[{self.name}] Subscribing to depth quotes (DOM) for {len(symbol_ids)} symbols: {symbol_ids}")
            request = Messages.ProtoOASubscribeDepthQuotesReq()
            request.ctidTraderAccountId = self._account_id

            # Add all symbol IDs to the repeated field
            for symbol_id in symbol_ids:
                request.symbolId.append(symbol_id)

            # Send subscription request (fire-and-forget, no response expected)
            # DOM events will arrive asynchronously via message stream
            logger.debug(f"[{self.name}] Sending depth quotes subscription request (fire-and-forget)")
            
            # Use very long timeout (3600s = 1 hour) since we don't expect a response
            # The library always creates a Deferred, but we don't wait for it
            deferred = self.client.send(request, responseTimeoutInSeconds=3600)
            
            # Add callbacks to consume the Deferred and prevent garbage collection errors
            if deferred:
                deferred.addCallback(lambda _: None)  # Ignore success
                deferred.addErrback(lambda err: logger.debug(f"[{self.name}] DOM subscription deferred error (ignored): {err}"))
            
            logger.info(
                f"[{self.name}] 📡 Subscribed to DOM for {len(symbols)} symbols\n"
                f"DOM events will stream continuously if supported by account."
            )
            return True  # Subscription sent successfully

        except Exception as e:
            logger.warning(f"[{self.name}] Error subscribing to depth quotes: {e} (DOM may not be available)")
            return False

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
            # Use custom timeout from parameter (not library's default 5s)
            self.client.send(request, clientMsgId=msg_id, responseTimeoutInSeconds=timeout)

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=timeout)

            # Check if response is a ProtoMessage wrapper
            if hasattr(response, 'payloadType') and hasattr(response, 'payload'):
                # This is a ProtoMessage wrapper - need to parse the payload
                logger.debug(f"[{self.name}] Response is ProtoMessage, payloadType={response.payloadType}")

                # Check if this is an error response (payloadType 2142 = AUTH_FAILURE)
                if response.payloadType == 2142:
                    error_msg = response.payload.decode('utf-8', errors='ignore') if isinstance(response.payload, bytes) else str(response.payload)
                    logger.error(f"[{self.name}] cTrader authorization error: {error_msg}")

                    # Check for "Trading account is not authorized" error
                    if "not authorized" in error_msg.lower() or "Trading account is not authorized" in error_msg:
                        raise CTraderAuthorizationError(f"Trading account is not authorized: {error_msg}")

                    # Other auth errors
                    raise RuntimeError(f"cTrader authentication failed: {error_msg}")

                # Try to parse the payload bytes into the expected response type
                try:
                    actual_response = response_type()
                    actual_response.ParseFromString(response.payload)
                    logger.debug(f"[{self.name}] Successfully parsed payload into {response_type.__name__}")
                    return actual_response
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to parse payload: {e}")
                    logger.debug(f"[{self.name}] Raw response: {response}")
                    return None

            # Direct response (not wrapped)
            if not isinstance(response, response_type):
                logger.error(
                    f"[{self.name}] Response type mismatch for msg {msg_id}: "
                    f"expected {response_type.__name__}, got {type(response).__name__}"
                )
                logger.debug(f"[{self.name}] Response object: {response}")
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
