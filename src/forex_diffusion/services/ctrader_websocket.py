"""
cTrader WebSocket Service with Twisted Integration

Implements full WebSocket connection to cTrader Open API with:
- Order book (DOM) streaming
- Volume data streaming
- Sentiment data (if available)
- Twisted reactor → asyncio bridge
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from datetime import datetime

from loguru import logger
from sqlalchemy import text

try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
        ProtoMessage, ProtoHeartbeatEvent, ProtoErrorRes
    )
    from ctrader_open_api.messages import OpenApiMessages_pb2 as Messages
    from twisted.internet import reactor, ssl as twisted_ssl
    from twisted.internet.protocol import ReconnectingClientFactory
    _HAS_CTRADER = True
except ImportError as e:
    _HAS_CTRADER = False
    logger.error(f"ctrader-open-api or twisted not installed: {e}")


class CTraderWebSocketService:
    """
    WebSocket service for cTrader with full Twisted integration.

    Features:
    - Non-blocking Twisted reactor in separate thread
    - Order book (DOM) streaming to database
    - Volume data aggregation
    - Sentiment tracking (from order flow)
    - Auto-reconnection
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id,  # Accept str or int - will be converted as needed
        db_engine,
        environment: str = "demo",
        symbols: Optional[List[str]] = None
    ):
        """
        Initialize cTrader WebSocket service.

        Args:
            client_id: cTrader application client ID
            client_secret: cTrader application client secret
            access_token: OAuth access token
            account_id: cTrader trading account ID (can be string like "a.taini" or int like 12345)
            db_engine: SQLAlchemy engine for database operations
            environment: 'demo' or 'live'
            symbols: List of symbols to stream (default: ['EURUSD', 'GBPUSD'])
        """
        if not _HAS_CTRADER:
            raise ImportError(
                "ctrader-open-api and twisted required. "
                "Install with: pip install ctrader-open-api twisted"
            )

        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token

        # Account ID handling:
        # - If numeric (int or numeric string), use it directly
        # - If non-numeric (like "a.taini"), fetch account list from cTrader API
        self.account_id_raw = account_id  # Store original value
        self.account_id: Optional[int] = None  # Will be set after fetching accounts
        self.auto_fetch_account = False

        if isinstance(account_id, str):
            try:
                # Try to parse as integer
                self.account_id = int(account_id)
                logger.debug(f"Using numeric account_id: {self.account_id}")
            except ValueError:
                # Non-numeric string (like "a.taini") - we'll fetch the account list
                logger.info(
                    f"Account ID '{account_id}' is not numeric. "
                    f"Will automatically fetch account list from cTrader API."
                )
                self.auto_fetch_account = True
        elif isinstance(account_id, int):
            self.account_id = account_id
        else:
            # None or other type - will fetch account list
            logger.info("No account ID provided. Will fetch account list from cTrader API.")
            self.auto_fetch_account = True

        self.db_engine = db_engine
        self.environment = environment
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY']

        # Twisted client
        self.client: Optional[Client] = None
        self.factory: Optional[ReconnectingClientFactory] = None

        # State
        self.connected = False
        self.authenticated = False
        self.subscribed_symbols: Dict[str, int] = {}  # symbol -> symbol_id
        self._skip_depth_quotes = False  # Flag to skip depth quotes if not supported

        # Message queue for asyncio communication
        self.message_queue: asyncio.Queue = None

        # Twisted reactor thread
        self._reactor_thread: Optional[threading.Thread] = None
        self._reactor_running = False

        # Data buffers
        self.order_book_buffer: Dict[str, Dict] = {}  # symbol -> latest DOM
        self.depth_quotes: Dict[str, Dict[int, Dict]] = {}  # symbol -> {quote_id -> quote_data}
        self.volume_buffer: Dict[str, deque] = {s: deque(maxlen=100) for s in self.symbols}
        self.sentiment_buffer: Dict[str, Dict] = {}  # symbol -> sentiment metrics

        # Callbacks
        self.on_order_book_update: Optional[Callable] = None
        self.on_volume_update: Optional[Callable] = None
        self.on_sentiment_update: Optional[Callable] = None

        # Determine host and port
        if environment == "demo":
            self.host = EndPoints.PROTOBUF_DEMO_HOST
        else:
            self.host = EndPoints.PROTOBUF_LIVE_HOST

        # Port is the same for both demo and live
        self.port = EndPoints.PROTOBUF_PORT

        logger.info(
            f"Initialized CTraderWebSocketService: "
            f"env={environment}, symbols={symbols}, account={account_id}"
        )

    def start(self):
        """Start WebSocket service (Twisted reactor in separate thread)."""
        if self._reactor_running:
            logger.warning("WebSocket service already running")
            return

        # Create message queue for asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.message_queue = asyncio.Queue(maxsize=10000)

        # Check if reactor is already running (from another service)
        if reactor.running:
            logger.info("Twisted reactor already running")
        else:
            # Start Twisted reactor in thread
            self._reactor_running = True
            self._reactor_thread = threading.Thread(
                target=self._run_twisted_reactor,
                daemon=True,
                name="CTraderTwisted"
            )
            self._reactor_thread.start()

            # Wait for reactor to start
            import time
            time.sleep(0.5)

        # Create client and set callbacks
        self.client = Client(self.host, self.port, TcpProtocol)
        self.client.setMessageReceivedCallback(self._on_message_received)
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)

        # Start client service (initiates connection)
        logger.info("Starting client service...")
        self.client.startService()

        logger.info("CTrader WebSocket service started")

    def stop(self):
        """Stop WebSocket service."""
        self._reactor_running = False

        if self.client and self.connected:
            try:
                # Disconnect gracefully
                reactor.callFromThread(self._disconnect)
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

        # Stop reactor
        if reactor.running:
            reactor.callFromThread(reactor.stop)

        if self._reactor_thread:
            self._reactor_thread.join(timeout=5.0)

        logger.info("CTrader WebSocket service stopped")

    def _run_twisted_reactor(self):
        """Run Twisted reactor (blocking call in separate thread)."""
        try:
            # Run reactor (blocking until stopped)
            logger.info("Starting Twisted reactor...")
            reactor.run(installSignalHandlers=False)
            logger.info("Twisted reactor stopped")

        except Exception as e:
            logger.exception(f"Twisted reactor error: {e}")
        finally:
            self._reactor_running = False

    def _on_connected(self, client):
        """Called when WebSocket connection established.

        Args:
            client: The cTrader client instance (passed by library)
        """
        self.connected = True
        logger.info("WebSocket connected to cTrader")

        # Authenticate
        reactor.callLater(0, self._authenticate)

    def _on_disconnected(self, client, reason):
        """Called when WebSocket disconnected.

        Args:
            client: The cTrader client instance (passed by library)
            reason: The disconnection reason
        """
        self.connected = False
        self.authenticated = False

        reason_str = str(reason)

        # Check if disconnection is due to depth quotes timeout (common for demo accounts)
        if 'TimeoutError' in reason_str or 'Deferred' in reason_str:
            logger.warning(
                f"⚠️  WebSocket disconnected due to timeout: {reason}\n"
                f"This is often caused by depth quotes (DOM) subscription not being supported.\n"
                f"The service will reconnect without attempting depth quotes subscription."
            )
            # Mark that we shouldn't try depth quotes on next connection
            self._skip_depth_quotes = True
        elif 'ConnectionDone' in reason_str:
            logger.info(f"WebSocket connection closed cleanly: {reason}")
        else:
            logger.warning(f"WebSocket disconnected from cTrader: {reason}")

        # Auto-reconnect will be handled by ReconnectingClientFactory

    def _authenticate(self):
        """Authenticate with cTrader API."""
        try:
            # Application auth
            app_auth_req = Messages.ProtoOAApplicationAuthReq()
            app_auth_req.clientId = self.client_id
            app_auth_req.clientSecret = self.client_secret

            self.client.send(app_auth_req)
            logger.info("Sent application auth request")

        except Exception as e:
            logger.error(f"Authentication error: {e}")

    def _fetch_account_list(self):
        """Fetch list of trading accounts from cTrader API."""
        try:
            logger.info("Fetching account list from cTrader API...")
            account_list_req = Messages.ProtoOAGetAccountListByAccessTokenReq()
            account_list_req.accessToken = self.access_token

            self.client.send(account_list_req)
            logger.info("Sent account list request")

        except Exception as e:
            logger.error(f"Error fetching account list: {e}")

    def _handle_account_list(self, message):
        """Handle account list response and select account to use."""
        try:
            if not hasattr(message, 'ctidTraderAccount') or len(message.ctidTraderAccount) == 0:
                logger.error("No trading accounts found for this access token")
                return

            # List all available accounts
            logger.info(f"Found {len(message.ctidTraderAccount)} trading account(s):")
            for i, account in enumerate(message.ctidTraderAccount):
                account_id = account.ctidTraderAccountId
                is_live = account.isLive if hasattr(account, 'isLive') else False
                account_type = "LIVE" if is_live else "DEMO"
                logger.info(f"  [{i+1}] Account ID: {account_id} (Type: {account_type})")

            # Select the first account that matches our environment
            selected_account = None
            is_demo_env = (self.environment.lower() == "demo")

            for account in message.ctidTraderAccount:
                is_live = account.isLive if hasattr(account, 'isLive') else False
                # Match account type with environment
                if is_demo_env and not is_live:
                    selected_account = account
                    break
                elif not is_demo_env and is_live:
                    selected_account = account
                    break

            # If no matching account, use the first one
            if not selected_account:
                logger.warning(
                    f"No {self.environment} account found. Using first available account."
                )
                selected_account = message.ctidTraderAccount[0]

            self.account_id = selected_account.ctidTraderAccountId
            is_live = selected_account.isLive if hasattr(selected_account, 'isLive') else False
            account_type = "LIVE" if is_live else "DEMO"

            logger.info(
                f"✓ Selected account: {self.account_id} (Type: {account_type})"
            )

            # Now authorize this account
            reactor.callLater(0, self._authorize_account)

        except Exception as e:
            logger.error(f"Error handling account list: {e}")

    def _authorize_account(self):
        """Authorize trading account after application auth."""
        try:
            if self.account_id is None:
                logger.error("Cannot authorize account: account_id is None")
                return

            account_auth_req = Messages.ProtoOAAccountAuthReq()
            account_auth_req.ctidTraderAccountId = self.account_id
            account_auth_req.accessToken = self.access_token

            self.client.send(account_auth_req)
            logger.info(f"Sent account auth request for account {self.account_id}")

        except Exception as e:
            logger.error(f"Account authorization error: {e}")

    def _subscribe_to_symbols(self):
        """Subscribe to spot quotes and symbols for all configured symbols."""
        try:
            for symbol in self.symbols:
                # Get symbols list first (to get symbol IDs)
                symbols_req = Messages.ProtoOASymbolsListReq()
                symbols_req.ctidTraderAccountId = self.account_id
                self.client.send(symbols_req)

            logger.info(f"Requested symbols list for subscription")

        except Exception as e:
            logger.error(f"Subscription error: {e}")

    def _subscribe_to_depth_quotes(self, symbol_ids: List[int]):
        """Subscribe to depth quotes (order book / DOM)."""
        try:
            if not symbol_ids:
                logger.warning("No symbol IDs to subscribe for depth quotes")
                return

            logger.info(f"Attempting to subscribe to depth quotes for {len(symbol_ids)} symbols: {symbol_ids}")

            # Create ONE request with ALL symbol IDs (symbolId is a repeated field)
            depth_req = Messages.ProtoOASubscribeDepthQuotesReq()
            depth_req.ctidTraderAccountId = self.account_id

            # Add all symbol IDs to the repeated field
            for symbol_id in symbol_ids:
                depth_req.symbolId.append(symbol_id)
                logger.debug(f"Added symbol ID {symbol_id} to depth quotes subscription")

            # Send ONE request with all symbols
            logger.info(
                f"Sending depth quotes subscription request: "
                f"account={self.account_id}, symbolIds={list(depth_req.symbolId)}"
            )
            self.client.send(depth_req)
            logger.info(f"✓ Sent depth quotes subscription request for {len(symbol_ids)} symbols")

        except Exception as e:
            logger.exception(f"Depth subscription error: {e}")
            logger.error(
                f"⚠️  Failed to subscribe to depth quotes. "
                f"Please verify your cTrader account has DOM access enabled."
            )

    def _on_message_received(self, client, message):
        """
        Handle incoming message from cTrader (called by Twisted).

        Args:
            client: The cTrader client instance (passed by library)
            message: The received message

        Dispatches to appropriate handler based on message type.
        """
        try:
            payload_type = message.payloadType

            # Log all message types for debugging
            message_type_name = str(payload_type).split('.')[-1] if hasattr(payload_type, '__class__') else str(payload_type)
            logger.debug(f"Received message: {message_type_name} (type={payload_type})")

            # Application auth response
            if payload_type == Messages.ProtoOAApplicationAuthRes:
                logger.info("✓ Application authenticated")
                # If we need to fetch account list, do it now; otherwise authorize directly
                if self.auto_fetch_account:
                    reactor.callLater(0, self._fetch_account_list)
                else:
                    reactor.callLater(0, self._authorize_account)

            # Account list response
            elif payload_type == Messages.ProtoOAGetAccountListByAccessTokenRes:
                self._handle_account_list(message)

            # Account auth response
            elif payload_type == Messages.ProtoOAAccountAuthRes:
                self.authenticated = True
                logger.info("✓ Account authorized")
                reactor.callLater(0, self._subscribe_to_symbols)

            # Symbols list response
            elif payload_type == Messages.ProtoOASymbolsListRes:
                self._handle_symbols_list(message)

            # Depth quotes subscription response
            elif payload_type == Messages.ProtoOASubscribeDepthQuotesRes:
                logger.info("✓ Successfully subscribed to depth quotes (DOM streaming active)")

            # Depth event (order book update)
            elif payload_type == Messages.ProtoOADepthEvent:
                logger.debug("Received ProtoOADepthEvent - processing order book update")
                self._handle_depth_event(message)

            # Execution event (for sentiment tracking)
            elif payload_type == Messages.ProtoOAExecutionEvent:
                logger.debug("Received ProtoOAExecutionEvent - processing trade execution")
                self._handle_execution_event(message)

            # Error response
            elif payload_type == Messages.ProtoOAErrorRes:
                error_code = message.errorCode if hasattr(message, 'errorCode') else 'unknown'
                error_desc = message.description if hasattr(message, 'description') else ''

                # Check if it's a depth quotes subscription error
                if 'DEPTH' in str(error_desc).upper() or 'DOM' in str(error_desc).upper():
                    logger.warning(
                        f"⚠️  Depth quotes (DOM) not available: {error_code} - {error_desc}\n"
                        f"This is normal for demo accounts or accounts without DOM access.\n"
                        f"Trading and pattern detection will continue without real-time order book data."
                    )
                else:
                    logger.error(f"cTrader error: {error_code} - {error_desc}")

            # Heartbeat (keep-alive)
            elif payload_type == ProtoHeartbeatEvent:
                logger.debug("Received heartbeat")

            # Unknown message type
            else:
                logger.debug(f"Unhandled message type: {message_type_name} (type={payload_type})")

        except Exception as e:
            logger.exception(f"Error handling message: {e}")

    def _handle_symbols_list(self, message):
        """Handle symbols list response and subscribe to relevant symbols."""
        try:
            symbol_ids = []

            for symbol in message.symbol:
                symbol_name = symbol.symbolName

                # Check if this is one of our tracked symbols
                # cTrader uses names like "EURUSD" without slash
                normalized_name = symbol_name.replace('/', '')

                for tracked_symbol in self.symbols:
                    if normalized_name == tracked_symbol.replace('/', ''):
                        self.subscribed_symbols[tracked_symbol] = symbol.symbolId
                        symbol_ids.append(symbol.symbolId)
                        logger.info(f"Found symbol {tracked_symbol}: ID={symbol.symbolId}")
                        break

            if symbol_ids:
                # Subscribe to depth quotes only if not skipped (e.g., after timeout)
                if self._skip_depth_quotes:
                    logger.info(
                        f"⏭️  Skipping depth quotes subscription (not supported by this account). "
                        f"Subscribed to {len(symbol_ids)} symbols for price/volume streaming only."
                    )
                else:
                    # Subscribe to depth quotes (order books) for these symbols
                    reactor.callLater(0, lambda: self._subscribe_to_depth_quotes(symbol_ids))

        except Exception as e:
            logger.error(f"Error handling symbols list: {e}")

    def _handle_spot_event(self, message):
        """
        Handle spot price + volume update.

        This is the main real-time data stream from cTrader.
        Includes bid/ask prices and tick volume.
        """
        try:
            symbol_id = message.symbolId
            timestamp_ms = message.timestamp  # Microseconds from epoch

            # Find symbol name from ID
            symbol_name = None
            for sym, sid in self.subscribed_symbols.items():
                if sid == symbol_id:
                    symbol_name = sym
                    break

            if not symbol_name:
                return

            # Extract prices (cTrader uses pips * 100000)
            bid = message.bid / 100000.0 if hasattr(message, 'bid') else None
            ask = message.ask / 100000.0 if hasattr(message, 'ask') else None

            # Extract volumes
            bid_volume = message.bidVolume if hasattr(message, 'bidVolume') else None
            ask_volume = message.askVolume if hasattr(message, 'askVolume') else None

            # Total tick volume (if available)
            tick_volume = message.volume if hasattr(message, 'volume') else 0

            # Create synthetic order book from bid/ask
            # Note: Real DOM would come from ProtoOADepthEvent (not always available)
            spread = (ask - bid) if (bid and ask) else 0

            synthetic_dom = {
                'symbol': symbol_name,
                'timestamp': int(timestamp_ms / 1000),  # Convert to milliseconds
                'bids': [[bid, bid_volume or 10000]] if bid else [],
                'asks': [[ask, ask_volume or 10000]] if ask else [],
                'mid_price': (bid + ask) / 2 if (bid and ask) else None,
                'spread': spread,
                'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume)
                             if (bid_volume and ask_volume) else 0.0
            }

            # Update buffer
            self.order_book_buffer[symbol_name] = synthetic_dom

            # Store to database (non-blocking)
            reactor.callInThread(self._store_order_book, synthetic_dom)

            # Volume tracking
            if tick_volume > 0:
                volume_data = {
                    'symbol': symbol_name,
                    'timestamp': int(timestamp_ms / 1000),
                    'tick_volume': tick_volume,
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'price': (bid + ask) / 2 if (bid and ask) else None
                }

                self.volume_buffer[symbol_name].append(volume_data)

                # Calculate sentiment from volume imbalance
                self._update_sentiment(symbol_name)

                # Callback
                if self.on_volume_update:
                    reactor.callInThread(self.on_volume_update, volume_data)

            # Callback for order book
            if self.on_order_book_update:
                reactor.callInThread(self.on_order_book_update, synthetic_dom)

            # Push to asyncio queue
            if self.message_queue and not self.message_queue.full():
                try:
                    self.message_queue.put_nowait({
                        'type': 'spot',
                        'data': synthetic_dom
                    })
                except:
                    pass

        except Exception as e:
            logger.error(f"Error handling spot event: {e}")

    def _handle_depth_event(self, message):
        """
        Handle DOM depth event from cTrader.

        ProtoOADepthEvent contains:
        - symbolId: symbol identifier
        - newQuotes: list of ProtoOADepthQuote (new or updated quotes)
        - deletedQuotes: list of quote IDs to remove

        ProtoOADepthQuote structure:
        - id: unique quote identifier (uint64)
        - size: volume in cents (uint64) - divide by 100 for actual volume
        - bid: optional bid price (uint64) - divide by 100000 for actual price
        - ask: optional ask price (uint64) - divide by 100000 for actual price
        """
        try:
            symbol_id = message.symbolId

            # Find symbol name
            symbol_name = None
            for sym, sid in self.subscribed_symbols.items():
                if sid == symbol_id:
                    symbol_name = sym
                    break

            if not symbol_name:
                return

            # Initialize depth quotes for this symbol if needed
            if symbol_name not in self.depth_quotes:
                self.depth_quotes[symbol_name] = {}

            # Process deleted quotes
            if hasattr(message, 'deletedQuotes'):
                for quote_id in message.deletedQuotes:
                    self.depth_quotes[symbol_name].pop(quote_id, None)
                    logger.debug(f"Deleted quote {quote_id} for {symbol_name}")

            # Process new/updated quotes
            if hasattr(message, 'newQuotes'):
                for quote in message.newQuotes:
                    quote_id = quote.id
                    size = quote.size / 100.0  # Convert from cents to actual volume

                    # Extract bid/ask prices (divide by 100000)
                    bid_price = quote.bid / 100000.0 if hasattr(quote, 'bid') and quote.bid else None
                    ask_price = quote.ask / 100000.0 if hasattr(quote, 'ask') and quote.ask else None

                    # Store quote
                    self.depth_quotes[symbol_name][quote_id] = {
                        'id': quote_id,
                        'size': size,
                        'bid': bid_price,
                        'ask': ask_price
                    }

                    logger.debug(
                        f"Updated quote {quote_id} for {symbol_name}: "
                        f"bid={bid_price}, ask={ask_price}, size={size}"
                    )

            # Rebuild order book from quotes
            bids = []
            asks = []

            for quote in self.depth_quotes[symbol_name].values():
                if quote['bid'] is not None:
                    bids.append([quote['bid'], quote['size']])
                if quote['ask'] is not None:
                    asks.append([quote['ask'], quote['size']])

            # Sort: bids descending (highest first), asks ascending (lowest first)
            bids.sort(reverse=True, key=lambda x: x[0])
            asks.sort(key=lambda x: x[0])

            # Calculate DOM metrics
            mid_price = None
            spread = None
            imbalance = 0.0

            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid

                total_bid_volume = sum(b[1] for b in bids)
                total_ask_volume = sum(a[1] for a in asks)

                if total_bid_volume + total_ask_volume > 0:
                    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

            # Create DOM data
            dom_data = {
                'symbol': symbol_name,
                'timestamp': int(time.time() * 1000),  # Current time in ms
                'bids': bids,
                'asks': asks,
                'mid_price': mid_price,
                'spread': spread,
                'imbalance': imbalance
            }

            # Update buffer and store
            self.order_book_buffer[symbol_name] = dom_data
            reactor.callInThread(self._store_order_book, dom_data)

            # Callback for order book update
            if self.on_order_book_update:
                reactor.callInThread(self.on_order_book_update, dom_data)

            logger.debug(
                f"Processed depth event for {symbol_name}: "
                f"{len(bids)} bids, {len(asks)} asks, "
                f"spread={spread:.5f if spread else 0}"
            )

        except Exception as e:
            logger.exception(f"Error handling depth event: {e}")

    def _handle_execution_event(self, message):
        """Handle execution event (for sentiment tracking from order flow)."""
        try:
            # Extract trade direction and volume
            if hasattr(message, 'order'):
                order = message.order
                trade_side = order.tradeSide if hasattr(order, 'tradeSide') else None
                volume = order.volume if hasattr(order, 'volume') else 0

                # Use this to track buying/selling pressure
                # This is simplified - full sentiment would need more data
                logger.debug(f"Execution event: side={trade_side}, volume={volume}")

        except Exception as e:
            logger.error(f"Error handling execution event: {e}")

    def _update_sentiment(self, symbol: str):
        """Calculate sentiment from recent volume data."""
        try:
            if symbol not in self.volume_buffer or len(self.volume_buffer[symbol]) < 10:
                return

            recent_volumes = list(self.volume_buffer[symbol])[-20:]  # Last 20 ticks

            total_buy = sum(v.get('bid_volume', 0) or 0 for v in recent_volumes)
            total_sell = sum(v.get('ask_volume', 0) or 0 for v in recent_volumes)

            if total_buy + total_sell == 0:
                return

            # Sentiment ratio (-1 to +1, where +1 = all buying)
            sentiment_ratio = (total_buy - total_sell) / (total_buy + total_sell)

            # Classify sentiment
            if sentiment_ratio > 0.3:
                sentiment = "bullish"
            elif sentiment_ratio < -0.3:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            sentiment_data = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'sentiment': sentiment,
                'ratio': sentiment_ratio,
                'buy_volume': total_buy,
                'sell_volume': total_sell,
                'confidence': abs(sentiment_ratio)
            }

            self.sentiment_buffer[symbol] = sentiment_data

            # Store to database (non-blocking)
            reactor.callInThread(self._store_sentiment, sentiment_data)

            # Callback
            if self.on_sentiment_update:
                reactor.callInThread(self.on_sentiment_update, sentiment_data)

            logger.debug(f"Sentiment for {symbol}: {sentiment} (ratio={sentiment_ratio:.2f})")

        except Exception as e:
            logger.error(f"Error updating sentiment: {e}")

    def _store_order_book(self, dom_data: Dict[str, Any]):
        """Store order book snapshot to database (called from thread pool)."""
        try:
            with self.db_engine.begin() as conn:
                query = text(
                    "INSERT INTO market_depth (symbol, ts_utc, bids, asks, mid_price, spread, imbalance, provider) "
                    "VALUES (:symbol, :ts_utc, :bids, :asks, :mid_price, :spread, :imbalance, :provider)"
                )

                conn.execute(query, {
                    'symbol': dom_data['symbol'],
                    'ts_utc': dom_data['timestamp'],
                    'bids': json.dumps(dom_data['bids']),
                    'asks': json.dumps(dom_data['asks']),
                    'mid_price': dom_data.get('mid_price'),
                    'spread': dom_data.get('spread'),
                    'imbalance': dom_data.get('imbalance'),
                    'provider': 'ctrader'
                })

            logger.debug(f"Stored order book for {dom_data['symbol']}")

        except Exception as e:
            logger.error(f"Error storing order book: {e}")

    def _store_sentiment(self, sentiment_data: Dict[str, Any]):
        """Store sentiment data to database (called from thread pool)."""
        try:
            with self.db_engine.begin() as conn:
                query = text(
                    "INSERT INTO sentiment_data (symbol, ts_utc, sentiment, ratio, "
                    "buy_volume, sell_volume, confidence, long_pct, short_pct, provider, ts_created_ms) "
                    "VALUES (:symbol, :ts_utc, :sentiment, :ratio, :buy_volume, "
                    ":sell_volume, :confidence, :long_pct, :short_pct, :provider, :ts_created_ms)"
                )

                # Calculate long/short percentages from volumes
                total_volume = sentiment_data['buy_volume'] + sentiment_data['sell_volume']
                long_pct = (sentiment_data['buy_volume'] / total_volume * 100) if total_volume > 0 else 50.0
                short_pct = 100.0 - long_pct

                conn.execute(query, {
                    'symbol': sentiment_data['symbol'],
                    'ts_utc': sentiment_data['timestamp'],
                    'sentiment': sentiment_data['sentiment'],
                    'ratio': sentiment_data['ratio'],
                    'buy_volume': sentiment_data['buy_volume'],
                    'sell_volume': sentiment_data['sell_volume'],
                    'confidence': sentiment_data['confidence'],
                    'long_pct': long_pct,
                    'short_pct': short_pct,
                    'provider': 'ctrader_orderflow',
                    'ts_created_ms': sentiment_data['timestamp']
                })

            logger.debug(
                f"Stored sentiment for {sentiment_data['symbol']}: "
                f"{sentiment_data['sentiment']} (ratio={sentiment_data['ratio']:.2f})"
            )

        except Exception as e:
            logger.error(f"Error storing sentiment: {e}")

    def _on_error(self, error):
        """Handle connection error."""
        logger.error(f"WebSocket error: {error}")
        self.connected = False

    def _disconnect(self):
        """Disconnect from cTrader."""
        try:
            if self.client:
                self.client.disconnect()
                self.connected = False
                self.authenticated = False
                logger.info("Disconnected from cTrader")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def get_latest_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest order book snapshot from buffer."""
        return self.order_book_buffer.get(symbol)

    def get_latest_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest sentiment data from buffer."""
        return self.sentiment_buffer.get(symbol)

    def get_recent_volumes(self, symbol: str, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent volume data points."""
        if symbol not in self.volume_buffer:
            return []
        return list(self.volume_buffer[symbol])[-count:]
