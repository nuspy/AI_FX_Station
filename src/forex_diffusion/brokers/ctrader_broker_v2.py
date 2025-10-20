"""
FxPro cTrader broker integration for live trading.

Provides order execution, position tracking, and real-time P&L calculation.
Uses cTrader Open API for trading operations.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio

from loguru import logger

# cTrader Open API imports
try:
    from ctrader_open_api import Client, Protobuf, EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOANewOrderReq,
        ProtoOAClosePositionReq,
        ProtoOAAmendPositionSLTPReq
    )
    CTRADER_AVAILABLE = True

    # cTrader API constants (these are integers, not enum classes)
    # Based on ProtoOA protocol specification
    class ProtoOAOrderType:
        """cTrader order type constants."""
        MARKET = 1
        LIMIT = 2
        STOP = 3
        STOP_LIMIT = 4
        MARKET_RANGE = 5

    class ProtoOATradeSide:
        """cTrader trade side constants."""
        BUY = 1
        SELL = 2

    class ProtoOAPayloadType:
        """cTrader payload type constants (for message handling)."""
        PROTO_OA_EXECUTION_EVENT = 2124
        PROTO_OA_SPOT_EVENT = 2128

except ImportError as e:
    CTRADER_AVAILABLE = False
    logger.warning(f"ctrader-open-api not available - install with: pip install ctrader-open-api (error: {e})")


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class PositionStatus(Enum):
    """Position status."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    volume: float  # In lots
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "pending"
    filled_volume: float = 0.0
    created_at: datetime = None
    filled_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Position:
    """Position representation."""
    position_id: str
    symbol: str
    side: OrderSide
    volume: float  # In lots
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = None
    closed_at: Optional[datetime] = None
    commission: float = 0.0
    swap: float = 0.0

    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.utcnow()

    def update_pnl(self, current_price: float, pip_value: float = 10.0):
        """
        Update unrealized P&L based on current price.

        Args:
            current_price: Current market price
            pip_value: Value of 1 pip in account currency (default: $10 for standard lot)
        """
        self.current_price = current_price

        # Calculate pip difference
        if self.side == OrderSide.BUY:
            pip_diff = (current_price - self.entry_price) * 10000  # Forex pips
        else:
            pip_diff = (self.entry_price - current_price) * 10000

        # Calculate P&L
        self.unrealized_pnl = pip_diff * pip_value * self.volume - self.commission - self.swap


class CTraderBroker:
    """
    cTrader broker integration for live trading.

    Provides:
    - Order placement (market, limit, stop)
    - Position management (modify SL/TP, close)
    - Real-time position tracking
    - P&L calculation
    - Account information
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id: int,
        environment: str = "demo",
        on_position_update: Optional[Callable] = None,
        on_order_filled: Optional[Callable] = None
    ):
        """
        Initialize cTrader broker.

        Args:
            client_id: cTrader app client ID
            client_secret: cTrader app client secret
            access_token: OAuth access token
            account_id: Trading account ID
            environment: 'demo' or 'live'
            on_position_update: Callback for position updates
            on_order_filled: Callback for order fills
        """
        if not CTRADER_AVAILABLE:
            raise ImportError("ctrader-open-api required")

        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id = account_id
        self.environment = environment

        # Callbacks
        self.on_position_update = on_position_update
        self.on_order_filled = on_order_filled

        # State
        self.client: Optional[Client] = None
        self.connected = False
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.account_info: Dict = {}

        logger.info(f"Initialized cTrader broker: account={account_id}, env={environment}")

    async def connect(self):
        """
        Connect to cTrader API.

        Strategy: Token first, OAuth fallback
        1. Try direct token authentication (faster, simpler)
        2. If fails, fallback to OAuth flow
        """
        # Strategy 1: Try direct token authentication
        try:
            logger.info("Attempting direct token connection...")
            await self._connect_with_token()
            logger.info("Successfully connected via direct token")
            return
        except Exception as e:
            logger.warning(f"Direct token connection failed: {e}, trying OAuth fallback...")

        # Strategy 2: Fallback to OAuth
        try:
            logger.info("Attempting OAuth connection...")
            await self._connect_with_oauth()
            logger.info("Successfully connected via OAuth")
        except Exception as e:
            logger.error(f"OAuth connection also failed: {e}")
            raise

    async def _connect_with_token(self):
        """Connect using direct token (no OAuth) - preferred method."""
        # Determine host
        if self.environment == "demo":
            host = EndPoints.PROTOBUF_DEMO_HOST
        else:
            host = EndPoints.PROTOBUF_LIVE_HOST

        # Create client
        self.client = Client(host, EndPoints.PROTOBUF_PORT)

        # Connect
        await self.client.connect()

        # Authenticate application
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAApplicationAuthReq
        auth_req = ProtoOAApplicationAuthReq()
        auth_req.clientId = self.client_id
        auth_req.clientSecret = self.client_secret
        await self.client.send(auth_req)

        # Authorize account using token (no OAuth)
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAAccountAuthReq
        account_req = ProtoOAAccountAuthReq()
        account_req.ctidTraderAccountId = self.account_id
        account_req.accessToken = self.access_token
        await self.client.send(account_req)

        self.connected = True
        logger.info("Connected to cTrader API via token")

        # Start event loop
        asyncio.create_task(self._event_loop())

    async def _connect_with_oauth(self):
        """Connect using OAuth flow - fallback method."""
        # Determine host
        if self.environment == "demo":
            host = EndPoints.PROTOBUF_DEMO_HOST
        else:
            host = EndPoints.PROTOBUF_LIVE_HOST

        # Create client
        self.client = Client(host, EndPoints.PROTOBUF_PORT)

        # Connect
        await self.client.connect()

        # Authenticate application
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAApplicationAuthReq
        auth_req = ProtoOAApplicationAuthReq()
        auth_req.clientId = self.client_id
        auth_req.clientSecret = self.client_secret
        await self.client.send(auth_req)

        # OAuth flow - authorize with access token
        # Note: This assumes access_token was obtained via OAuth beforehand
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAAccountAuthReq
        account_req = ProtoOAAccountAuthReq()
        account_req.ctidTraderAccountId = self.account_id
        account_req.accessToken = self.access_token
        await self.client.send(account_req)

        self.connected = True
        logger.info("Connected to cTrader API via OAuth")

        # Start event loop
        asyncio.create_task(self._event_loop())

    async def disconnect(self):
        """Disconnect from cTrader API."""
        if self.client and self.connected:
            await self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from cTrader API")

    async def _event_loop(self):
        """Process incoming events from cTrader."""
        while self.connected:
            try:
                message = await self.client.receive()

                # Handle different message types
                if message.payloadType == ProtoOAPayloadType.PROTO_OA_EXECUTION_EVENT:
                    await self._handle_execution_event(message)
                elif message.payloadType == ProtoOAPayloadType.PROTO_OA_SPOT_EVENT:
                    await self._handle_spot_event(message)

            except Exception as e:
                logger.error(f"Error in event loop: {e}")
                await asyncio.sleep(1)

    async def _handle_execution_event(self, message):
        """Handle execution events (order fills, position updates)."""
        # Parse execution event
        execution = message.execution

        if execution.executionType == "ORDER_FILLED":
            # Order filled
            order_id = str(execution.orderId)
            if order_id in self.orders:
                order = self.orders[order_id]
                order.status = "filled"
                order.filled_at = datetime.utcnow()
                order.filled_volume = execution.volume / 100  # Convert to lots

                # Create position
                position = Position(
                    position_id=str(execution.positionId),
                    symbol=order.symbol,
                    side=order.side,
                    volume=order.filled_volume,
                    entry_price=execution.executionPrice,
                    current_price=execution.executionPrice,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit
                )
                self.positions[position.position_id] = position

                logger.info(f"Order filled: {order_id} -> Position {position.position_id}")

                # Callback
                if self.on_order_filled:
                    self.on_order_filled(order, position)

        elif execution.executionType == "ORDER_CANCELLED":
            order_id = str(execution.orderId)
            if order_id in self.orders:
                self.orders[order_id].status = "cancelled"
                logger.info(f"Order cancelled: {order_id}")

    async def _handle_spot_event(self, message):
        """Handle spot price updates."""
        # Update position P&L with new prices
        symbol = message.symbolId  # Need symbol ID to name mapping
        bid = message.bid / 100000
        ask = message.ask / 100000

        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                # Use bid for sell positions, ask for buy positions
                current_price = bid if position.side == OrderSide.SELL else ask
                position.update_pnl(current_price)

                # Callback
                if self.on_position_update:
                    self.on_position_update(position)

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Order:
        """
        Place market order.

        Args:
            symbol: Symbol name (e.g., "EURUSD")
            side: BUY or SELL
            volume: Volume in lots
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price

        Returns:
            Order object
        """
        if not self.connected:
            raise RuntimeError("Not connected to cTrader")

        # Get symbol ID (need symbol mapping)
        symbol_id = await self._get_symbol_id(symbol)

        # Create order
        order = Order(
            order_id=f"order_{datetime.utcnow().timestamp()}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Convert to cTrader format
        trade_side = ProtoOATradeSide.BUY if side == OrderSide.BUY else ProtoOATradeSide.SELL
        volume_units = int(volume * 100000)  # Convert lots to units

        # Send order request
        request = ProtoOANewOrderReq(
            ctidTraderAccountId=self.account_id,
            symbolId=symbol_id,
            orderType=ProtoOAOrderType.MARKET,
            tradeSide=trade_side,
            volume=volume_units,
            stopLoss=int(stop_loss * 100000) if stop_loss else None,
            takeProfit=int(take_profit * 100000) if take_profit else None
        )

        await self.client.send(request)

        # Store order
        self.orders[order.order_id] = order

        logger.info(f"Placed market order: {symbol} {side.value} {volume} lots")

        return order

    async def close_position(self, position_id: str, volume: Optional[float] = None) -> bool:
        """
        Close position (fully or partially).

        Args:
            position_id: Position ID to close
            volume: Optional partial volume (None = full close)

        Returns:
            True if successful
        """
        if not self.connected:
            raise RuntimeError("Not connected to cTrader")

        if position_id not in self.positions:
            logger.warning(f"Position not found: {position_id}")
            return False

        position = self.positions[position_id]

        # Determine volume to close
        close_volume = volume if volume else position.volume
        volume_units = int(close_volume * 100000)

        # Send close request
        request = ProtoOAClosePositionReq(
            ctidTraderAccountId=self.account_id,
            positionId=int(position_id),
            volume=volume_units
        )

        await self.client.send(request)

        # Update position
        if volume and volume < position.volume:
            position.volume -= volume
            position.status = PositionStatus.PARTIALLY_CLOSED
        else:
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.utcnow()
            position.realized_pnl = position.unrealized_pnl

        logger.info(f"Closed position: {position_id} ({close_volume} lots)")

        return True

    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify position stop loss and/or take profit.

        Args:
            position_id: Position ID
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            True if successful
        """
        if not self.connected:
            raise RuntimeError("Not connected to cTrader")

        if position_id not in self.positions:
            logger.warning(f"Position not found: {position_id}")
            return False

        # Send modify request
        request = ProtoOAAmendPositionSLTPReq(
            ctidTraderAccountId=self.account_id,
            positionId=int(position_id),
            stopLoss=int(stop_loss * 100000) if stop_loss else None,
            takeProfit=int(take_profit * 100000) if take_profit else None
        )

        await self.client.send(request)

        # Update local position
        position = self.positions[position_id]
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit

        logger.info(f"Modified position {position_id}: SL={stop_loss}, TP={take_profit}")

        return True

    async def get_account_info(self) -> Dict:
        """
        Get account information.

        Returns:
            Dictionary with account details (balance, equity, margin, etc.)
        """
        if not self.connected:
            raise RuntimeError("Not connected to cTrader")

        # Request account info
        # Implementation depends on cTrader API message structure
        # This is a placeholder
        return {
            "balance": 10000.0,
            "equity": 10500.0,
            "margin_used": 500.0,
            "margin_free": 10000.0,
            "currency": "USD"
        }

    async def get_open_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of open positions
        """
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def _calculate_total_pnl(self) -> Tuple[float, float]:
        """
        Calculate total P&L (realized + unrealized) - synchronous version.

        Returns:
            (realized_pnl, unrealized_pnl)
        """
        realized = sum(p.realized_pnl for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values() if p.status == PositionStatus.OPEN)

        return realized, unrealized

    async def get_total_pnl(self) -> Tuple[float, float]:
        """
        Get total P&L (realized + unrealized).

        Returns:
            (realized_pnl, unrealized_pnl)
        """
        return self._calculate_total_pnl()

    async def _get_symbol_id(self, symbol: str) -> int:
        """Get symbol ID from symbol name."""
        # This requires symbol list from cTrader
        # Placeholder implementation
        symbol_map = {
            "EURUSD": 1,
            "GBPUSD": 2,
            "USDJPY": 3,
            # ... more symbols
        }
        return symbol_map.get(symbol, 1)


class BrokerSimulator(CTraderBroker):
    """
    Simulated broker for testing without real connection.

    Uses same interface as CTraderBroker but simulates execution.
    """

    def __init__(self, initial_balance: float = 10000.0):
        """Initialize simulator."""
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.connected = True
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.current_prices: Dict[str, float] = {}

        logger.info(f"Initialized broker simulator: balance=${initial_balance}")

    async def connect(self):
        """Simulated connection."""
        self.connected = True
        logger.info("Simulator connected")

    async def disconnect(self):
        """Simulated disconnection."""
        self.connected = False
        logger.info("Simulator disconnected")

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Order:
        """Simulate market order."""
        # Get current price (simulated)
        current_price = self.current_prices.get(symbol, 1.1000)

        # Create and fill order immediately
        order = Order(
            order_id=f"sim_order_{len(self.orders)}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="filled",
            filled_volume=volume,
            filled_at=datetime.utcnow()
        )

        # Create position
        position = Position(
            position_id=f"sim_pos_{len(self.positions)}",
            symbol=symbol,
            side=side,
            volume=volume,
            entry_price=current_price,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.orders[order.order_id] = order
        self.positions[position.position_id] = position

        logger.info(f"[SIM] Filled order: {symbol} {side.value} {volume} @ {current_price}")

        return order

    async def close_position(self, position_id: str, volume: Optional[float] = None) -> bool:
        """Simulate position close."""
        if position_id not in self.positions:
            return False

        position = self.positions[position_id]
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.utcnow()
        position.realized_pnl = position.unrealized_pnl

        self.balance += position.realized_pnl

        logger.info(f"[SIM] Closed position {position_id}: P&L=${position.realized_pnl:.2f}")

        return True

    def update_price(self, symbol: str, price: float):
        """Update simulated market price."""
        self.current_prices[symbol] = price

        # Update position P&L
        for position in self.positions.values():
            if position.symbol == symbol and position.status == PositionStatus.OPEN:
                position.update_pnl(price)

        # Update equity
        _, unrealized = self._calculate_total_pnl()
        self.equity = self.balance + unrealized


def get_ctrader_accounts(client_id: str, client_secret: str, access_token: str, environment: str = "demo") -> List[Dict]:
    """
    Get list of cTrader accounts using API credentials.

    Strategy: Token first, OAuth fallback
    1. Try direct connection with access_token (no OAuth)
    2. If fails, fallback to OAuth flow

    Args:
        client_id: cTrader app client ID
        client_secret: cTrader app client secret
        access_token: OAuth access token (or direct token)
        environment: 'demo' or 'live'

    Returns:
        List of account dictionaries with id, type, balance, currency
    """
    if not CTRADER_AVAILABLE:
        logger.warning("cTrader API not available")
        return []

    # Strategy 1: Try direct token connection (no OAuth)
    try:
        logger.info(f"Attempting direct token connection to cTrader {environment}...")
        accounts = _get_accounts_with_token(client_id, client_secret, access_token, environment)
        if accounts:
            logger.info(f"Successfully retrieved {len(accounts)} accounts via direct token")
            return accounts
    except Exception as e:
        logger.warning(f"Direct token connection failed: {e}, trying OAuth fallback...")

    # Strategy 2: Fallback to OAuth flow
    try:
        logger.info(f"Attempting OAuth connection to cTrader {environment}...")
        accounts = _get_accounts_with_oauth(client_id, client_secret, access_token, environment)
        if accounts:
            logger.info(f"Successfully retrieved {len(accounts)} accounts via OAuth")
            return accounts
    except Exception as e:
        logger.error(f"OAuth connection also failed: {e}")

    return []


def _get_accounts_with_token(client_id: str, client_secret: str, access_token: str, environment: str) -> List[Dict]:
    """
    Get accounts using direct token (no OAuth flow).

    This is the preferred method - faster and simpler.
    """
    import asyncio
    from ctrader_open_api import Client, EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAGetAccountListByAccessTokenReq
    )

    async def _fetch_accounts():
        # Determine host
        host = EndPoints.PROTOBUF_DEMO_HOST if environment == "demo" else EndPoints.PROTOBUF_LIVE_HOST
        port = EndPoints.PROTOBUF_PORT

        # Create client
        client = Client(host, port)

        try:
            # Connect
            await client.connect()
            logger.debug(f"Connected to {host}:{port}")

            # Authenticate application
            auth_req = ProtoOAApplicationAuthReq()
            auth_req.clientId = client_id
            auth_req.clientSecret = client_secret

            await client.send(auth_req)
            logger.debug("Application authenticated")

            # Get accounts by access token (no OAuth)
            accounts_req = ProtoOAGetAccountListByAccessTokenReq()
            accounts_req.accessToken = access_token

            response = await client.send(accounts_req)

            # Parse accounts from response
            accounts = []
            if hasattr(response, 'ctidTraderAccount'):
                for acc in response.ctidTraderAccount:
                    account_info = {
                        'id': str(acc.ctidTraderAccountId),
                        'type': 'demo' if acc.isLive == False else 'live',
                        'balance': acc.balance / 100 if hasattr(acc, 'balance') else 0,  # Convert cents to dollars
                        'currency': acc.depositAssetId if hasattr(acc, 'depositAssetId') else 'USD'
                    }
                    accounts.append(account_info)

            return accounts

        finally:
            await client.disconnect()
            logger.debug("Disconnected from cTrader")

    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_fetch_accounts())
    finally:
        loop.close()


def _get_accounts_with_oauth(client_id: str, client_secret: str, access_token: str, environment: str) -> List[Dict]:
    """
    Get accounts using OAuth flow.

    This is the fallback method when direct token doesn't work.
    """
    import asyncio
    from ctrader_open_api import Client, EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAGetAccountListReq
    )

    async def _fetch_accounts():
        # Determine host
        host = EndPoints.PROTOBUF_DEMO_HOST if environment == "demo" else EndPoints.PROTOBUF_LIVE_HOST
        port = EndPoints.PROTOBUF_PORT

        # Create client
        client = Client(host, port)

        try:
            # Connect
            await client.connect()
            logger.debug(f"Connected to {host}:{port} for OAuth")

            # Authenticate application
            auth_req = ProtoOAApplicationAuthReq()
            auth_req.clientId = client_id
            auth_req.clientSecret = client_secret

            await client.send(auth_req)
            logger.debug("Application authenticated for OAuth")

            # Note: OAuth flow requires user interaction
            # This is a simplified version - real OAuth needs browser redirect
            # For now, we try to use the access_token as if it's already authorized

            # Get accounts list (requires prior OAuth authorization)
            accounts_req = ProtoOAGetAccountListReq()
            accounts_req.accessToken = access_token

            response = await client.send(accounts_req)

            # Parse accounts from response
            accounts = []
            if hasattr(response, 'ctidTraderAccount'):
                for acc in response.ctidTraderAccount:
                    account_info = {
                        'id': str(acc.ctidTraderAccountId),
                        'type': 'demo' if acc.isLive == False else 'live',
                        'balance': acc.balance / 100 if hasattr(acc, 'balance') else 0,
                        'currency': acc.depositAssetId if hasattr(acc, 'depositAssetId') else 'USD'
                    }
                    accounts.append(account_info)

            return accounts

        finally:
            await client.disconnect()
            logger.debug("Disconnected from cTrader OAuth")

    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_fetch_accounts())
    finally:
        loop.close()
