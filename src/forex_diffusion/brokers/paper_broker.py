"""
Paper Trading Broker - Simulated trading with realistic fills and slippage.
"""
from __future__ import annotations

import time
import json
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from loguru import logger

from .base import (
    BrokerBase, Order, Position, AccountInfo,
    OrderType, OrderSide, OrderStatus, PositionSide
)
from ..utils.user_settings import SETTINGS_DIR


class PaperBroker(BrokerBase):
    """
    Paper trading broker with realistic simulation.

    Features:
    - Simulated fills with configurable slippage
    - Position tracking
    - P&L calculation
    - Order management (limit, stop, market orders)
    - Persistence across sessions
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        currency: str = "USD",
        leverage: int = 30,
        slippage_pips: float = 0.5,
        state_file: Optional[Path] = None
    ):
        """
        Initialize paper broker.

        Args:
            initial_balance: Starting account balance
            currency: Account currency
            leverage: Account leverage
            slippage_pips: Simulated slippage in pips
            state_file: Path to save/load state
        """
        self.state_file = state_file or (SETTINGS_DIR / "paper_broker_state.json")

        self.initial_balance = initial_balance
        self.currency = currency
        self.leverage = leverage
        self.slippage_pips = slippage_pips

        # Account state
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin_used = 0.0

        # Orders and positions
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}

        # Connection state
        self._connected = False

        # Load saved state
        self._load_state()

    def _load_state(self):
        """Load broker state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                self.balance = data.get('balance', self.initial_balance)
                self.equity = data.get('equity', self.balance)
                self.margin_used = data.get('margin_used', 0.0)

                # Load orders
                for order_data in data.get('orders', []):
                    order = self._dict_to_order(order_data)
                    self.orders[order.order_id] = order

                # Load positions
                for pos_data in data.get('positions', []):
                    pos = self._dict_to_position(pos_data)
                    self.positions[pos.symbol] = pos

                logger.info(f"Loaded paper broker state: {len(self.orders)} orders, {len(self.positions)} positions")
        except Exception as e:
            logger.warning(f"Failed to load paper broker state: {e}")

    def _save_state(self):
        """Save broker state to file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'balance': self.balance,
                'equity': self.equity,
                'margin_used': self.margin_used,
                'orders': [self._order_to_dict(o) for o in self.orders.values()],
                'positions': [self._position_to_dict(p) for p in self.positions.values()],
                'timestamp': datetime.now().isoformat(),
            }

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save paper broker state: {e}")

    def connect(self) -> bool:
        """Connect (always succeeds for paper broker)"""
        self._connected = True
        logger.info("Paper broker connected")
        return True

    def disconnect(self) -> bool:
        """Disconnect and save state"""
        self._save_state()
        self._connected = False
        logger.info("Paper broker disconnected")
        return True

    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connected

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        # Update equity based on positions
        self._update_equity()

        return AccountInfo(
            account_id="PAPER",
            balance=self.balance,
            equity=self.equity,
            margin_used=self.margin_used,
            margin_available=self.equity - self.margin_used,
            currency=self.currency,
            leverage=self.leverage,
            unrealized_pnl=sum(p.unrealized_pnl for p in self.positions.values()),
        )

    def place_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """Place an order"""
        # Generate order ID
        order_id = f"PAPER_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        order.order_id = order_id
        order.status = OrderStatus.PENDING
        order.timestamp = datetime.now()

        # For market orders, fill immediately
        if order.order_type == OrderType.MARKET:
            # Simulate slippage
            fill_price = self._apply_slippage(order.price or 1.0, order.side)
            order.avg_fill_price = fill_price
            order.filled_quantity = order.quantity
            order.status = OrderStatus.FILLED

            # Update position
            self._update_position_from_order(order)

        else:
            # Limit/stop orders stay pending
            order.status = OrderStatus.OPEN

        self.orders[order_id] = order
        self._save_state()

        logger.info(f"Paper order placed: {order_id} - {order.symbol} {order.side.value} {order.quantity}")
        return True, order_id, None

    def cancel_order(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel an order"""
        if order_id not in self.orders:
            return False, "Order not found"

        order = self.orders[order_id]

        if order.status not in (OrderStatus.OPEN, OrderStatus.PARTIAL):
            return False, f"Cannot cancel order with status {order.status.value}"

        order.status = OrderStatus.CANCELLED
        self._save_state()

        logger.info(f"Paper order cancelled: {order_id}")
        return True, None

    def modify_order(self, order_id: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Modify an order"""
        if order_id not in self.orders:
            return False, "Order not found"

        order = self.orders[order_id]

        if order.status != OrderStatus.OPEN:
            return False, f"Cannot modify order with status {order.status.value}"

        # Update order fields
        if 'price' in kwargs:
            order.price = kwargs['price']
        if 'stop_price' in kwargs:
            order.stop_price = kwargs['stop_price']
        if 'take_profit' in kwargs:
            order.take_profit = kwargs['take_profit']
        if 'stop_loss' in kwargs:
            order.stop_loss = kwargs['stop_loss']

        self._save_state()

        logger.info(f"Paper order modified: {order_id}")
        return True, None

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        orders = [
            o for o in self.orders.values()
            if o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)
        ]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all positions"""
        positions = list(self.positions.values())

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        return positions

    def close_position(self, symbol: str, quantity: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """Close a position"""
        if symbol not in self.positions:
            return False, "Position not found"

        position = self.positions[symbol]

        # Determine close quantity
        close_qty = quantity if quantity else position.quantity

        if close_qty > position.quantity:
            return False, "Close quantity exceeds position size"

        # Create closing order
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        close_order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=close_qty,
            price=position.current_price,
            comment="Close position"
        )

        success, order_id, error = self.place_order(close_order)

        if success:
            # Realize P&L
            pnl = position.unrealized_pnl * (close_qty / position.quantity)
            self.balance += pnl

            # Update or remove position
            if close_qty >= position.quantity:
                del self.positions[symbol]
            else:
                position.quantity -= close_qty
                position.realized_pnl += pnl

            self._save_state()

        return success, error

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information (simplified for paper broker)"""
        return {
            'symbol': symbol,
            'pip_size': 0.0001,
            'min_volume': 0.01,
            'max_volume': 100.0,
            'volume_step': 0.01,
            'digits': 5,
            'description': f'Paper trading: {symbol}',
        }

    def get_available_symbols(self) -> List[str]:
        """Get available symbols (simplified)"""
        return [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
            'AUD/USD', 'USD/CAD', 'NZD/USD',
            'XAU/USD', 'XAG/USD',
        ]

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply simulated slippage to fill price"""
        pip_size = 0.0001  # Simplified
        slippage = self.slippage_pips * pip_size

        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage

    def _update_position_from_order(self, order: Order):
        """Update position based on filled order"""
        symbol = order.symbol

        if symbol not in self.positions:
            # Create new position
            side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT

            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=order.filled_quantity,
                entry_price=order.avg_fill_price,
                current_price=order.avg_fill_price,
                unrealized_pnl=0.0,
                timestamp=order.timestamp
            )
        else:
            # Update existing position
            position = self.positions[symbol]

            # Check if same direction
            order_is_long = order.side == OrderSide.BUY
            position_is_long = position.side == PositionSide.LONG

            if order_is_long == position_is_long:
                # Add to position
                total_qty = position.quantity + order.filled_quantity
                weighted_price = (
                    (position.entry_price * position.quantity + order.avg_fill_price * order.filled_quantity) / total_qty
                )
                position.quantity = total_qty
                position.entry_price = weighted_price
            else:
                # Reduce position or reverse
                if order.filled_quantity >= position.quantity:
                    # Close and potentially reverse
                    pnl = self._calculate_pnl(position, order.avg_fill_price)
                    self.balance += pnl

                    remaining = order.filled_quantity - position.quantity

                    if remaining > 0:
                        # Reverse position
                        position.side = PositionSide.SHORT if position_is_long else PositionSide.LONG
                        position.quantity = remaining
                        position.entry_price = order.avg_fill_price
                        position.unrealized_pnl = 0.0
                    else:
                        # Full close
                        del self.positions[symbol]
                else:
                    # Partial close
                    pnl = self._calculate_pnl(position, order.avg_fill_price) * (order.filled_quantity / position.quantity)
                    self.balance += pnl
                    position.quantity -= order.filled_quantity
                    position.realized_pnl += pnl

    def _calculate_pnl(self, position: Position, close_price: float) -> float:
        """Calculate P&L for a position"""
        price_diff = close_price - position.entry_price

        if position.side == PositionSide.SHORT:
            price_diff = -price_diff

        # Simplified: 1 pip = $10 per lot
        pips = price_diff / 0.0001
        pnl = pips * 10 * position.quantity

        return pnl

    def _update_equity(self):
        """Update equity based on current positions"""
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        self.equity = self.balance + unrealized_pnl

    def _order_to_dict(self, order: Order) -> Dict:
        """Convert Order to dict for serialization"""
        return order.to_dict()

    def _dict_to_order(self, data: Dict) -> Order:
        """Convert dict to Order"""
        return Order(
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            order_type=OrderType(data['order_type']),
            quantity=data['quantity'],
            price=data.get('price'),
            stop_price=data.get('stop_price'),
            take_profit=data.get('take_profit'),
            stop_loss=data.get('stop_loss'),
            order_id=data.get('order_id'),
            status=OrderStatus(data['status']),
            filled_quantity=data.get('filled_quantity', 0.0),
            avg_fill_price=data.get('avg_fill_price'),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None,
            comment=data.get('comment'),
        )

    def _position_to_dict(self, position: Position) -> Dict:
        """Convert Position to dict for serialization"""
        return position.to_dict()

    def _dict_to_position(self, data: Dict) -> Position:
        """Convert dict to Position"""
        return Position(
            symbol=data['symbol'],
            side=PositionSide(data['side']),
            quantity=data['quantity'],
            entry_price=data['entry_price'],
            current_price=data['current_price'],
            unrealized_pnl=data['unrealized_pnl'],
            realized_pnl=data.get('realized_pnl', 0.0),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None,
        )
