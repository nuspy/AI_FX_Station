"""
Base broker interface and common types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order side (direction)"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(Enum):
    """Position side"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Order:
    """Order representation"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'take_profit': self.take_profit,
            'stop_loss': self.stop_loss,
            'order_id': self.order_id,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'comment': self.comment,
        }


@dataclass
class Position:
    """Position representation"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class AccountInfo:
    """Account information"""
    account_id: str
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    currency: str
    leverage: int
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'account_id': self.account_id,
            'balance': self.balance,
            'equity': self.equity,
            'margin_used': self.margin_used,
            'margin_available': self.margin_available,
            'currency': self.currency,
            'leverage': self.leverage,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
        }


class BrokerBase(ABC):
    """
    Abstract base class for broker integrations.

    All broker implementations must inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker API.

        Returns:
            bool: True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from broker API.

        Returns:
            bool: True if disconnection successful
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to broker API.

        Returns:
            bool: True if connected
        """
        pass

    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account information.

        Returns:
            AccountInfo or None if error
        """
        pass

    @abstractmethod
    def place_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Place an order.

        Args:
            order: Order object

        Returns:
            Tuple of (success, order_id, error_message)
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Tuple of (success, error_message)
        """
        pass

    @abstractmethod
    def modify_order(self, order_id: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            **kwargs: Fields to modify (price, stop_loss, take_profit, etc.)

        Returns:
            Tuple of (success, error_message)
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        pass

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of Order objects
        """
        pass

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all open positions.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str, quantity: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Close a position (or partial position).

        Args:
            symbol: Symbol to close
            quantity: Quantity to close (None = close all)

        Returns:
            Tuple of (success, error_message)
        """
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol/instrument information.

        Args:
            symbol: Symbol name

        Returns:
            Dictionary with symbol info (pip_size, min_volume, max_volume, etc.)
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.

        Returns:
            List of symbol names
        """
        pass

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol name to broker-specific format.

        Args:
            symbol: Input symbol (e.g., "EUR/USD", "EURUSD")

        Returns:
            Normalized symbol name for this broker
        """
        # Default implementation - override in subclasses
        return symbol.replace('/', '').upper()

    def denormalize_symbol(self, symbol: str) -> str:
        """
        Convert broker symbol to standard format.

        Args:
            symbol: Broker-specific symbol

        Returns:
            Standard symbol format (e.g., "EUR/USD")
        """
        # Default implementation - override in subclasses
        if len(symbol) == 6:
            return f"{symbol[:3]}/{symbol[3:]}"
        return symbol
