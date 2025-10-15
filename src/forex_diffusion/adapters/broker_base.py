"""
Broker Adapter Base Classes

Unified interface for multiple broker integrations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELED = "canceled"


@dataclass
class Quote:
    """Market quote."""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None


@dataclass
class Order:
    """Order details."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_fill_price: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class Position:
    """Position details."""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class AccountInfo:
    """Account information."""
    account_id: str
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    positions: List[Position]


class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters.

    All broker implementations must inherit from this class.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to broker.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get current quote for symbol.

        Args:
            symbol: Symbol to query

        Returns:
            Quote or None if unavailable
        """
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to Quote
        """
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Place an order.

        Args:
            symbol: Symbol to trade
            side: 'buy' or 'sell'
            size: Order size
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Order object or None if failed
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account information.

        Returns:
            AccountInfo or None if unavailable
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """
        Close position for symbol.

        Args:
            symbol: Symbol to close

        Returns:
            True if successful
        """
        pass

    def subscribe_quotes(self, symbols: List[str], callback: callable) -> bool:
        """
        Subscribe to real-time quotes (optional).

        Args:
            symbols: Symbols to subscribe
            callback: Function to call with new quotes

        Returns:
            True if successful
        """
        # Default implementation does nothing
        return False

    def unsubscribe_quotes(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from quotes (optional).

        Args:
            symbols: Symbols to unsubscribe

        Returns:
            True if successful
        """
        return False
