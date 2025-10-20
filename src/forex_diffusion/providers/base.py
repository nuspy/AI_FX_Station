"""
Base provider interface and capability definitions.

Defines abstract base class for all market data providers and
enumeration of supported capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import AsyncIterator, Dict, List, Optional, Any
from datetime import datetime

import pandas as pd


class ProviderCapability(Enum):
    """Enumeration of provider capabilities."""

    # Market data capabilities
    QUOTES = auto()          # Real-time bid/ask quotes
    BARS = auto()            # OHLCV candlestick bars
    TICKS = auto()           # Tick-by-tick data
    VOLUMES = auto()         # Volume data (tick + real)
    DOM = auto()             # Depth of Market (Level 2)

    # Supplementary data
    SENTIMENT = auto()       # Market sentiment indicators
    NEWS = auto()            # News feed
    CALENDAR = auto()        # Economic calendar

    # Streaming
    WEBSOCKET = auto()       # WebSocket real-time streaming

    # Historical data
    HISTORICAL_BARS = auto()  # Historical candlestick data
    HISTORICAL_TICKS = auto() # Historical tick data


class ProviderHealth:
    """Health status for a provider."""

    def __init__(self):
        self.is_connected: bool = False
        self.latency_ms: Optional[float] = None
        self.data_rate_msg_per_sec: float = 0.0
        self.error_rate_pct: float = 0.0
        self.uptime_seconds: float = 0.0
        self.last_message_ts: Optional[datetime] = None
        self.errors: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_connected": self.is_connected,
            "latency_ms": self.latency_ms,
            "data_rate_msg_per_sec": self.data_rate_msg_per_sec,
            "error_rate_pct": self.error_rate_pct,
            "uptime_seconds": self.uptime_seconds,
            "last_message_ts": self.last_message_ts.isoformat() if self.last_message_ts else None,
            "recent_errors": self.errors[-5:],  # Last 5 errors
        }


class BaseProvider(ABC):
    """
    Abstract base class for all market data providers.

    Each provider must:
    - Declare supported capabilities
    - Implement async methods for data retrieval
    - Handle connection lifecycle (connect/disconnect)
    - Provide health monitoring
    - Return None for unsupported features
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize provider.

        Args:
            name: Provider name (e.g., "tiingo", "ctrader")
            config: Provider-specific configuration
        """
        self.name = name
        self.config = config or {}
        self.health = ProviderHealth()
        self._start_time: Optional[datetime] = None

    @property
    @abstractmethod
    def capabilities(self) -> List[ProviderCapability]:
        """Return list of supported capabilities."""
        pass

    def supports(self, capability: ProviderCapability) -> bool:
        """Check if provider supports a given capability."""
        return capability in self.capabilities

    # Connection lifecycle

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to provider.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from provider and cleanup resources."""
        pass

    async def reconnect(self) -> bool:
        """Reconnect to provider (disconnect + connect)."""
        await self.disconnect()
        return await self.connect()

    # Health monitoring

    def get_health(self) -> ProviderHealth:
        """Get current health status."""
        if self._start_time and self.health.is_connected:
            self.health.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        return self.health

    def is_healthy(self) -> bool:
        """Check if provider is healthy (connected and no critical errors)."""
        return self.health.is_connected and self.health.error_rate_pct < 50.0

    # Market data methods - all return None if not supported

    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price quote.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")

        Returns:
            Dict with keys: symbol, ts_utc, bid, ask, price (mid)
            None if not supported
        """
        if not self.supports(ProviderCapability.QUOTES):
            return None
        return await self._get_current_price_impl(symbol)

    async def _get_current_price_impl(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Implementation of get_current_price - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement get_current_price")

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start_ts_ms: int,
        end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            start_ts_ms: Start timestamp (milliseconds UTC)
            end_ts_ms: End timestamp (milliseconds UTC)

        Returns:
            DataFrame with columns: ts_utc, open, high, low, close, volume
            None if not supported
        """
        if not self.supports(ProviderCapability.HISTORICAL_BARS):
            return None
        return await self._get_historical_bars_impl(symbol, timeframe, start_ts_ms, end_ts_ms)

    async def _get_historical_bars_impl(
        self, symbol: str, timeframe: str, start_ts_ms: int, end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """Implementation of get_historical_bars - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement get_historical_bars")

    async def get_historical_ticks(
        self,
        symbol: str,
        start_ts_ms: int,
        end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """
        Get historical tick data.

        Args:
            symbol: Trading symbol
            start_ts_ms: Start timestamp (milliseconds UTC)
            end_ts_ms: End timestamp (milliseconds UTC)

        Returns:
            DataFrame with columns: ts_utc, price, bid, ask, volume
            None if not supported
        """
        if not self.supports(ProviderCapability.HISTORICAL_TICKS):
            return None
        return await self._get_historical_ticks_impl(symbol, start_ts_ms, end_ts_ms)

    async def _get_historical_ticks_impl(
        self, symbol: str, start_ts_ms: int, end_ts_ms: int
    ) -> Optional[pd.DataFrame]:
        """Implementation of get_historical_ticks - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement get_historical_ticks")

    async def get_market_depth(self, symbol: str, levels: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get market depth (DOM/Level 2 data).

        Args:
            symbol: Trading symbol
            levels: Number of price levels to retrieve

        Returns:
            Dict with keys: symbol, ts_utc, bids [(price, volume), ...], asks [(price, volume), ...]
            None if not supported
        """
        if not self.supports(ProviderCapability.DOM):
            return None
        return await self._get_market_depth_impl(symbol, levels)

    async def _get_market_depth_impl(self, symbol: str, levels: int) -> Optional[Dict[str, Any]]:
        """Implementation of get_market_depth - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement get_market_depth")

    async def get_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market sentiment data.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with keys: symbol, ts_utc, long_pct, short_pct, total_traders
            None if not supported
        """
        if not self.supports(ProviderCapability.SENTIMENT):
            return None
        return await self._get_sentiment_impl(symbol)

    async def _get_sentiment_impl(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Implementation of get_sentiment - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement get_sentiment")

    async def get_news(self, currency: Optional[str] = None, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Get news feed.

        Args:
            currency: Filter by currency (e.g., "USD", "EUR")
            limit: Maximum number of news items

        Returns:
            List of dicts with keys: ts_utc, title, content, impact, currency
            None if not supported
        """
        if not self.supports(ProviderCapability.NEWS):
            return None
        return await self._get_news_impl(currency, limit)

    async def _get_news_impl(self, currency: Optional[str], limit: int) -> Optional[List[Dict[str, Any]]]:
        """Implementation of get_news - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement get_news")

    async def get_economic_calendar(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        currency: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get economic calendar events.

        Args:
            start_date: Start date for events
            end_date: End date for events
            currency: Filter by currency

        Returns:
            List of dicts with keys: event_id, ts_utc, event_name, currency,
            forecast, actual, previous, impact
            None if not supported
        """
        if not self.supports(ProviderCapability.CALENDAR):
            return None
        return await self._get_economic_calendar_impl(start_date, end_date, currency)

    async def _get_economic_calendar_impl(
        self, start_date: Optional[datetime], end_date: Optional[datetime], currency: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Implementation of get_economic_calendar - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement get_economic_calendar")

    # WebSocket streaming

    async def stream_quotes(self, symbols: List[str]) -> Optional[AsyncIterator[Dict[str, Any]]]:
        """
        Stream real-time quotes via WebSocket.

        Args:
            symbols: List of symbols to subscribe

        Yields:
            Dicts with keys: symbol, ts_utc, bid, ask, price
            None if not supported
        """
        if not self.supports(ProviderCapability.WEBSOCKET):
            return None
        return self._stream_quotes_impl(symbols)

    async def _stream_quotes_impl(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Implementation of stream_quotes - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement stream_quotes")
        # Suppress warning for async generator
        yield {}  # pragma: no cover

    async def stream_market_depth(self, symbols: List[str]) -> Optional[AsyncIterator[Dict[str, Any]]]:
        """
        Stream real-time market depth updates.

        Args:
            symbols: List of symbols to subscribe

        Yields:
            Dicts with market depth data
            None if not supported
        """
        if not self.supports(ProviderCapability.DOM):
            return None
        return self._stream_market_depth_impl(symbols)

    async def _stream_market_depth_impl(self, symbols: List[str]) -> AsyncIterator[Dict[str, Any]]:
        """Implementation of stream_market_depth - override in subclass."""
        raise NotImplementedError(f"{self.name} does not implement stream_market_depth")
        # Suppress warning for async generator
        yield {}  # pragma: no cover
