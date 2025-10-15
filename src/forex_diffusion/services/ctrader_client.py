"""
CTrader Client for MarketDataService

Synchronous wrapper around CTraderProvider for use in MarketDataService.
Similar to TiingoClient but uses cTrader Open API for historical data.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
from loguru import logger


class CTraderClient:
    """
    Synchronous cTrader client for historical data fetching.

    Wraps CTraderProvider with sync interface for MarketDataService compatibility.
    Uses singleton pattern to ensure only one connection instance exists.
    """

    _instance = None
    _lock = None

    def __new__(cls, config: Optional[dict] = None):
        """
        Singleton pattern: return existing instance if available.
        """
        if cls._instance is None:
            cls._instance = super(CTraderClient, cls).__new__(cls)
            cls._instance._singleton_initialized = False
        return cls._instance

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize cTrader client (only runs once due to singleton).

        Args:
            config: Optional config dict with:
                - client_id: cTrader application client ID
                - client_secret: cTrader application client secret
                - access_token: cTrader access token
                - account_id: cTrader account ID
                - environment: 'demo' or 'live' (default: 'demo')
        """
        # Only initialize once
        if self._singleton_initialized:
            # Update config if provided
            if config:
                self.config.update(config)
                logger.debug("Updated CTraderClient config (singleton)")
            return

        self.config = config or {}
        self._provider = None
        self._initialized = False
        self._event_loop = None
        self._singleton_initialized = True
        logger.debug("CTraderClient singleton instance created")

    def _ensure_initialized(self):
        """Ensure provider is initialized (lazy initialization)."""
        if self._initialized:
            return

        try:
            from ..providers.ctrader_provider import CTraderProvider

            # Get configuration from settings if not provided
            if not self.config:
                try:
                    from ..utils.user_settings import get_setting
                    self.config = {
                        'client_id': get_setting('ctrader_client_id', ''),
                        'client_secret': get_setting('ctrader_client_secret', ''),
                        'access_token': get_setting('ctrader_access_token', ''),
                        'account_id': get_setting('ctrader_account_id', None),
                        'environment': get_setting('ctrader_environment', 'demo'),
                    }
                except Exception as e:
                    logger.warning(f"Could not load cTrader settings: {e}")
                    raise ValueError("cTrader credentials not configured in settings")

            # Create provider
            self._provider = CTraderProvider(config=self.config)

            # Connect provider (async)
            success = self._run_async(self._provider.connect())
            if not success:
                raise RuntimeError("Failed to connect to cTrader API")

            self._initialized = True
            logger.info("CTraderClient initialized successfully")

        except ImportError as e:
            logger.error(f"ctrader-open-api not installed: {e}")
            raise ImportError(
                "ctrader-open-api package not installed. "
                "Install with: pip install ctrader-open-api twisted protobuf"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CTraderClient: {e}")
            raise

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (in async context), create task
                return asyncio.ensure_future(coro)
        except RuntimeError:
            pass

        # Reuse existing event loop if available (for provider connection lifetime)
        if self._event_loop and not self._event_loop.is_closed():
            asyncio.set_event_loop(self._event_loop)
            return self._event_loop.run_until_complete(coro)

        # Create new event loop for sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Store the loop to keep it alive for provider callbacks
        self._event_loop = loop

        try:
            return loop.run_until_complete(coro)
        except Exception as e:
            # Don't close loop on error - provider may still need it
            logger.error(f"Error in _run_async: {e}")
            raise
        # DON'T close the loop here - it needs to stay alive for Twisted callbacks

    def get_candles(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        resample_freq: str = "1min",
        fmt: str = "json"
    ) -> pd.DataFrame:
        """
        Query cTrader for historical candles (trendbars).

        Args:
            symbol: Symbol like "EUR/USD"
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            resample_freq: Timeframe like '1min', '5min', '15min', '1hour', '1day'
            fmt: Format (json or other) - currently only json supported

        Returns:
            DataFrame with columns: ts_utc, open, high, low, close, volume
        """
        self._ensure_initialized()

        try:
            # Convert dates to timestamps (milliseconds)
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            # Convert resample_freq to cTrader timeframe format
            # Examples: '1min' -> '1m', '5min' -> '5m', '1hour' -> '1h', '1day' -> '1d'
            timeframe = resample_freq.replace('min', 'm').replace('hour', 'h').replace('day', 'd')

            # Get historical bars from provider (async)
            logger.info(f"Fetching cTrader candles for {symbol} from {start_date} to {end_date} ({timeframe})")
            df = self._run_async(
                self._provider.get_historical_bars(symbol, timeframe, start_ms, end_ms)
            )

            if df is None or df.empty:
                logger.warning(f"cTrader returned empty data for {symbol} {start_date}-{end_date}")
                return pd.DataFrame()

            # Ensure required columns
            required_cols = ["ts_utc", "open", "high", "low", "close"]
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column {col} in cTrader response")
                    return pd.DataFrame()

            # Add volume if not present
            if "volume" not in df.columns:
                if "tick_volume" in df.columns:
                    df["volume"] = df["tick_volume"]
                else:
                    df["volume"] = 0

            # Select and order columns
            cols = ["ts_utc", "open", "high", "low", "close", "volume"]
            result = df[cols].copy()

            logger.info(f"Retrieved {len(result)} candles from cTrader for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to get candles from cTrader: {e}")
            return pd.DataFrame()

    def get_ticks(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fmt: str = "json"
    ) -> pd.DataFrame:
        """
        Query cTrader for tick data (both BID and ASK).

        ⚠️ DEPRECATED: This method uses REST API for historical tick data.
        DO NOT USE for historical data - use get_candles() instead.
        For real-time data, use WebSocket stream_quotes() method in CTraderProvider.

        This method should ONLY be used if explicitly required for legacy compatibility.

        Args:
            symbol: Symbol like "EUR/USD"
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            fmt: Format (json or other) - currently only json supported

        Returns:
            DataFrame with columns: ts_utc, bid, ask, price
            - bid: BID price for display in GUI
            - ask: ASK price for display in GUI
            - price: Mid price for sampling/analysis
        """
        logger.warning(
            f"[{self.__class__.__name__}] get_ticks() called for historical data - "
            "this is DEPRECATED. Use get_candles() for historical data or WebSocket for realtime."
        )
        self._ensure_initialized()

        try:
            # Convert dates to timestamps
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            # Get historical ticks from provider (async)
            logger.info(f"Fetching cTrader ticks for {symbol} from {start_date} to {end_date}")
            df = self._run_async(
                self._provider.get_historical_ticks(symbol, start_ms, end_ms)
            )

            if df is None or df.empty:
                logger.warning(f"cTrader returned empty tick data for {symbol} {start_date}-{end_date}")
                return pd.DataFrame()

            logger.info(f"Retrieved {len(df)} ticks from cTrader for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to get ticks from cTrader: {e}")
            return pd.DataFrame()

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful when credentials change)."""
        if cls._instance is not None:
            # Disconnect existing provider
            try:
                if cls._instance._provider and cls._instance._initialized:
                    cls._instance._run_async(cls._instance._provider.disconnect())
                    logger.info("Disconnected existing CTraderClient instance")
            except Exception as e:
                logger.warning(f"Error disconnecting CTraderClient during reset: {e}")

            # Close the event loop
            try:
                if cls._instance._event_loop and not cls._instance._event_loop.is_closed():
                    cls._instance._event_loop.close()
                    logger.debug("Closed event loop for CTraderClient")
            except Exception as e:
                logger.warning(f"Error closing event loop during reset: {e}")

            cls._instance = None
            logger.info("CTraderClient singleton instance reset")

    def __del__(self):
        """Cleanup on deletion."""
        if self._provider and self._initialized:
            try:
                self._run_async(self._provider.disconnect())
            except Exception as e:
                logger.debug(f"Error disconnecting CTraderClient: {e}")

        # Close event loop if it exists
        try:
            if self._event_loop and not self._event_loop.is_closed():
                self._event_loop.close()
        except Exception as e:
            logger.debug(f"Error closing event loop in __del__: {e}")
