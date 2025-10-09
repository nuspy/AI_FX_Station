"""
Provider Manager - Factory pattern for provider instantiation and lifecycle management.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from loguru import logger

from .base import BaseProvider, ProviderCapability
from .tiingo_provider import TiingoProvider
from .ctrader_provider import CTraderProvider


class ProviderManager:
    """
    Factory and lifecycle manager for market data providers.

    Responsibilities:
    - Instantiate providers by name
    - Manage provider configurations
    - Handle failover between providers
    - Track provider health
    """

    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._primary_provider: Optional[str] = None
        self._secondary_provider: Optional[str] = None

        # Provider registry
        self._provider_classes = {
            "tiingo": TiingoProvider,
            "ctrader": CTraderProvider,
        }

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self._provider_classes.keys())

    def create_provider(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseProvider:
        """
        Create a provider instance.

        Args:
            name: Provider name ("tiingo", "ctrader", etc.)
            config: Provider-specific configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider name not recognized
        """
        name_lower = name.lower()

        if name_lower not in self._provider_classes:
            raise ValueError(
                f"Unknown provider: {name}. "
                f"Available: {', '.join(self.get_available_providers())}"
            )

        provider_class = self._provider_classes[name_lower]
        provider = provider_class(config=config)

        # Store in registry
        self._providers[name_lower] = provider

        logger.info(f"Created provider: {name_lower}")
        return provider

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """
        Get existing provider instance.

        Args:
            name: Provider name

        Returns:
            Provider instance if exists, None otherwise
        """
        return self._providers.get(name.lower())

    def remove_provider(self, name: str) -> None:
        """
        Remove and disconnect a provider.

        Args:
            name: Provider name
        """
        name_lower = name.lower()

        if name_lower in self._providers:
            # Async disconnect would need event loop
            # For now, just remove from registry
            del self._providers[name_lower]
            logger.info(f"Removed provider: {name_lower}")

    def set_primary_provider(self, name: str) -> None:
        """Set primary provider for data acquisition."""
        if name.lower() not in self._providers:
            raise ValueError(f"Provider not found: {name}")

        self._primary_provider = name.lower()
        logger.info(f"Primary provider set to: {name}")

    def set_secondary_provider(self, name: str) -> None:
        """Set secondary provider for failover."""
        if name.lower() not in self._providers:
            raise ValueError(f"Provider not found: {name}")

        self._secondary_provider = name.lower()
        logger.info(f"Secondary provider set to: {name}")

    def get_primary_provider(self) -> Optional[BaseProvider]:
        """Get primary provider instance."""
        if self._primary_provider:
            return self._providers.get(self._primary_provider)
        return None

    def get_secondary_provider(self) -> Optional[BaseProvider]:
        """Get secondary provider instance."""
        if self._secondary_provider:
            return self._providers.get(self._secondary_provider)
        return None

    def get_providers_by_capability(self, capability: ProviderCapability) -> List[BaseProvider]:
        """
        Get all providers that support a given capability.

        Args:
            capability: Required capability

        Returns:
            List of provider instances
        """
        return [
            provider for provider in self._providers.values()
            if provider.supports(capability)
        ]

    def get_all_providers(self) -> List[BaseProvider]:
        """Get all registered provider instances."""
        return list(self._providers.values())

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health summary for all providers.

        Returns:
            Dict mapping provider names to health status
        """
        summary = {}

        for name, provider in self._providers.items():
            health = provider.get_health()
            summary[name] = health.to_dict()

        return summary

    async def failover_to_secondary(self) -> bool:
        """
        Failover from primary to secondary provider.

        Returns:
            True if failover successful, False otherwise
        """
        if not self._secondary_provider:
            logger.warning("No secondary provider configured for failover")
            return False

        secondary = self.get_secondary_provider()
        if not secondary:
            logger.error("Secondary provider not found")
            return False

        try:
            # Connect to secondary
            success = await secondary.connect()

            if success:
                # Swap primary and secondary
                old_primary = self._primary_provider
                self._primary_provider = self._secondary_provider
                self._secondary_provider = old_primary

                logger.info(
                    f"Failover successful: primary={self._primary_provider}, "
                    f"secondary={self._secondary_provider}"
                )
                return True

            logger.error("Secondary provider connection failed")
            return False

        except Exception as e:
            logger.error(f"Failover failed: {e}")
            return False

    async def get_realtime_quotes_with_fallback(self, symbols: List[str]) -> Optional[Any]:
        """
        Get realtime quotes with WebSocket first, REST fallback strategy.

        Strategy:
        1. Try WebSocket streaming (faster, lower latency)
        2. If fails, fallback to REST polling

        Args:
            symbols: List of symbols to get quotes for

        Returns:
            AsyncIterator for quotes or None if both fail
        """
        primary = self.get_primary_provider()
        if not primary:
            logger.error("No primary provider configured")
            return None

        # Strategy 1: Try WebSocket streaming
        try:
            logger.info(f"Attempting WebSocket streaming for {symbols}...")
            if primary.supports(ProviderCapability.WEBSOCKET):
                stream = await primary.stream_quotes(symbols)
                if stream:
                    logger.info("Successfully started WebSocket streaming")
                    return stream
        except Exception as e:
            logger.warning(f"WebSocket streaming failed: {e}, trying REST fallback...")

        # Strategy 2: Fallback to REST polling
        try:
            logger.info(f"Attempting REST polling for {symbols}...")
            if primary.supports(ProviderCapability.QUOTES):
                # Note: REST doesn't return a stream, but single quotes
                # Caller should poll this method periodically
                quotes = await primary.get_quotes(symbols)
                logger.info("Successfully got quotes via REST")
                return quotes
        except Exception as e:
            logger.error(f"REST polling also failed: {e}")

        return None

    async def disconnect_all(self) -> None:
        """Disconnect all providers."""
        for name, provider in self._providers.items():
            try:
                await provider.disconnect()
                logger.info(f"Disconnected provider: {name}")
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")


# Global instance
_global_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Get global ProviderManager instance."""
    global _global_manager

    if _global_manager is None:
        _global_manager = ProviderManager()

    return _global_manager
