"""
Multi-provider data acquisition system.

This package implements a flexible provider architecture supporting multiple
data sources (Tiingo, cTrader, etc.) with unified interfaces.
"""

from .base import BaseProvider, ProviderCapability
from .manager import ProviderManager, get_provider_manager
from .tiingo_provider import TiingoProvider
from .ctrader_provider import CTraderProvider

__all__ = [
    "BaseProvider",
    "ProviderCapability",
    "ProviderManager",
    "get_provider_manager",
    "TiingoProvider",
    "CTraderProvider",
]
