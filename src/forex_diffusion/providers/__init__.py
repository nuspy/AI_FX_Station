"""
Multi-provider data acquisition system.

This package implements a flexible provider architecture supporting multiple
data sources (Tiingo, cTrader, etc.) with unified interfaces.
"""

from .base import BaseProvider, ProviderCapability
from .manager import ProviderManager, get_provider_manager

__all__ = [
    "BaseProvider",
    "ProviderCapability",
    "ProviderManager",
    "get_provider_manager",
]
