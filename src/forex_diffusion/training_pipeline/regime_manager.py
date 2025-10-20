"""
Regime Manager - Stub for E2E testing.

This is a placeholder module to allow E2E tests to run.
Full implementation will be added in future sprint.
"""

from __future__ import annotations
from loguru import logger


class RegimeManager:
    """Stub class for regime management."""

    def __init__(self, *args, **kwargs):
        logger.warning("Using stub RegimeManager - full implementation pending")
        pass

    def detect_regime(self, *args, **kwargs) -> str:
        """Stub method."""
        return "all"

    def get_best_model_for_regime(self, regime: str) -> Optional[int]:
        """Stub method."""
        return None
