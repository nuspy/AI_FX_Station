# src/forex_diffusion/features/indicators.py
"""
DEPRECATED: Use consolidated_indicators.py instead

HIGH-001: This file is kept for backward compatibility only.
All new code should import from consolidated_indicators.

This module redirects to the consolidated implementation.
"""
from __future__ import annotations

import warnings
from .consolidated_indicators import (
    sma as _sma,
    ema as _ema,
    rsi as _rsi,
    macd as _macd,
    atr as _atr,
    bollinger as _bollinger,
    stochastic as _stochastic,
    adx as _adx,
    calculate_indicators,
    get_available_indicators,
    get_backend_info
)

# Re-export with deprecation warnings

def sma(*args, **kwargs):
    warnings.warn(
        "indicators.sma is deprecated, use consolidated_indicators.sma instead",
        DeprecationWarning, stacklevel=2
    )
    return _sma(*args, **kwargs)


def ema(*args, **kwargs):
    warnings.warn(
        "indicators.ema is deprecated, use consolidated_indicators.ema instead",
        DeprecationWarning, stacklevel=2
    )
    return _ema(*args, **kwargs)


def rsi(*args, **kwargs):
    warnings.warn(
        "indicators.rsi is deprecated, use consolidated_indicators.rsi instead",
        DeprecationWarning, stacklevel=2
    )
    return _rsi(*args, **kwargs)


def macd(*args, **kwargs):
    warnings.warn(
        "indicators.macd is deprecated, use consolidated_indicators.macd instead",
        DeprecationWarning, stacklevel=2
    )
    return _macd(*args, **kwargs)


def atr(*args, **kwargs):
    warnings.warn(
        "indicators.atr is deprecated, use consolidated_indicators.atr instead",
        DeprecationWarning, stacklevel=2
    )
    return _atr(*args, **kwargs)


def bollinger(*args, **kwargs):
    warnings.warn(
        "indicators.bollinger is deprecated, use consolidated_indicators.bollinger instead",
        DeprecationWarning, stacklevel=2
    )
    return _bollinger(*args, **kwargs)


def stochastic(*args, **kwargs):
    warnings.warn(
        "indicators.stochastic is deprecated, use consolidated_indicators.stochastic instead",
        DeprecationWarning, stacklevel=2
    )
    return _stochastic(*args, **kwargs)


def adx(*args, **kwargs):
    warnings.warn(
        "indicators.adx is deprecated, use consolidated_indicators.adx instead",
        DeprecationWarning, stacklevel=2
    )
    return _adx(*args, **kwargs)


# Export all for compatibility
__all__ = [
    'sma', 'ema', 'rsi', 'macd', 'atr', 'bollinger', 'stochastic', 'adx',
    'calculate_indicators', 'get_available_indicators', 'get_backend_info'
]
