"""
Regime Detection System for ForexGPT.

Provides unsupervised machine learning based market regime detection
using logical groups of technical indicators for all timeframes.
"""

from .regime_detector import (
    RegimeDetector,
    RegimeState,
    IndicatorGroup,
    TechnicalIndicatorCalculator
)

__all__ = [
    'RegimeDetector',
    'RegimeState',
    'IndicatorGroup',
    'TechnicalIndicatorCalculator'
]