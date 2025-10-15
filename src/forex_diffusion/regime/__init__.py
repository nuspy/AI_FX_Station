"""Regime detection module"""
from .hmm_detector import HMMRegimeDetector, RegimeType, RegimeState, detect_regimes
from .adaptive_window import AdaptiveWindowSizer, WindowConfig, MarketConditions, calculate_adaptive_window
from .coherence_validator import (
    CoherenceValidator,
    CoherenceResult,
    CoherenceLevel,
    TimeframeRegime,
    validate_multi_timeframe_regimes,
)

__all__ = [
    "HMMRegimeDetector",
    "RegimeType",
    "RegimeState",
    "detect_regimes",
    "AdaptiveWindowSizer",
    "WindowConfig",
    "MarketConditions",
    "calculate_adaptive_window",
    "CoherenceValidator",
    "CoherenceResult",
    "CoherenceLevel",
    "TimeframeRegime",
    "validate_multi_timeframe_regimes",
]
