"""
Centralized feature engineering package.

This package consolidates duplicate feature engineering code from training modules.
"""

from .feature_utils import (
    ensure_dt_index,
    timeframe_to_timedelta,
    coerce_indicator_tfs,
    resample_ohlc,
    validate_ohlc_dataframe,
    align_to_base_timeframe,
)

from .feature_engineering import (
    relative_ohlc,
    temporal_features,
    realized_volatility_feature,
    returns_features,
    price_momentum_features,
    volume_features,
    ohlc_range_features,
    standardize_features_no_leakage,
)

__all__ = [
    # Utility functions
    "ensure_dt_index",
    "timeframe_to_timedelta",
    "coerce_indicator_tfs",
    "resample_ohlc",
    "validate_ohlc_dataframe",
    "align_to_base_timeframe",
    # Feature engineering functions
    "relative_ohlc",
    "temporal_features",
    "realized_volatility_feature",
    "returns_features",
    "price_momentum_features",
    "volume_features",
    "ohlc_range_features",
    "standardize_features_no_leakage",
]
