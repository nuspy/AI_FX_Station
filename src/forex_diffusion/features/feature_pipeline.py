"""
Unified Feature Pipeline - Single Entry Point

This module serves as the main entry point for feature engineering,
consolidating functionality from multiple pipeline files:
- pipeline.py (legacy, to be deprecated)
- unified_pipeline.py (training/inference consistency)
- feature_engineering.py (consolidated feature functions)
- consolidated_indicators.py (indicator computation)

Usage:
    from forex_diffusion.features.feature_pipeline import (
        compute_features,
        compute_indicators,
        FeatureConfig
    )

Architecture:
    feature_pipeline.py (THIS FILE - main API)
    ├── feature_engineering.py (core feature functions)
    ├── consolidated_indicators.py (indicator computation)
    ├── unified_pipeline.py (training pipeline)
    └── pipeline.py (DEPRECATED - legacy support only)

Status: ALPHA - Under active development
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger

# Import from consolidated modules
from .feature_engineering import (
    relative_ohlc,
    temporal_features,
    realized_volatility_feature,
    rolling_features,
    Standardizer
)

from .consolidated_indicators import (
    atr,
    rsi,
    macd,
    bollinger,
    sma,
    ema,
    stochastic,
    adx,
    calculate_indicators,
    get_available_indicators
)

from .unified_pipeline import (
    FeatureConfig,
    compute_features as unified_compute_features,
    compute_inference_features
)

# Re-export for convenience
__all__ = [
    # Configuration
    'FeatureConfig',
    
    # High-level APIs
    'compute_features',
    'compute_inference_features',
    'get_available_indicators',
    
    # Core feature functions
    'relative_ohlc',
    'temporal_features',
    'realized_volatility_feature',
    'rolling_features',
    
    # Indicators
    'atr',
    'rsi',
    'macd',
    'bollinger',
    'sma',
    'ema',
    'stochastic',
    'adx',
    'calculate_indicators',
    
    # Standardization
    'Standardizer',
]


def compute_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    standardize: bool = True
) -> Tuple[pd.DataFrame, Optional[Standardizer]]:
    """
    Main entry point for feature computation.
    
    This function computes a complete feature set including:
    - Relative OHLC normalization
    - Temporal features (cyclical encoding)
    - Technical indicators (ATR, RSI, MACD, Bollinger, etc.)
    - Realized volatility
    - Standardization (optional)
    
    Args:
        df: Input DataFrame with OHLC data and 'ts_utc' column
        config: Feature configuration (uses defaults if None)
        standardize: Whether to apply standardization
        
    Returns:
        Tuple of (features_df, standardizer)
        
    Example:
        >>> config = FeatureConfig()
        >>> features, scaler = compute_features(df, config)
        >>> print(features.shape)
        (1000, 45)  # 45 features
    """
    if config is None:
        config = FeatureConfig()
    
    # Delegate to unified pipeline (maintains training/inference consistency)
    return unified_compute_features(df, config, standardize=standardize)


def compute_minimal_features(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute minimal feature set for quick prototyping.
    
    This is a lightweight alternative to compute_features() that:
    - Skips standardization
    - Only computes specified indicators
    - Returns DataFrame directly (no scaler)
    
    Args:
        df: Input DataFrame with OHLC data and 'ts_utc' column
        indicators: List of indicator names (default: ['atr', 'rsi'])
        
    Returns:
        DataFrame with minimal feature set
        
    Example:
        >>> features = compute_minimal_features(df, indicators=['atr', 'rsi', 'macd'])
    """
    if indicators is None:
        indicators = ['atr', 'rsi']
    
    features = pd.DataFrame(index=df.index)
    
    # Add relative OHLC
    features = pd.concat([features, relative_ohlc(df)], axis=1)
    
    # Add temporal features
    features = pd.concat([features, temporal_features(df, use_cyclical=True)], axis=1)
    
    # Add requested indicators
    ind_results = calculate_indicators(df, indicators=indicators)
    for col_name, series in ind_results.items():
        features[col_name] = series
    
    return features


def get_feature_names(config: Optional[FeatureConfig] = None) -> List[str]:
    """
    Get list of feature names that will be computed.
    
    Args:
        config: Feature configuration (uses defaults if None)
        
    Returns:
        List of feature column names
        
    Example:
        >>> config = FeatureConfig()
        >>> names = get_feature_names(config)
        >>> print(len(names))
        45
    """
    if config is None:
        config = FeatureConfig()
    
    # This is a placeholder - actual implementation would parse config
    # and return exact list of features
    feature_names = []
    
    # Base features
    if config.config["base_features"]["relative_ohlc"]:
        feature_names.extend(['r_open', 'r_high', 'r_low', 'r_close'])
    
    if config.config["base_features"]["time_features"]:
        feature_names.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])
    
    # Indicators
    for indicator, ind_config in config.config["indicators"].items():
        if ind_config.get("enabled", False):
            if indicator == "atr":
                feature_names.append(f"atr_{ind_config['n']}")
            elif indicator == "rsi":
                feature_names.append(f"rsi_{ind_config['n']}")
            elif indicator == "macd":
                feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
            elif indicator == "bollinger":
                n = ind_config['n']
                feature_names.extend([
                    f'bb_upper_{n}', f'bb_lower_{n}', 
                    f'bb_width_{n}', f'bb_pctb_{n}'
                ])
    
    return feature_names


def validate_input_data(df: pd.DataFrame) -> None:
    """
    Validate input DataFrame has required columns.
    
    Args:
        df: Input DataFrame
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ['open', 'high', 'low', 'close', 'ts_utc']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if len(df) < 100:
        logger.warning(f"Input data has only {len(df)} rows, may be insufficient for some indicators")


# Convenience aliases for backward compatibility
from .pipeline import (
    log_returns,
    rolling_std,
    hurst_feature,
    donchian
)

__all__.extend(['log_returns', 'rolling_std', 'hurst_feature', 'donchian'])
