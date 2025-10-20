"""
Gradient Boosting Models Integration

Adds LightGBM and XGBoost for improved non-linear pattern detection.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from loguru import logger

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")


def get_lightgbm_regressor(
    learning_rate: float = 0.05,
    max_depth: int = 5,
    n_estimators: int = 500,
    subsample: float = 0.8,
    **kwargs
) -> Optional[LGBMRegressor]:
    """
    Get configured LightGBM regressor.

    Args:
        learning_rate: Learning rate (default: 0.05, conservative)
        max_depth: Maximum tree depth (default: 5)
        n_estimators: Number of boosting rounds (default: 500)
        subsample: Row subsampling ratio (default: 0.8)
        **kwargs: Additional LightGBM parameters

    Returns:
        Configured LGBMRegressor or None if unavailable
    """
    if not LIGHTGBM_AVAILABLE:
        return None

    default_params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'subsample': subsample,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    default_params.update(kwargs)

    return LGBMRegressor(**default_params)


def get_xgboost_regressor(
    learning_rate: float = 0.05,
    max_depth: int = 5,
    n_estimators: int = 500,
    subsample: float = 0.8,
    **kwargs
) -> Optional[XGBRegressor]:
    """
    Get configured XGBoost regressor.

    Args:
        learning_rate: Learning rate (default: 0.05)
        max_depth: Maximum tree depth (default: 5)
        n_estimators: Number of boosting rounds (default: 500)
        subsample: Row subsampling ratio (default: 0.8)
        **kwargs: Additional XGBoost parameters

    Returns:
        Configured XGBRegressor or None if unavailable
    """
    if not XGBOOST_AVAILABLE:
        return None

    default_params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'subsample': subsample,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',  # Fast histogram-based
        'verbosity': 0
    }

    default_params.update(kwargs)

    return XGBRegressor(**default_params)


def get_available_models() -> Dict[str, Any]:
    """
    Get dictionary of available gradient boosting models.

    Returns:
        Dict mapping model name to factory function
    """
    models = {}

    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = get_lightgbm_regressor
        logger.info("✓ LightGBM available")

    if XGBOOST_AVAILABLE:
        models['xgboost'] = get_xgboost_regressor
        logger.info("✓ XGBoost available")

    return models
