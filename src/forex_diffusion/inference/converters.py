"""Shared utilities to convert Lightning forecasts into returns and price paths."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def denormalize_lightning_outputs(
    raw_predictions: Dict[int, float],
    price_mu: Optional[float],
    price_sigma: Optional[float]
) -> Dict[int, float]:
    """
    Convert raw Lightning outputs (z-scores) into absolute prices.

    Args:
        raw_predictions: mapping horizon -> raw model output
        price_mu: mean of close channel from training normalization
        price_sigma: std of close channel from training normalization

    Returns:
        Dict[horizon, price]
    """

    if price_mu is None or price_sigma is None:
        return dict(raw_predictions)

    return {
        horizon: float(value) * price_sigma + price_mu
        for horizon, value in raw_predictions.items()
    }


def returns_from_prices(
    price_predictions: Dict[int, float],
    last_close: float
) -> Dict[int, float]:
    """Convert absolute price forecast into relative returns vs last_close."""

    if last_close is None or last_close == 0.0 or np.isnan(last_close):
        return dict(price_predictions)

    return {
        horizon: (float(price) - last_close) / last_close
        for horizon, price in price_predictions.items()
    }


def build_price_path(
    returns: Iterable[float],
    anchor_price: float,
    horizon_count: int
) -> np.ndarray:
    """Create price trajectory from returns and anchor price."""

    returns_array = np.asarray(list(returns), dtype=float)

    if horizon_count and returns_array.shape[0] != horizon_count:
        if returns_array.shape[0] < horizon_count:
            returns_array = np.pad(returns_array, (0, horizon_count - returns_array.shape[0]), mode="edge")
        else:
            returns_array = returns_array[:horizon_count]

    # Calculate cumulative product of (1 + returns) and multiply by anchor
    cumulative_returns = np.cumprod(1.0 + returns_array)
    prices = anchor_price * cumulative_returns
    return prices
