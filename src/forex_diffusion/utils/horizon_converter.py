"""
Horizon conversion utilities for consistent training-inference horizon handling.

Handles conversion between:
- Training format: horizon in bars (e.g., 5)
- Inference format: time labels (e.g., ["1m", "5m", "15m"])
"""
from __future__ import annotations

from typing import List, Union, Tuple
import pandas as pd
from loguru import logger


def timeframe_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    tf = str(tf).strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    elif tf.endswith("h"):
        return int(tf[:-1]) * 60
    elif tf.endswith("d"):
        return int(tf[:-1]) * 24 * 60
    elif tf.endswith("w"):
        return int(tf[:-1]) * 7 * 24 * 60
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")


def minutes_to_timeframe(minutes: int) -> str:
    """Convert minutes to timeframe string."""
    if minutes < 60:
        return f"{minutes}m"
    elif minutes < 24 * 60:
        hours = minutes // 60
        if minutes % 60 == 0:
            return f"{hours}h"
        else:
            return f"{minutes}m"
    else:
        days = minutes // (24 * 60)
        if minutes % (24 * 60) == 0:
            return f"{days}d"
        else:
            return f"{minutes}m"


def horizon_bars_to_time_labels(
    horizon_bars: int,
    base_timeframe: str,
    target_timeframes: List[str] = None
) -> List[str]:
    """
    Convert horizon in bars to time labels.

    Args:
        horizon_bars: Number of bars to predict ahead
        base_timeframe: Base timeframe of the model (e.g., "1m")
        target_timeframes: Desired output timeframes, if None uses standard progression

    Returns:
        List of time labels corresponding to the horizon
    """
    if target_timeframes is None:
        # Standard progression based on base timeframe
        base_minutes = timeframe_to_minutes(base_timeframe)

        # Create progression: 1x, 5x, 15x, 60x of base timeframe
        multipliers = [1, 5, 15, 60] if base_minutes == 1 else [1, 3, 5, 15]
        target_timeframes = []

        for mult in multipliers:
            target_minutes = base_minutes * mult
            tf_label = minutes_to_timeframe(target_minutes)
            target_timeframes.append(tf_label)
            if len(target_timeframes) >= horizon_bars:
                break

    # Limit to the number of horizon bars
    return target_timeframes[:horizon_bars]


def time_labels_to_horizon_bars(
    time_labels: List[str],
    base_timeframe: str
) -> List[int]:
    """
    Convert time labels to horizon bars relative to base timeframe.

    Args:
        time_labels: List of time labels (e.g., ["1m", "5m", "15m"])
        base_timeframe: Base timeframe of the model

    Returns:
        List of horizon bars corresponding to each time label
    """
    base_minutes = timeframe_to_minutes(base_timeframe)
    horizon_bars = []

    for label in time_labels:
        label_minutes = timeframe_to_minutes(label)
        bars = max(1, label_minutes // base_minutes)
        horizon_bars.append(bars)

    return horizon_bars


def convert_horizons_for_inference(
    horizons: Union[List[str], List[int], int],
    base_timeframe: str,
    model_horizon_bars: int = None
) -> Tuple[List[str], List[int]]:
    """
    Convert various horizon formats to consistent inference format.

    Args:
        horizons: Horizons in various formats
        base_timeframe: Base timeframe for conversion
        model_horizon_bars: Original model horizon in bars (for validation)

    Returns:
        Tuple of (time_labels, horizon_bars)
    """
    if isinstance(horizons, int):
        # Single horizon bar value
        time_labels = horizon_bars_to_time_labels(horizons, base_timeframe)
        horizon_bars = [horizons]

    elif isinstance(horizons, list):
        if not horizons:
            # Default horizons
            time_labels = ["1m", "5m", "15m"]
            horizon_bars = time_labels_to_horizon_bars(time_labels, base_timeframe)

        elif isinstance(horizons[0], str):
            # Time labels format
            time_labels = horizons
            horizon_bars = time_labels_to_horizon_bars(time_labels, base_timeframe)

        elif isinstance(horizons[0], (int, float)):
            # Bars format
            horizon_bars = [int(h) for h in horizons]
            time_labels = []
            for bars in horizon_bars:
                labels = horizon_bars_to_time_labels(bars, base_timeframe)
                time_labels.extend(labels)
            time_labels = time_labels[:len(horizon_bars)]

        else:
            raise ValueError(f"Unsupported horizon format: {type(horizons[0])}")
    else:
        raise ValueError(f"Unsupported horizon type: {type(horizons)}")

    # Validation if model horizon is known
    if model_horizon_bars is not None:
        max_bars = max(horizon_bars) if horizon_bars else 1
        if max_bars > model_horizon_bars:
            logger.warning(
                f"Requested horizon {max_bars} bars exceeds model training horizon "
                f"{model_horizon_bars} bars. Results may be unreliable."
            )

    return time_labels, horizon_bars


def create_future_timestamps(
    last_timestamp_ms: int,
    base_timeframe: str,
    time_labels: List[str]
) -> List[int]:
    """
    Create future timestamps for predictions.

    Args:
        last_timestamp_ms: Last timestamp in milliseconds
        base_timeframe: Base timeframe for the data
        time_labels: Target time labels for predictions

    Returns:
        List of future timestamps in milliseconds
    """
    base_dt = pd.to_datetime(last_timestamp_ms, unit="ms", utc=True)
    future_timestamps = []

    for label in time_labels:
        try:
            future_dt = base_dt + pd.to_timedelta(label)
            future_timestamps.append(int(future_dt.value // 1_000_000))
        except Exception:
            # Fallback to base timeframe increment
            base_minutes = timeframe_to_minutes(base_timeframe)
            future_dt = base_dt + pd.to_timedelta(f"{base_minutes}m")
            future_timestamps.append(int(future_dt.value // 1_000_000))

    return future_timestamps


def validate_horizon_compatibility(
    inference_horizons: List[str],
    training_horizon_bars: int,
    base_timeframe: str
) -> dict:
    """
    Validate compatibility between inference horizons and training horizon.

    Returns:
        Dict with validation results
    """
    _, inference_bars = convert_horizons_for_inference(
        inference_horizons, base_timeframe
    )

    max_inference_bars = max(inference_bars) if inference_bars else 1

    results = {
        "compatible": True,
        "warnings": [],
        "max_horizon_bars": max_inference_bars,
        "training_horizon_bars": training_horizon_bars
    }

    if max_inference_bars > training_horizon_bars:
        results["compatible"] = False
        results["warnings"].append(
            f"Maximum inference horizon ({max_inference_bars} bars) exceeds "
            f"training horizon ({training_horizon_bars} bars)"
        )

    return results