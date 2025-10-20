"""
Centralized feature engineering utility functions.

Consolidates duplicate utility functions from train_sklearn.py and train_sklearn_btalib.py.
These utilities support timeframe manipulation, data preprocessing, and indicator configuration.
"""
from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List

import pandas as pd


def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has datetime index based on ts_utc column.

    This is the centralized version consolidating duplicates from:
    - train_sklearn.py (line 98)
    - train_sklearn_btalib.py (line 86)

    Args:
        df: DataFrame with ts_utc column (milliseconds UTC timestamp)

    Returns:
        DataFrame with datetime index

    Example:
        >>> df = pd.DataFrame({'ts_utc': [1609459200000, 1609459260000], 'close': [1.2, 1.3]})
        >>> df_indexed = ensure_dt_index(df)
        >>> isinstance(df_indexed.index, pd.DatetimeIndex)
        True
    """
    out = df.copy()
    out.index = pd.to_datetime(out["ts_utc"], unit="ms", utc=True)
    return out


def timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    """
    Convert timeframe string to pandas Timedelta.

    This is the centralized version consolidating duplicates from:
    - train_sklearn.py (line 104)
    - train_sklearn_btalib.py (line 93)

    Supported formats:
    - Milliseconds: "1000ms"
    - Seconds: "30s"
    - Minutes: "5m", "15m"
    - Hours: "1h", "4h"
    - Days: "1d"
    - Weeks: "1w"
    - Numeric fallback: "5" -> 5 minutes

    Args:
        tf: Timeframe string

    Returns:
        pandas Timedelta object

    Raises:
        ValueError: If timeframe format is invalid

    Example:
        >>> timeframe_to_timedelta("5m")
        Timedelta('0 days 00:05:00')
        >>> timeframe_to_timedelta("1h")
        Timedelta('0 days 01:00:00')
    """
    tf = str(tf).strip().lower()

    # Milliseconds
    if tf.endswith("ms"):
        return pd.Timedelta(milliseconds=int(tf[:-2]))

    # Seconds (but not milliseconds)
    if tf.endswith("s") and not tf.endswith("ms"):
        return pd.Timedelta(seconds=int(tf[:-1]))

    # Minutes
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))

    # Hours
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))

    # Days
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))

    # Weeks (from btalib version)
    if tf.endswith("w"):
        return pd.Timedelta(weeks=int(tf[:-1]))

    # Fallback: try parsing as numeric minutes
    try:
        return pd.Timedelta(minutes=int(tf))
    except Exception:
        pass

    raise ValueError(f"Unsupported timeframe format: {tf}")


def coerce_indicator_tfs(raw_value: Any) -> Dict[str, List[str]]:
    """
    Coerce indicator timeframes configuration to standard format.

    This is the centralized version consolidating duplicates from:
    - train_sklearn.py (line 119)
    - train_sklearn_btalib.py (line 408)

    Handles multiple input formats:
    - dict: {"rsi": ["5m", "15m"], "atr": ["5m"]}
    - JSON string: '{"rsi": ["5m", "15m"]}'
    - Empty/None: returns {}

    Normalizes to: {"indicator_name": ["tf1", "tf2", ...]}

    Args:
        raw_value: Raw configuration value (dict, str, or None)

    Returns:
        Normalized dictionary mapping indicator names to list of timeframes

    Example:
        >>> coerce_indicator_tfs({"rsi": "5m"})
        {'rsi': ['5m']}
        >>> coerce_indicator_tfs('{"rsi": ["5m", "15m"]}')
        {'rsi': ['5m', '15m']}
        >>> coerce_indicator_tfs(None)
        {}
    """
    if not raw_value:
        return {}

    # Parse input
    data: Dict[str, Any]
    if isinstance(raw_value, dict):
        data = raw_value
    elif isinstance(raw_value, str):
        try:
            data = json.loads(raw_value)
            if not isinstance(data, dict):
                warnings.warn(
                    "indicator_tfs JSON must be a dict; using empty dict",
                    RuntimeWarning
                )
                return {}
        except Exception:
            warnings.warn(
                "indicator_tfs not parseable JSON; using empty dict",
                RuntimeWarning
            )
            return {}
    else:
        warnings.warn(
            f"indicator_tfs must be dict or JSON string, got {type(raw_value)}; using empty dict",
            RuntimeWarning
        )
        return {}

    # Normalize to dict of lists
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not v:
            continue

        key = str(k).lower().strip()

        # Convert to list
        if isinstance(v, (list, tuple, set)):
            vals = [str(x).strip() for x in v if str(x).strip()]
        else:
            vals = [str(v).strip()]

        # Deduplicate while preserving order
        dedup: List[str] = []
        for tf in vals:
            tf_norm = tf.strip()
            if tf_norm and tf_norm not in dedup:
                dedup.append(tf_norm)

        if dedup:
            out[key] = dedup

    return out


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLC data to a different timeframe.

    This is the centralized version from train_sklearn.py (line 159).

    Aggregation rules:
    - open: first
    - high: max
    - low: min
    - close: last
    - volume: sum

    Args:
        df: DataFrame with OHLC data and ts_utc column
        timeframe: Target timeframe (e.g., "5m", "1h")

    Returns:
        Resampled DataFrame with same structure

    Raises:
        ValueError: If timeframe format is not supported

    Example:
        >>> df_1m = pd.DataFrame({
        ...     'ts_utc': [1609459200000, 1609459260000, 1609459320000],
        ...     'open': [1.2, 1.21, 1.22],
        ...     'high': [1.21, 1.22, 1.23],
        ...     'low': [1.19, 1.20, 1.21],
        ...     'close': [1.20, 1.21, 1.22],
        ...     'volume': [100, 150, 120]
        ... })
        >>> df_5m = resample_ohlc(df_1m, "5m")
        >>> len(df_5m) < len(df_1m)
        True
    """
    # Determine resample rule
    if timeframe.endswith("m"):
        rule = f"{int(timeframe[:-1])}min"
    elif timeframe.endswith("h"):
        rule = f"{int(timeframe[:-1])}h"
    elif timeframe.endswith("d"):
        rule = f"{int(timeframe[:-1])}D"
    else:
        raise ValueError(f"Unsupported timeframe for resampling: {timeframe}")

    # Create datetime index
    x = df.copy()
    x.index = pd.to_datetime(x["ts_utc"], unit="ms", utc=True)

    # Resample OHLC
    ohlc = x[["open", "high", "low", "close"]].astype(float).resample(
        rule, label="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    # Resample volume if available
    if "volume" in x.columns:
        ohlc["volume"] = x["volume"].astype(float).resample(rule, label="right").sum()

    # Convert index back to ts_utc
    ohlc["ts_utc"] = (ohlc.index.view("int64") // 10**6).astype("int64")

    return ohlc.reset_index(drop=True)


def validate_ohlc_dataframe(df: pd.DataFrame, require_volume: bool = False) -> None:
    """
    Validate that DataFrame has required OHLC columns.

    Args:
        df: DataFrame to validate
        require_volume: Whether volume column is required

    Raises:
        ValueError: If required columns are missing or invalid

    Example:
        >>> df = pd.DataFrame({'ts_utc': [1], 'open': [1.2], 'high': [1.3], 'low': [1.1], 'close': [1.25]})
        >>> validate_ohlc_dataframe(df)  # Should pass
        >>> validate_ohlc_dataframe(df, require_volume=True)  # Raises ValueError
        Traceback (most recent call last):
        ...
        ValueError: Missing required column: volume
    """
    required = ["ts_utc", "open", "high", "low", "close"]
    if require_volume:
        required.append("volume")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    # Validate numeric types
    for col in ["open", "high", "low", "close"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric, got {df[col].dtype}")


def align_to_base_timeframe(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    base_tf: str,
    feature_tf: str,
    direction: str = "nearest"
) -> pd.DataFrame:
    """
    Align feature DataFrame to base DataFrame timeframe using merge_asof.

    This is a common pattern used in both train_sklearn.py and train_sklearn_btalib.py
    for aligning indicators computed on different timeframes to the base timeframe.

    Args:
        base_df: Base DataFrame with ts_utc column
        feature_df: Feature DataFrame to align
        base_tf: Base timeframe string
        feature_tf: Feature timeframe string
        direction: Direction for merge_asof ("nearest", "backward", "forward")

    Returns:
        DataFrame aligned to base timeframe

    Example:
        >>> base = pd.DataFrame({'ts_utc': [1000, 2000, 3000], 'close': [1.2, 1.3, 1.4]})
        >>> feat = pd.DataFrame({'ts_utc': [1500, 3500], 'indicator': [0.5, 0.6]})
        >>> aligned = align_to_base_timeframe(base, feat, "1m", "5m")
        >>> len(aligned) == len(base)
        True
    """
    base = ensure_dt_index(base_df)
    base_lookup = base[["ts_utc"]].copy()

    feature = ensure_dt_index(feature_df)

    # Calculate tolerance for merge_asof
    try:
        base_delta = timeframe_to_timedelta(base_tf)
        feature_delta = timeframe_to_timedelta(feature_tf)
        tolerance = max(feature_delta, base_delta)
    except Exception:
        tolerance = pd.Timedelta("5min")  # Default fallback

    # Merge
    merged = pd.merge_asof(
        left=base_lookup,
        right=feature,
        left_index=True,
        right_index=True,
        direction=direction,
        tolerance=tolerance
    )

    # Clean up columns
    merged = merged.reset_index(drop=True)
    merged = merged.drop(columns=["ts_utc_y"], errors="ignore")
    merged = merged.rename(columns={"ts_utc_x": "ts_utc"}, errors="ignore")
    merged = merged.drop(columns=["ts_utc"], errors="ignore")

    return merged
