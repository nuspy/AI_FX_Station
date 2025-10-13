"""
Centralized feature engineering functions.

Consolidates duplicate feature computation logic from train_sklearn.py and train_sklearn_btalib.py.
These functions compute various technical and statistical features from OHLC data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_utils import ensure_dt_index


def relative_ohlc(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Compute relative OHLC features (log returns from previous close).

    This is the centralized version consolidating duplicates from:
    - train_sklearn.py (line 171)
    - train_sklearn_btalib.py (line 114) - simpler version

    The train_sklearn.py version is more robust with log-space normalization
    relative to previous close, which is better for financial data.

    Args:
        df: DataFrame with OHLC columns
        eps: Small epsilon to prevent division by zero

    Returns:
        DataFrame with relative OHLC features:
        - r_open: log(open / prev_close)
        - r_high: log(high / open)
        - r_low: log(low / open)
        - r_close: log(close / open)

    Example:
        >>> df = pd.DataFrame({
        ...     'open': [1.20, 1.21, 1.22],
        ...     'high': [1.22, 1.23, 1.24],
        ...     'low': [1.19, 1.20, 1.21],
        ...     'close': [1.21, 1.22, 1.23]
        ... })
        >>> rel_ohlc = relative_ohlc(df)
        >>> 'r_open' in rel_ohlc.columns
        True
    """
    prev_close = df["close"].shift(1).astype(float).clip(lower=eps)
    o = df["open"].astype(float).clip(lower=eps)
    h = df["high"].astype(float).clip(lower=eps)
    l = df["low"].astype(float).clip(lower=eps)
    c = df["close"].astype(float).clip(lower=eps)

    out = pd.DataFrame(index=df.index)
    out["r_open"] = np.log(o / prev_close)
    out["r_high"] = np.log(h / o)
    out["r_low"] = np.log(l / o)
    out["r_close"] = np.log(c / o)

    return out


def temporal_features(df: pd.DataFrame, use_cyclical: bool = True) -> pd.DataFrame:
    """
    Create temporal features from timestamp.

    This is the centralized version consolidating duplicates from:
    - train_sklearn.py (line 185) - uses sin/cos encoding
    - train_sklearn_btalib.py (line 125) - uses raw values

    Supports both cyclical (sin/cos) and raw encoding.

    Args:
        df: DataFrame with ts_utc column
        use_cyclical: If True, use sin/cos encoding. If False, use raw values.

    Returns:
        DataFrame with temporal features

    Example:
        >>> df = pd.DataFrame({'ts_utc': [1609459200000]})  # 2021-01-01 00:00 UTC
        >>> feats = temporal_features(df, use_cyclical=True)
        >>> 'hour_sin' in feats.columns and 'hour_cos' in feats.columns
        True
    """
    ts = pd.to_datetime(df["ts_utc"], unit="ms", utc=True)
    out = pd.DataFrame(index=df.index)

    if use_cyclical:
        # Cyclical encoding (better for ML models)
        hour = ts.dt.hour
        dow = ts.dt.dayofweek

        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    else:
        # Raw values (simpler, used in btalib version)
        out["hour"] = ts.dt.hour
        out["dow"] = ts.dt.dayofweek
        out["dom"] = ts.dt.day
        out["month"] = ts.dt.month
        out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    return out


def realized_volatility_feature(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute realized volatility feature.

    This is the centralized version consolidating duplicates from:
    - train_sklearn.py (line 150)
    - train_sklearn_btalib.py (line 137)

    Both implementations use slightly different formulas:
    - train_sklearn.py: sqrt(sum(log_ret^2)) over window
    - train_sklearn_btalib.py: std(log_ret) * sqrt(window)

    The btalib version is more standard for financial volatility estimation.

    Args:
        df: DataFrame with close column
        window: Rolling window size

    Returns:
        DataFrame with realized volatility feature

    Example:
        >>> df = pd.DataFrame({'close': [1.20, 1.21, 1.22, 1.23, 1.24]})
        >>> rv = realized_volatility_feature(df, window=3)
        >>> 'rv_3' in rv.columns
        True
    """
    if window <= 1:
        return pd.DataFrame(index=df.index)

    c = df["close"].astype(float)

    # Compute log returns
    log_returns = np.log(c / c.shift(1))

    # Annualized realized volatility using standard formula
    # std(log_ret) * sqrt(window) gives volatility estimate
    rv = log_returns.rolling(window=window, min_periods=2).std() * np.sqrt(window)

    feature = pd.DataFrame({f"rv_{window}": rv}, index=df.index)
    return feature


def returns_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Compute log return features over multiple windows.

    Args:
        df: DataFrame with close column
        windows: List of window sizes for return computation (default: [1, 5, 10, 20])

    Returns:
        DataFrame with return features for each window

    Example:
        >>> df = pd.DataFrame({'close': [1.20, 1.21, 1.22, 1.23, 1.24]})
        >>> ret_feats = returns_features(df, windows=[1, 3])
        >>> 'ret_1' in ret_feats.columns and 'ret_3' in ret_feats.columns
        True
    """
    if windows is None:
        windows = [1, 5, 10, 20]

    c = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)

    for w in windows:
        if w <= 0:
            continue
        # Log return over w periods
        out[f"ret_{w}"] = np.log(c / c.shift(w))

    return out


def price_momentum_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Compute price momentum features (percentage change).

    Args:
        df: DataFrame with close column
        windows: List of window sizes (default: [5, 10, 20, 50])

    Returns:
        DataFrame with momentum features

    Example:
        >>> df = pd.DataFrame({'close': [1.20, 1.21, 1.22, 1.23, 1.24, 1.25]})
        >>> mom = price_momentum_features(df, windows=[3, 5])
        >>> 'momentum_3' in mom.columns
        True
    """
    if windows is None:
        windows = [5, 10, 20, 50]

    c = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)

    for w in windows:
        if w <= 0 or w >= len(df):
            continue
        # Percentage change over w periods
        out[f"momentum_{w}"] = (c / c.shift(w) - 1.0) * 100

    return out


def volume_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Compute volume-based features.

    Args:
        df: DataFrame with volume column
        windows: List of window sizes for volume MA (default: [10, 20, 50])

    Returns:
        DataFrame with volume features

    Example:
        >>> df = pd.DataFrame({'volume': [100, 150, 120, 180, 200, 160]})
        >>> vol_feats = volume_features(df, windows=[3])
        >>> 'volume_ma_3' in vol_feats.columns
        True
    """
    if "volume" not in df.columns:
        return pd.DataFrame(index=df.index)

    if windows is None:
        windows = [10, 20, 50]

    vol = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)

    for w in windows:
        if w <= 0 or w >= len(df):
            continue

        # Volume moving average
        out[f"volume_ma_{w}"] = vol.rolling(window=w, min_periods=1).mean()

        # Volume ratio (current / MA)
        vol_ma = vol.rolling(window=w, min_periods=1).mean()
        out[f"volume_ratio_{w}"] = vol / (vol_ma + 1e-12)

    return out


def ohlc_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute OHLC range and spread features.

    Args:
        df: DataFrame with OHLC columns

    Returns:
        DataFrame with range features:
        - hl_range: (high - low) / close
        - oc_range: abs(open - close) / close
        - body_pct: (close - open) / (high - low + eps)

    Example:
        >>> df = pd.DataFrame({
        ...     'open': [1.20, 1.21],
        ...     'high': [1.22, 1.23],
        ...     'low': [1.19, 1.20],
        ...     'close': [1.21, 1.22]
        ... })
        >>> range_feats = ohlc_range_features(df)
        >>> 'hl_range' in range_feats.columns
        True
    """
    out = pd.DataFrame(index=df.index)
    eps = 1e-12

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    # High-Low range as percentage of close
    out["hl_range"] = (h - l) / (c + eps)

    # Open-Close range as percentage of close
    out["oc_range"] = np.abs(o - c) / (c + eps)

    # Body percentage of total range (0 = doji, 1 = no wicks)
    out["body_pct"] = np.abs(c - o) / (h - l + eps)

    # Upper/lower wick ratios
    out["upper_wick"] = (h - np.maximum(o, c)) / (h - l + eps)
    out["lower_wick"] = (np.minimum(o, c) - l) / (h - l + eps)

    return out


def standardize_features_no_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    val_frac: float
) -> tuple:
    """
    Standardize features ensuring NO look-ahead bias.

    CRITICAL: Computes mean/std ONLY on training set, then applies to validation.
    This prevents information leakage from future data.

    This is the robust version from train_sklearn.py with KS test verification.

    Args:
        X: Feature DataFrame
        y: Target Series
        val_frac: Validation fraction (e.g., 0.2 for 20%)

    Returns:
        Tuple of ((Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, metadata))

    Example:
        >>> X = pd.DataFrame({'f1': [1, 2, 3, 4, 5], 'f2': [10, 20, 30, 40, 50]})
        >>> y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> (Xtr, ytr), (Xva, yva), (mu, sigma, meta) = standardize_features_no_leakage(X, y, 0.2)
        >>> meta['ks_test_median_p'] is not None
        True
    """
    from sklearn.model_selection import train_test_split
    from scipy import stats
    import warnings

    # Split WITHOUT shuffling to maintain temporal order (prevent look-ahead bias)
    Xtr, Xva, ytr, yva = train_test_split(
        X.values, y.values, test_size=val_frac, shuffle=False
    )

    # Compute statistics ONLY on training set (NO look-ahead bias)
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma == 0] = 1.0  # Prevent division by zero (BUG-001 fix)

    # Apply standardization
    Xtr_scaled = (Xtr - mu) / sigma
    Xva_scaled = (Xva - mu) / sigma

    # VERIFICATION: Statistical test for look-ahead bias detection
    # If train and validation distributions are too similar, likely bias present
    p_values = []
    for i in range(min(10, Xtr_scaled.shape[1])):  # Test first 10 features
        if Xtr_scaled.shape[0] > 20 and Xva_scaled.shape[0] > 20:
            # Kolmogorov-Smirnov test: different distributions should have low p-value
            _, p_val = stats.ks_2samp(Xtr_scaled[:, i], Xva_scaled[:, i])
            p_values.append(p_val)

    # Metadata for debugging
    metadata = {
        "train_size": Xtr.shape[0],
        "val_size": Xva.shape[0],
        "train_mean": mu.tolist(),
        "train_std": sigma.tolist(),
        "ks_test_p_values": p_values,
        "ks_test_median_p": float(np.median(p_values)) if p_values else None,
    }

    # WARNING: If distributions too similar, potential look-ahead bias
    if metadata["ks_test_median_p"] is not None:
        if metadata["ks_test_median_p"] > 0.8:
            warnings.warn(
                f"⚠️ POTENTIAL LOOK-AHEAD BIAS DETECTED!\n"
                f"Train/Val distributions suspiciously similar (KS median p-value={metadata['ks_test_median_p']:.3f}).\n"
                f"Expected p < 0.5 for different time periods. Verify train_test_split has shuffle=False.",
                RuntimeWarning
            )

    return ((Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, metadata))
