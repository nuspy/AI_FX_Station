"""
Centralized indicator computation pipeline.

ISSUE-001b: Consolidates _indicators() function that was duplicated across:
- training/train_sklearn.py (original implementation)
- ui/workers/forecast_worker.py (imported)
- features/incremental_updater.py (imported)

Provides multi-timeframe indicator computation with caching and efficient data fetching.

HIGH-002: All feature names follow lowercase_underscore convention.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from loguru import logger
from .feature_name_utils import standardize_dataframe_columns

# Import TA library if available
try:
    import ta
    _HAS_TA = True
except Exception:
    _HAS_TA = False
    warnings.warn(
        "Package 'ta' not found: advanced indicators limited.", RuntimeWarning
    )


def compute_indicators(
    df: pd.DataFrame,
    ind_cfg: Dict[str, Any],
    indicator_tfs: Dict[str, List[str]],
    base_tf: str,
    symbol: str = None,
    days_history: int = None,
) -> pd.DataFrame:
    """
    Compute technical indicators across multiple timeframes.

    ISSUE-001b: Centralized implementation formerly in train_sklearn.py.

    Features:
    - Multi-timeframe support with automatic resampling
    - Timeframe caching to avoid redundant DB queries (OPT-001)
    - Support for both 'ta' library and fallback implementations
    - Proper timestamp alignment via merge_asof

    Args:
        df: Base timeframe candlestick data with columns [ts_utc, open, high, low, close, volume]
        ind_cfg: Indicator configuration dict mapping indicator names to parameters
            Example: {"rsi": {"n": 14}, "atr": {"n": 14}, "bollinger": {"n": 20, "dev": 2.0}}
        indicator_tfs: Mapping of indicators to timeframes to compute
            Example: {"rsi": ["5m", "15m"], "atr": ["5m", "1h"]}
        base_tf: Base timeframe string (e.g., "5m", "1h")
        symbol: Trading symbol for fetching higher timeframe data (optional)
        days_history: Days of history to fetch for higher timeframes (optional)

    Returns:
        DataFrame with computed indicator features aligned to base timeframe index

    Raises:
        RuntimeError: If data fetching fails
        ValueError: If invalid timeframe specified

    Example:
        >>> df = fetch_candles_from_db("EUR/USD", "5m", 90)
        >>> ind_cfg = {"rsi": {"n": 14}, "atr": {"n": 14}}
        >>> indicator_tfs = {"rsi": ["5m", "15m"], "atr": ["5m"]}
        >>> features = compute_indicators(df, ind_cfg, indicator_tfs, "5m", "EUR/USD", 90)
        >>> print(features.columns)
        # ['rsi_5m_14', 'rsi_15m_14', 'atr_5m_14']
    """
    from .feature_utils import ensure_dt_index, timeframe_to_timedelta
    from ..data.data_loader import fetch_candles_from_db

    logger.debug(f"compute_indicators called with df shape: {df.shape}, base_tf: {base_tf}")
    frames: List[pd.DataFrame] = []
    base = ensure_dt_index(df)
    base_lookup = base[["ts_utc"]].copy()
    try:
        base_delta = timeframe_to_timedelta(base_tf)
    except Exception:
        base_delta = pd.Timedelta("1min")

    # OPTIMIZATION (OPT-001): Pre-fetch all timeframes needed (cache to avoid redundant DB queries)
    # BUG-003 FIX: Cache is local-scope (recreated on each call), no cross-symbol contamination
    timeframe_cache: Dict[str, pd.DataFrame] = {base_tf: df.copy()}

    # Collect all unique timeframes needed
    unique_tfs = set([base_tf])
    for name in ind_cfg.keys():
        key = str(name).lower()
        tfs = indicator_tfs.get(key) or indicator_tfs.get(name, []) or [base_tf]
        unique_tfs.update(tfs)

    # Pre-fetch all non-base timeframes ONCE
    for tf in unique_tfs:
        if tf != base_tf and tf not in timeframe_cache:
            try:
                if symbol and days_history:
                    logger.info(
                        f"[CACHE] Pre-fetching {tf} candles from DB (will be reused for all indicators)"
                    )
                    timeframe_cache[tf] = fetch_candles_from_db(
                        symbol, tf, days_history
                    )
                    logger.debug(
                        f"[CACHE] Cached {tf}: {timeframe_cache[tf].shape[0]} rows"
                    )
                else:
                    # Fallback to resample if symbol/days_history not provided
                    logger.warning(
                        f"Symbol/days_history not provided, falling back to resample for {tf}"
                    )
                    timeframe_cache[tf] = _resample(df, tf)
            except Exception as e:
                logger.exception(f"Failed to pre-fetch {tf} candles: {e}")
                # Don't cache failed fetches

    logger.info(
        f"[CACHE] Pre-fetched {len(timeframe_cache)} timeframes: {list(timeframe_cache.keys())}"
    )

    # Now process indicators using cached data
    for name, params in ind_cfg.items():
        key = str(name).lower()
        tfs = indicator_tfs.get(key) or indicator_tfs.get(name, []) or [base_tf]
        for tf in tfs:
            logger.debug(f"Processing indicator {key} for TF {tf}")

            # Use cached data instead of fetching again
            if tf in timeframe_cache:
                tmp = timeframe_cache[tf].copy()
                logger.debug(f"[CACHE HIT] Using cached {tf} data for {key}")
            else:
                logger.warning(
                    f"[CACHE MISS] TF {tf} not in cache, skipping indicator {key}_{tf}"
                )
                continue

            tmp = ensure_dt_index(tmp)
            logger.debug(
                f"Indicator {key}_{tf}: final tmp shape = {tmp.shape} (need ≥14 for ATR/RSI)"
            )
            cols: Dict[str, pd.Series] = {}

            if not _HAS_TA:
                if key == "rsi":
                    n = int(params.get("n", 14))
                    delta = tmp["close"].diff()
                    up = delta.clip(lower=0.0).rolling(n).mean()
                    down = (-delta.clip(upper=0.0)).rolling(n).mean()
                    rs = up / (down + 1e-12)
                    cols[f"rsi_{tf}_{n}"] = 100 - (100 / (1 + rs))
                elif key == "atr":
                    n = int(params.get("n", 14))
                    hl = (tmp["high"] - tmp["low"]).abs()
                    hc = (tmp["high"] - tmp["close"].shift(1)).abs()
                    lc = (tmp["low"] - tmp["close"].shift(1)).abs()
                    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                    cols[f"atr_{tf}_{n}"] = tr.rolling(n).mean()
            else:
                if key == "rsi":
                    n = int(params.get("n", 14))
                    cols[f"rsi_{tf}_{n}"] = ta.momentum.RSIIndicator(
                        close=tmp["close"], window=n
                    ).rsi()
                elif key == "atr":
                    n = int(params.get("n", 14))
                    cols[f"atr_{tf}_{n}"] = ta.volatility.AverageTrueRange(
                        high=tmp["high"], low=tmp["low"], close=tmp["close"], window=n
                    ).average_true_range()
                elif key == "bollinger":
                    n = int(params.get("n", 20))
                    dev = float(params.get("dev", 2.0))
                    bb = ta.volatility.BollingerBands(
                        close=tmp["close"], window=n, window_dev=dev
                    )
                    cols[f"bb_m_{tf}_{n}_{dev}"] = bb.bollinger_mavg()
                    cols[f"bb_h_{tf}_{n}_{dev}"] = bb.bollinger_hband()
                    cols[f"bb_l_{tf}_{n}_{dev}"] = bb.bollinger_lband()
                elif key == "macd":
                    f = int(params.get("fast", 12))
                    s = int(params.get("slow", 26))
                    sig = int(params.get("signal", 9))
                    macd = ta.trend.MACD(
                        close=tmp["close"],
                        window_fast=f,
                        window_slow=s,
                        window_sign=sig,
                    )
                    cols[f"macd_{tf}_{f}_{s}_{sig}"] = macd.macd()
                    cols[f"macd_sig_{tf}_{f}_{s}_{sig}"] = macd.macd_signal()
                    cols[f"macd_diff_{tf}_{f}_{s}_{sig}"] = macd.macd_diff()
                elif key == "donchian":
                    n = int(params.get("n", 20))
                    u = tmp["high"].rolling(n).max()
                    l = tmp["low"].rolling(n).min()
                    cols[f"donch_mid_{tf}_{n}"] = (u + l) / 2.0
                elif key == "keltner":
                    ema = int(params.get("ema", 20))
                    atr_n = int(params.get("atr", 10))
                    mult = float(params.get("mult", 1.5))
                    mid = tmp["close"].ewm(span=ema, adjust=False).mean()
                    hl = (tmp["high"] - tmp["low"]).abs()
                    hc = (tmp["high"] - tmp["close"].shift(1)).abs()
                    lc = (tmp["low"] - tmp["close"].shift(1)).abs()
                    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                    atr = tr.rolling(atr_n).mean()
                    cols[f"kelt_mid_{tf}_{ema}_{atr_n}_{mult}"] = mid
                    cols[f"kelt_up_{tf}_{ema}_{atr_n}_{mult}"] = mid + mult * atr
                    cols[f"kelt_lo_{tf}_{ema}_{atr_n}_{mult}"] = mid - mult * atr
                elif key == "hurst":
                    w = int(params.get("window", 128))
                    series = tmp["close"].astype(float)
                    roll = series.rolling(w)

                    def _h(x: pd.Series) -> float:
                        vals = x.values
                        if len(vals) < 2:
                            return np.nan
                        vals = vals - vals.mean()
                        z = np.cumsum(vals)
                        R = z.max() - z.min()
                        S = vals.std() + 1e-12
                        return math.log((R / S) + 1e-12) / math.log(len(vals) + 1e-12)

                    cols[f"hurst_{tf}_{w}"] = roll.apply(_h, raw=False)
                elif key == "ema":
                    fast = int(params.get("fast", 12))
                    slow = int(params.get("slow", 26))
                    ema_fast = tmp["close"].ewm(span=fast, adjust=False).mean()
                    ema_slow = tmp["close"].ewm(span=slow, adjust=False).mean()
                    cols[f"ema_fast_{tf}_{fast}"] = ema_fast
                    cols[f"ema_slow_{tf}_{slow}"] = ema_slow
                    cols[f"ema_slope_{tf}_{fast}"] = ema_fast.diff().fillna(0.0)
            if not cols:
                continue
            feat = pd.DataFrame(cols)
            feat["ts_utc"] = tmp.index.view("int64") // 10**6
            right = ensure_dt_index(feat)
            try:
                tol = max(timeframe_to_timedelta(tf), base_delta)
            except Exception:
                tol = base_delta
            merged = pd.merge_asof(
                left=base_lookup,
                right=right,
                left_index=True,
                right_index=True,
                direction="nearest",
                tolerance=tol,
            )
            merged = (
                merged.reset_index(drop=True)
                .drop(columns=["ts_utc_y"], errors="ignore")
                .rename(columns={"ts_utc_x": "ts_utc"})
            )
            frames.append(merged.drop(columns=["ts_utc"], errors="ignore"))
    # HIGH-002: Standardize all feature names to lowercase_underscore
    result = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
    result = standardize_dataframe_columns(result, inplace=False)
    return result


def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLC data to a higher timeframe.

    Args:
        df: Base timeframe data
        timeframe: Target timeframe (e.g., "15m", "1h", "4h")

    Returns:
        Resampled OHLC DataFrame

    Raises:
        ValueError: If timeframe format is invalid
    """
    if timeframe.endswith("m"):
        rule = f"{int(timeframe[:-1])}min"
    elif timeframe.endswith("h"):
        rule = f"{int(timeframe[:-1])}h"
    else:
        raise ValueError(f"Timeframe not supported: {timeframe}")
    x = df.copy()
    x.index = pd.to_datetime(x["ts_utc"], unit="ms", utc=True)
    ohlc = (
        x[["open", "high", "low", "close"]]
        .astype(float)
        .resample(rule, label="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    )
    if "volume" in x.columns:
        ohlc["volume"] = x["volume"].astype(float).resample(rule, label="right").sum()
    ohlc["ts_utc"] = ohlc.index.view("int64") // 10**6
    return ohlc.reset_index(drop=True)
