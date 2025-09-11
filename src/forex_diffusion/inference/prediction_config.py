# src/forex_diffusion/inference/prediction_config.py
"""
Helper utilities to ensure required features for prediction exist for a given timeframe.
Provides:
 - REQUIRED_FEATURES grouped by categories
 - ensure_features_for_prediction(df, timeframe, features_list): compute missing cols in-place (returns df)
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np
import pandas as pd

REQUIRED_FEATURES = {
    "time": ["day_of_week", "hour", "hour_sin", "hour_cos", "is_asia", "is_eu", "is_us"],
    "base_candle": ["open", "high", "low", "close"],
    "volatility": ["r", "hl_range", "atr", "rv", "gk_vol", "vol_mean"],
    "ema": ["ema_fast", "ema_slow", "ema_slope"],
    "macd": ["macd", "macd_signal", "macd_hist"],
    "momentum": ["rsi"],
    "bollinger": ["bb_upper", "bb_lower", "bb_width", "bb_pctb"],
    "keltner": ["kelt_upper", "kelt_lower"],
    "donchian": ["don_upper", "don_lower"],
    "realized": ["realized_skew", "realized_kurt"],
    "hurst": ["hurst"],
}

def _hour_session_cols(ts_series: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(ts_series.astype("int64"), unit="ms", utc=True)
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    hours_no_min = hour
    hour_sin = np.sin(2 * np.pi * hours_no_min / 24.0)
    hour_cos = np.cos(2 * np.pi * hours_no_min / 24.0)
    is_asia = ((hours_no_min >= 0) & (hours_no_min < 9)).astype(int)
    is_eu = ((hours_no_min >= 7) & (hours_no_min < 16)).astype(int)
    is_us = ((hours_no_min >= 13) & (hours_no_min < 22)).astype(int)
    return pd.DataFrame({
        "day_of_week": dow,
        "hour": hours_no_min,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_asia": is_asia,
        "is_eu": is_eu,
        "is_us": is_us,
    }, index=ts_series.index)

def _compute_basic_indicators(df: pd.DataFrame, r_col: str = "r", std_window: int = 60) -> pd.DataFrame:
    tmp = df.copy()
    # log returns
    if r_col not in tmp.columns:
        tmp[r_col] = np.log(tmp["close"]).diff().fillna(0.0)
    # hl_range
    if "hl_range" not in tmp.columns:
        tmp["hl_range"] = (tmp["high"] - tmp["low"]).fillna(0.0)
    # rolling std of returns (rv)
    if "rv" not in tmp.columns:
        tmp["rv"] = tmp[r_col].rolling(window=std_window, min_periods=1).std(ddof=1).fillna(0.0)
    # simple atr (Wilder-like) approx using high/low/prev close
    if "atr" not in tmp.columns:
        prior = tmp["close"].shift(1)
        tr1 = tmp["high"] - tmp["low"]
        tr2 = (tmp["high"] - prior).abs()
        tr3 = (tmp["low"] - prior).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0.0)
        tmp["atr"] = tr.ewm(alpha=1.0/14, adjust=False).mean().fillna(0.0)
    # garman-klass per-bar then rolling sqrt of mean
    if "gk_vol" not in tmp.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_hl = np.log((tmp["high"] / tmp["low"]).replace([np.inf, -np.inf], np.nan)).fillna(0.0)
            ln_co = np.log((tmp["close"] / tmp["open"]).replace([np.inf, -np.inf], np.nan)).fillna(0.0)
            var_bar = 0.5 * (ln_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (ln_co ** 2)
        tmp["_gk_var"] = var_bar.fillna(0.0)
        tmp["gk_vol"] = np.sqrt(tmp["_gk_var"].rolling(window=std_window, min_periods=1).mean().clip(lower=0.0)).fillna(0.0)
        tmp.drop(columns=["_gk_var"], inplace=True, errors="ignore")
    # vol_mean
    if "vol_mean" not in tmp.columns and "volume" in tmp.columns:
        tmp["vol_mean"] = tmp["volume"].rolling(window=std_window, min_periods=1).mean().fillna(0.0)
    elif "vol_mean" not in tmp.columns:
        tmp["vol_mean"] = 0.0
    return tmp

def _compute_ema_and_slope(df: pd.DataFrame, fast_span: int = 12, slow_span: int = 26) -> pd.DataFrame:
    tmp = df.copy()
    if "ema_fast" not in tmp.columns:
        tmp["ema_fast"] = tmp["close"].ewm(span=fast_span, adjust=False).mean()
    if "ema_slow" not in tmp.columns:
        tmp["ema_slow"] = tmp["close"].ewm(span=slow_span, adjust=False).mean()
    if "ema_slope" not in tmp.columns:
        # slope approximate as first difference of ema_fast
        tmp["ema_slope"] = tmp["ema_fast"].diff().fillna(0.0)
    return tmp

def _compute_macd_rsi_bollinger(df: pd.DataFrame, rsi_n: int = 14, bb_n: int = 20, bb_k: float = 2.0) -> pd.DataFrame:
    tmp = df.copy()
    # MACD
    if "macd" not in tmp.columns:
        ema_fast = tmp["close"].ewm(span=12, adjust=False).mean()
        ema_slow = tmp["close"].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        tmp["macd"] = macd
        tmp["macd_signal"] = macd_signal
        tmp["macd_hist"] = macd - macd_signal
    # RSI (Wilder)
    if "rsi" not in tmp.columns:
        delta = tmp["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0/rsi_n, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/rsi_n, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        tmp["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        tmp["rsi"] = tmp["rsi"].fillna(50.0)
    # Bollinger
    if f"bb_pctb_{bb_n}" not in tmp.columns and f"bb_width_{bb_n}" not in tmp.columns:
        ma = tmp["close"].rolling(window=bb_n, min_periods=1).mean()
        std = tmp["close"].rolling(window=bb_n, min_periods=1).std(ddof=1)
        upper = ma + bb_k * std
        lower = ma - bb_k * std
        tmp[f"bb_upper_{bb_n}"] = upper
        tmp[f"bb_lower_{bb_n}"] = lower
        tmp[f"bb_width_{bb_n}"] = (upper - lower)
        tmp[f"bb_pctb_{bb_n}"] = (tmp["close"] - lower) / tmp[f"bb_width_{bb_n}"].replace(0, np.nan)
        tmp[f"bb_pctb_{bb_n}"] = tmp[f"bb_pctb_{bb_n}"].fillna(0.5)
    return tmp

def _compute_donchian_keltner_realized_hurst(df: pd.DataFrame, don_n: int = 20, hurst_window: int = 64, rv_window: int = 60) -> pd.DataFrame:
    tmp = df.copy()
    # Donchian
    if f"don_upper_{don_n}" not in tmp.columns:
        tmp[f"don_upper_{don_n}"] = tmp["high"].rolling(window=don_n, min_periods=1).max()
        tmp[f"don_lower_{don_n}"] = tmp["low"].rolling(window=don_n, min_periods=1).min()
    # Keltner approx: EMA of close ± multiplier * ATR
    if "kelt_upper" not in tmp.columns:
        ema = tmp["close"].ewm(span=20, adjust=False).mean()
        atr = tmp["atr"] if "atr" in tmp.columns else _compute_basic_indicators(tmp)["atr"]
        tmp["kelt_upper"] = ema + 1.5 * atr
        tmp["kelt_lower"] = ema - 1.5 * atr
    # realized moments
    if "realized_skew" not in tmp.columns:
        r = tmp["r"].fillna(0.0)
        tmp["realized_skew"] = r.rolling(window=rv_window, min_periods=1).skew().fillna(0.0)
    if "realized_kurt" not in tmp.columns:
        r = tmp["r"].fillna(0.0)
        tmp["realized_kurt"] = r.rolling(window=rv_window, min_periods=1).kurt().fillna(0.0)
    # hurst: preserve if present else nan (caller may compute differently)
    if "hurst" not in tmp.columns:
        tmp["hurst"] = np.nan
    return tmp

def ensure_features_for_prediction(df: pd.DataFrame, timeframe: str, features_list: Sequence[str]) -> pd.DataFrame:
    """
    Ensure the DataFrame contains all features in features_list by computing common ones.
    Returns augmented DataFrame (may be the same object).
    Conservative: computes only safe, causal approximations.
    """
    tmp = df.copy().reset_index(drop=True)
    # time features
    if any(col in features_list for col in REQUIRED_FEATURES["time"]):
        ts = tmp["ts_utc"]
        tfcols = _hour_session_cols(ts)
        for c in tfcols.columns:
            if c not in tmp.columns:
                tmp[c] = tfcols[c].values
    # basic indicators and vol
    if any(col in features_list for col in REQUIRED_FEATURES["volatility"]):
        tmp = _compute_basic_indicators(tmp, r_col="r", std_window=max(1, int(60)))
    # ema + slope
    if any(col in features_list for col in REQUIRED_FEATURES["ema"]):
        tmp = _compute_ema_and_slope(tmp)
    # macd, rsi, bollinger
    if any(col in features_list for col in REQUIRED_FEATURES["macd"] + REQUIRED_FEATURES["momentum"] + REQUIRED_FEATURES["bollinger"]):
        tmp = _compute_macd_rsi_bollinger(tmp)
    # donchian, keltner, realized, hurst
    if any(col in features_list for col in (REQUIRED_FEATURES["donchian"] + REQUIRED_FEATURES["keltner"] + REQUIRED_FEATURES["realized"] + REQUIRED_FEATURES["hurst"])):
        tmp = _compute_donchian_keltner_realized_hurst(tmp, don_n=20, hurst_window=64, rv_window=60)
    # ensure any requested features missing are added as zeros/nans
    for f in features_list:
        if f not in tmp.columns:
            tmp[f] = np.nan if "hurst" in f else 0.0
    return tmp
