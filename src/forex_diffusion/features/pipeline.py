"""
Feature pipeline for MagicForex.

Contains:
- causal resampling (resample_candles -> uses pandas resample with right-closed bins)
- technical indicators computed in a causal way (ATR Wilder, Bollinger, MACD, RSI Wilder, Donchian)
- aggregated-variance Hurst estimator
- time cyclic features and session dummies
- Standardizer class for causal standardization (fit on train only)
- pipeline_process to run the full feature engineering for a dataframe
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.config import get_config

# Map timeframe labels to pandas resample rules
TF_TO_PANDAS = {
    "1m": "1T",
    "2m": "2T",
    "3m": "3T",
    "4m": "4T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "60m": "60T",
    "1h": "60T",
    "2h": "120T",
    "4h": "240T",
    "1d": "1D",
    "1D": "1D",
}


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has datetime index converted from ts_utc in milliseconds UTC.
    """
    tmp = df.copy()
    if "ts_utc" not in tmp.columns:
        raise ValueError("ts_utc column required")
    tmp["ts_dt"] = pd.to_datetime(tmp["ts_utc"].astype("int64"), unit="ms", utc=True)
    tmp = tmp.set_index("ts_dt").sort_index()
    return tmp


def resample_causal(df: pd.DataFrame, src_tf: str, tgt_tf: str) -> pd.DataFrame:
    """
    Causal resampling from src_tf to tgt_tf:
    - open: first
    - close: last
    - high: max
    - low: min
    - volume: sum (if present)
    Uses right-closed intervals so period t corresponds to up to t inclusive.
    """
    if src_tf == tgt_tf:
        return df.copy()
    if tgt_tf not in TF_TO_PANDAS:
        raise ValueError(f"Unsupported target timeframe: {tgt_tf}")

    tmp = to_datetime_index(df)
    rule = TF_TO_PANDAS[tgt_tf]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in tmp.columns:
        agg["volume"] = "sum"
    # Use label='right', closed='right' to make the bar timestamp the period end (causal aggregation)
    res = tmp.resample(rule, label="right", closed="right").agg(agg)
    res = res.dropna(subset=["open", "close"]).reset_index()
    res["ts_utc"] = (res["ts_dt"].view("int64") // 1_000_000).astype("int64")
    cols = ["ts_utc", "open", "high", "low", "close"]
    if "volume" in res.columns:
        cols.append("volume")
    return res[cols]


def log_returns(df: pd.DataFrame, col: str = "close", out_col: str = "r") -> pd.DataFrame:
    """
    Compute log returns r_t = ln(close_t) - ln(close_{t-1}).
    """
    tmp = df.copy()
    tmp[out_col] = np.log(tmp[col]).diff()
    return tmp


def rolling_std(df: pd.DataFrame, col: str = "r", window: int = 20, out_col: Optional[str] = None) -> pd.DataFrame:
    """
    Rolling standard deviation with unbiased estimator (ddof=1).
    Causal: uses .rolling(window, min_periods=1).std(ddof=1) with center=False.
    """
    tmp = df.copy()
    if out_col is None:
        out_col = f"{col}_std_{window}"
    tmp[out_col] = tmp[col].rolling(window=window, min_periods=1).std(ddof=1)
    return tmp


def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder moving average via exponential smoothing with alpha = 1/period (adjust=False).
    """
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14, out_col: str = "atr") -> pd.DataFrame:
    """
    Average True Range (Wilder) computed causally.
    """
    tmp = df.copy()
    prior_close = tmp["close"].shift(1)
    tr1 = tmp["high"] - tmp["low"]
    tr2 = (tmp["high"] - prior_close).abs()
    tr3 = (tmp["low"] - prior_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tmp[out_col] = _wilder_ema(tr.fillna(0.0), n)
    return tmp


def bollinger(df: pd.DataFrame, n: int = 20, k: float = 2.0, out_prefix: str = "bb") -> pd.DataFrame:
    """
    Bollinger bands (MA + k * std), causal moving average via simple rolling mean.
    """
    tmp = df.copy()
    tmp[f"{out_prefix}_ma_{n}"] = tmp["close"].rolling(window=n, min_periods=1).mean()
    tmp[f"{out_prefix}_std_{n}"] = tmp["close"].rolling(window=n, min_periods=1).std(ddof=1)
    tmp[f"{out_prefix}_upper_{n}"] = tmp[f"{out_prefix}_ma_{n}"] + k * tmp[f"{out_prefix}_std_{n}"]
    tmp[f"{out_prefix}_lower_{n}"] = tmp[f"{out_prefix}_ma_{n}"] - k * tmp[f"{out_prefix}_std_{n}"]
    tmp[f"{out_prefix}_width_{n}"] = tmp[f"{out_prefix}_upper_{n}"] - tmp[f"{out_prefix}_lower_{n}"]
    tmp[f"{out_prefix}_pctb_{n}"] = (tmp["close"] - tmp[f"{out_prefix}_lower_{n}"]) / (tmp[f"{out_prefix}_width_{n}"].replace(0, np.nan))
    return tmp


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD = EMA_fast(close) - EMA_slow(close); signal = EMA_signal(MACD); hist = MACD - signal
    All EMA computed causally (adjust=False).
    """
    tmp = df.copy()
    ema_fast = tmp["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = tmp["close"].ewm(span=slow, adjust=False).mean()
    tmp["macd"] = ema_fast - ema_slow
    tmp["macd_signal"] = tmp["macd"].ewm(span=signal, adjust=False).mean()
    tmp["macd_hist"] = tmp["macd"] - tmp["macd_signal"]
    return tmp


def rsi_wilder(df: pd.DataFrame, n: int = 14, out_col: str = "rsi") -> pd.DataFrame:
    """
    RSI using Wilder smoothing.
    """
    tmp = df.copy()
    delta = tmp["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_ema(gain.fillna(0.0), n)
    avg_loss = _wilder_ema(loss.fillna(0.0), n)
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    tmp[out_col] = 100.0 - (100.0 / (1.0 + rs))
    tmp[out_col] = tmp[out_col].fillna(50.0)  # neutral for initial periods
    return tmp


def donchian(df: pd.DataFrame, n: int = 20, out_prefix: str = "don") -> pd.DataFrame:
    """
    Donchian channel upper/lower over last n bars (causal).
    """
    tmp = df.copy()
    tmp[f"{out_prefix}_upper_{n}"] = tmp["high"].rolling(window=n, min_periods=1).max()
    tmp[f"{out_prefix}_lower_{n}"] = tmp["low"].rolling(window=n, min_periods=1).min()
    return tmp


def hurst_aggvar(ts: pd.Series, min_chunks: int = 4) -> float:
    """
    Estimate Hurst exponent using aggregated variance method.
    Returns H estimate (float). For short series returns nan.
    """
    n = len(ts)
    if n < 32:
        return float("nan")
    # use powers of two chunk sizes
    max_k = int(np.floor(np.log2(n)))
    sizes = [2 ** k for k in range(1, max_k + 1)]
    variances = []
    ks = []
    for s in sizes:
        if s >= n or s < 2:
            continue
        # reshape into blocks of size s (drop remainder)
        m = n // s
        if m < 2:
            continue
        chunks = ts.values[: m * s].reshape(m, s)
        means = chunks.mean(axis=1)
        variances.append(means.var(ddof=1))
        ks.append(s)
    if len(ks) < min_chunks:
        return float("nan")
    log_vars = np.log(variances)
    log_sizes = np.log(ks)
    # slope = 2H - 2 => H = (slope + 2)/2
    slope, intercept = np.polyfit(log_sizes, log_vars, 1)
    H = (slope + 2.0) / 2.0
    return float(H)


def hurst_feature(df: pd.DataFrame, window: int = 256, out_col: str = "hurst") -> pd.DataFrame:
    """
    Rolling Hurst estimator applied to log-returns.
    """
    tmp = df.copy()
    if "r" not in tmp.columns:
        tmp = log_returns(tmp, col="close", out_col="r")
    rs = []
    series = tmp["r"].fillna(0.0)
    for i in range(len(series)):
        if i + 1 < window:
            rs.append(float("nan"))
            continue
        window_series = series.iloc[i + 1 - window : i + 1]
        rs.append(hurst_aggvar(window_series))
    tmp[out_col] = rs
    return tmp


def time_cyclic_and_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour sin/cos and session dummies (tokyo, london, ny) based on UTC hour.
    Sessions defined as approximate UTC ranges.
    """
    tmp = df.copy()
    dt = pd.to_datetime(tmp["ts_utc"].astype("int64"), unit="ms", utc=True)
    hours = dt.dt.hour + dt.dt.minute / 60.0
    tmp["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    tmp["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Session ranges in UTC (approx)
    tmp["session_tokyo"] = ((hours >= 0) & (hours < 9)).astype(int)
    tmp["session_london"] = ((hours >= 7) & (hours < 16)).astype(int)
    tmp["session_ny"] = ((hours >= 13) & (hours < 22)).astype(int)
    return tmp


@dataclass
class Standardizer:
    """
    Causal standardizer: fit on training dataframe only (no leakage).
    Then transform applies same mean/std to new data.
    """
    cols: List[str]
    mu: Optional[Dict[str, float]] = None
    sigma: Optional[Dict[str, float]] = None
    eps: float = 1e-9

    def fit(self, df: pd.DataFrame) -> None:
        """
        Compute mean and std on provided df (train only).
        """
        self.mu = {}
        self.sigma = {}
        for c in self.cols:
            vals = df[c].dropna().astype(float)
            self.mu[c] = float(vals.mean()) if len(vals) > 0 else 0.0
            self.sigma[c] = float(vals.std(ddof=0)) if len(vals) > 0 else 1.0
            if self.sigma[c] < self.eps:
                self.sigma[c] = 1.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standardization using fitted mu/sigma. Returns a new DataFrame.
        """
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Standardizer must be fitted before transform")
        tmp = df.copy()
        for c in self.cols:
            if c in tmp.columns:
                tmp[c] = (tmp[c].astype(float) - self.mu[c]) / self.sigma[c]
        return tmp

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


def pipeline_process(
    df: pd.DataFrame,
    timeframe: str,
    features_config: Optional[Dict] = None,
    standardize_on: Optional[List[str]] = None,
    standardizer: Optional[Standardizer] = None,
) -> Tuple[pd.DataFrame, Standardizer]:
    """
    Run full feature engineering pipeline:
    - compute log-returns
    - rolling features (std, atr)
    - indicators: bollinger, macd, rsi, donchian
    - hurst rolling
    - time cyclic & session features
    - drop warmup bars (based on features_config or default)
    - fit or apply provided Standardizer

    Returns (features_df, standardizer)
    """
    cfg = get_config()
    fc = features_config or {}
    warmup = fc.get("warmup_bars", 512)

    tmp = df.copy()
    # Ensure ts_dt index for rolling based on time (not strictly necessary for rolling counts)
    tmp = tmp.sort_values("ts_utc").reset_index(drop=True)

    tmp = log_returns(tmp, col="close", out_col="r")
    std_window = fc.get("standardization", {}).get("window_bars", 1000)
    tmp = rolling_std(tmp, col="r", window=std_window, out_col=f"r_std_{std_window}")
    tmp = atr(tmp, n=fc.get("indicators", {}).get("atr", {}).get("n", 14))
    tmp = bollinger(tmp, n=fc.get("indicators", {}).get("bollinger", {}).get("n", 20), k=fc.get("indicators", {}).get("bollinger", {}).get("k", 2.0))
    tmp = macd(tmp, fast=fc.get("indicators", {}).get("macd", {}).get("fast", 12), slow=fc.get("indicators", {}).get("macd", {}).get("slow", 26), signal=fc.get("indicators", {}).get("macd", {}).get("signal", 9))
    tmp = rsi_wilder(tmp, n=fc.get("indicators", {}).get("rsi", {}).get("n", 14))
    tmp = donchian(tmp, n=fc.get("indicators", {}).get("donchian", {}).get("n", 20))
    tmp = hurst_feature(tmp, window=fc.get("indicators", {}).get("hurst", {}).get("window", 256))
    tmp = time_cyclic_and_session(tmp)

    # Define feature columns (preserve order)
    base_features = ["r", f"r_std_{std_window}", "atr", "hour_sin", "hour_cos", "session_tokyo", "session_london", "session_ny", "rsi", "macd", "macd_hist", "hurst"]
    # add bollinger pctb and donchian
    base_features += [f"bb_pctb_{fc.get('indicators', {}).get('bollinger', {}).get('n', 20)}", f"don_upper_{fc.get('indicators', {}).get('donchian', {}).get('n', 20)}", f"don_lower_{fc.get('indicators', {}).get('donchian', {}).get('n', 20)}"]

    # Keep only existing columns
    features = [c for c in base_features if c in tmp.columns]

    # Drop warmup bars to avoid NaNs from indicators
    if warmup > 0:
        if len(tmp) <= warmup:
            logger.warning("Pipeline: data length (%d) <= warmup (%d). Returning empty features.", len(tmp), warmup)
            features_df = tmp.iloc[0:0][features].copy()
        else:
            features_df = tmp.iloc[warmup:][features].copy()
    else:
        features_df = tmp[features].copy()

    # Standardization
    if standardizer is None:
        if standardize_on is None:
            standardize_on = features  # default: standardize all features
        standardizer = Standardizer(cols=standardize_on)
        # Fit with causality: caller should pass training slice; here we fit on entire features_df for convenience
        standardizer.fit(features_df)
    features_df = standardizer.transform(features_df)

    return features_df.reset_index(drop=True), standardizer
