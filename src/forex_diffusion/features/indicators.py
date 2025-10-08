# src/forex_diffusion/features/indicators.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    sma_series = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
    upper = sma_series + n_std * std
    lower = sma_series - n_std * std
    return upper, lower


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1.0/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Consolidated from multiple implementations across the codebase.
    This is the canonical ATR implementation - use this instead of local copies.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        n: Period for ATR calculation (default 14)

    Returns:
        Series of ATR values
    """
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    prev_close = np.roll(close, 1)

    # True Range: max of (high-low, |high-prev_close|, |low-prev_close|)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )
    tr[0] = high[0] - low[0]  # First TR is just high-low

    # ATR is EMA of TR
    atr_series = pd.Series(tr).ewm(span=n, adjust=False).mean()
    return atr_series
