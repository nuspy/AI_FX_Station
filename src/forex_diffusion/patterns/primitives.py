from __future__ import annotations
from typing import Tuple, List
import numpy as np
import pandas as pd

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Average True Range (ATR).
    
    HIGH-001: Use consolidated implementation from features.consolidated_indicators
    """
    # Import here to avoid circular dependencies
    from ..features.consolidated_indicators import atr as consolidated_atr
    return consolidated_atr(df, n=n)

def zigzag_pivots(df: pd.DataFrame, atr_mult: float = 2.0, n_atr: int = 14) -> List[Tuple[int,int]]:
    """
    Causal zigzag: returns list of (index, typ) where typ=+1 swing-high, -1 swing-low.
    Threshold based on ATR to be scale-invariant.
    """
    a = atr(df, n_atr).to_numpy()
    c = df["close"].astype(float).to_numpy()
    piv: List[Tuple[int,int]] = []
    if len(c) < 3:
        return piv
    last = 0
    mode = 0 # 0 unk, +1 up leg, -1 down leg
    for i in range(1, len(c)):
        thr = max(a[i] * atr_mult, 1e-9)
        if mode >= 0 and c[i] <= c[last] - thr:
            if last > 0: piv.append((last, +1))  # last was swing-high
            mode = -1
            last = i
        elif mode <= 0 and c[i] >= c[last] + thr:
            if last > 0: piv.append((last, -1))  # last was swing-low
            mode = +1
            last = i
        else:
            # extend leg if new extreme found
            if (mode >= 0 and c[i] > c[last]) or (mode <= 0 and c[i] < c[last]):
                last = i
    return piv

def fit_line_indices(y: np.ndarray, i0: int, i1: int) -> Tuple[float, float]:
    """Return slope, intercept of least-squares fit over indices [i0, i1] inclusive."""
    x = np.arange(i0, i1+1, dtype=float)
    yy = y[i0:i1+1].astype(float)
    if len(x) < 2:
        return 0.0, float(yy[-1] if len(yy) else 0.0)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, yy, rcond=None)[0]
    return float(slope), float(intercept)


def time_array(df):
    if df is None or len(df)==0:
        return np.array([], dtype='datetime64[ns]')
    if 'ts_utc' in df.columns:
        try:
            return pd.to_datetime(df['ts_utc'], unit='ms').to_numpy()
        except Exception:
            return pd.to_datetime(df['ts_utc']).to_numpy()
    if 'time' in df.columns:
        return pd.to_datetime(df['time']).to_numpy()
    return pd.to_datetime(df.index).to_numpy()

def safe_tz_convert(ts, target_tz=None):
    """
    Safely convert timezone for different datetime types.
    Handles pandas Series, DatetimeIndex, and numpy arrays.
    """
    try:
        if hasattr(ts, 'dt'):
            # Pandas Series with datetime
            return ts.dt.tz_convert(target_tz)
        elif hasattr(ts, 'tz_convert'):
            # DatetimeIndex
            return ts.tz_convert(target_tz)
        else:
            # Numpy array - convert to pandas and then back
            if target_tz is None:
                return pd.to_datetime(ts, utc=True).tz_convert(None).to_numpy()
            else:
                return pd.to_datetime(ts, utc=True).tz_convert(target_tz).to_numpy()
    except Exception:
        # If all else fails, just return the original
        return ts
