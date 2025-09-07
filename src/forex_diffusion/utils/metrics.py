"""
Utility metrics for MagicForex.

Provides:
- annualized_sharpe: compute annualized Sharpe from per-bar returns
- max_drawdown: compute maximum drawdown on equity series
- crps_sample_np: sample-based CRPS estimator (numpy)
- pit_ks_np: PIT values and KS test p-value against Uniform(0,1)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy import stats


def annualized_sharpe(returns: np.ndarray, bars_per_day: float = 1440.0, annual_days: int = 252) -> float:
    """
    Annualized Sharpe ratio from per-bar returns (array-like).
    Returns NaN if std is zero or insufficient data.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return float("nan")
    mean_r = np.nanmean(r)
    std_r = np.nanstd(r, ddof=1)
    if std_r == 0 or math.isnan(std_r):
        return float("nan")
    factor = math.sqrt(bars_per_day * annual_days)
    return float((mean_r / std_r) * factor)


def max_drawdown(equity: np.ndarray) -> float:
    """
    Compute maximum drawdown from equity series (array of equity levels).
    Returns value in (0,1] representing fraction drawdown.
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-12)
    return float(np.max(dd))


def crps_sample_np(samples: np.ndarray, y: np.ndarray) -> float:
    """
    Sample-based CRPS estimator (numpy).
    samples: shape (N, B) or (N, ) for univariate single obs
    y: shape (B, ) or scalar
    Returns scalar CRPS averaged over batch/dims.
    """
    s = np.asarray(samples)
    y_arr = np.asarray(y)
    if s.ndim == 1:
        s = s[:, None]  # (N,1)
    N, B = s.shape
    if y_arr.ndim == 0:
        y_arr = np.full((B,), float(y_arr))
    if y_arr.shape[0] != B:
        # try to broadcast last value
        if y_arr.size == 1:
            y_arr = np.full((B,), float(y_arr))
        else:
            raise ValueError("y shape incompatible with samples")
    # term1
    term1 = np.mean(np.abs(s - y_arr[None, :]), axis=0).mean()
    # term2
    if N <= 2048:
        s1 = s[:, None, :]  # (N,1,B)
        s2 = s[None, :, :]  # (1,N,B)
        pair_abs = np.abs(s1 - s2)  # (N,N,B)
        term2 = 0.5 * pair_abs.mean(axis=(0, 1)).mean()
    else:
        # approximate by sampling pairs
        idx1 = np.random.randint(0, N, size=1024)
        idx2 = np.random.randint(0, N, size=1024)
        pair_abs = np.abs(s[idx1, :] - s[idx2, :])
        term2 = 0.5 * pair_abs.mean()
    return float(term1 - term2)


def pit_ks_np(samples: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Compute PIT values and KS test statistic/p-value vs Uniform(0,1).
    samples: (N, B) or (N,) ; y: (B,) or scalar
    Returns (pit_values array shape (B,), ks_stat, ks_pvalue)
    """
    s = np.asarray(samples)
    if s.ndim == 1:
        s = s[:, None]
    N, B = s.shape
    y_arr = np.asarray(y)
    if y_arr.ndim == 0:
        y_arr = np.full((B,), float(y_arr))
    if y_arr.shape[0] != B:
        raise ValueError("y shape incompatible with samples")

    pit = np.empty(B, dtype=float)
    for b in range(B):
        pit[b] = float(np.mean(s[:, b] <= y_arr[b]))
    ks_stat, ks_p = stats.kstest(pit, "uniform")
    return pit, float(ks_stat), float(ks_p)
