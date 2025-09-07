"""
Uncertainty postprocessing utilities.

Provides:
- empirical_quantiles: compute empirical quantiles from sampled trajectories
- compute_crps_numpy: sample-based CRPS estimator (numpy)
- weighted_icp_calibrate: weighted Inductive Conformal Prediction (ICP) for interval adjustment
- apply_conformal_adjustment: adjust predicted quantiles with delta
- credibility_score: compute credibility in [0,1] per spec
- compute_pit_ks: PIT values and KS p-value against Uniform(0,1)
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from loguru import logger


def empirical_quantiles(samples: np.ndarray, quantiles: Iterable[float]) -> Dict[float, np.ndarray]:
    """
    Compute empirical quantiles from samples.

    samples: array shape (N_samples, B) or (N_samples, B, D)
    quantiles: iterable of floats in (0,1)
    returns dict {q: array shape (B,) or (B,D)}
    """
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    qs = sorted(list(quantiles))
    # If samples has shape (N, B, D) or (N, B)
    if samples.ndim == 2:
        # (N, B)
        N, B = samples.shape
        out = {}
        for q in qs:
            out[q] = np.quantile(samples, q, axis=0)
        return out
    elif samples.ndim == 3:
        # (N, B, D)
        out = {}
        for q in qs:
            out[q] = np.quantile(samples, q, axis=0)  # shape (B, D)
        return out
    else:
        raise ValueError("samples must be 2D (N,B) or 3D (N,B,D)")


def compute_crps_numpy(samples: np.ndarray, y: np.ndarray) -> float:
    """
    Sample-based CRPS estimator (averaged across batch and dims).

    samples: (N, B) or (N, B, D)
    y: (B,) or (B,D) or (B,1)
    Returns scalar CRPS (float)
    """
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if samples.ndim == 2:
        # (N, B) -> treat as D=1
        N, B = samples.shape
        samples_ = samples[:, :, np.newaxis]  # (N,B,1)
        y_ = y[:, np.newaxis] if y.ndim == 1 else y
        if y_.ndim == 1:
            y_ = y_[:, np.newaxis]
    elif samples.ndim == 3:
        N, B, D = samples.shape
        samples_ = samples
        if y.ndim == 1:
            y_ = y[:, np.newaxis]
        else:
            y_ = y
    else:
        raise ValueError("samples must be 2D or 3D numpy array")

    # term1: mean_i |S_i - y|
    term1 = np.abs(samples_ - y_[np.newaxis, ...]).mean(axis=0).mean()  # scalar

    # term2: 0.5 * mean_{i,j} |S_i - S_j|
    # compute pairwise differences efficiently
    # shape (N,N,B,D) -> may be large; do pairwise via broadcasting but try memory-safe approach
    # We can compute mean absolute difference via expectation formula using sorting:
    # But for simplicity and clarity, compute via pairwise for moderate N.
    if N <= 2048:
        s1 = samples_[:, np.newaxis, ...]  # (N,1,B,D)
        s2 = samples_[np.newaxis, :, ...]  # (1,N,B,D)
        pair_abs = np.abs(s1 - s2)  # (N,N,B,D)
        term2 = 0.5 * pair_abs.mean(axis=(0, 1)).mean()
    else:
        # approximate term2 by sampling subset of pairs
        idx1 = np.random.randint(0, N, size=1024)
        idx2 = np.random.randint(0, N, size=1024)
        s1 = samples_[idx1, ...]  # (K,B,D)
        s2 = samples_[idx2, ...]  # (K,B,D)
        term2 = 0.5 * np.abs(s1 - s2).mean()

    crps = term1 - term2
    return float(crps)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Compute weighted quantile of 1D arrays.
    values: shape (n,)
    weights: shape (n,), must be non-negative and sum > 0
    q: quantile in (0,1)
    """
    if len(values) == 0:
        return 0.0
    sorter = np.argsort(values)
    values_s = values[sorter]
    weights_s = weights[sorter]
    cumw = np.cumsum(weights_s)
    total = cumw[-1]
    if total <= 0:
        # fallback to unweighted
        return float(np.quantile(values_s, q))
    threshold = q * total
    idx = np.searchsorted(cumw, threshold, side="right")
    idx = min(max(idx, 0), len(values_s) - 1)
    return float(values_s[idx])


def weighted_icp_calibrate(
    q05_hist: np.ndarray,
    q95_hist: np.ndarray,
    y_hist: np.ndarray,
    ts_hist_ms: np.ndarray,
    t_now_ms: Optional[int] = None,
    half_life_days: float = 30.0,
    alpha: float = 0.10,
    mondrian_buckets: Optional[np.ndarray] = None,
) -> Dict:
    """
    Weighted ICP calibration.

    Inputs:
      - q05_hist, q95_hist: arrays shape (M, ) predicted quantiles at time i
      - y_hist: observed targets array shape (M,)
      - ts_hist_ms: timestamps of observations in ms (M,)
      - t_now_ms: reference time in ms for weight decay (defaults to last ts)
      - half_life_days: half-life in days for exponential decay
      - alpha: miscoverage level (e.g., 0.10)
      - mondrian_buckets: optional array shape (M,) of group labels (str/int) for Mondrian ICP

    Returns dict with:
      - delta_global: scalar delta (float)
      - cov_hat (weighted coverage)
      - weights (M,)
      - delta_per_bucket (if mondrian provided)
    """
    # checks and conversions
    q05 = np.asarray(q05_hist).astype(float)
    q95 = np.asarray(q95_hist).astype(float)
    y = np.asarray(y_hist).astype(float)
    ts = np.asarray(ts_hist_ms).astype(float)

    if t_now_ms is None:
        t_now_ms = float(ts.max()) if ts.size > 0 else 0.0

    # compute nonconformity s_i = max(q05 - y, 0, y - q95)
    s = np.maximum.reduce([q05 - y, np.zeros_like(y), y - q95])

    # compute lambda from half-life
    seconds_per_day = 86400.0
    half_life_seconds = half_life_days * seconds_per_day
    if half_life_seconds <= 0:
        lam = 0.0
    else:
        lam = math.log(2.0) / half_life_seconds

    # Compute weights w_i ∝ exp(-lam * (t_now - t_i))
    delta_seconds = (t_now_ms - ts) / 1000.0
    raw_w = np.exp(-lam * delta_seconds)
    # normalize
    if raw_w.sum() <= 0:
        weights = np.ones_like(raw_w) / len(raw_w)
    else:
        weights = raw_w / raw_w.sum()

    # weighted coverage: proportion of observations inside [q05, q95], weighted
    inside = ((y >= q05) & (y <= q95)).astype(float)
    cov_hat = float((weights * inside).sum())

    result: Dict = {"cov_hat": cov_hat, "weights": weights, "alpha": alpha}

    # Global delta: weighted quantile of s at level 1 - alpha
    if s.size == 0:
        delta_global = 0.0
    else:
        delta_global = _weighted_quantile(s, weights, 1.0 - alpha)
    result["delta_global"] = float(delta_global)

    # Mondrian per-bucket deltas if buckets provided
    if mondrian_buckets is not None:
        buckets = {}
        for b in np.unique(mondrian_buckets):
            mask = mondrian_buckets == b
            s_b = s[mask]
            w_b = weights[mask]
            if s_b.size == 0:
                d_b = 0.0
            else:
                # renormalize bucket weights
                if w_b.sum() <= 0:
                    w_b_norm = np.ones_like(w_b) / len(w_b)
                else:
                    w_b_norm = w_b / w_b.sum()
                d_b = _weighted_quantile(s_b, w_b_norm, 1.0 - alpha)
            buckets[str(b)] = float(d_b)
        result["delta_mondrian"] = buckets

    # also include unweighted empirical coverage and basic stats
    result["coverage_unweighted"] = float(((y >= q05) & (y <= q95)).mean()) if y.size > 0 else float("nan")
    result["s_mean"] = float(np.mean(s)) if s.size > 0 else 0.0
    result["s_std"] = float(np.std(s)) if s.size > 0 else 0.0

    return result


def apply_conformal_adjustment(
    q05: Union[np.ndarray, float],
    q50: Union[np.ndarray, float],
    q95: Union[np.ndarray, float],
    delta: float,
    asymmetric: bool = True,
) -> Dict:
    """
    Apply conformal delta to quantiles.
    If asymmetric=True, returns q05 - delta, q95 + delta.
    If asymmetric=False (symmetric), compute center = q50 and width = (q95 - q05)/2, then expand by delta.
    Returns dict with adjusted quantiles arrays/scalars.
    """
    q05_arr = np.asarray(q05)
    q50_arr = np.asarray(q50)
    q95_arr = np.asarray(q95)

    if asymmetric:
        adj_low = q05_arr - delta
        adj_high = q95_arr + delta
    else:
        # symmetric around median
        base_half = (q95_arr - q05_arr) / 2.0
        adj_low = q50_arr - (base_half + delta)
        adj_high = q50_arr + (base_half + delta)

    return {"q05_adj": adj_low, "q50": q50_arr, "q95_adj": adj_high, "delta": float(delta)}


def _entropy_of_samples(samples: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute normalized entropy of samples per-batch aggregated.
    samples: (N, B) or (N, B, D) -> flatten across B and D for simplicity
    Returns entropy H normalized to [0,1] by dividing by log(n_bins)
    """
    s = np.asarray(samples)
    if s.ndim >= 2:
        s = s.reshape(s.shape[0], -1)
        s = s.flatten()
    if s.size == 0:
        return 0.0
    hist, _ = np.histogram(s, bins=n_bins, density=False)
    probs = hist / hist.sum()
    probs = probs[probs > 0.0]
    ent = -np.sum(probs * np.log(probs))
    max_ent = math.log(n_bins) if n_bins > 1 else 1.0
    return float(ent / max_ent)


def credibility_score(
    q05: float,
    q95: float,
    calibration_cov_hat: float,
    alpha: float,
    samples_for_entropy: Optional[np.ndarray] = None,
    sigma_realized: Optional[float] = None,
    d_regime: Optional[float] = None,
    a: float = 5.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    kappa: float = 1.0,
) -> float:
    """
    Compute credibility score in [0,1] per the spec:
    z = a * |cov_hat - target| + b * max(0, w - 1) + c * H(pi) + d * |d_regime|
    credibility = sigmoid(-z)

    where:
      - target = 1 - alpha
      - w = (q95 - q05) / (sigma_realized * kappa)
      - H(pi) entropy of sample clusters (use histogram entropy normalized)
      - d_regime is z-score of volatility etc (abs)

    Parameters a,b,c,d,kappa are tunable weights.
    """
    target = 1.0 - float(alpha)
    cov_hat = float(calibration_cov_hat)
    term1 = a * abs(cov_hat - target)

    width = float(q95 - q05)
    if sigma_realized is None or sigma_realized <= 0:
        norm_w = 0.0
    else:
        norm_w = width / (sigma_realized * kappa)
    term2 = b * max(0.0, norm_w - 1.0)

    if samples_for_entropy is None:
        H = 0.0
    else:
        H = _entropy_of_samples(samples_for_entropy, n_bins=20)
    term3 = c * H

    term4 = d * abs(float(d_regime)) if d_regime is not None else 0.0

    z = term1 + term2 + term3 + term4
    # logistic sigmoid on -z
    cred = 1.0 / (1.0 + math.exp(z))
    return float(cred)


def compute_pit_ks(samples: np.ndarray, y: np.ndarray) -> Dict:
    """
    Compute PIT values and KS test p-value against Uniform(0,1)

    samples: (N, B) or (N, B, D) - for univariate we expect D=1 or flatten
    y: (B,) or (B,D)
    Returns:
      - pit_values: array shape (B,) of PIT u = F_hat(y)
      - ks_stat: KS statistic
      - ks_pvalue: p-value of KS test
    """
    s = np.asarray(samples)
    if s.ndim == 3:
        # collapse last dim if D==1
        if s.shape[2] == 1:
            s = s[:, :, 0]
        else:
            # reduce by using last dimension target if needed
            s = s[:, :, 0]

    N, B = s.shape
    y_arr = np.asarray(y).squeeze()
    if y_arr.ndim > 1:
        y_arr = y_arr.squeeze()

    pit = []
    for b in range(B):
        samp = s[:, b]
        u = (samp <= y_arr[b]).mean()
        pit.append(u)
    pit = np.asarray(pit)

    # KS test comparing pit distribution to Uniform(0,1)
    try:
        ks_stat, ks_p = stats.kstest(pit, "uniform")
    except Exception as e:
        logger.warning("KS test failed: {}", e)
        ks_stat, ks_p = float("nan"), float("nan")

    return {"pit": pit, "ks_stat": float(ks_stat), "ks_pvalue": float(ks_p)}
