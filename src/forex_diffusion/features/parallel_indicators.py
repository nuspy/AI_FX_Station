"""
Parallel indicator computation for multi-timeframe analysis.

OPT-002: Parallelizes indicator calculation across timeframes using ThreadPoolExecutor.
Expected 2-4x speedup for workloads with 10+ indicator×timeframe combinations.
"""
from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Try importing TA library
try:
    import ta

    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    warnings.warn(
        "Package 'ta' not found: limited indicator support", RuntimeWarning
    )


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has DateTime index from ts_utc column."""
    out = df.copy()
    out.index = pd.to_datetime(out["ts_utc"], unit="ms", utc=True)
    return out


def _timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    """Convert timeframe string to pandas Timedelta."""
    tf = str(tf).strip().lower()
    if tf.endswith("ms"):
        return pd.Timedelta(milliseconds=int(tf[:-2]))
    if tf.endswith("s") and not tf.endswith("ms"):
        return pd.Timedelta(seconds=int(tf[:-1]))
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {tf}")


def compute_single_indicator(
    indicator_name: str,
    params: Dict[str, Any],
    tf: str,
    df_cache: pd.DataFrame,
    base_lookup: pd.DataFrame,
    base_tf: str,
    base_delta: pd.Timedelta,
) -> Optional[pd.DataFrame]:
    """
    Compute a single indicator for a single timeframe.

    This function is designed to be called in parallel via ThreadPoolExecutor.

    Args:
        indicator_name: Indicator name (e.g., "rsi", "atr", "macd")
        params: Indicator parameters dict
        tf: Timeframe for this computation
        df_cache: Cached OHLCV data for this timeframe
        base_lookup: Base timeframe DataFrame for merging
        base_tf: Base timeframe string
        base_delta: Base timeframe as Timedelta

    Returns:
        DataFrame with computed indicator columns, or None on failure
    """
    key = str(indicator_name).lower()

    try:
        tmp = df_cache.copy()
        tmp = _ensure_dt_index(tmp)

        logger.debug(
            f"[Parallel] Computing indicator {key}_{tf}: tmp shape = {tmp.shape}"
        )

        cols: Dict[str, pd.Series] = {}

        # Compute indicator based on type
        if not _HAS_TA:
            # Fallback implementations without TA library
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
            # Full implementations with TA library
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

                def _h(x):
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
            logger.warning(f"[Parallel] No columns generated for {key}_{tf}")
            return None

        # Create feature DataFrame
        feat = pd.DataFrame(cols)
        feat["ts_utc"] = tmp.index.view("int64") // 10**6
        right = _ensure_dt_index(feat)

        # Merge with base timeframe
        try:
            tol = max(_timeframe_to_timedelta(tf), base_delta)
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

        result = merged.drop(columns=["ts_utc"], errors="ignore")
        logger.debug(f"[Parallel] ✓ {key}_{tf}: generated {len(result.columns)} columns")

        return result

    except Exception as e:
        logger.error(f"[Parallel] Failed to compute {key}_{tf}: {e}")
        return None


def indicators_parallel(
    df: pd.DataFrame,
    ind_cfg: Dict[str, Any],
    indicator_tfs: Dict[str, List[str]],
    base_tf: str,
    timeframe_cache: Dict[str, pd.DataFrame],
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Compute indicators in parallel using ThreadPoolExecutor.

    OPT-002: Parallelizes indicator computation for 2-4x speedup.

    Args:
        df: Base timeframe DataFrame
        ind_cfg: Indicator configuration dict
        indicator_tfs: Mapping of indicator name → list of timeframes
        base_tf: Base timeframe string
        timeframe_cache: Pre-fetched timeframe data cache
        max_workers: Number of parallel workers (default: 4)

    Returns:
        DataFrame with all computed indicator features
    """
    from loguru import logger

    logger.info(
        f"[Parallel Indicators] Starting parallel computation with {max_workers} workers"
    )

    base = _ensure_dt_index(df)
    base_lookup = base[["ts_utc"]].copy()

    try:
        base_delta = _timeframe_to_timedelta(base_tf)
    except Exception:
        base_delta = pd.Timedelta("1min")

    # Collect all (indicator, timeframe) combinations
    tasks = []
    for name, params in ind_cfg.items():
        key = str(name).lower()
        tfs = indicator_tfs.get(key) or indicator_tfs.get(name, []) or [base_tf]

        for tf in tfs:
            if tf not in timeframe_cache:
                logger.warning(
                    f"[Parallel] TF {tf} not in cache, skipping indicator {key}_{tf}"
                )
                continue

            tasks.append((name, params, tf, timeframe_cache[tf]))

    if not tasks:
        logger.warning("[Parallel Indicators] No tasks to execute")
        return pd.DataFrame(index=df.index)

    logger.info(f"[Parallel Indicators] Queued {len(tasks)} indicator×timeframe tasks")

    # Execute in parallel
    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for name, params, tf, df_cache in tasks:
            future = executor.submit(
                compute_single_indicator,
                name,
                params,
                tf,
                df_cache,
                base_lookup,
                base_tf,
                base_delta,
            )
            futures[future] = (name, tf)

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            name, tf = futures[future]

            try:
                result = future.result()
                if result is not None and not result.empty:
                    frames.append(result)
                    logger.debug(
                        f"[Parallel] [{completed}/{len(tasks)}] ✓ {name}_{tf}"
                    )
                else:
                    logger.warning(
                        f"[Parallel] [{completed}/{len(tasks)}] ✗ {name}_{tf} returned empty"
                    )
            except Exception as e:
                logger.error(
                    f"[Parallel] [{completed}/{len(tasks)}] ✗ {name}_{tf} failed: {e}"
                )

    logger.info(
        f"[Parallel Indicators] Completed: {len(frames)}/{len(tasks)} indicators succeeded"
    )

    if not frames:
        return pd.DataFrame(index=df.index)

    return pd.concat(frames, axis=1)
