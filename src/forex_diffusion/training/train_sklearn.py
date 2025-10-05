
import argparse, json, math, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Import neural encoders
try:
    from .encoders import SklearnAutoencoder, SklearnVAE
except ImportError:
    try:
        from encoders import SklearnAutoencoder, SklearnVAE
    except ImportError:
        SklearnAutoencoder = None
        SklearnVAE = None

# Use project's MarketDataService (SQLAlchemy) to load candles for training.
# This avoids file-based adapters and ensures a single DB access mode across the app.
try:
    from forex_diffusion.services.marketdata import MarketDataService  # type: ignore
except Exception:
    # fallback relative import if run as module inside repo
    from ..services.marketdata import MarketDataService  # type: ignore

import datetime
from sqlalchemy import text

def fetch_candles_from_db(symbol: str, timeframe: str, days_history: int) -> pd.DataFrame:
    """
    Fetch candles using SQLAlchemy engine from MarketDataService.
    Returns DataFrame with columns ['ts_utc','open','high','low','close','volume'] ordered ASC.
    """
    # Get engine
    try:
        ms = MarketDataService()
        engine = getattr(ms, "engine", None)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate MarketDataService: {e}")

    if engine is None:
        raise RuntimeError("Database engine not available from MarketDataService")

    # compute start timestamp (ms)
    try:
        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        start_ms = now_ms - int(max(0, int(days_history)) * 24 * 3600 * 1000)
    except Exception:
        start_ms = None

    # Query DB
    try:
        with engine.connect() as conn:
            if start_ms is None:
                q = text(
                    "SELECT ts_utc, open, high, low, close, COALESCE(volume,0) AS volume "
                    "FROM market_data_candles "
                    "WHERE symbol = :symbol AND timeframe = :timeframe "
                    "ORDER BY ts_utc ASC"
                )
                rows = conn.execute(q, {"symbol": symbol, "timeframe": timeframe}).fetchall()
            else:
                q = text(
                    "SELECT ts_utc, open, high, low, close, COALESCE(volume,0) AS volume "
                    "FROM market_data_candles "
                    "WHERE symbol = :symbol AND timeframe = :timeframe AND ts_utc >= :start_ms "
                    "ORDER BY ts_utc ASC"
                )
                rows = conn.execute(q, {"symbol": symbol, "timeframe": timeframe, "start_ms": int(start_ms)}).fetchall()
    except Exception as e:
        raise RuntimeError(f"Failed to query market_data_candles: {e}")

    if not rows:
        raise RuntimeError(f"No candles found for {symbol} {timeframe} in last {days_history} days")

    df = pd.DataFrame(rows, columns=["ts_utc", "open", "high", "low", "close", "volume"])
    df["ts_utc"] = pd.to_numeric(df["ts_utc"], errors="coerce").astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["ts_utc", "open", "high", "low", "close"]).sort_values("ts_utc").reset_index(drop=True)
    return df[["ts_utc", "open", "high", "low", "close", "volume"]]

try:
    import ta
    _HAS_TA = True
except Exception:
    _HAS_TA = False
    warnings.warn("Package 'ta' non trovato: indicatori avanzati limitati.", RuntimeWarning)


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out["ts_utc"], unit="ms", utc=True)
    return out


def _timeframe_to_timedelta(tf: str) -> pd.Timedelta:
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
    raise ValueError(f"Timeframe non supportato: {tf}")


def _coerce_indicator_tfs(raw_value: Any) -> Dict[str, List[str]]:
    if not raw_value:
        return {}
    data: Dict[str, Any]
    if isinstance(raw_value, dict):
        data = raw_value
    else:
        try:
            data = json.loads(str(raw_value))
        except Exception:
            warnings.warn("indicator_tfs non parseable; uso dizionario vuoto", RuntimeWarning)
            return {}
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not v:
            continue
        key = str(k).lower()
        if isinstance(v, (list, tuple, set)):
            vals = [str(x) for x in v if str(x).strip()]
        else:
            vals = [str(v)]
        dedup: List[str] = []
        for tf in vals:
            tf_norm = tf.strip()
            if tf_norm and tf_norm not in dedup:
                dedup.append(tf_norm)
        if dedup:
            out[key] = dedup
    return out


def _realized_vol_feature(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return pd.DataFrame(index=df.index)
    close = pd.to_numeric(df["close"], errors="coerce").replace(0.0, np.nan).fillna(method="ffill")
    log_ret = np.log(close).diff().fillna(0.0)
    rv = log_ret.pow(2).rolling(window=window, min_periods=2).sum().pow(0.5)
    feature = pd.DataFrame({f"rv_{window}": rv}, index=df.index)
    return feature

def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe.endswith("m"): rule = f"{int(timeframe[:-1])}min"
    elif timeframe.endswith("h"): rule = f"{int(timeframe[:-1])}h"
    else: raise ValueError(f"Timeframe non supportato: {timeframe}")
    x = df.copy(); x.index = pd.to_datetime(x["ts_utc"], unit="ms", utc=True)
    ohlc = x[["open","high","low","close"]].astype(float).resample(rule, label="right").agg(
        {"open":"first","high":"max","low":"min","close":"last"}
    )
    if "volume" in x.columns: ohlc["volume"] = x["volume"].astype(float).resample(rule, label="right").sum()
    ohlc["ts_utc"] = (ohlc.index.view("int64") // 10**6)
    return ohlc.reset_index(drop=True)

def _relative_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    prev_close = df["close"].shift(1).astype(float).clip(lower=eps)
    o = df["open"].astype(float).clip(lower=eps)
    h = df["high"].astype(float).clip(lower=eps)
    l = df["low"].astype(float).clip(lower=eps)
    c = df["close"].astype(float).clip(lower=eps)
    out = pd.DataFrame(index=df.index)
    out["r_open"]  = np.log(o / prev_close)
    out["r_high"]  = np.log(h / o)
    out["r_low"]   = np.log(l / o)
    out["r_close"] = np.log(c / o)
    return out

def _temporal_feats(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["ts_utc"], unit="ms", utc=True)
    hour = ts.dt.hour; dow = ts.dt.dayofweek
    out = pd.DataFrame(index=df.index)
    out["hour_sin"] = np.sin(2*np.pi*hour/24.0); out["hour_cos"] = np.cos(2*np.pi*hour/24.0)
    out["dow_sin"]  = np.sin(2*np.pi*dow/7.0);   out["dow_cos"]  = np.cos(2*np.pi*dow/7.0)
    return out


def _indicators(df: pd.DataFrame, ind_cfg: Dict[str, Any], indicator_tfs: Dict[str, List[str]], base_tf: str, symbol: str = None, days_history: int = None) -> pd.DataFrame:
    from loguru import logger
    logger.debug(f"_indicators called with df shape: {df.shape}, base_tf: {base_tf}")
    frames: List[pd.DataFrame] = []
    base = _ensure_dt_index(df)
    base_lookup = base[["ts_utc"]].copy()
    try:
        base_delta = _timeframe_to_timedelta(base_tf)
    except Exception:
        base_delta = pd.Timedelta("1min")

    # OPTIMIZATION: Pre-fetch all timeframes needed (cache to avoid redundant DB queries)
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
                    logger.info(f"[CACHE] Pre-fetching {tf} candles from DB (will be reused for all indicators)")
                    timeframe_cache[tf] = fetch_candles_from_db(symbol, tf, days_history)
                    logger.debug(f"[CACHE] Cached {tf}: {timeframe_cache[tf].shape[0]} rows")
                else:
                    # Fallback to resample if symbol/days_history not provided
                    logger.warning(f"Symbol/days_history not provided, falling back to resample for {tf}")
                    timeframe_cache[tf] = _resample(df, tf)
            except Exception as e:
                logger.exception(f"Failed to pre-fetch {tf} candles: {e}")
                # Don't cache failed fetches

    logger.info(f"[CACHE] Pre-fetched {len(timeframe_cache)} timeframes: {list(timeframe_cache.keys())}")

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
                logger.warning(f"[CACHE MISS] TF {tf} not in cache, skipping indicator {key}_{tf}")
                continue

            tmp = _ensure_dt_index(tmp)
            logger.debug(f"Indicator {key}_{tf}: final tmp shape = {tmp.shape} (need ≥14 for ATR/RSI)")
            cols: Dict[str, pd.Series] = {}

            if not _HAS_TA:
                if key == "rsi":
                    n = int(params.get("n", 14))
                    delta = tmp["close"].diff()
                    up = delta.clip(lower=0.0).rolling(n).mean()
                    down = (-delta.clip(upper=0.0)).rolling(n).mean()
                    rs = (up / (down + 1e-12))
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
                    cols[f"rsi_{tf}_{n}"] = ta.momentum.RSIIndicator(close=tmp["close"], window=n).rsi()
                elif key == "atr":
                    n = int(params.get("n", 14))
                    cols[f"atr_{tf}_{n}"] = ta.volatility.AverageTrueRange(
                        high=tmp["high"], low=tmp["low"], close=tmp["close"], window=n
                    ).average_true_range()
                elif key == "bollinger":
                    n = int(params.get("n", 20))
                    dev = float(params.get("dev", 2.0))
                    bb = ta.volatility.BollingerBands(close=tmp["close"], window=n, window_dev=dev)
                    cols[f"bb_m_{tf}_{n}_{dev}"] = bb.bollinger_mavg()
                    cols[f"bb_h_{tf}_{n}_{dev}"] = bb.bollinger_hband()
                    cols[f"bb_l_{tf}_{n}_{dev}"] = bb.bollinger_lband()
                elif key == "macd":
                    f = int(params.get("fast", 12))
                    s = int(params.get("slow", 26))
                    sig = int(params.get("signal", 9))
                    macd = ta.trend.MACD(close=tmp["close"], window_fast=f, window_slow=s, window_sign=sig)
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
                continue
            feat = pd.DataFrame(cols)
            feat["ts_utc"] = (tmp.index.view("int64") // 10**6)
            right = _ensure_dt_index(feat)
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
                tolerance=tol
            )
            merged = merged.reset_index(drop=True).drop(columns=["ts_utc_y"], errors="ignore").rename(columns={"ts_utc_x": "ts_utc"})
            frames.append(merged.drop(columns=["ts_utc"], errors="ignore"))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)



def _build_features(candles: pd.DataFrame, args):
    """
    Build feature matrix from candles.

    CRITICAL (TASK 1.3): Validates ALL features are saved, no silent dropping.
    """
    from loguru import logger

    H = int(args.horizon)
    if H <= 0:
        raise ValueError("horizon deve essere > 0")
    c = pd.to_numeric(candles["close"], errors="coerce").astype(float)
    if len(c) <= H:
        raise ValueError(f"Non abbastanza barre ({len(c)}) per orizzonte {H}")
    y = (c.shift(-H) / c) - 1.0

    # Track all feature groups for validation
    feature_groups = {}
    feats: List[pd.DataFrame] = []

    if getattr(args, "use_relative_ohlc", True):
        rel_ohlc = _relative_ohlc(candles)
        feats.append(rel_ohlc)
        feature_groups["relative_ohlc"] = list(rel_ohlc.columns)
        logger.debug(f"[Features] Relative OHLC: {list(rel_ohlc.columns)}")

    if getattr(args, "use_temporal_features", True):
        temp_feats = _temporal_feats(candles)
        feats.append(temp_feats)
        feature_groups["temporal"] = list(temp_feats.columns)
        logger.debug(f"[Features] Temporal: {list(temp_feats.columns)}")

    rv_window = int(getattr(args, "rv_window", 0) or 0)
    if rv_window > 1:
        rv_feats = _realized_vol_feature(candles, rv_window)
        feats.append(rv_feats)
        feature_groups["realized_vol"] = list(rv_feats.columns)
        logger.debug(f"[Features] Realized Vol: {list(rv_feats.columns)}")

    # Volume Profile features (TASK 2.1)
    vp_window = int(getattr(args, "vp_window", 0) or 0)
    vp_bins = int(getattr(args, "vp_bins", 50))
    if vp_window > 1 and "volume" in candles.columns:
        from forex_diffusion.features.volume_profile import VolumeProfile
        vp_calculator = VolumeProfile(n_bins=vp_bins)
        vp_feats = vp_calculator.calculate_rolling(candles, window=vp_window, step=1)
        feats.append(vp_feats)
        feature_groups["volume_profile"] = list(vp_feats.columns)
        logger.debug(f"[Features] Volume Profile: {list(vp_feats.columns)}")

    # VSA features (TASK 2.2)
    use_vsa = getattr(args, "use_vsa", False)
    if use_vsa and "volume" in candles.columns:
        from forex_diffusion.features.vsa import VSAAnalyzer
        vsa_analyzer = VSAAnalyzer(
            volume_ma_period=int(getattr(args, "vsa_volume_ma", 20)),
            spread_ma_period=int(getattr(args, "vsa_spread_ma", 20)),
        )
        vsa_feats = vsa_analyzer.analyze_dataframe(candles)
        feats.append(vsa_feats)
        feature_groups["vsa"] = list(vsa_feats.columns)
        logger.debug(f"[Features] VSA: {list(vsa_feats.columns)}")

    # Smart Money Detection (TASK 2.3)
    use_smart_money = getattr(args, "use_smart_money", False)
    if use_smart_money and "volume" in candles.columns:
        from forex_diffusion.features.smart_money import SmartMoneyDetector
        sm_detector = SmartMoneyDetector(
            volume_ma_period=int(getattr(args, "sm_volume_ma", 20)),
            volume_std_threshold=float(getattr(args, "sm_volume_threshold", 2.0)),
        )
        sm_feats = sm_detector.analyze_dataframe(candles)
        feats.append(sm_feats)
        feature_groups["smart_money"] = list(sm_feats.columns)
        logger.debug(f"[Features] Smart Money: {list(sm_feats.columns)}")

    # HMM Regime Detection (TASK 4.1)
    use_regime = getattr(args, "use_regime_detection", False)
    if use_regime:
        from forex_diffusion.regime import HMMRegimeDetector
        regime_detector = HMMRegimeDetector(
            n_regimes=int(getattr(args, "n_regimes", 4)),
            min_history=int(getattr(args, "regime_min_history", 100)),
        )
        if len(candles) >= regime_detector.min_history:
            regime_detector.fit(candles)
            regime_feats = regime_detector.predict(candles)
            feats.append(regime_feats)
            feature_groups["regime"] = list(regime_feats.columns)
            logger.debug(f"[Features] Regime Detection: {list(regime_feats.columns)}")
        else:
            logger.warning(f"Not enough data for regime detection ({len(candles)} < {regime_detector.min_history})")

    indicator_tfs = _coerce_indicator_tfs(getattr(args, "indicator_tfs", {}))
    ind_cfg: Dict[str, Dict[str, Any]] = {}
    if "atr" in indicator_tfs:
        ind_cfg["atr"] = {"n": int(args.atr_n)}
    if "rsi" in indicator_tfs:
        ind_cfg["rsi"] = {"n": int(args.rsi_n)}
    if "bollinger" in indicator_tfs:
        ind_cfg["bollinger"] = {"n": int(args.bb_n), "dev": 2.0}
    if "macd" in indicator_tfs:
        ind_cfg["macd"] = {"fast": 12, "slow": 26, "signal": 9}
    if "donchian" in indicator_tfs:
        ind_cfg["donchian"] = {"n": 20}
    if "keltner" in indicator_tfs:
        ind_cfg["keltner"] = {"ema": 20, "atr": 10, "mult": 1.5}
    if "hurst" in indicator_tfs:
        ind_cfg["hurst"] = {"window": int(args.hurst_window)}
    if ind_cfg:
        ind_feats = _indicators(candles, ind_cfg, indicator_tfs, args.timeframe, symbol=args.symbol, days_history=args.days_history)
        feats.append(ind_feats)
        feature_groups["indicators"] = list(ind_feats.columns)
        logger.debug(f"[Features] Indicators: {list(ind_feats.columns)}")

    if not feats:
        raise RuntimeError("Nessuna feature disponibile per il training")

    # Log total features before concatenation
    total_features_expected = sum(len(cols) for cols in feature_groups.values())
    logger.info(f"[Features] Total expected: {total_features_expected} features from {len(feature_groups)} groups")

    X = pd.concat(feats, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)

    # rimuovi colonne quasi vuote prima di forzare dropna
    coverage = X.notna().mean()
    min_cov = float(getattr(args, "min_feature_coverage", 0.15) or 0.0)
    dropped_feats: List[str] = []
    if min_cov > 0.0:
        low_cov = coverage[coverage < min_cov]
        if not low_cov.empty:
            dropped_feats = list(low_cov.index)
            X = X.drop(columns=dropped_feats, errors="ignore")
            warnings.warn(f"Feature con coverage < {min_cov:.2f} drop: {dropped_feats}", RuntimeWarning)

    X = X.dropna()
    y = y.loc[X.index].dropna()
    # riallinea X e y su indice comune dopo dropna
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    if getattr(args, "warmup_bars", 0) > 0 and len(X) > args.warmup_bars:
        X = X.iloc[int(args.warmup_bars):]
        y = y.iloc[int(args.warmup_bars):]
    if X.empty or y.empty:
        raise RuntimeError("Dataset vuoto dopo il preprocessing; controlla warmup/horizon")

    # VALIDATION (TASK 1.3): Verify all expected features survived
    final_features = set(X.columns)
    expected_features = set()
    for group_feats in feature_groups.values():
        expected_features.update(group_feats)

    # Features that were expected but missing after processing
    missing_features = expected_features - final_features - set(dropped_feats)

    if missing_features:
        # CRITICAL: Some features were silently dropped
        logger.error(f"❌ FEATURE LOSS DETECTED! {len(missing_features)} features missing:")
        logger.error(f"   Missing: {sorted(missing_features)}")
        logger.error(f"   Expected groups: {list(feature_groups.keys())}")
        logger.error(f"   Dropped (low coverage): {dropped_feats}")
        raise RuntimeError(
            f"Feature loss bug: {len(missing_features)} features silently dropped! "
            f"Missing: {sorted(missing_features)}"
        )

    logger.info(f"[Features] ✓ Validation passed: {len(X.columns)} features saved")
    logger.info(f"[Features] Final feature list: {list(X.columns)}")

    meta = {
        "features": list(X.columns),
        "feature_groups": feature_groups,
        "indicator_tfs": indicator_tfs,
        "dropped_features": dropped_feats,
        "total_expected": total_features_expected,
        "total_saved": len(X.columns),
        "args_used": vars(args)
    }
    return X, y, meta


def _standardize_train_val(X: pd.DataFrame, y: pd.Series, val_frac: float):
    """
    Standardize features ensuring NO look-ahead bias.

    CRITICAL: Computes mean/std ONLY on training set, then applies to validation.
    This prevents information leakage from future data.

    Returns:
        Tuple of ((Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, scaler_metadata))
    """
    from scipy import stats

    # Split WITHOUT shuffling to maintain temporal order
    Xtr, Xva, ytr, yva = train_test_split(X.values, y.values, test_size=val_frac, shuffle=False)

    # Compute statistics ONLY on training set (NO look-ahead bias)
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma == 0] = 1.0  # Prevent division by zero

    # Apply standardization
    Xtr_scaled = (Xtr - mu) / sigma
    Xva_scaled = (Xva - mu) / sigma

    # VERIFICATION: Statistical test for look-ahead bias detection
    # If train and test distributions are too similar, likely bias present
    p_values = []
    for i in range(min(10, Xtr_scaled.shape[1])):  # Test first 10 features
        if Xtr_scaled.shape[0] > 20 and Xva_scaled.shape[0] > 20:
            # Kolmogorov-Smirnov test: different distributions should have low p-value
            _, p_val = stats.ks_2samp(Xtr_scaled[:, i], Xva_scaled[:, i])
            p_values.append(p_val)

    # Metadata for debugging
    scaler_metadata = {
        "train_size": Xtr.shape[0],
        "val_size": Xva.shape[0],
        "train_mean": mu.tolist(),
        "train_std": sigma.tolist(),
        "ks_test_p_values": p_values,
        "ks_test_median_p": float(np.median(p_values)) if p_values else None,
    }

    # WARNING: If distributions too similar, potential look-ahead bias
    if scaler_metadata["ks_test_median_p"] is not None:
        if scaler_metadata["ks_test_median_p"] > 0.8:
            warnings.warn(
                f"⚠️ POTENTIAL LOOK-AHEAD BIAS DETECTED!\n"
                f"Train/Val distributions suspiciously similar (KS median p-value={scaler_metadata['ks_test_median_p']:.3f}).\n"
                f"Expected p < 0.5 for different time periods. Verify train_test_split has shuffle=False.",
                RuntimeWarning
            )

    return ((Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, scaler_metadata))

def _fit_model(algo: str, Xtr, ytr, args):
    if   algo == "ridge":      return Ridge(alpha=float(args.alpha), random_state=args.random_state)
    elif algo == "lasso":      return Lasso(alpha=float(args.alpha), random_state=args.random_state)
    elif algo == "elasticnet": return ElasticNet(alpha=float(args.alpha), l1_ratio=float(args.l1_ratio), random_state=args.random_state)
    elif algo == "rf":         return RandomForestRegressor(n_estimators=int(args.n_estimators), max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=args.random_state)
    else: raise ValueError(f"Algo non supportato: {algo}")


def _optimize_genetic_basic(algo: str, Xtr, ytr, Xva, yva, args):
    """
    Simple genetic algorithm for single-objective optimization (minimize MAE).
    Search space: hyperparameters specific to each algorithm.
    Returns: best_model, best_params, best_score
    """
    from scipy.optimize import differential_evolution

    print(f"[Optimization] Running genetic-basic for {algo} (gen={args.gen}, pop={args.pop})")

    # Define search space based on algorithm
    if algo == "ridge":
        bounds = [(1e-6, 10.0)]  # alpha
        param_names = ["alpha"]
    elif algo == "lasso":
        bounds = [(1e-6, 10.0)]  # alpha
        param_names = ["alpha"]
    elif algo == "elasticnet":
        bounds = [(1e-6, 10.0), (0.01, 0.99)]  # alpha, l1_ratio
        param_names = ["alpha", "l1_ratio"]
    elif algo == "rf":
        bounds = [(50, 500), (2, 20), (2, 50)]  # n_estimators, max_depth, min_samples_leaf
        param_names = ["n_estimators", "max_depth", "min_samples_leaf"]
    else:
        raise ValueError(f"Optimization not supported for algo: {algo}")

    # Objective function: minimize validation MAE
    def objective(params):
        try:
            if algo == "ridge":
                model = Ridge(alpha=params[0], random_state=args.random_state)
            elif algo == "lasso":
                model = Lasso(alpha=params[0], random_state=args.random_state)
            elif algo == "elasticnet":
                model = ElasticNet(alpha=params[0], l1_ratio=params[1], random_state=args.random_state)
            elif algo == "rf":
                model = RandomForestRegressor(
                    n_estimators=int(params[0]),
                    max_depth=int(params[1]) if params[1] < 50 else None,
                    min_samples_leaf=int(params[2]),
                    n_jobs=-1,
                    random_state=args.random_state
                )

            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            mae = mean_absolute_error(yva, pred)
            return mae
        except Exception as e:
            print(f"[Optimization] Error evaluating params {params}: {e}")
            return 1e10  # Return high error on failure

    # Run differential evolution (genetic algorithm)
    result = differential_evolution(
        objective,
        bounds,
        maxiter=args.gen,
        popsize=args.pop,
        seed=args.random_state,
        polish=False,
        workers=1,
        updating='deferred'
    )

    best_params = result.x
    best_score = result.fun

    # Build best model with optimal params
    if algo == "ridge":
        best_model = Ridge(alpha=best_params[0], random_state=args.random_state)
    elif algo == "lasso":
        best_model = Lasso(alpha=best_params[0], random_state=args.random_state)
    elif algo == "elasticnet":
        best_model = ElasticNet(alpha=best_params[0], l1_ratio=best_params[1], random_state=args.random_state)
    elif algo == "rf":
        best_model = RandomForestRegressor(
            n_estimators=int(best_params[0]),
            max_depth=int(best_params[1]) if best_params[1] < 50 else None,
            min_samples_leaf=int(best_params[2]),
            n_jobs=-1,
            random_state=args.random_state
        )

    best_model.fit(Xtr, ytr)

    params_dict = {name: val for name, val in zip(param_names, best_params)}
    print(f"[Optimization] Best params: {params_dict}, Best MAE: {best_score:.6f}")

    return best_model, params_dict, best_score


def _optimize_nsga2(algo: str, Xtr, ytr, Xva, yva, args):
    """
    NSGA-II multi-objective optimization.
    Objectives: 1) Minimize MAE, 2) Minimize model complexity
    Returns: best_model (from Pareto front), best_params, best_score
    """
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    print(f"[Optimization] Running NSGA-II for {algo} (gen={args.gen}, pop={args.pop})")

    # Define search space and complexity metric
    if algo == "ridge":
        xl = np.array([1e-6])  # alpha lower bound
        xu = np.array([10.0])  # alpha upper bound
        param_names = ["alpha"]

        def complexity_metric(params):
            # Ridge: complexity ≈ 1/alpha (less regularization = more complex)
            return 1.0 / (params[0] + 1e-9)

    elif algo == "lasso":
        xl = np.array([1e-6])
        xu = np.array([10.0])
        param_names = ["alpha"]

        def complexity_metric(params):
            return 1.0 / (params[0] + 1e-9)

    elif algo == "elasticnet":
        xl = np.array([1e-6, 0.01])  # alpha, l1_ratio
        xu = np.array([10.0, 0.99])
        param_names = ["alpha", "l1_ratio"]

        def complexity_metric(params):
            return 1.0 / (params[0] + 1e-9)

    elif algo == "rf":
        xl = np.array([50, 2, 2])  # n_estimators, max_depth, min_samples_leaf
        xu = np.array([500, 20, 50])
        param_names = ["n_estimators", "max_depth", "min_samples_leaf"]

        def complexity_metric(params):
            # RF: complexity ≈ n_estimators × max_depth / min_samples_leaf
            n_est, max_d, min_leaf = params
            return (n_est * max_d) / (min_leaf + 1e-9)
    else:
        raise ValueError(f"NSGA-II not supported for algo: {algo}")

    # Define multi-objective problem
    class HyperparameterProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(xl), n_obj=2, n_constr=0, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            try:
                # Build model with current params
                if algo == "ridge":
                    model = Ridge(alpha=x[0], random_state=args.random_state)
                elif algo == "lasso":
                    model = Lasso(alpha=x[0], random_state=args.random_state)
                elif algo == "elasticnet":
                    model = ElasticNet(alpha=x[0], l1_ratio=x[1], random_state=args.random_state)
                elif algo == "rf":
                    model = RandomForestRegressor(
                        n_estimators=int(x[0]),
                        max_depth=int(x[1]) if x[1] < 50 else None,
                        min_samples_leaf=int(x[2]),
                        n_jobs=-1,
                        random_state=args.random_state
                    )

                model.fit(Xtr, ytr)
                pred = model.predict(Xva)
                mae = mean_absolute_error(yva, pred)
                complexity = complexity_metric(x)

                # Objectives: minimize MAE and complexity
                out["F"] = [mae, complexity]
            except Exception as e:
                print(f"[NSGA-II] Error evaluating {x}: {e}")
                out["F"] = [1e10, 1e10]  # Penalize failures

    problem = HyperparameterProblem()

    # Configure NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=args.pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', args.gen),
        seed=args.random_state,
        verbose=False
    )

    # Extract Pareto front
    if res.F is None or len(res.F) == 0:
        raise RuntimeError("NSGA-II optimization failed to find solutions")

    # Select solution from Pareto front: lowest MAE with reasonable complexity
    # Strategy: pick solution with lowest MAE among those with complexity < median
    complexity_values = res.F[:, 1]
    mae_values = res.F[:, 0]
    complexity_threshold = np.median(complexity_values)

    # Filter solutions below complexity threshold
    mask = complexity_values <= complexity_threshold
    if mask.sum() == 0:
        # If no solutions below threshold, just pick lowest MAE
        best_idx = np.argmin(mae_values)
    else:
        filtered_mae = mae_values[mask]
        filtered_indices = np.where(mask)[0]
        best_idx = filtered_indices[np.argmin(filtered_mae)]

    best_params = res.X[best_idx]
    best_score = mae_values[best_idx]
    best_complexity = complexity_values[best_idx]

    # Build final model
    if algo == "ridge":
        best_model = Ridge(alpha=best_params[0], random_state=args.random_state)
    elif algo == "lasso":
        best_model = Lasso(alpha=best_params[0], random_state=args.random_state)
    elif algo == "elasticnet":
        best_model = ElasticNet(alpha=best_params[0], l1_ratio=best_params[1], random_state=args.random_state)
    elif algo == "rf":
        best_model = RandomForestRegressor(
            n_estimators=int(best_params[0]),
            max_depth=int(best_params[1]) if best_params[1] < 50 else None,
            min_samples_leaf=int(best_params[2]),
            n_jobs=-1,
            random_state=args.random_state
        )

    best_model.fit(Xtr, ytr)

    params_dict = {name: val for name, val in zip(param_names, best_params)}
    print(f"[NSGA-II] Pareto front size: {len(res.F)}")
    print(f"[NSGA-II] Best params: {params_dict}")
    print(f"[NSGA-II] Best MAE: {best_score:.6f}, Complexity: {best_complexity:.2f}")

    return best_model, params_dict, best_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--algo", choices=["ridge","lasso","elasticnet","rf"], required=True)
    ap.add_argument("--pca", type=int, default=0, help="PCA components (0=disabled)")
    ap.add_argument("--encoder", type=str, choices=["none", "pca", "autoencoder", "vae", "latents"], default="none", help="Encoder type")
    ap.add_argument("--latent_dim", type=int, default=16, help="Latent dimension for autoencoder/VAE")
    ap.add_argument("--encoder_epochs", type=int, default=50, help="Training epochs for neural encoders")
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--warmup_bars", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--l1_ratio", type=float, default=0.5)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--days_history", type=int, default=60)
    ap.add_argument("--atr_n", type=int, default=14)
    ap.add_argument("--rsi_n", type=int, default=14)
    ap.add_argument("--bb_n", type=int, default=20)
    ap.add_argument("--hurst_window", type=int, default=64)
    ap.add_argument("--rv_window", type=int, default=60)
    ap.add_argument("--min_feature_coverage", type=float, default=0.15)
    ap.add_argument("--indicator_tfs", type=str, default="{}")
    ap.add_argument("--use_relative_ohlc", action="store_true", default=True)
    ap.add_argument("--use_temporal_features", action="store_true", default=True)

    # Additional feature parameters (from new training tab)
    ap.add_argument("--returns_window", type=int, default=5, help="Window for returns calculation")
    ap.add_argument("--session_overlap", type=int, default=30, help="Session overlap in minutes")
    ap.add_argument("--higher_tf", type=str, default="15m", help="Higher timeframe for candlestick patterns")
    ap.add_argument("--vp_bins", type=int, default=50, help="Number of bins for volume profile")
    ap.add_argument("--vp_window", type=int, default=100, help="Window size for volume profile calculation")
    ap.add_argument("--use_vsa", action="store_true", help="Enable Volume Spread Analysis (VSA)")
    ap.add_argument("--vsa_volume_ma", type=int, default=20, help="Volume MA period for VSA")
    ap.add_argument("--vsa_spread_ma", type=int, default=20, help="Spread MA period for VSA")
    ap.add_argument("--use_smart_money", action="store_true", help="Enable Smart Money Detection")
    ap.add_argument("--sm_volume_ma", type=int, default=20, help="Volume MA period for smart money")
    ap.add_argument("--sm_volume_threshold", type=float, default=2.0, help="Volume z-score threshold")
    ap.add_argument("--use_regime_detection", action="store_true", help="Enable HMM Regime Detection")
    ap.add_argument("--n_regimes", type=int, default=4, help="Number of market regimes")
    ap.add_argument("--regime_min_history", type=int, default=100, help="Min bars for regime training")

    # Optimization parameters
    ap.add_argument("--optimization", type=str, choices=["none", "genetic-basic", "nsga2"], default="none", help="Hyperparameter optimization strategy")
    ap.add_argument("--gen", type=int, default=5, help="Number of generations for genetic algorithm")
    ap.add_argument("--pop", type=int, default=8, help="Population size for genetic algorithm")

    # GPU Support
    ap.add_argument("--use-gpu", action="store_true", default=False, help="Use GPU for encoder training (requires CUDA)")

    args = ap.parse_args()

    candles = fetch_candles_from_db(args.symbol, args.timeframe, args.days_history)
    req = {"ts_utc","open","high","low","close"}
    if not isinstance(candles, pd.DataFrame) or not req.issubset(candles.columns):
        raise ValueError(f"Candles mancanti colonne: {req}")

    X, y, meta = _build_features(candles, args)
    (Xtr, ytr), (Xva, yva), (mu, sigma, scaler_metadata) = _standardize_train_val(X, y, args.val_frac)

    # Log scaler metadata for debugging
    print(f"[Scaler] Train size: {scaler_metadata['train_size']}, Val size: {scaler_metadata['val_size']}")
    if scaler_metadata.get('ks_test_median_p') is not None:
        print(f"[Scaler] KS test median p-value: {scaler_metadata['ks_test_median_p']:.4f} (< 0.5 expected for no bias)")

    # Ensure numpy arrays
    Xtr = np.asarray(Xtr, dtype=float)
    ytr = np.asarray(ytr, dtype=float)
    Xva = np.asarray(Xva, dtype=float)
    yva = np.asarray(yva, dtype=float)

    # Remove rows containing NaN / inf in X or y for both train and val
    def _filter_finite(Xa, ya):
        if Xa.ndim == 1:
            ok = np.isfinite(Xa) & np.isfinite(ya)
            Xa_f = Xa[ok]
            ya_f = ya[ok]
            return Xa_f.reshape(-1, 1), ya_f
        mask = np.isfinite(Xa).all(axis=1) & np.isfinite(ya)
        return Xa[mask], ya[mask]

    Xtr_f, ytr_f = _filter_finite(Xtr, ytr)
    Xva_f, yva_f = _filter_finite(Xva, yva)

    # Log removed rows if any
    removed_tr = Xtr.shape[0] - Xtr_f.shape[0]
    removed_va = Xva.shape[0] - Xva_f.shape[0]
    if removed_tr > 0:
        warnings.warn(f"Removed {removed_tr} training rows containing NaN/Inf after standardization.", RuntimeWarning)
    if removed_va > 0:
        warnings.warn(f"Removed {removed_va} validation rows containing NaN/Inf after standardization.", RuntimeWarning)

    # Validate enough data remains
    if Xtr_f.shape[0] < 2:
        raise RuntimeError("Not enough training rows after NaN/Inf filtering; aborting training.")
    if Xva_f.shape[0] < 1:
        raise RuntimeError("Not enough validation rows after NaN/Inf filtering; aborting training.")

    Xtr, ytr = Xtr_f, ytr_f
    Xva, yva = Xva_f, yva_f

    # Apply encoder/dimensionality reduction
    encoder_model = None
    encoder_type = args.encoder if hasattr(args, 'encoder') else "none"

    # Legacy support: if --pca is set but --encoder is "none", use PCA
    if int(args.pca) > 0 and encoder_type == "none":
        encoder_type = "pca"

    if encoder_type == "pca":
        # PCA encoder
        ncomp = int(args.latent_dim) if hasattr(args, 'latent_dim') else int(args.pca)
        if ncomp <= 0:
            ncomp = 16  # Default
        ncomp = min(ncomp, Xtr.shape[1], Xtr.shape[0])
        if ncomp > 0:
            print(f"[Encoder] Training PCA with {ncomp} components...")
            encoder_model = PCA(n_components=ncomp, whiten=False, random_state=args.random_state)
            Xtr = encoder_model.fit_transform(Xtr)
            Xva = encoder_model.transform(Xva)
            print(f"[Encoder] PCA reduced features: {Xtr_f.shape[1]} -> {Xtr.shape[1]}")

    elif encoder_type == "autoencoder":
        # Autoencoder
        if SklearnAutoencoder is None:
            raise RuntimeError("Autoencoder not available. Install PyTorch: pip install torch")

        latent_dim = int(args.latent_dim) if hasattr(args, 'latent_dim') else 16
        epochs = int(args.encoder_epochs) if hasattr(args, 'encoder_epochs') else 50
        use_gpu = getattr(args, 'use_gpu', False)
        device_str = "cuda" if use_gpu else "cpu"
        print(f"[Encoder] Training Autoencoder with {latent_dim} latent dimensions for {epochs} epochs on {device_str.upper()}...")
        encoder_model = SklearnAutoencoder(
            latent_dim=latent_dim,
            hidden_dims=[128, 64],
            epochs=epochs,
            batch_size=64,
            learning_rate=0.001,
            device=device_str,
            verbose=True
        )
        Xtr = encoder_model.fit_transform(Xtr)
        Xva = encoder_model.transform(Xva)
        print(f"[Encoder] Autoencoder reduced features: {Xtr_f.shape[1]} -> {Xtr.shape[1]}")

    elif encoder_type == "vae":
        # Variational Autoencoder
        if SklearnVAE is None:
            raise RuntimeError("VAE not available. Install PyTorch: pip install torch")

        latent_dim = int(args.latent_dim) if hasattr(args, 'latent_dim') else 16
        epochs = int(args.encoder_epochs) if hasattr(args, 'encoder_epochs') else 50
        use_gpu = getattr(args, 'use_gpu', False)
        device_str = "cuda" if use_gpu else "cpu"
        print(f"[Encoder] Training VAE with {latent_dim} latent dimensions for {epochs} epochs on {device_str.upper()}...")
        encoder_model = SklearnVAE(
            latent_dim=latent_dim,
            hidden_dims=[128, 64],
            epochs=epochs,
            batch_size=64,
            learning_rate=0.001,
            device=device_str,
            beta=1.0,
            verbose=True
        )
        Xtr = encoder_model.fit_transform(Xtr)
        Xva = encoder_model.transform(Xva)
        print(f"[Encoder] VAE reduced features: {Xtr_f.shape[1]} -> {Xtr.shape[1]}")

    elif encoder_type == "latents":
        # Placeholder for pre-trained encoder (user must load separately)
        print("[Encoder] Using 'latents' mode - no encoder training, expecting pre-trained encoder at inference time")
        encoder_model = None

    else:
        # No encoder
        encoder_model = None

    # Train model with or without optimization
    optimization_strategy = getattr(args, 'optimization', 'none')
    optimized_params = {}

    if optimization_strategy == 'genetic-basic':
        print(f"[Training] Using genetic-basic optimization")
        model, optimized_params, mae = _optimize_genetic_basic(args.algo, Xtr, ytr, Xva, yva, args)
    elif optimization_strategy == 'nsga2':
        print(f"[Training] Using NSGA-II multi-objective optimization")
        model, optimized_params, mae = _optimize_nsga2(args.algo, Xtr, ytr, Xva, yva, args)
    else:
        # Standard training without optimization
        print(f"[Training] Training {args.algo} with default parameters (no optimization)")
        model = _fit_model(args.algo, Xtr, ytr, args)
        model.fit(Xtr, ytr)
        val_pred = model.predict(Xva)
        mae = float(mean_absolute_error(yva, val_pred))

    out_dir = Path(args.artifacts_dir) / "models"; out_dir.mkdir(parents=True, exist_ok=True)

    # Build run name with encoder info
    encoder_suffix = ""
    if encoder_type == "pca":
        ncomp = encoder_model.n_components_ if encoder_model else 0
        encoder_suffix = f"_pca{ncomp}"
    elif encoder_type == "autoencoder":
        encoder_suffix = f"_ae{args.latent_dim}"
    elif encoder_type == "vae":
        encoder_suffix = f"_vae{args.latent_dim}"
    elif encoder_type == "latents":
        encoder_suffix = "_latents"

    run_name = f"{args.symbol.replace('/','')}_{args.timeframe}_d{args.days_history}_h{args.horizon}_{args.algo}{encoder_suffix}"

    # Save payload with encoder, optimization info, and scaler metadata
    payload = {
        "model_type": args.algo,
        "model": model,
        "scaler_mu": mu,
        "scaler_sigma": sigma,
        "scaler_metadata": scaler_metadata,  # NEW: metadata for bias verification
        "encoder": encoder_model,  # Generic 'encoder' key instead of just 'pca'
        "pca": encoder_model if encoder_type == "pca" else None,  # Keep 'pca' for backward compatibility
        "encoder_type": encoder_type,
        "features": meta["features"],
        "indicator_tfs": meta["indicator_tfs"],
        "params_used": meta["args_used"],
        "val_mae": mae,
        "optimization_strategy": optimization_strategy,
        "optimized_params": optimized_params if optimization_strategy != 'none' else None
    }

    out_path = out_dir / f"{run_name}.pkl"
    dump(payload, out_path, compress=3)
    print(f"[OK] saved model to {out_path} (val_mae={mae:.6f}, encoder={encoder_type})")

if __name__ == "__main__": main()
