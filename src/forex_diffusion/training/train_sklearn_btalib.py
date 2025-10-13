"""
Enhanced training system with bta-lib indicators integration
Supports 80+ professional indicators with smart data filtering and configuration management
"""

import argparse, json, math, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Import new indicators system
from ..features.indicators_btalib import BTALibIndicators, IndicatorConfig, IndicatorCategories, DataRequirement

# Use project's MarketDataService (SQLAlchemy) to load candles for training.
try:
    from forex_diffusion.services.marketdata import MarketDataService  # type: ignore
except Exception:
    from ..services.marketdata import MarketDataService  # type: ignore

import datetime
from sqlalchemy import text


def fetch_candles_from_db(symbol: str, timeframe: str, days_history: int) -> pd.DataFrame:
    """
    Fetch candles using SQLAlchemy engine from MarketDataService.
    Returns DataFrame with columns ['ts_utc','open','high','low','close','volume'] ordered ASC.
    """
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


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has datetime index"""
    out = df.copy()
    out.index = pd.to_datetime(out["ts_utc"], unit="ms", utc=True)
    return out


def _timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    """Convert timeframe string to pandas Timedelta"""
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
    if tf.endswith("w"):
        return pd.Timedelta(weeks=int(tf[:-1]))
    try:
        return pd.Timedelta(minutes=int(tf))
    except Exception:
        return pd.Timedelta(minutes=5)


def _relative_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Create relative OHLC features"""
    return pd.DataFrame({
        "rel_high": (df["high"] / df["close"]) - 1.0,
        "rel_low": (df["low"] / df["close"]) - 1.0,
        "rel_open": (df["open"] / df["close"]) - 1.0,
        "hl_pct": (df["high"] - df["low"]) / df["close"],
        "oc_pct": (df["open"] - df["close"]) / df["close"],
    }, index=df.index)


def _temporal_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features"""
    dt_idx = _ensure_dt_index(df).index
    return pd.DataFrame({
        "hour": dt_idx.hour,
        "dow": dt_idx.dayofweek,
        "dom": dt_idx.day,
        "month": dt_idx.month,
        "is_weekend": (dt_idx.dayofweek >= 5).astype(int),
    }, index=df.index)


def _realized_vol_feature(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Create realized volatility feature"""
    c = df["close"].astype(float)
    log_returns = np.log(c / c.shift(1))
    rv = log_returns.rolling(window).std() * np.sqrt(window)
    return pd.DataFrame({"realized_vol": rv}, index=df.index)


class BTALibIndicatorsTraining:
    """
    Enhanced training class that integrates bta-lib indicators system
    with intelligent data filtering and configuration management
    """

    def __init__(self, indicators_config: Optional[Dict[str, Any]] = None):
        """
        Initialize training system with indicators configuration

        Args:
            indicators_config: Configuration dict with available_data and indicators settings
        """
        self.indicators_config = indicators_config or {}
        self.available_data = self.indicators_config.get('available_data', ['open', 'high', 'low', 'close'])
        self.indicators_system = BTALibIndicators(self.available_data)

        # Load indicators configuration if provided
        if 'indicators' in self.indicators_config:
            self.indicators_system.load_config_dict(self.indicators_config['indicators'])

    def calculate_indicators_multi_timeframe(self,
                                           df: pd.DataFrame,
                                           timeframes: Dict[str, List[str]],
                                           base_tf: str,
                                           symbol: str = None,
                                           days_history: int = None) -> pd.DataFrame:
        """
        Calculate indicators across multiple timeframes using bta-lib

        Args:
            df: Base OHLC(V) DataFrame
            timeframes: Dict mapping timeframe strings to lists of indicator names
            base_tf: Base timeframe for alignment
            symbol: Symbol to fetch from DB (for direct queries)
            days_history: Number of days history to fetch from DB

        Returns:
            DataFrame with all calculated indicators
        """
        from loguru import logger
        frames: List[pd.DataFrame] = []
        base = _ensure_dt_index(df)
        base_lookup = base[["ts_utc"]].copy()
        base_delta = _timeframe_to_timedelta(base_tf)

        enabled_indicators = self.indicators_system.get_enabled_indicators()

        for tf, indicator_names in timeframes.items():
            if tf == base_tf:
                # Same timeframe - use original data
                tf_data = df.copy()
            else:
                # NEW STRATEGY: Query DB directly for this timeframe instead of resampling
                # This avoids data quality issues with gaps in resampled data
                try:
                    if symbol and days_history:
                        logger.info(f"Fetching {tf} candles directly from DB (no resampling)")
                        from .train_sklearn import fetch_candles_from_db
                        tf_data = fetch_candles_from_db(symbol, tf, days_history)
                        logger.debug(f"After DB fetch for {tf}, shape: {tf_data.shape}")
                    else:
                        # Fallback to resample if symbol/days_history not provided (backward compatibility)
                        logger.warning(f"Symbol/days_history not provided, falling back to resample for {tf}")
                        tf_delta = _timeframe_to_timedelta(tf)
                        resampled = base.resample(tf_delta).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()

                        # Convert back to expected format
                        tf_data = resampled.reset_index()
                        tf_data['ts_utc'] = (tf_data['index'].view('int64') // 10**6)
                        tf_data = tf_data.drop('index', axis=1)
                except Exception as e:
                    logger.exception(f"Failed to fetch {tf} candles from DB: {e}")
                    warnings.warn(f"Failed to fetch/resample to {tf}: {e}")
                    continue

            # Calculate indicators for this timeframe
            cols = {}

            for indicator_name in indicator_names:
                if indicator_name not in enabled_indicators:
                    warnings.warn(f"Indicator {indicator_name} not enabled or available")
                    continue

                config = enabled_indicators[indicator_name]

                try:
                    # Calculate indicator using bta-lib
                    indicator_results = self.indicators_system.calculate_indicator(
                        tf_data, indicator_name
                    )

                    # Add results with timeframe suffix
                    for result_name, result_series in indicator_results.items():
                        col_name = f"{result_name}_{tf}"
                        cols[col_name] = result_series

                except Exception as e:
                    warnings.warn(f"Failed to calculate {indicator_name} for {tf}: {e}")
                    continue

            if not cols:
                continue

            # Create feature DataFrame
            feat = pd.DataFrame(cols)
            feat["ts_utc"] = (tf_data.index.view("int64") // 10**6) if hasattr(tf_data, 'index') else tf_data["ts_utc"]
            right = _ensure_dt_index(feat)

            try:
                tol = max(_timeframe_to_timedelta(tf), base_delta)
            except Exception:
                tol = base_delta

            # Align with base timeframe
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

    def build_features(self, candles: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Build comprehensive feature set including bta-lib indicators

        Args:
            candles: OHLC(V) DataFrame
            args: Training arguments

        Returns:
            Tuple of (features, targets, metadata)
        """
        H = int(args.horizon)
        if H <= 0:
            raise ValueError("horizon deve essere > 0")

        c = pd.to_numeric(candles["close"], errors="coerce").astype(float)
        if len(c) <= H:
            raise ValueError(f"Non abbastanza barre ({len(c)}) per orizzonte {H}")

        # Target: future returns
        y = (c.shift(-H) / c) - 1.0

        feats: List[pd.DataFrame] = []

        # Basic features
        if getattr(args, "use_relative_ohlc", True):
            feats.append(_relative_ohlc(candles))

        if getattr(args, "use_temporal_features", True):
            feats.append(_temporal_feats(candles))

        # Realized volatility
        rv_window = int(getattr(args, "rv_window", 0) or 0)
        if rv_window > 1:
            feats.append(_realized_vol_feature(candles, rv_window))

        # Enhanced indicators using bta-lib
        indicator_tfs = getattr(args, "indicator_tfs", {})
        if indicator_tfs and self.indicators_system:
            try:
                indicators_df = self.calculate_indicators_multi_timeframe(
                    candles, indicator_tfs, args.timeframe, symbol=args.symbol, days_history=args.days
                )
                if not indicators_df.empty:
                    feats.append(indicators_df)
            except Exception as e:
                warnings.warn(f"Failed to calculate indicators: {e}")

        if not feats:
            raise RuntimeError("Nessuna feature disponibile per il training")

        X = pd.concat(feats, axis=1)
        X = X.replace([np.inf, -np.inf], np.nan)

        # Remove features with low coverage
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

        # Align X and y on common index
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Apply warmup
        if getattr(args, "warmup_bars", 0) > 0 and len(X) > args.warmup_bars:
            X = X.iloc[int(args.warmup_bars):]
            y = y.iloc[int(args.warmup_bars):]

        if X.empty or y.empty:
            raise RuntimeError("Dataset vuoto dopo il preprocessing; controlla warmup/horizon")

        # Create comprehensive metadata
        enabled_indicators = self.indicators_system.get_enabled_indicators()
        meta = {
            "features": list(X.columns),
            "indicator_tfs": indicator_tfs,
            "dropped_features": dropped_feats,
            "available_data": self.available_data,
            "enabled_indicators": {
                name: {
                    "enabled": config.enabled,
                    "weight": config.weight,
                    "data_requirement": config.data_requirement.value,
                    "category": config.category
                }
                for name, config in enabled_indicators.items()
            },
            "indicators_config": self.indicators_config,
            "args_used": vars(args)
        }

        return X, y, meta

    def get_indicators_summary(self) -> Dict[str, Any]:
        """Get summary of indicators configuration"""
        if not self.indicators_system:
            return {}

        enabled = self.indicators_system.get_enabled_indicators()
        available = self.indicators_system.get_available_indicators()
        data_summary = self.indicators_system.get_data_requirements_summary()

        return {
            "total_indicators": len(self.indicators_system.indicators_config),
            "enabled_indicators": len(enabled),
            "available_indicators": len([c for c in available.values() if c.enabled]),
            "data_requirements": data_summary,
            "categories": {
                category: len(self.indicators_system.get_indicators_by_category(category))
                for category in [IndicatorCategories.OVERLAP, IndicatorCategories.MOMENTUM,
                               IndicatorCategories.VOLATILITY, IndicatorCategories.TREND,
                               IndicatorCategories.VOLUME, IndicatorCategories.PRICE_TRANSFORM,
                               IndicatorCategories.STATISTICS, IndicatorCategories.CYCLE]
            }
        }


def _coerce_indicator_tfs(raw: Any) -> Dict[str, List[str]]:
    """Coerce indicator timeframes configuration"""
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return {k: v if isinstance(v, list) else [v] for k, v in parsed.items()}
        except Exception:
            return {}
    if isinstance(raw, dict):
        return {k: v if isinstance(v, list) else [v] for k, v in raw.items()}
    return {}


def _standardize_train_val(X: pd.DataFrame, y: pd.Series, val_frac: float):
    """
    Standardize training and validation sets ensuring NO look-ahead bias.

    CRITICAL: Computes mean/std ONLY on training set, then applies to validation.
    This prevents information leakage from future data.

    Returns:
        Tuple of ((Xtr_scaled, ytr), (Xva_scaled, yva), (mu, sigma, scaler_metadata))
    """
    from scipy import stats

    # Split WITHOUT shuffling to maintain temporal order (prevent look-ahead bias)
    Xtr, Xva, ytr, yva = train_test_split(X.values, y.values, test_size=val_frac, shuffle=False)

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
    """Fit model based on algorithm choice"""
    if algo == "ridge":
        return Ridge(alpha=float(args.alpha), random_state=args.random_state).fit(Xtr, ytr)
    elif algo == "lasso":
        return Lasso(alpha=float(args.alpha), random_state=args.random_state).fit(Xtr, ytr)
    elif algo == "elasticnet":
        return ElasticNet(alpha=float(args.alpha), l1_ratio=float(args.l1_ratio), random_state=args.random_state).fit(Xtr, ytr)
    elif algo == "rf":
        return RandomForestRegressor(n_estimators=int(args.n_estimators), max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=args.random_state).fit(Xtr, ytr)
    else:
        raise ValueError(f"Algo non supportato: {algo}")


def main():
    """Main training function with enhanced bta-lib indicators"""
    ap = argparse.ArgumentParser(description="Enhanced training with bta-lib indicators")

    # Core arguments
    ap.add_argument("--symbol", required=True, help="Trading symbol")
    ap.add_argument("--timeframe", required=True, help="Base timeframe")
    ap.add_argument("--horizon", type=int, required=True, help="Prediction horizon")
    ap.add_argument("--algo", choices=["ridge", "lasso", "elasticnet", "rf"], required=True, help="Algorithm")
    ap.add_argument("--artifacts_dir", required=True, help="Output directory")

    # Data arguments
    ap.add_argument("--days_history", type=int, default=60, help="Days of history to fetch")
    ap.add_argument("--available_data", nargs='+', default=['open', 'high', 'low', 'close'],
                   help="Available data columns")

    # Feature arguments
    ap.add_argument("--use_relative_ohlc", type=bool, default=True, help="Use relative OHLC features")
    ap.add_argument("--use_temporal_features", type=bool, default=True, help="Use temporal features")
    ap.add_argument("--rv_window", type=int, default=60, help="Realized volatility window")
    ap.add_argument("--warmup_bars", type=int, default=64, help="Warmup bars")
    ap.add_argument("--min_feature_coverage", type=float, default=0.15, help="Minimum feature coverage")

    # Model arguments
    ap.add_argument("--pca", type=int, default=0, help="PCA components (0 = disabled)")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--alpha", type=float, default=0.001, help="Regularization alpha")
    ap.add_argument("--l1_ratio", type=float, default=0.5, help="ElasticNet L1 ratio")
    ap.add_argument("--n_estimators", type=int, default=400, help="RF n_estimators")
    ap.add_argument("--random_state", type=int, default=0, help="Random state")

    # Indicators arguments
    ap.add_argument("--indicators_config", type=str, help="Path to indicators configuration JSON")
    ap.add_argument("--indicator_tfs", type=str, default='{}', help="Indicator timeframes JSON")

    args = ap.parse_args()

    print(f"🚀 Starting enhanced training with bta-lib indicators...")
    print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}, Horizon: {args.horizon}")

    # Load indicators configuration
    indicators_config = {}
    if args.indicators_config and Path(args.indicators_config).exists():
        with open(args.indicators_config, 'r') as f:
            indicators_config = json.load(f)
            print(f"📊 Loaded indicators config from {args.indicators_config}")
    else:
        # Default configuration
        indicators_config = {
            'available_data': args.available_data,
            'indicators': {}
        }

    # Parse indicator timeframes
    indicator_tfs = _coerce_indicator_tfs(args.indicator_tfs)

    # Initialize enhanced training system
    trainer = BTALibIndicatorsTraining(indicators_config)

    # Print indicators summary
    summary = trainer.get_indicators_summary()
    print(f"📈 Indicators Summary:")
    print(f"   Total available: {summary.get('total_indicators', 0)}")
    print(f"   Enabled: {summary.get('enabled_indicators', 0)}")
    print(f"   Available data: {trainer.available_data}")

    # Fetch data
    print(f"📊 Fetching {args.days_history} days of data...")
    candles = fetch_candles_from_db(args.symbol, args.timeframe, args.days_history)
    print(f"✅ Loaded {len(candles)} candles")

    # Build features
    print(f"🔧 Building features with enhanced indicators...")
    X, y, meta = trainer.build_features(candles, args)
    print(f"✅ Created {X.shape[1]} features from {len(candles)} candles")
    print(f"   Final dataset: {len(X)} samples")

    # Standardize and split
    print(f"📊 Splitting and standardizing data...")
    (Xtr, ytr), (Xva, yva), (mu, sigma, scaler_metadata) = _standardize_train_val(X, y, args.val_frac)
    print(f"   Training: {len(Xtr)} samples")
    print(f"   Validation: {len(Xva)} samples")

    # Log KS test results for look-ahead bias detection
    if scaler_metadata.get('ks_test_median_p') is not None:
        print(f"   KS test median p-value: {scaler_metadata['ks_test_median_p']:.4f} (< 0.5 expected for no bias)")

    # Apply PCA if requested
    pca_model = None
    if args.pca > 0:
        print(f"🔄 Applying PCA with {args.pca} components...")
        pca_model = PCA(n_components=args.pca, random_state=args.random_state)
        Xtr = pca_model.fit_transform(Xtr)
        Xva = pca_model.transform(Xva)
        print(f"✅ PCA: {X.shape[1]} → {args.pca} dimensions")

    # Train model
    print(f"🤖 Training {args.algo} model...")
    model = _fit_model(args.algo, Xtr, ytr, args)

    # Evaluate
    ytr_pred = model.predict(Xtr)
    yva_pred = model.predict(Xva)
    mae_tr = mean_absolute_error(ytr, ytr_pred)
    mae_va = mean_absolute_error(yva, yva_pred)

    print(f"📊 Training Results:")
    print(f"   Training MAE: {mae_tr:.6f}")
    print(f"   Validation MAE: {mae_va:.6f}")
    print(f"   Overfitting ratio: {mae_va/mae_tr:.3f}")

    # Save artifacts
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = artifacts_dir / "model.joblib"
    dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # Save preprocessing info
    preprocessing = {
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "pca_model": pca_model
    }
    preprocessing_path = artifacts_dir / "preprocessing.joblib"
    dump(preprocessing, preprocessing_path)
    print(f"✅ Preprocessing saved to {preprocessing_path}")

    # Save enhanced metadata
    meta.update({
        "training_results": {
            "mae_train": mae_tr,
            "mae_validation": mae_va,
            "overfitting_ratio": mae_va / mae_tr,
            "n_features_final": Xtr.shape[1],
            "n_samples_train": len(Xtr),
            "n_samples_validation": len(Xva)
        },
        "indicators_summary": summary
    })

    meta_path = artifacts_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"✅ Enhanced metadata saved to {meta_path}")

    print(f"🎉 Training completed successfully!")
    print(f"   Enhanced with {summary.get('enabled_indicators', 0)} bta-lib indicators")
    print(f"   Final performance: {mae_va:.6f} MAE")


if __name__ == "__main__":
    main()