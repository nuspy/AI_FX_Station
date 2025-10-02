"""
ForecastWorker - Background worker for model inference.

Handles:
- Single model inference
- Parallel ensemble inference
- Feature computation and caching
- Enhanced multi-horizon predictions
- Performance tracking
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Callable, List, Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from PySide6.QtCore import QRunnable
from loguru import logger

import pickle
import hashlib

from ...services.marketdata import MarketDataService
from ...inference.parallel_inference import get_parallel_engine
from ...utils.horizon_converter import convert_horizons_for_inference, create_future_timestamps

from ...utils.horizon_converter import (
    convert_horizons_for_inference,
    create_future_timestamps,
    convert_single_to_multi_horizon,
    get_trading_scenarios
)
from ...services.performance_registry import get_performance_registry

# Ensure features helper
_ensure_features_for_prediction: Optional[Callable] = None
try:
    from forex_diffusion.inference.prediction_config import ensure_features_for_prediction as _ensure_features_for_prediction
except Exception:
    try:
        from ...inference.prediction_config import ensure_features_for_prediction as _ensure_features_for_prediction
    except Exception:
        _ensure_features_for_prediction = None


class ForecastWorker(QRunnable):
    """
    Esegue inferenza locale:
     - recupera ultime candele dal DB
     - carica il modello dal path risolto
     - costruisce features senza rifittare alcun standardizer
     - emette forecastReady(df_candles, quantiles)
    """

    def __init__(self, engine_url: str, payload: dict, market_service: MarketDataService, signals):
        super().__init__()
        self.engine_url = engine_url.rstrip("/")
        self.payload = payload
        self.market_service = market_service
        self.signals = signals

    def _get_param_with_override(self, param_name: str, default_value, model_paths: List[str] = None):
        """
        Get parameter value with override logic:
        - If override_<param_name> is True: use payload value
        - If override_<param_name> is False: try to load from model metadata
        - Fallback to default_value if metadata not available
        """
        override_key = f"override_{param_name}"
        override_enabled = self.payload.get(override_key, True)  # Default: override enabled

        if override_enabled:
            # Use value from payload (UI override)
            return self.payload.get(param_name, default_value)

        # Try to load from model metadata (only for first model if multi-model)
        if model_paths and len(model_paths) > 0:
            try:
                from ...models.metadata_manager import MetadataManager
                manager = MetadataManager()
                metadata = manager.load_metadata(model_paths[0])

                if metadata and hasattr(metadata, 'preprocessing_config'):
                    config = metadata.preprocessing_config
                    if param_name in config:
                        logger.debug(f"Using {param_name}={config[param_name]} from model metadata (override disabled)")
                        return config[param_name]
            except Exception as e:
                logger.warning(f"Failed to load {param_name} from model metadata: {e}")

        # Fallback to default
        logger.debug(f"Using default {param_name}={default_value} (metadata not available)")
        return default_value

    def run(self):
        try:
            self.signals.status.emit("Forecast: running local inference...")
            df_local, quantiles_local = self._local_infer()
            self.signals.status.emit("Forecast: ready (local)")
            self.signals.forecastReady.emit(df_local, quantiles_local)
        except Exception as e:
            logger.exception("Forecast worker failed: {}", e)
            self.signals.error.emit(str(e))
            self.signals.status.emit("Forecast: failed")

    # -------- helpers: risoluzione modello e dati -------------------------- #

    def _known_model_dirs(self) -> List[Path]:
        dirs: List[Path] = []
        # config artifacts_dir se presente
        try:
            from ...utils.config import get_config
            cfg = get_config()
            model_cfg = getattr(cfg, "model", None)
            art = None
            if isinstance(model_cfg, dict):
                art = model_cfg.get("artifacts_dir")
            else:
                art = getattr(model_cfg, "artifacts_dir", None)
            if art:
                p = Path(str(art))
                if p.exists():
                    dirs.append(p)
        except Exception:
            pass
        # cwd + artifacts/models
        try:
            if Path.cwd().exists():
                dirs.append(Path.cwd())
                am = Path.cwd() / "artifacts" / "models"
                if am.exists():
                    dirs.append(am)
        except Exception:
            pass
        # cartelle dei "consentiti" (allowed list)
        for a in (self.payload.get("allowed_models") or []):
            try:
                ap = Path(str(a)).expanduser().resolve()
                if ap.parent.exists():
                    dirs.append(ap.parent)
            except Exception:
                pass
        # dedup case-insensitive su Windows
        out, seen = [], set()
        for d in dirs:
            key = str(d).lower() if sys.platform.startswith("win") else str(d)
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    def _resolve_model_path(self, raw_path: str) -> Path:
        """Resolve model path using standardized path resolver with legacy fallback."""
        from ...models.model_path_resolver import ModelPathResolver

        try:
            # Use standardized resolver first
            resolver = ModelPathResolver()
            settings = {"model_path": raw_path}
            resolved_paths = resolver.resolve_model_paths(settings)

            if resolved_paths:
                resolved_path = Path(resolved_paths[0])
                logger.info("Model path resolved (standardized): {} -> {}", raw_path, str(resolved_path))
                return resolved_path

        except Exception as e:
            logger.debug(f"Standardized resolution failed, using legacy: {e}")

        # Legacy fallback resolution
        raw = (raw_path or "").strip().strip('"').strip("'")
        def canon(s: str) -> Path:
            try:
                p = Path(os.path.expandvars(os.path.expanduser(s)))
                return p if p.is_absolute() else (Path.cwd() / p).resolve()
            except Exception:
                return Path(s)
        norm = canon(raw)
        # esiste già?
        try:
            if norm.exists():
                logger.info("Model path resolved (exact): {} -> {}", raw, str(norm))
                return norm
        except Exception:
            pass
        # prova raw
        try:
            rp = Path(raw)
            if rp.exists():
                return rp.resolve()
        except Exception:
            pass
        # match su allowed (exact, casefold, basename)
        allowed = []
        try:
            allowed = [Path(os.path.expandvars(os.path.expanduser(str(a)))) for a in (self.payload.get("allowed_models") or []) if a]
        except Exception:
            pass
        try:
            # exact
            for ap in allowed:
                if str(ap) == str(norm) and ap.exists():
                    return ap
            # case-insensitive (Windows)
            if sys.platform.startswith("win"):
                rn = str(norm).lower()
                for ap in allowed:
                    if str(ap).lower() == rn and ap.exists():
                        return ap
            # basename
            base = Path(raw).name.lower()
            if base:
                for ap in allowed:
                    if ap.name.lower() == base and ap.exists():
                        return ap
        except Exception:
            pass
        # scan cartelle note per basename
        try:
            base = Path(raw).name
            for d in self._known_model_dirs():
                cand = d / base
                if cand.exists():
                    logger.info("Model path resolved by search: {} -> {}", raw, str(cand))
                    return cand.resolve()
        except Exception:
            pass
        logger.error("Model path not found (strict): raw='{}' norm='{}'", raw, str(norm))
        return norm

    def _fetch_recent_candles(self, engine, symbol: str, timeframe: str, n_bars: int = 512, end_ts: Optional[int] = None) -> pd.DataFrame:
        try:
            from sqlalchemy import MetaData, select, text
            meta = MetaData()
            meta.reflect(bind=engine, only=["market_data_candles"])
            tbl = meta.tables.get("market_data_candles")
            if tbl is None:
                return pd.DataFrame()
            with engine.connect() as conn:
                if end_ts is None:
                    stmt = select(tbl.c.ts_utc, tbl.c.open, tbl.c.high, tbl.c.low, tbl.c.close, tbl.c.volume)\
                        .where(tbl.c.symbol == symbol).where(tbl.c.timeframe == timeframe)\
                        .order_by(tbl.c.ts_utc.desc()).limit(n_bars)
                    rows = conn.execute(stmt).fetchall()
                else:
                    q = text(
                        "SELECT ts_utc, open, high, low, close, volume FROM market_data_candles "
                        "WHERE symbol = :symbol AND timeframe = :timeframe AND ts_utc <= :end_ts "
                        "ORDER BY ts_utc DESC LIMIT :limit"
                    )
                    rows = conn.execute(q, {"symbol": symbol, "timeframe": timeframe, "end_ts": int(end_ts), "limit": int(n_bars)}).fetchall()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=["ts_utc", "open", "high", "low", "close", "volume"])
                return df.sort_values("ts_utc").reset_index(drop=True)
        except Exception as e:
            logger.exception("Failed to fetch recent candles: {}", e)
            return pd.DataFrame()

    def _dict_to_candles(self, candles_list) -> pd.DataFrame:
        """Convert candles list to DataFrame."""
        import pandas as _pd
        return _pd.DataFrame(candles_list).copy()

    # --------------------------- local inference --------------------------- #

    def _local_infer(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Fallback locale (basic). Flusso:
        candele -> pipeline (NO standardization fit) -> ensure -> z-score con μ/σ del modello -> predict -> prezzi/ts -> quantili
        """
        # Check if this is a parallel inference request
        if self.payload.get("parallel_inference", False):
            return self._parallel_infer()




        sym = self.payload.get("symbol")
        tf = (self.payload.get("timeframe") or "1m")
        horizons_raw = self.payload.get("horizons", ["5m"])
        limit = int(self.payload.get("limit_candles", 512))
        ftype = str(self.payload.get("forecast_type", "basic")).lower()

        # Converti horizons al formato corretto
        horizons_time_labels, horizons_bars = convert_horizons_for_inference(horizons_raw, tf)
        logger.debug(f"Horizons: {len(horizons_time_labels)} points from {horizons_raw}")

        # 1) dati (ancorati all'eventuale timestamp del click)
        anchor_ts = None
        try:
            a = self.payload.get("testing_point_ts", None)
            if a is None:
                a = self.payload.get("requested_at_ms", None)
            if a is not None:
                anchor_ts = int(a)
        except Exception:
            anchor_ts = None

        # Check if multi-timeframe indicators are configured (need more data from DB)
        has_multi_tf_indicators = bool(self.payload.get("indicator_tfs", {}))

        # Sorgente dati: override esplicito o fetch dal DB; se anchor_ts è definito, non includere barre successive
        if isinstance(self.payload.get("candles_override"), (list, tuple)):
            df_candles = self._dict_to_candles(self.payload["candles_override"])
            # If candles_override too small for multi-timeframe, fetch from DB
            if len(df_candles) < limit or has_multi_tf_indicators:
                logger.info(f"candles_override has {len(df_candles)} rows, fetching {limit} from DB for multi-timeframe indicators")
                df_candles_db = self._fetch_recent_candles(
                    self.market_service.engine, sym, tf,
                    n_bars=limit,
                    end_ts=anchor_ts if anchor_ts is not None else None
                )
                if df_candles_db is not None and not df_candles_db.empty:
                    df_candles = df_candles_db
                    logger.debug(f"Fetched {len(df_candles)} candles from DB for single model inference")
        else:
            df_candles = self._fetch_recent_candles(
                self.market_service.engine, sym, tf,
                n_bars=limit,
                end_ts=anchor_ts if anchor_ts is not None else None
            )

        if df_candles is None or df_candles.empty:
            raise RuntimeError("No candles available for local inference")

        # Rename 'price' to OHLC for tick data
        if "price" in df_candles.columns and "close" not in df_candles.columns:
            df_candles["close"] = df_candles["price"]
            df_candles["open"] = df_candles["price"]
            df_candles["high"] = df_candles["price"]
            df_candles["low"] = df_candles["price"]
            logger.debug("Converted tick 'price' data to OHLC format for single model inference")

        # Normalizza ordine ASC e, se presente anchor_ts, taglia le barre > anchor
        df_candles = df_candles.sort_values("ts_utc").reset_index(drop=True)
        if anchor_ts is not None:
            try:
                df_candles["ts_utc"] = pd.to_numeric(df_candles["ts_utc"], errors="coerce").astype("int64")
                df_candles = df_candles[df_candles["ts_utc"] <= int(anchor_ts)].reset_index(drop=True)
            except Exception:
                pass
        if df_candles.empty:
            raise RuntimeError("No candles available at anchor timestamp")

        # 2) carica modello usando standardized loader (se non RW)
        used_model_path_str = ""
        model_sha16 = None
        payload_obj = {}
        model = None
        model_data = None  # Initialize for later access
        features_list: List[str] = []
        std_mu: Dict[str, float] = {}
        std_sigma: Dict[str, float] = {}
        preprocessor = None  # Initialize preprocessor outside try-except

        if ftype != "rw":
            from ...models.standardized_loader import get_model_loader

            mp = self.payload.get("model_path") or self.payload.get("model")
            p = self._resolve_model_path(str(mp))
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            used_model_path_str = str(p)

            # Use standardized loader
            try:
                loader = get_model_loader()
                model_data = loader.load_single_model(str(p))

                # Extract standardized data
                model = model_data['model']
                features_list = model_data.get('features', [])

                # Extract any preprocessor/encoder (PCA, autoencoder, etc.)
                preprocessor = None
                if 'pca' in model_data and model_data['pca'] is not None:
                    preprocessor = model_data['pca']
                    logger.debug(f"Found PCA preprocessor with {preprocessor.n_components_} components")
                elif 'encoder' in model_data and model_data['encoder'] is not None:
                    preprocessor = model_data['encoder']
                    logger.debug(f"Found encoder preprocessor: {type(preprocessor).__name__}")
                elif 'preprocessor' in model_data and model_data['preprocessor'] is not None:
                    preprocessor = model_data['preprocessor']
                    logger.debug(f"Found generic preprocessor: {type(preprocessor).__name__}")

                # Handle standardizer/scaler with legacy compatibility
                scaler = model_data.get('scaler') or model_data.get('standardizer')
                if scaler:
                    std_mu = scaler.get('mu') if isinstance(scaler, dict) else {}
                    std_sigma = scaler.get('sigma') if isinstance(scaler, dict) else {}
                else:
                    # Legacy support
                    payload_obj = model_data.get('raw_data', {})
                    std_mu = payload_obj.get("std_mu") or payload_obj.get("std") or {}
                    std_sigma = payload_obj.get("std_sigma") or ({c: 1.0 for c in features_list} if "std" in payload_obj else {})

                # Also check for scaler_mu/scaler_sigma (direct keys in model_data)
                if not std_mu and 'scaler_mu' in model_data:
                    std_mu = model_data['scaler_mu']
                if not std_sigma and 'scaler_sigma' in model_data:
                    std_sigma = model_data['scaler_sigma']

                # Calculate model hash
                try:
                    model_sha16 = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
                except Exception:
                    model_sha16 = None

                # Validation
                validation = model_data.get('validation', {})
                if not validation.get('valid', True):
                    logger.warning(f"Model validation issues: {validation.get('errors', [])}")

                if model is None or not features_list:
                    raise RuntimeError("Model payload missing 'model' or 'features'")

                logger.debug(f"Loaded model: {model_data.get('model_type', 'unknown')} from {Path(p).name}")

            except Exception as e:
                logger.error(f"Standardized loader failed with error: {type(e).__name__}: {e}")
                logger.error(f"Full traceback:", exc_info=True)
                logger.error(f"Model path: {p}")
                logger.error(f"File exists: {p.exists()}")
                if p.exists():
                    logger.error(f"File size: {p.stat().st_size} bytes")

                # Try to understand why it failed
                try:
                    import joblib
                    test_data = joblib.load(p)
                    logger.error(f"Joblib can load the file. Keys: {list(test_data.keys()) if isinstance(test_data, dict) else type(test_data)}")
                except Exception as joblib_err:
                    logger.error(f"Even joblib fails to load: {joblib_err}")

                # DO NOT use legacy fallback - fail explicitly so user can fix the issue
                raise RuntimeError(
                    f"Model loading failed. Please check the model file format.\n"
                    f"Error: {type(e).__name__}: {e}\n"
                    f"Model path: {p}\n"
                    f"This is a critical error that needs to be fixed in the model file or loader."
                ) from e
        else:
            # RW baseline
            std_mu, std_sigma = {}, {}

        # 3) features usando STESSO sistema del training (train_sklearn.py) CON CACHING
        from ...training.train_sklearn import _relative_ohlc, _temporal_feats, _realized_vol_feature, _indicators, _coerce_indicator_tfs
        from ...features.feature_cache import get_feature_cache
        from ...features.unified_pipeline import FeatureConfig, hierarchical_multi_timeframe_pipeline

        # Configura la cache delle features
        feature_cache = get_feature_cache()

        # Extract multi-timeframe config from model metadata if available
        model_indicator_tfs = {}
        if ftype != "rw" and model_data:
            # Get multi-timeframe config from model metadata
            metadata = model_data.get('metadata', {})
            # metadata can be ModelMetadata object or dict
            if hasattr(metadata, 'multi_timeframe_config'):
                mtf_config = metadata.multi_timeframe_config
            else:
                mtf_config = metadata.get('multi_timeframe_config', {}) if isinstance(metadata, dict) else {}

            if mtf_config and 'indicator_tfs' in mtf_config:
                model_indicator_tfs = mtf_config['indicator_tfs']
                logger.info(f"Using multi-timeframe config from model metadata: {len(model_indicator_tfs)} indicators")

        # Crea configurazione per il caching
        # Use model's indicator_tfs if available, otherwise fall back to payload
        indicator_tfs_to_use = model_indicator_tfs if model_indicator_tfs else self.payload.get("indicator_tfs", {})

        cache_config = {
            "use_relative_ohlc": self.payload.get("use_relative_ohlc", True),
            "use_temporal_features": self.payload.get("use_temporal_features", True),
            "rv_window": int(self.payload.get("rv_window", 60)),
            "indicator_tfs": indicator_tfs_to_use,
            "advanced": self.payload.get("advanced", False),
            "use_advanced_features": self.payload.get("use_advanced_features", False),
            "enable_ema_features": self.payload.get("enable_ema_features", False),
            "enable_donchian": self.payload.get("enable_donchian", False),
            "enable_keltner": self.payload.get("enable_keltner", False),
            "enable_hurst_advanced": self.payload.get("enable_hurst_advanced", False),
            "atr_n": int(self.payload.get("atr_n", 14)),
            "rsi_n": int(self.payload.get("rsi_n", 14)),
            "bb_n": int(self.payload.get("bb_n", 20)),
            "don_n": int(self.payload.get("don_n", 20)),
            "keltner_ema": int(self.payload.get("keltner_ema", 20)),
            "keltner_atr": int(self.payload.get("keltner_atr", 10)),
            "keltner_k": float(self.payload.get("keltner_k", 1.5)),
            "hurst_window": int(self.payload.get("hurst_window", 64)),
            "ema_fast": int(self.payload.get("ema_fast", 12)),
            "ema_slow": int(self.payload.get("ema_slow", 26))
        }

        # Controlla se usare il sistema multi-timeframe hierarchical
        use_hierarchical = self.payload.get("use_hierarchical_multitf", False)
        query_timeframe = self.payload.get("query_timeframe", tf)

        # Controlla cache prima di calcolare features
        cached_result = feature_cache.get_cached_features(df_candles, cache_config, tf)

        if cached_result is not None:
            feats_df, feature_metadata = cached_result
            logger.debug(f"Features loaded from cache for {sym} {tf}")
        else:
            # Calcola features (cache miss)
            logger.debug(f"Computing features for {sym} {tf} (cache miss)")

            if use_hierarchical:
                # USA SISTEMA HIERARCHICAL MULTI-TIMEFRAME
                logger.info(f"Using hierarchical multi-timeframe system: query_tf={query_timeframe}")

                # Configura FeatureConfig per hierarchical
                feature_config = FeatureConfig({
                    "base_features": {
                        "relative_ohlc": cache_config["use_relative_ohlc"],
                        "log_returns": True,
                        "time_features": cache_config["use_temporal_features"],
                        "session_features": True
                    },
                    "indicators": {
                        "atr": {"enabled": True, "n": cache_config["atr_n"]},
                        "rsi": {"enabled": True, "n": cache_config["rsi_n"]},
                        "bollinger": {"enabled": True, "n": cache_config["bb_n"], "k": 2.0},
                        "macd": {"enabled": True, "fast": 12, "slow": 26, "signal": 9},
                        "donchian": {"enabled": cache_config.get("enable_donchian", False), "n": cache_config["don_n"]},
                        "keltner": {"enabled": cache_config.get("enable_keltner", False),
                                  "ema": cache_config["keltner_ema"], "atr": cache_config["keltner_atr"], "mult": cache_config["keltner_k"]},
                        "hurst": {"enabled": cache_config.get("enable_hurst_advanced", False), "window": cache_config["hurst_window"]},
                        "ema": {"enabled": cache_config.get("enable_ema_features", False),
                               "fast": cache_config["ema_fast"], "slow": cache_config["ema_slow"]}
                    },
                    "multi_timeframe": {
                        "enabled": True,
                        "timeframes": self.payload.get("hierarchical_timeframes", ["1m", "5m", "15m", "1h"]),
                        "base_timeframe": tf,
                        "query_timeframe": query_timeframe,
                        "indicators": list(cache_config.get("indicator_tfs", {}).keys()),
                        "hierarchical_mode": True,
                        "exclude_children": self.payload.get("exclude_children", True),
                        "auto_group_selection": True
                    },
                    "rv_window": cache_config["rv_window"],
                    "standardization": {"enabled": True}
                })

                # Applica il pipeline hierarchical
                feats_df, standardizer, feature_names, hierarchy = hierarchical_multi_timeframe_pipeline(
                    df_candles, feature_config, tf, standardizer=None, fit_standardizer=False
                )

                logger.info(f"Hierarchical pipeline completed: {len(feature_names)} features, {len(feats_df)} samples")

            else:
                # USA SISTEMA TRADIZIONALE (come prima)
                # Usa gli stessi metodi del training per garantire coerenza
                feats_list = []

                # Relative OHLC (come in training)
                if cache_config["use_relative_ohlc"]:
                    feats_list.append(_relative_ohlc(df_candles))

                # Temporal features (come in training)
                if cache_config["use_temporal_features"]:
                    feats_list.append(_temporal_feats(df_candles))

                # Realized volatility (come in training)
                if cache_config["rv_window"] > 1:
                    feats_list.append(_realized_vol_feature(df_candles, cache_config["rv_window"]))

                # Multi-timeframe indicators (come in training)
                indicator_tfs_raw = cache_config["indicator_tfs"]
                indicator_tfs = _coerce_indicator_tfs(indicator_tfs_raw)

                # Advanced mode: aggiungi indicators extra se abilitato
                is_advanced = cache_config["advanced"] or cache_config["use_advanced_features"]

                if is_advanced:
                    # In advanced mode, aggiungi indicators anche se non in indicator_tfs
                    if not indicator_tfs:
                        indicator_tfs = {}

                    # Abilita features avanzate se richieste
                    if cache_config["enable_ema_features"]:
                        indicator_tfs.setdefault("ema", [tf])

                    if cache_config["enable_donchian"]:
                        indicator_tfs.setdefault("donchian", [tf])

                    if cache_config["enable_keltner"]:
                        indicator_tfs.setdefault("keltner", [tf])

                    if cache_config["enable_hurst_advanced"]:
                        indicator_tfs.setdefault("hurst", [tf])

                if indicator_tfs:
                    # Crea la stessa configurazione del training
                    ind_cfg = {}
                    if "atr" in indicator_tfs:
                        ind_cfg["atr"] = {"n": cache_config["atr_n"]}
                    if "rsi" in indicator_tfs:
                        ind_cfg["rsi"] = {"n": cache_config["rsi_n"]}
                    if "bollinger" in indicator_tfs:
                        ind_cfg["bollinger"] = {"n": cache_config["bb_n"], "dev": 2.0}
                    if "macd" in indicator_tfs:
                        ind_cfg["macd"] = {"fast": 12, "slow": 26, "signal": 9}
                    if "donchian" in indicator_tfs:
                        ind_cfg["donchian"] = {"n": cache_config["don_n"]}
                    if "keltner" in indicator_tfs:
                        ind_cfg["keltner"] = {
                            "ema": cache_config["keltner_ema"],
                            "atr": cache_config["keltner_atr"],
                            "mult": cache_config["keltner_k"]
                        }
                    if "hurst" in indicator_tfs:
                        ind_cfg["hurst"] = {"window": cache_config["hurst_window"]}
                    if "ema" in indicator_tfs:
                        ind_cfg["ema"] = {
                            "fast": cache_config["ema_fast"],
                            "slow": cache_config["ema_slow"]
                        }

                    if ind_cfg:
                        # Calculate days_history from limit for better data fetching
                        # For 1h TF: 24 bars = 1 day, for 1m TF: 1440 bars = 1 day
                        tf_bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1, "1w": 1/7}
                        bars_per_day = tf_bars_per_day.get(tf, 24)
                        days_history_est = max(1, int(limit / bars_per_day)) if bars_per_day > 0 else 7
                        feats_list.append(_indicators(df_candles, ind_cfg, indicator_tfs, tf, symbol=sym, days_history=days_history_est))

                if not feats_list:
                    raise RuntimeError("No features configured for inference")

                # Combina tutte le features (come in training)
                feats_df = pd.concat(feats_list, axis=1)
                feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

                # Salva in cache per riuso futuro (solo per sistema tradizionale)
                feature_metadata = {
                    "config": cache_config,
                    "timestamp": df_candles["ts_utc"].iat[-1] if len(df_candles) > 0 else 0,
                    "symbol": sym,
                    "timeframe": tf
                }

                try:
                    feature_cache.cache_features(df_candles, feats_df, feature_metadata, cache_config, tf)
                    logger.debug(f"Features cached for {sym} {tf}")
                except Exception as e:
                    logger.warning(f"Failed to cache features: {e}")

        # Applica coverage filtering (come in training)
        coverage = feats_df.notna().mean()
        min_cov = float(self.payload.get("min_feature_coverage", 0.15))
        if min_cov > 0.0:
            low_cov = coverage[coverage < min_cov]
            if not low_cov.empty:
                feats_df = feats_df.drop(columns=list(low_cov.index), errors="ignore")

        feats_df = feats_df.dropna()

        # Applica warmup (come in training)
        warmup_bars = int(self.payload.get("warmup_bars", 16))
        if warmup_bars > 0 and len(feats_df) > warmup_bars:
            feats_df = feats_df.iloc[warmup_bars:]

        if feats_df.empty:
            raise RuntimeError("No features computed after preprocessing")

        # 4) ensure schema and feature order (only if helper is available)
        if callable(_ensure_features_for_prediction) and features_list:
            try:
                feats_df = _ensure_features_for_prediction(
                    feats_df,
                    timeframe=tf,
                    features_list=features_list,
                    adv_cfg={
                        "rv_window": int(self.payload.get("rv_window", 60)),
                        "rsi_n": int(self.payload.get("rsi_n", 14)),
                        "bb_n": int(self.payload.get("bb_n", 20)),
                        "don_n": int(self.payload.get("don_n", 20)),
                        "hurst_window": int(self.payload.get("hurst_window", 64)),
                        "keltner_k": float(self.payload.get("keltner_k", 1.5)),
                        "ema_fast": int(self.payload.get("ema_fast", 12)),
                        "ema_slow": int(self.payload.get("ema_slow", 26)),
                    },
                )
            except Exception as e:
                logger.warning("ensure_features_for_prediction failed: {}", e)

        # 5) completa mancanti con μ e z-score
        for c in features_list:
            if c not in feats_df.columns:
                feats_df[c] = float(std_mu.get(c, 0.0))
        X = feats_df[features_list].astype(float).replace([float("inf"), float("-inf")], float("nan")).fillna(0.0).copy()
        for c in features_list:
            if c in std_mu and c in std_sigma:
                denom = float(std_sigma[c]) if float(std_sigma[c]) != 0.0 else 1.0
                X[c] = (X[c] - float(std_mu[c])) / denom
        X_arr = X.to_numpy(dtype=float)

        # 6) inferenza
        preds = None
        if model is not None:
            # Usa l'ultimo sample per la predizione
            X_last = X_arr[-1:, :]  # Shape: (1, n_features)

            # Apply preprocessor/encoder transformation if present
            if preprocessor is not None:
                if hasattr(preprocessor, 'transform'):
                    # Scikit-learn style transformer (PCA, StandardScaler, etc.)
                    X_last = preprocessor.transform(X_last)
                    logger.debug(f"Applied {type(preprocessor).__name__} transformation: {X_arr.shape[1]} -> {X_last.shape[1]} features")
                elif hasattr(preprocessor, 'encode') or hasattr(preprocessor, 'predict'):
                    # Neural network encoder (autoencoder, VAE, etc.)
                    if hasattr(preprocessor, 'encode'):
                        X_last = preprocessor.encode(X_last)
                    else:
                        X_last = preprocessor.predict(X_last)
                    logger.debug(f"Applied {type(preprocessor).__name__} encoding")
                else:
                    logger.warning(f"Preprocessor {type(preprocessor).__name__} has no transform/encode/predict method, skipping")

            # torch?
            try:
                import torch
                if hasattr(model, "eval"):
                    model.eval()
                with torch.no_grad():
                    t_in = torch.tensor(X_last, dtype=torch.float32)
                    out = model(t_in)
                    preds = np.ravel(out.detach().cpu().numpy())
            except Exception:
                # sklearn?
                if hasattr(model, "predict"):
                    preds = np.ravel(model.predict(X_last))

        if preds is None:
            # fallback: zero-returns
            preds = np.zeros((1,), dtype=float)

        # === ENHANCED MULTI-HORIZON PREDICTION SYSTEM ===

        # Check for scenario-based or enhanced multi-horizon prediction
        scenario = self.payload.get("trading_scenario")
        use_enhanced_scaling = self.payload.get("use_enhanced_scaling", True)
        scaling_mode = self.payload.get("scaling_mode", "smart_adaptive")

        if use_enhanced_scaling and len(preds) == 1 and len(horizons_bars) > 1:
            # Use Enhanced Multi-Horizon System
            base_pred = preds[0]

            # Get recent market data for regime detection
            market_data = df_candles.tail(100) if len(df_candles) >= 100 else df_candles

            try:
                # Convert single prediction to multi-horizon using smart scaling
                multi_horizon_results = convert_single_to_multi_horizon(
                    base_prediction=base_pred,
                    base_timeframe=tf,
                    target_horizons=horizons_time_labels,
                    scenario=scenario,
                    scaling_mode=scaling_mode,
                    market_data=market_data,
                    uncertainty_bands=True
                )

                # Extract predictions and uncertainty info
                scaled_preds = []
                uncertainty_data = {}

                for i, horizon in enumerate(horizons_time_labels):
                    if horizon in multi_horizon_results:
                        result = multi_horizon_results[horizon]
                        scaled_preds.append(result["prediction"])
                        uncertainty_data[horizon] = {
                            "lower": result["lower"],
                            "upper": result["upper"],
                            "confidence": result["confidence"],
                            "regime": result["regime"],
                            "scaling_mode": result["scaling_mode"]
                        }
                    else:
                        # Fallback to linear scaling
                        bars = horizons_bars[i]
                        scale_factor = bars / horizons_bars[0] if horizons_bars[0] > 0 else 1.0
                        scaled_preds.append(base_pred * scale_factor)

                preds = np.array(scaled_preds)

                # Store uncertainty data for later use
                self.payload["enhanced_uncertainty"] = uncertainty_data

                logger.info(f"Enhanced multi-horizon scaling completed for {len(horizons_time_labels)} horizons using {scaling_mode}")

            except Exception as e:
                logger.warning(f"Enhanced scaling failed, using linear fallback: {e}")
                # Fallback to linear scaling
                base_pred = preds[0]
                scaled_preds = []
                for i, bars in enumerate(horizons_bars):
                    scale_factor = bars / horizons_bars[0] if horizons_bars[0] > 0 else 1.0
                    scaled_preds.append(base_pred * scale_factor)
                preds = np.array(scaled_preds)

        elif len(preds) == 1 and len(horizons_bars) > 1:
            # Legacy linear scaling
            base_pred = preds[0]
            scaled_preds = []
            for i, bars in enumerate(horizons_bars):
                scale_factor = bars / horizons_bars[0] if horizons_bars[0] > 0 else 1.0
                scaled_preds.append(base_pred * scale_factor)
            preds = np.array(scaled_preds)
        elif len(preds) < len(horizons_bars):
            # Estendi la predizione per coprire tutti gli horizons
            preds = np.pad(preds, (0, len(horizons_bars) - len(preds)), mode='edge')

        # 7) prezzi e quantili usando conversione horizon corretta
        last_close = float(df_candles["close"].iat[-1])

        # Usa le predizioni convertite
        seq = preds[:len(horizons_bars)]
        prices = []
        p = last_close
        for r in seq:
            p *= (1.0 + float(r))
            prices.append(p)
        q50 = np.asarray(prices, dtype=float)

        # Volatilità realizzata per bande
        logret = pd.Series(df_candles["close"], dtype=float).pipe(lambda s: np.log(s).diff()).dropna()
        sigma_base = float(logret.tail(512).std() if len(logret) else 0.0)
        z = 1.645 if bool(self.payload.get("apply_conformal", True)) else 1.0

        # Scala volatilità per horizon bars (volatilità aumenta con √time)
        band_rel = []
        for i, bars in enumerate(horizons_bars):
            vol_scale = np.sqrt(bars) if bars > 0 else 1.0
            band = np.clip(z * sigma_base * vol_scale, 1e-6, 0.2)
            band_rel.append(band)

        band_rel = np.array(band_rel)
        q05 = np.maximum(1e-12, q50 * (1.0 - band_rel))
        q95 = np.maximum(1e-12, q50 * (1.0 + band_rel))

        # 8) future_ts usando converter
        last_ts_ms = int(df_candles["ts_utc"].iat[-1])
        future_ts = create_future_timestamps(last_ts_ms, tf, horizons_time_labels)

        display_name = str(self.payload.get("name") or self.payload.get("source_label") or "forecast")

        # === PERFORMANCE TRACKING INTEGRATION ===
        try:
            # Record predictions for performance tracking
            performance_registry = get_performance_registry()

            # Get regime and volatility info from enhanced uncertainty data
            enhanced_uncertainty = self.payload.get("enhanced_uncertainty", {})

            # Record each horizon prediction
            for i, (horizon, prediction) in enumerate(zip(horizons_time_labels, q50)):
                horizon_data = enhanced_uncertainty.get(horizon, {})

                performance_registry.record_prediction(
                    model_name=display_name,
                    symbol=sym,
                    timeframe=tf,
                    horizon=horizon,
                    prediction=float(prediction),
                    regime=horizon_data.get("regime", "unknown"),
                    volatility=horizon_data.get("volatility", sigma_base),
                    confidence=horizon_data.get("confidence", 0.5),
                    scaling_mode=horizon_data.get("scaling_mode", "linear"),
                    metadata={
                        "model_path": used_model_path_str,
                        "model_sha16": model_sha16,
                        "anchor_ts": anchor_ts,
                        "scenario": scenario
                    }
                )

            logger.debug(f"Recorded {len(horizons_time_labels)} predictions for performance tracking")

        except Exception as e:
            logger.warning(f"Failed to record predictions for performance tracking: {e}")

        # Enhanced quantiles with uncertainty data
        quantiles = {
            "q50": q50.tolist(),
            "q05": q05.tolist(),
            "q95": q95.tolist(),
            "future_ts": future_ts,
            "source": display_name,
            "label": display_name,
            "model_path_used": used_model_path_str,
            "model_sha16": model_sha16,
            "enhanced_uncertainty": enhanced_uncertainty,  # Include enhanced uncertainty info
            "trading_scenario": scenario,
            "scaling_mode": scaling_mode,
            "use_enhanced_scaling": use_enhanced_scaling,
            "requested_at_ms": self.payload.get("requested_at_ms"),  # For connecting forecast to price line
            "anchor_price": self.payload.get("anchor_price")  # Alt+Click Y coordinate
        }
        return df_candles, quantiles

    def _parallel_infer(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Parallel inference using multiple models for ensemble predictions.
        """

        try:

            symbol = str(self.payload.get("symbol", "EUR/USD"))
            tf = str(self.payload.get("timeframe", "1m"))
            horizons_raw = self.payload.get("horizons", ["5m"])
            limit = int(self.payload.get("limit_candles", 512))

            # Initial limit - will be adjusted for multi-timeframe if needed
            base_limit = limit

            logger.info(f"Starting parallel inference for {symbol} {tf}")

            # Map 'tick' and 'auto' to 1m for horizon conversion
            tf_for_conversion = "1m" if tf in ("tick", "auto") else tf

            # Converti horizons al formato corretto
            horizons_time_labels, horizons_bars = convert_horizons_for_inference(horizons_raw, tf_for_conversion)

            # Get candles (reuse existing logic)
            anchor_ts = None
            try:
                a = self.payload.get("testing_point_ts", None)
                if a is None:
                    a = self.payload.get("requested_at_ms", None)
                if a is not None:
                    anchor_ts = int(a)
            except Exception:
                anchor_ts = None

            # Check if any model needs multi-timeframe indicators and calculate required data
            has_multi_tf_indicators = False
            max_tf_minutes = 1  # Track highest timeframe in minutes

            try:
                model_paths = self.payload.get("model_paths", [])
                if model_paths and len(model_paths) > 0:
                    from ...models.standardized_loader import get_model_loader
                    loader = get_model_loader()
                    first_model_data = loader.load_single_model(str(model_paths[0]))
                    metadata = first_model_data.get('metadata', {})
                    # metadata can be ModelMetadata object or dict
                    if hasattr(metadata, 'multi_timeframe_config'):
                        mtf_config = metadata.multi_timeframe_config
                    else:
                        mtf_config = metadata.get('multi_timeframe_config', {}) if isinstance(metadata, dict) else {}

                    if mtf_config and 'indicator_tfs' in mtf_config:
                        has_multi_tf_indicators = bool(mtf_config['indicator_tfs'])

                        # Calculate max timeframe across all indicators
                        for indicator, tfs_list in mtf_config['indicator_tfs'].items():
                            for tf_str in tfs_list:
                                tf_str = str(tf_str).strip().lower()
                                if tf_str.endswith('m'):
                                    mins = int(tf_str[:-1])
                                elif tf_str.endswith('h'):
                                    mins = int(tf_str[:-1]) * 60
                                elif tf_str.endswith('d'):
                                    mins = int(tf_str[:-1]) * 1440
                                else:
                                    continue
                                max_tf_minutes = max(max_tf_minutes, mins)

                        # ATR needs window=14, so we need 14 * max_timeframe worth of base data
                        # Plus margin for resampling alignment
                        required_bars = max(limit, max_tf_minutes * 20)  # 20x to ensure enough after resampling

                        if required_bars > limit:
                            limit = required_bars
                            logger.info(f"Increased data limit to {limit} bars for {max_tf_minutes}m max timeframe (MTF indicators: {has_multi_tf_indicators})")
                        else:
                            logger.info(f"Detected multi-timeframe indicators with max TF {max_tf_minutes}m (limit {limit} sufficient)")
            except Exception as e:
                logger.warning(f"Could not check for multi-timeframe indicators: {e}")
                has_multi_tf_indicators = False

            # Get data - prefer candles_override if sufficient, else fetch from DB
            if isinstance(self.payload.get("candles_override"), (list, tuple)):
                df_candles = self._dict_to_candles(self.payload["candles_override"])
                # If candles_override too small, fetch from DB
                if len(df_candles) < limit:
                    logger.info(f"candles_override has {len(df_candles)} rows, fetching {limit} from DB")
                    df_candles_db = self._fetch_recent_candles(
                        self.market_service.engine, symbol, tf, limit, anchor_ts
                    )
                    if df_candles_db is not None and not df_candles_db.empty:
                        df_candles = df_candles_db
                        logger.debug(f"Fetched {len(df_candles)} candles from DB")
                else:
                    logger.info(f"Using candles_override with {len(df_candles)} rows for multi-timeframe indicators")
            else:
                # No override provided, fetch from DB
                logger.info(f"No candles_override provided, fetching {limit} from DB")
                df_candles = self._fetch_recent_candles(
                    self.market_service.engine, symbol, tf, limit, anchor_ts
                )

            if df_candles is None or df_candles.empty:
                raise RuntimeError("No candles available for parallel inference")

            # Normalize data
            df_candles = df_candles.sort_values("ts_utc").reset_index(drop=True)
            df_candles["ts_utc"] = pd.to_numeric(df_candles["ts_utc"], errors="coerce").astype("int64")

            # Rename 'price' to 'close' for tick data - feature functions expect 'close' column
            if "price" in df_candles.columns and "close" not in df_candles.columns:
                df_candles["close"] = df_candles["price"]
                df_candles["open"] = df_candles["price"]
                df_candles["high"] = df_candles["price"]
                df_candles["low"] = df_candles["price"]
                logger.debug("Converted tick 'price' data to OHLC format for feature calculation")

            # Store full dataframe for feature calculation - DO NOT filter by anchor yet
            # Multi-timeframe indicators need full historical data
            df_candles_full = df_candles.copy()

            # Find the index of the anchor timestamp for later extraction
            anchor_idx = None
            if anchor_ts is not None:
                try:
                    anchor_idx = df_candles_full[df_candles_full["ts_utc"] <= int(anchor_ts)].index[-1]
                    logger.debug(f"Anchor timestamp {anchor_ts} corresponds to index {anchor_idx}")
                except (IndexError, KeyError):
                    raise RuntimeError(f"No candles available at anchor timestamp {anchor_ts}")

            logger.debug(f"Parallel inference: {len(df_candles_full)} total candles for feature calculation")

            # Prepare features using the same system as single model inference
            from ...training.train_sklearn import _relative_ohlc, _temporal_feats, _realized_vol_feature, _indicators, _coerce_indicator_tfs
            from ...features.feature_cache import get_feature_cache

            # Feature cache and config (reuse from single model logic)
            feature_cache = get_feature_cache()

            # Extract multi-timeframe config from first model's metadata
            model_indicator_tfs = {}
            try:
                model_paths = self.payload.get("model_paths", [])
                if model_paths and len(model_paths) > 0:
                    from ...models.standardized_loader import get_model_loader
                    loader = get_model_loader()
                    first_model_data = loader.load_single_model(str(model_paths[0]))
                    metadata = first_model_data.get('metadata', {})
                    # metadata can be ModelMetadata object or dict
                    if hasattr(metadata, 'multi_timeframe_config'):
                        mtf_config = metadata.multi_timeframe_config
                    else:
                        mtf_config = metadata.get('multi_timeframe_config', {}) if isinstance(metadata, dict) else {}

                    if mtf_config and 'indicator_tfs' in mtf_config:
                        model_indicator_tfs = mtf_config['indicator_tfs']
                        logger.info(f"✓ Parallel inference using multi-timeframe config from model metadata: {len(model_indicator_tfs)} indicators")
                        logger.debug(f"  indicator_tfs: {model_indicator_tfs}")
                    else:
                        logger.warning(f"✗ No indicator_tfs found in model metadata: mtf_config={mtf_config}")
            except Exception as e:
                logger.warning(f"✗ Could not extract indicator_tfs from model metadata for parallel inference: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            indicator_tfs_to_use = model_indicator_tfs if model_indicator_tfs else self.payload.get("indicator_tfs", {})
            logger.info(f"Using indicator_tfs: {indicator_tfs_to_use if indicator_tfs_to_use else 'NONE'}")

            # Get model_paths for metadata lookup
            model_paths = self.payload.get("model_paths", [])

            cache_config = {
                "use_relative_ohlc": self.payload.get("use_relative_ohlc", True),
                "use_temporal_features": self.payload.get("use_temporal_features", True),
                "rv_window": int(self._get_param_with_override("rv_window", 60, model_paths)),
                "indicator_tfs": indicator_tfs_to_use,
                "advanced": self.payload.get("advanced", False),
                "atr_n": int(self.payload.get("atr_n", 14)),
                "rsi_n": int(self.payload.get("rsi_n", 14)),
                "bb_n": int(self.payload.get("bb_n", 20))
            }

            # Check cache first - use FULL dataframe for cache key
            cached_result = feature_cache.get_cached_features(df_candles_full, cache_config, tf)

            if cached_result is not None:
                feats_df, feature_metadata = cached_result
                logger.debug(f"Features loaded from cache for parallel inference {symbol} {tf}")
            else:
                # Compute features (cache miss) - use FULL dataframe for multi-timeframe indicators
                feats_list = []

                # Build features using same logic as single model
                if cache_config["use_relative_ohlc"]:
                    feats_list.append(_relative_ohlc(df_candles_full))

                if cache_config["use_temporal_features"]:
                    feats_list.append(_temporal_feats(df_candles_full))

                if cache_config["rv_window"] > 1:
                    feats_list.append(_realized_vol_feature(df_candles_full, cache_config["rv_window"]))

                # Multi-timeframe indicators - MUST use full dataframe for resampling
                indicator_tfs_raw = cache_config["indicator_tfs"]
                indicator_tfs = _coerce_indicator_tfs(indicator_tfs_raw)

                if indicator_tfs:
                    # Build full indicator config from model metadata
                    ind_cfg = {}
                    if "atr" in indicator_tfs:
                        ind_cfg["atr"] = {"n": cache_config["atr_n"]}
                    if "rsi" in indicator_tfs:
                        ind_cfg["rsi"] = {"n": cache_config["rsi_n"]}
                    if "bollinger" in indicator_tfs:
                        ind_cfg["bollinger"] = {"n": cache_config["bb_n"], "dev": 2.0}
                    if "macd" in indicator_tfs:
                        ind_cfg["macd"] = {"fast": 12, "slow": 26, "signal": 9}
                    if "stochastic" in indicator_tfs:
                        ind_cfg["stochastic"] = {"k": 14, "d": 3, "smooth_k": 3}
                    if "cci" in indicator_tfs:
                        ind_cfg["cci"] = {"n": 20}
                    if "williamsr" in indicator_tfs:
                        ind_cfg["williamsr"] = {"n": 14}
                    if "adx" in indicator_tfs:
                        ind_cfg["adx"] = {"n": 14}
                    if "mfi" in indicator_tfs:
                        ind_cfg["mfi"] = {"n": 14}
                    if "obv" in indicator_tfs:
                        ind_cfg["obv"] = {}
                    if "trix" in indicator_tfs:
                        ind_cfg["trix"] = {"n": 15}
                    if "ultimate" in indicator_tfs:
                        ind_cfg["ultimate"] = {"short": 7, "medium": 14, "long": 28}
                    if "donchian" in indicator_tfs:
                        ind_cfg["donchian"] = {"n": 20}
                    if "keltner" in indicator_tfs:
                        ind_cfg["keltner"] = {"n": 20, "atr_n": 10, "mult": 1.5}
                    if "ema" in indicator_tfs:
                        ind_cfg["ema"] = {"n": 20}
                    if "sma" in indicator_tfs:
                        ind_cfg["sma"] = {"n": 20}
                    if "hurst" in indicator_tfs:
                        ind_cfg["hurst"] = {"window": 64}
                    if "vwap" in indicator_tfs:
                        ind_cfg["vwap"] = {}

                    if ind_cfg:
                        # Calculate days_history from limit for better data fetching
                        # For 1h TF: 24 bars = 1 day, for 1m TF: 1440 bars = 1 day
                        tf_bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1, "1w": 1/7}
                        bars_per_day = tf_bars_per_day.get(tf, 24)
                        days_history_est = max(1, int(limit / bars_per_day)) if bars_per_day > 0 else 7
                        feats_list.append(_indicators(df_candles_full, ind_cfg, indicator_tfs, tf, symbol=symbol, days_history=days_history_est))

                if not feats_list:
                    # Fallback to basic features
                    feats_list.append(_relative_ohlc(df_candles))
                    feats_list.append(_temporal_feats(df_candles))

                # Combine features
                feats_df = pd.concat(feats_list, axis=1)
                feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

                # Debug: log ATR features before filtering
                atr_features = [c for c in feats_df.columns if 'atr' in str(c)]
                logger.debug(f"ATR features before filtering: {atr_features}")
                if atr_features:
                    atr_coverage = feats_df[atr_features].notna().mean()
                    logger.debug(f"ATR coverage: {dict(atr_coverage)}")

                # Save to cache
                feature_metadata = {
                    "config": cache_config,
                    "timestamp": df_candles_full["ts_utc"].iat[-1] if len(df_candles_full) > 0 else 0,
                    "symbol": symbol,
                    "timeframe": tf
                }

                try:
                    feature_cache.cache_features(df_candles_full, feats_df, feature_metadata, cache_config, tf)
                except Exception as e:
                    logger.warning(f"Failed to cache features for parallel inference: {e}")

            # Apply preprocessing (same as single model) on FULL feature set
            coverage = feats_df.notna().mean()
            min_cov = float(self._get_param_with_override("min_feature_coverage", 0.15, model_paths))
            if min_cov > 0.0:
                low_cov = coverage[coverage < min_cov]
                if not low_cov.empty:
                    logger.debug(f"Dropping {len(low_cov)} low-coverage features (< {min_cov}): {list(low_cov.index)}")
                    feats_df = feats_df.drop(columns=list(low_cov.index), errors="ignore")

            # Fill NaN values with forward fill, then backward fill, then 0
            # This is necessary for multi-timeframe features which have sparse coverage
            feats_df = feats_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Apply warmup on full dataset
            # Check if override is disabled - if so, try to load from model metadata
            warmup_bars = self._get_param_with_override("warmup_bars", 16, model_paths)
            if warmup_bars > 0 and len(feats_df) > warmup_bars:
                feats_df = feats_df.iloc[warmup_bars:]

            if feats_df.empty:
                raise RuntimeError("No features computed for parallel inference")

            # NOW extract only the row at anchor index (after all preprocessing)
            if anchor_idx is not None:
                # Adjust anchor_idx if warmup was applied
                adjusted_idx = anchor_idx - warmup_bars if warmup_bars > 0 else anchor_idx
                if adjusted_idx >= 0 and adjusted_idx < len(feats_df):
                    feats_df = feats_df.iloc[adjusted_idx:adjusted_idx+1]  # Extract single row as DataFrame
                    logger.debug(f"Extracted feature row at anchor index {adjusted_idx} (original: {anchor_idx})")
                else:
                    raise RuntimeError(f"Anchor index {adjusted_idx} out of range after preprocessing (feats_df has {len(feats_df)} rows)")

            # Setup parallel inference
            model_paths = self.payload.get("model_paths", [])
            if not model_paths:
                raise RuntimeError("No model paths provided for parallel inference")

            max_workers = int(self.payload.get("max_parallel_workers", 4))
            max_workers = min(max_workers, len(model_paths))

            # Create parallel settings
            parallel_settings = {
                "model_paths": model_paths,
                "max_workers": max_workers,
                "timeout": 120,  # 2 minutes timeout
                "fallback_enabled": True
            }

            # Initialize parallel engine
            parallel_engine = get_parallel_engine(max_workers)

            # Run parallel inference
            parallel_results = parallel_engine.run_parallel_inference(
                parallel_settings, feats_df, symbol, tf, horizons_raw
            )

            # Check if we should combine models or keep them separate
            combine_models = self.payload.get("combine_models", True)

            # Extract ensemble predictions
            ensemble_preds = parallel_results.get("ensemble_predictions")
            if ensemble_preds is None:
                raise RuntimeError("Parallel inference failed to produce ensemble predictions")

            # Get individual results for separate forecasts
            individual_results = parallel_results.get("individual_results", [])

            # Convert ensemble predictions to price forecasts
            last_close = float(df_candles["close"].iat[-1])
            mean_returns = np.array(ensemble_preds["mean"])
            std_returns = np.array(ensemble_preds["std"])

            # Scale to match horizon count
            if len(mean_returns) != len(horizons_bars):
                if len(mean_returns) == 1:
                    # Single prediction, scale for each horizon
                    base_return = mean_returns[0]
                    base_std = std_returns[0]
                    mean_returns = []
                    std_returns = []
                    for bars in horizons_bars:
                        scale_factor = bars / horizons_bars[0] if horizons_bars[0] > 0 else 1.0
                        mean_returns.append(base_return * scale_factor)
                        std_returns.append(base_std * np.sqrt(scale_factor))
                    mean_returns = np.array(mean_returns)
                    std_returns = np.array(std_returns)
                else:
                    # Extend or truncate to match
                    target_len = len(horizons_bars)
                    mean_returns = np.pad(mean_returns, (0, max(0, target_len - len(mean_returns))), mode='edge')[:target_len]
                    std_returns = np.pad(std_returns, (0, max(0, target_len - len(std_returns))), mode='edge')[:target_len]

            # Convert to prices
            prices = []
            p = last_close
            for r in mean_returns:
                p *= (1.0 + float(r))
                prices.append(p)
            q50 = np.asarray(prices, dtype=float)

            # Calculate bands using ensemble uncertainty
            z = 1.645 if bool(self.payload.get("apply_conformal", True)) else 1.0
            band_rel = np.clip(z * std_returns, 1e-6, 0.2)

            q05 = np.maximum(1e-12, q50 * (1.0 - band_rel))
            q95 = np.maximum(1e-12, q50 * (1.0 + band_rel))

            # Create future timestamps
            last_ts_ms = int(df_candles["ts_utc"].iat[-1])
            future_ts = create_future_timestamps(last_ts_ms, tf, horizons_time_labels)

            display_name = str(self.payload.get("name") or "Parallel Ensemble")
            quantiles = {
                "q50": q50.tolist(),
                "q05": q05.tolist(),
                "q95": q95.tolist(),
                "future_ts": future_ts,
                "source": display_name,
                "label": display_name,
                "model_path_used": ", ".join([str(Path(p).name) for p in model_paths[:3]]) + ("..." if len(model_paths) > 3 else ""),
                "model_sha16": None,  # Not applicable for ensemble
                "ensemble_info": {
                    "model_count": len(model_paths),
                    "execution_summary": parallel_results.get("execution_summary", {}),
                    "individual_predictions": ensemble_preds.get("individual", [])
                },
                "parallel_inference": True,
                "requested_at_ms": self.payload.get("requested_at_ms"),  # For connecting forecast to price line
                "anchor_price": self.payload.get("anchor_price")  # Pass through Alt+Click Y coordinate
            }

            logger.info(f"Parallel inference completed for {symbol} {tf}: "
                       f"{len(model_paths)} models, mean accuracy: {parallel_results.get('mean_accuracy', 'N/A')}")

            # If combine_models is False, emit individual forecasts for each model
            if not combine_models and individual_results:
                logger.info(f"Emitting {len([r for r in individual_results if r.get('success')])} separate forecasts (combine_models=False)")
                for idx, result in enumerate(individual_results):
                    if not result.get('success'):
                        continue

                    # Extract model-specific predictions
                    model_path = result.get('model_path', '')
                    model_name = Path(model_path).stem if model_path else f"Model_{idx+1}"
                    predictions = result.get('predictions', [])

                    if predictions is None or len(predictions) == 0:
                        continue

                    # Convert returns to prices for this model
                    model_prices = []
                    p = last_close
                    for r in predictions:
                        p *= (1.0 + float(r))
                        model_prices.append(p)

                    # Pad to match horizons if needed
                    if len(model_prices) < len(horizons_bars):
                        model_prices.extend([model_prices[-1]] * (len(horizons_bars) - len(model_prices)))
                    model_prices = model_prices[:len(horizons_bars)]

                    q50_model = np.asarray(model_prices, dtype=float)

                    # Simple confidence bands (±5% for individual models without ensemble uncertainty)
                    q05_model = q50_model * 0.95
                    q95_model = q50_model * 1.05

                    # Create quantiles for this model
                    model_quantiles = {
                        "q50": q50_model.tolist(),
                        "q05": q05_model.tolist(),
                        "q95": q95_model.tolist(),
                        "future_ts": future_ts,
                        "source": model_name,
                        "label": model_name,
                        "model_path_used": Path(model_path).name if model_path else model_name,
                        "model_sha16": None,
                        "parallel_inference": True,
                        "separate_model": True,  # Mark as individual model forecast
                        "requested_at_ms": self.payload.get("requested_at_ms"),  # For connecting forecast to price line
                        "anchor_price": self.payload.get("anchor_price")
                    }

                    # Emit individual forecast
                    self.signals.forecastReady.emit(df_candles, model_quantiles)

                # Return the ensemble as well (but it was already emitted individually)
                return df_candles, quantiles

            # Default: return combined ensemble
            return df_candles, quantiles

        except Exception as e:
            logger.error(f"Parallel inference failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # NO FALLBACK! Parallel inference must succeed or fail completely
            # RW is only for benchmarking model efficiency, NOT as a fallback predictor
            raise