# ui/controllers.py
# Controller to bind UI menu actions to background workers and services.
from __future__ import annotations

# ensure-features helper (robust import)
from typing import Optional, Callable, List, Dict, Tuple
_ensure_features_for_prediction: Optional[Callable] = None
try:
    from forex_diffusion.inference.prediction_config import ensure_features_for_prediction as _ensure_features_for_prediction
except Exception:
    try:
        from ..inference.prediction_config import ensure_features_for_prediction as _ensure_features_for_prediction  # type: ignore
    except Exception:
        _ensure_features_for_prediction = None  # no-op fallback

from pathlib import Path

import os
import sys
import pandas as pd
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Slot
from loguru import logger

from ..services.marketdata import MarketDataService
from .prediction_settings_dialog import PredictionSettingsDialog

# pipeline & utils
try:
    from forex_diffusion.features.pipeline import pipeline_process, Standardizer
except Exception:
    from ..features.pipeline import pipeline_process, Standardizer  # type: ignore

# ensure-features helper
try:
    from forex_diffusion.inference.prediction_config import ensure_features_for_prediction
except Exception:
    ensure_features_for_prediction = None  # type: ignore

# --- warmup import per evitare deadlock di unpickle in thread ---
def _preimport_sklearn_for_unpickle():
    try:
        import sklearn  # noqa: F401
        # Importa i tipi che compaiono nei tuoi artifact
        from sklearn.linear_model import Ridge, Lasso, ElasticNet  # noqa: F401
        from sklearn.ensemble import RandomForestRegressor         # noqa: F401
        # Se usi anche PCA nei payload:
        from sklearn.decomposition import PCA                      # noqa: F401
    except Exception:
        # Non bloccare l'app se sklearn non è installato; semplicemente niente warmup.
        pass


class UIControllerSignals(QObject):
    forecastReady = Signal(object, object)  # (pd.DataFrame, quantiles_dict)
    error = Signal(str)
    status = Signal(str)


# ----------------------------- Forecast Worker ----------------------------- #

class ForecastWorker(QRunnable):
    """
    Esegue inferenza locale:
     - recupera ultime candele dal DB
     - carica il modello dal path risolto
     - costruisce features senza rifittare alcun standardizer
     - emette forecastReady(df_candles, quantiles)
    """
    def __init__(self, engine_url: str, payload: dict, market_service: MarketDataService, signals: UIControllerSignals):
        super().__init__()
        self.engine_url = engine_url.rstrip("/")
        self.payload = payload
        self.market_service = market_service
        self.signals = signals

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
            from ..utils.config import get_config
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
        from ..models.model_path_resolver import ModelPathResolver

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

    # --------------------------- local inference --------------------------- #

    def _local_infer(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Fallback locale (basic). Flusso:
        candele -> pipeline (NO standardization fit) -> ensure -> z-score con μ/σ del modello -> predict -> prezzi/ts -> quantili
        """
        # Check if this is a parallel inference request
        if self.payload.get("parallel_inference", False):
            return self._parallel_infer()

        import numpy as np
        import pickle
        import hashlib

        from ..utils.horizon_converter import (
            convert_horizons_for_inference,
            create_future_timestamps,
            convert_single_to_multi_horizon,
            get_trading_scenarios
        )
        from ..services.performance_registry import get_performance_registry

        sym = self.payload.get("symbol")
        tf = (self.payload.get("timeframe") or "1m")
        horizons_raw = self.payload.get("horizons", ["5m"])
        limit = int(self.payload.get("limit_candles", 512))
        ftype = str(self.payload.get("forecast_type", "basic")).lower()

        # Converti horizons al formato corretto
        horizons_time_labels, horizons_bars = convert_horizons_for_inference(horizons_raw, tf)

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

        # Sorgente dati: override esplicito o fetch dal DB; se anchor_ts è definito, non includere barre successive
        if isinstance(self.payload.get("candles_override"), (list, tuple)):
            import pandas as _pd
            df_candles = _pd.DataFrame(self.payload["candles_override"]).copy()
        else:
            df_candles = self._fetch_recent_candles(
                self.market_service.engine, sym, tf,
                n_bars=limit,
                end_ts=anchor_ts if anchor_ts is not None else None
            )

        if df_candles is None or df_candles.empty:
            raise RuntimeError("No candles available for local inference")

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
        features_list: List[str] = []
        std_mu: Dict[str, float] = {}
        std_sigma: Dict[str, float] = {}
        if ftype != "rw":
            from ..models.standardized_loader import get_model_loader

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
                logger.error(f"Standardized loader failed, falling back to legacy loader: {e}")
                # Fallback to legacy loading
                with open(p, "rb") as f:
                    payload_obj = pickle.load(f)
                model = payload_obj.get("model")
                features_list = payload_obj.get("features") or []
                std_mu = payload_obj.get("std_mu") or payload_obj.get("std") or {}
                std_sigma = payload_obj.get("std_sigma") or ({c: 1.0 for c in features_list} if "std" in payload_obj else {})

                # Calculate model hash
                try:
                    model_sha16 = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
                except Exception:
                    model_sha16 = None

                if model is None or not features_list:
                    raise RuntimeError("Model payload missing 'model' or 'features'")
        else:
            # RW baseline
            std_mu, std_sigma = {}, {}

        # 3) features usando STESSO sistema del training (train_sklearn.py) CON CACHING
        from ..training.train_sklearn import _relative_ohlc, _temporal_feats, _realized_vol_feature, _indicators, _coerce_indicator_tfs
        from ..features.feature_cache import get_feature_cache
        from ..features.unified_pipeline import FeatureConfig, hierarchical_multi_timeframe_pipeline

        # Configura la cache delle features
        feature_cache = get_feature_cache()

        # Crea configurazione per il caching
        cache_config = {
            "use_relative_ohlc": self.payload.get("use_relative_ohlc", True),
            "use_temporal_features": self.payload.get("use_temporal_features", True),
            "rv_window": int(self.payload.get("rv_window", 60)),
            "indicator_tfs": self.payload.get("indicator_tfs", {}),
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
            logger.debug(f"Computing features for {symbol} {tf} (cache miss)")

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
                        feats_list.append(_indicators(df_candles, ind_cfg, indicator_tfs, tf))

                if not feats_list:
                    raise RuntimeError("No features configured for inference")

                # Combina tutte le features (come in training)
                feats_df = pd.concat(feats_list, axis=1)
                feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

                # Salva in cache per riuso futuro (solo per sistema tradizionale)
                feature_metadata = {
                    "config": cache_config,
                    "timestamp": df_candles["ts_utc"].iat[-1] if len(df_candles) > 0 else 0,
                    "symbol": symbol,
                    "timeframe": tf
                }

                try:
                    feature_cache.cache_features(df_candles, feats_df, feature_metadata, cache_config, tf)
                    logger.debug(f"Features cached for {symbol} {tf}")
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
        import numpy as np
        preds = None
        if model is not None:
            # Usa l'ultimo sample per la predizione
            X_last = X_arr[-1:, :]  # Shape: (1, n_features)

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
            "use_enhanced_scaling": use_enhanced_scaling
        }
        return df_candles, quantiles

    def _parallel_infer(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Parallel inference using multiple models for ensemble predictions.
        """
        try:
            from ..inference.parallel_inference import get_parallel_engine
            from ..utils.horizon_converter import convert_horizons_for_inference, create_future_timestamps

            symbol = str(self.payload.get("symbol", "EUR/USD"))
            tf = str(self.payload.get("timeframe", "1m"))
            horizons_raw = self.payload.get("horizons", ["5m"])
            limit = int(self.payload.get("limit_candles", 512))

            logger.info(f"Starting parallel inference for {symbol} {tf}")

            # Converti horizons al formato corretto
            horizons_time_labels, horizons_bars = convert_horizons_for_inference(horizons_raw, tf)

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

            # Get data
            if isinstance(self.payload.get("candles_override"), (list, tuple)):
                df_candles = self._dict_to_candles(self.payload["candles_override"])
            else:
                df_candles = self._fetch_recent_candles(
                    self.engine, symbol, tf, limit, anchor_ts
                ).copy()

            if df_candles.empty:
                raise RuntimeError("No candles available for parallel inference")

            # Prepare features using the same system as single model inference
            # (reuse the feature computation logic from _local_infer)
            from ..training.train_sklearn import _relative_ohlc, _temporal_feats, _realized_vol_feature, _indicators, _coerce_indicator_tfs
            from ..features.feature_cache import get_feature_cache
            from ..features.unified_pipeline import FeatureConfig, hierarchical_multi_timeframe_pipeline

            # Configure feature cache
            feature_cache = get_feature_cache()

            # Create configuration for caching
            cache_config = {
                "use_relative_ohlc": self.payload.get("use_relative_ohlc", True),
                "use_temporal_features": self.payload.get("use_temporal_features", True),
                "rv_window": int(self.payload.get("rv_window", 60)),
                "indicator_tfs": self.payload.get("indicator_tfs", {}),
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

            # Check if using hierarchical multi-timeframe
            use_hierarchical = self.payload.get("use_hierarchical_multitf", False)
            query_timeframe = self.payload.get("query_timeframe", tf)

            # Check cache first
            cached_result = feature_cache.get_cached_features(df_candles, cache_config, tf)

            if cached_result is not None:
                feats_df, feature_metadata = cached_result
                logger.debug(f"Features loaded from cache for parallel inference {symbol} {tf}")
            else:
                # Compute features (same as single model)

                if use_hierarchical:
                    # USA SISTEMA HIERARCHICAL MULTI-TIMEFRAME nel parallel inference
                    logger.info(f"Using hierarchical multi-timeframe in parallel inference: query_tf={query_timeframe}")

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

                    logger.info(f"Hierarchical pipeline in parallel inference completed: {len(feature_names)} features, {len(feats_df)} samples")

                else:
                    # Sistema tradizionale per parallel inference
                    feats_list = []

                    if cache_config["use_relative_ohlc"]:
                        feats_list.append(_relative_ohlc(df_candles))

                    if cache_config["use_temporal_features"]:
                        feats_list.append(_temporal_feats(df_candles))

                    if cache_config["rv_window"] > 1:
                        feats_list.append(_realized_vol_feature(df_candles, cache_config["rv_window"]))

                    # Indicators
                    indicator_tfs_raw = cache_config["indicator_tfs"]
                    indicator_tfs = _coerce_indicator_tfs(indicator_tfs_raw)

                    is_advanced = cache_config["advanced"] or cache_config["use_advanced_features"]

                    if is_advanced:
                        if not indicator_tfs:
                            indicator_tfs = {}

                        if cache_config["enable_ema_features"]:
                            indicator_tfs.setdefault("ema", [tf])
                        if cache_config["enable_donchian"]:
                            indicator_tfs.setdefault("donchian", [tf])
                        if cache_config["enable_keltner"]:
                            indicator_tfs.setdefault("keltner", [tf])
                        if cache_config["enable_hurst_advanced"]:
                            indicator_tfs.setdefault("hurst", [tf])

                    if indicator_tfs:
                        # Same indicator config as single model
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
                            feats_list.append(_indicators(df_candles, ind_cfg, indicator_tfs, tf))

                    if not feats_list:
                        raise RuntimeError("No features configured for parallel inference")

                    feats_df = pd.concat(feats_list, axis=1)
                    feats_df = feats_df.replace([np.inf, -np.inf], np.nan)

                    # Cache the features (solo per sistema tradizionale)
                    feature_metadata = {
                        "config": cache_config,
                        "timestamp": df_candles["ts_utc"].iat[-1] if len(df_candles) > 0 else 0,
                        "symbol": symbol,
                        "timeframe": tf
                    }

                    try:
                        feature_cache.cache_features(df_candles, feats_df, feature_metadata, cache_config, tf)
                    except Exception as e:
                        logger.warning(f"Failed to cache features for parallel inference: {e}")

            # Coverage filtering
            coverage = feats_df.notna().mean()
            min_cov = float(self.payload.get("min_feature_coverage", 0.15))
            keep_feats = coverage[coverage >= min_cov].index.tolist()
            if not keep_feats:
                logger.warning("No features meet coverage requirement, using all features")
                keep_feats = feats_df.columns.tolist()

            feats_df = feats_df[keep_feats]

            # Initialize parallel engine
            max_workers = self.payload.get("max_parallel_workers", None)
            parallel_engine = get_parallel_engine(max_workers)

            # Create settings for parallel inference
            parallel_settings = {
                "model_paths": self.payload.get("model_paths", [])
            }

            # Run parallel inference
            parallel_results = parallel_engine.run_parallel_inference(
                parallel_settings,
                feats_df,
                symbol,
                tf,
                horizons_raw
            )

            # Extract ensemble predictions
            ensemble_preds = parallel_results.get("ensemble_predictions")
            if ensemble_preds is None:
                raise RuntimeError("Parallel inference failed to produce ensemble predictions")

            # Convert ensemble predictions to price forecasts
            last_close = float(df_candles["close"].iat[-1])
            mean_returns = np.array(ensemble_preds["mean"])
            std_returns = np.array(ensemble_preds["std"])

            # Convert returns to prices
            prices = []
            p = last_close
            for r in mean_returns:
                p *= (1.0 + float(r))
                prices.append(p)
            q50 = np.array(prices)

            # Create confidence bands using ensemble uncertainty
            confidence_level = 1.645 if self.payload.get("apply_conformal", True) else 1.0

            # Calculate price bands based on return volatility
            price_bands_lower = []
            price_bands_upper = []
            p = last_close

            for i, (r_mean, r_std) in enumerate(zip(mean_returns, std_returns)):
                r_lower = r_mean - confidence_level * r_std
                r_upper = r_mean + confidence_level * r_std

                p_mean = p * (1.0 + r_mean)
                p_lower = p * (1.0 + r_lower)
                p_upper = p * (1.0 + r_upper)

                price_bands_lower.append(max(1e-12, p_lower))
                price_bands_upper.append(max(1e-12, p_upper))

                p = p_mean  # Update base price for next horizon

            q05 = np.array(price_bands_lower)
            q95 = np.array(price_bands_upper)

            # Create future timestamps
            last_ts_ms = int(df_candles["ts_utc"].iat[-1])
            future_ts = create_future_timestamps(last_ts_ms, tf, horizons_time_labels)

            # Create quantiles result with ensemble information
            display_name = str(self.payload.get("name") or "Parallel Ensemble")
            quantiles = {
                "q50": q50.tolist(),
                "q05": q05.tolist(),
                "q95": q95.tolist(),
                "future_ts": future_ts,
                "source": display_name,
                "label": display_name,
                "ensemble_info": {
                    "model_count": parallel_results.get("total_models", 0),
                    "successful_models": parallel_results.get("successful_models", 0),
                    "model_weights": parallel_results.get("model_weights", []),
                    "execution_summary": parallel_results.get("execution_summary", {}),
                    "individual_predictions": ensemble_preds.get("individual", [])
                },
                "parallel_inference": True
            }

            logger.info(f"Parallel inference completed for {symbol} {tf}: "
                       f"{parallel_results.get('successful_models', 0)}/{parallel_results.get('total_models', 0)} models succeeded")

            return df_candles, quantiles

        except Exception as e:
            logger.error(f"Parallel inference failed: {e}")
            # Fallback to RW prediction
            import numpy as np
            from ..utils.horizon_converter import create_future_timestamps

            symbol = str(self.payload.get("symbol", "EUR/USD"))
            tf = str(self.payload.get("timeframe", "1m"))
            horizons_raw = self.payload.get("horizons", ["5m"])

            # Get basic candle data for fallback
            try:
                df_candles = self._fetch_recent_candles(self.engine, symbol, tf, 100, None).copy()
                if df_candles.empty:
                    raise RuntimeError("No candles available")

                last_close = float(df_candles["close"].iat[-1])
                logret = pd.Series(df_candles["close"], dtype=float).pipe(lambda s: np.log(s).diff()).dropna()
                sigma = float(logret.tail(100).std() if len(logret) else 0.01)

                # Simple random walk forecasts
                n_horizons = len(horizons_raw)
                random_walks = np.random.randn(n_horizons) * sigma * 0.1
                prices = [last_close * (1 + rw) for rw in random_walks]

                # Convert horizons for timestamps
                from ..utils.horizon_converter import convert_horizons_for_inference
                horizons_time_labels, _ = convert_horizons_for_inference(horizons_raw, tf)

                last_ts_ms = int(df_candles["ts_utc"].iat[-1])
                future_ts = create_future_timestamps(last_ts_ms, tf, horizons_time_labels)

                quantiles = {
                    "q50": prices,
                    "q05": [p * 0.95 for p in prices],
                    "q95": [p * 1.05 for p in prices],
                    "future_ts": future_ts,
                    "source": "Parallel Ensemble (Fallback)",
                    "label": "Parallel Ensemble (Fallback)",
                    "error": str(e),
                    "parallel_inference": True,
                    "fallback": True
                }

                return df_candles, quantiles

            except Exception as fallback_error:
                logger.error(f"Parallel inference fallback also failed: {fallback_error}")
                raise RuntimeError(f"Parallel inference failed: {e}. Fallback also failed: {fallback_error}")


# ----------------------------- UI Controller ------------------------------ #

class UIController:
    def __init__(self, main_window, market_service=None, engine_url="http://127.0.0.1:8000", db_writer=None):
        # Pre-import nel main thread per evitare deadlock nei worker
        try:
            _preimport_sklearn_for_unpickle()
        except Exception:
            pass
        self.main_window = main_window
        self.market_service = market_service or MarketDataService()
        self.engine_url = engine_url
        self.signals = UIControllerSignals()
        self.pool = QThreadPool.globalInstance()
        try:
            self.pool.setMaxThreadCount(max(2, min(self.pool.maxThreadCount(), 4)))
        except Exception:
            pass
        self.db_writer = db_writer
        self._forecast_active = 0
        # default indicators settings
        self.indicators_settings: dict = {
            "use_atr": True, "atr_n": 14,
            "use_rsi": True, "rsi_n": 14,
            "use_bollinger": True, "bb_n": 20, "bb_k": 2,
            "use_hurst": True, "hurst_window": 64,
        }
        # Load persisted indicators settings if available
        try:
            from ..utils.user_settings import get_setting
            saved = get_setting("indicators.last_settings", {}) or {}
            if isinstance(saved, dict) and saved:
                self.indicators_settings.update(saved)
        except Exception:
            pass
        try:
            self.signals.forecastReady.connect(self._on_forecast_finished)
            self.signals.error.connect(self._on_forecast_failed)
        except Exception:
            pass

            # === Menu handlers (stub/implementazioni leggere) ====================== #

            @Slot()
            def handle_ingest_requested(self):
                """Esegue un backfill rapido in background usando MarketDataService."""
                from PySide6.QtCore import QRunnable
                from loguru import logger

                class _IngestRunner(QRunnable):
                    def __init__(self, outer):
                        super().__init__()
                        self.outer = outer

                def run(self):
                    try:
                        self.outer.signals.status.emit("Backfill: running...")
                        # simboli di default (o prova a leggere da settings)
                        try:
                            from ..utils.user_settings import get_setting
                            symbols = get_setting("user_symbols", []) or ["EUR/USD"]
                        except Exception:
                            symbols = ["EUR/USD"]
                        for sym in symbols:
                            try:
                                # abilita REST se il servizio lo supporta
                                if hasattr(self.outer.market_service, "rest_enabled"):
                                    setattr(self.outer.market_service, "rest_enabled", True)
                                # fai almeno il daily per popolare il DB
                                self.outer.market_service.backfill_symbol_timeframe(sym, "1d", force_full=False)
                                self.outer.signals.status.emit(f"Backfill {sym}: done")
                            except Exception as e:
                                logger.warning("Backfill error for {}: {}", sym, e)
                        # spegni REST
                        try:
                            if hasattr(self.outer.market_service, "rest_enabled"):
                                setattr(self.outer.market_service, "rest_enabled", False)
                        except Exception:
                            pass
                        self.outer.signals.status.emit("Backfill: completed")
                    except Exception as e:
                        logger.exception("Backfill failed: {}", e)
                        self.outer.signals.error.emit(str(e))
                        self.outer.signals.status.emit("Backfill failed")

            self.pool.start(_IngestRunner(self))

        @Slot()
        def handle_train_requested(self):
            """Apre la Training tab se disponibile, altrimenti logga."""
            self.signals.status.emit("Train requested")
            try:
                if hasattr(self.main_window, "open_training_tab"):
                    self.main_window.open_training_tab()
                elif hasattr(self.main_window, "tabs") and hasattr(self.main_window.tabs, "show_training"):
                    self.main_window.tabs.show_training()
                else:
                    # Nessun tab manager: logga soltanto
                    from loguru import logger
                    logger.info("Training tab not wired; use Training menu to configure.")
            except Exception:
                pass

        @Slot()
        def handle_calibration_requested(self):
            self.signals.status.emit("Calibration requested (not implemented).")

        @Slot()
        def handle_backtest_requested(self):
            self.signals.status.emit("Backtest requested (not implemented).")

        @Slot(bool)
        def handle_realtime_toggled(self, enabled: bool):
            self.signals.status.emit(f"Realtime toggled: {'ON' if enabled else 'OFF'}")

        @Slot()
        def handle_config_requested(self):
            self.signals.status.emit("Config requested (not implemented).")

        @Slot()
        def handle_prediction_settings_requested(self):
            """Apre la finestra delle impostazioni di prediction."""
            try:
                from .prediction_settings_dialog import PredictionSettingsDialog
                dialog = PredictionSettingsDialog(self.main_window)
                dialog.exec()
            except Exception as e:
                self.signals.error.emit(str(e))

    @Slot()
    def handle_prediction_settings_requested(self):
        dialog = PredictionSettingsDialog(self.main_window)
        dialog.exec()

    from PySide6.QtCore import Slot

    @Slot()
    def handle_indicators_requested(self):
        """
        Apre il dialog degli indicatori se disponibile.
        In fallback mostra un messaggio per confermare che il click funziona.
        """
        try:
            # tenta il dialog “ufficiale”
            from .indicators_dialog import IndicatorsDialog  # deve esistere nel tuo repo
            # Se hai uno stato locale per gli indicatori, leggilo/aggiorna qui:
            initial = getattr(self, "indicators_settings", {}) or {}
            res = IndicatorsDialog.edit(self.main_window, initial=initial)
            if res is None:
                self.signals.status.emit("Indicators: annullato")
                return
            # salva le scelte
            self.indicators_settings = dict(res)
            # persist between sessions
            try:
                from ..utils.user_settings import set_setting
                set_setting("indicators.last_settings", dict(self.indicators_settings))
            except Exception:
                pass
            self.signals.status.emit("Indicators aggiornati")
            try:
                from loguru import logger
                logger.info("Indicators updated: {}", self.indicators_settings)
            except Exception:
                pass
        except ModuleNotFoundError:
            # fallback: nessun dialog disponibile → mostra un messaggio
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self.main_window, "Indicators", "Dialog degli indicatori non disponibile.")
            except Exception:
                pass
            self.signals.status.emit("Indicators: dialog non disponibile")
        except Exception as e:
            self.signals.status.emit("Indicators: errore")
            try:
                from loguru import logger
                logger.exception("Indicators dialog failed: {}", e)
            except Exception:
                pass
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self.main_window, "Indicators", str(e))
            except Exception:
                pass

    # ---- Forecast da menu (usa le settings correnti) ---------------------- #
    @Slot()
    def handle_forecast_requested(self):
        settings = PredictionSettingsDialog.get_settings() or {}

        # Usa ModelPathResolver unificato
        from ..models.model_path_resolver import ModelPathResolver

        resolver = ModelPathResolver()
        models = resolver.resolve_model_paths(settings)

        if not models:
            self.signals.error.emit("Prediction settings not configured or model path(s) missing.")
            self.handle_prediction_settings_requested()
            return

        # symbol/timeframe dal chart se disponibili
        chart_tab = getattr(self, "chart_tab", None)
        if chart_tab and getattr(chart_tab, "symbol", None) and getattr(chart_tab, "timeframe", None):
            symbol, timeframe = chart_tab.symbol, chart_tab.timeframe
        else:
            symbol, timeframe = "EUR/USD", "1m"

        # label univoche per basename
        counts: Dict[str, int] = {}
        label_map: Dict[str, str] = {}
        for m in models:
            base = os.path.splitext(os.path.basename(m))[0]
            counts[base] = counts.get(base, 0) + 1
            label_map[m] = base if counts[base] == 1 else f"{base}#{counts[base]}"

        self.signals.status.emit(f"Forecast requested for {symbol} {timeframe} (models={len(models)})")
        logger.info("Forecast (menu) models: {}", {m: label_map[m] for m in models})

        # Determina forecast types basato sui settings del dialog
        forecast_types = settings.get("forecast_types", ["basic"])

        for mp in models:
            for forecast_type in forecast_types:
                # Determina se usare advanced features
                is_advanced = forecast_type == "advanced"

                payload = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "horizons": settings.get("horizons", ["1m","5m","15m"]),
                    "N_samples": settings.get("N_samples", 200),
                    "apply_conformal": settings.get("apply_conformal", True),
                    "limit_candles": settings.get("limit_candles", 512),
                    "model_path": mp,
                    "source_label": f"{label_map[mp]}_{forecast_type}",
                    "name": f"{label_map[mp]}_{forecast_type}",
                    "forecast_type": forecast_type,
                    "advanced": is_advanced,
                    "allowed_models": list(models),

                    # Advanced features configuration
                    "use_advanced_features": is_advanced,
                    "enable_ema_features": is_advanced,
                    "enable_donchian": is_advanced,
                    "enable_keltner": is_advanced,
                    "enable_hurst_advanced": is_advanced,

                    # Passa tutti i parametri dal dialog
                    **{k: v for k, v in settings.items() if k.startswith(('ema_', 'don_', 'keltner_', 'hurst_'))}
                }
            logger.info("Forecast (menu) launching: file='{}' label='{}'", mp, label_map[mp])
            fw = ForecastWorker(engine_url=self.engine_url, payload=payload, market_service=self.market_service, signals=self.signals)
            self._forecast_active += 1
            self.pool.start(fw)

    # ---- Forecast da ChartTab (payload) ----------------------------------- #
    @Slot(dict)
    def handle_forecast_payload(self, payload: dict):
        """
        Avvia una previsione usando il payload emesso dalla ChartTab.
        Supporta selezione multipla: lancia un worker per ogni file modello selezionato.
        """
        try:
            settings = PredictionSettingsDialog.get_settings() or {}
            chart_tab = getattr(self, "chart_tab", None)
            if chart_tab:
                payload.setdefault("symbol", getattr(chart_tab, "symbol", None))
                payload.setdefault("timeframe", getattr(chart_tab, "timeframe", None))

            # defaults
            payload.setdefault("horizons", settings.get("horizons", ["1m","5m","15m"]))
            payload.setdefault("N_samples", settings.get("N_samples", 200))
            payload.setdefault("apply_conformal", settings.get("apply_conformal", True))
            payload.setdefault("limit_candles", 512)

            logger.info("Forecast payload received: symbol={}, tf={}, has_model_path={}, has_model_paths={}, has_model={}, has_models={}",
                        payload.get("symbol"), payload.get("timeframe"),
                        bool(payload.get("model_path")), bool(payload.get("model_paths")),
                        bool(payload.get("model")), bool(payload.get("models")))
            self.signals.status.emit("Forecast: payload received")

            # Unione di TUTTE le sorgenti
            import re
            def _norm_paths(src) -> List[str]:
                try:
                    if not src: return []
                    if isinstance(src, (list, tuple, set)):
                        return [str(s).strip() for s in src if str(s).strip()]
                    s = str(src).strip()
                    if any(sep in s for sep in [",",";","\n"]):
                        return [t.strip() for t in re.split(r"[,\n;]+", s) if t.strip()]
                    return [s] if s else []
                except Exception:
                    return []

            models: List[str] = []
            models += _norm_paths(payload.get("model_paths"))
            models += _norm_paths(payload.get("models"))
            models += _norm_paths(settings.get("model_paths"))
            models += _norm_paths(settings.get("models"))
            try:
                models += _norm_paths(PredictionSettingsDialog.get_model_paths())
            except Exception:
                pass
            models += _norm_paths(payload.get("model_path") or payload.get("model") or settings.get("model_path") or settings.get("model"))

            if not payload.get("symbol") or not payload.get("timeframe"):
                self.signals.error.emit("Missing symbol/timeframe for forecast.")
                return
            if not models and payload.get("forecast_type") != "rw":
                self.signals.error.emit("Missing model_path(s). Open Prediction Settings.")
                return

            # normalizza + dedup (casefold su Windows)
            def _canon(p: str) -> str:
                try:
                    s = str(p).strip().strip('"').strip("'")
                    s = os.path.expandvars(os.path.expanduser(s))
                    if not os.path.isabs(s):
                        s = os.path.abspath(s)
                    return os.path.realpath(s)
                except Exception:
                    return str(p)
            canon = [_canon(m) for m in models if m]
            seen, models = set(), []
            for rp in canon:
                key = rp.lower() if sys.platform.startswith("win") else rp
                if key not in seen:
                    seen.add(key)
                    models.append(rp)

            # log esistenza (non blocca: il worker risolve)
            exist_map = {m: os.path.exists(m) for m in models}
            logger.info("Forecast models (normalized): {}", exist_map)
            missing = [m for m, ok in exist_map.items() if not ok]
            if missing:
                self.signals.status.emit(f"Forecast: {len(missing)} model path(s) not found on disk, attempting search")
                logger.warning("Forecast: missing paths (will attempt search/load anyway): {}", missing)

            # label uniche
            counts: Dict[str, int] = {}
            label_map: Dict[str, str] = {}
            for m in models:
                base = os.path.splitext(os.path.basename(m))[0]
                counts[base] = counts.get(base, 0) + 1
                label_map[m] = base if counts[base] == 1 else f"{base}#{counts[base]}"
            logger.info("Forecast (payload) models: {}", {m: label_map[m] for m in models})

            # tipo: se >1 modello forza basic
            adv = bool(payload.get("advanced", False))
            forecast_type = "advanced" if (adv and len(models) == 1) else "basic"

            # fallback: inietta indicator settings globali se mancanti
            try:
                ind = getattr(self, "indicators_settings", {}) or {}
                if "use_atr" not in payload: payload["use_atr"] = bool(ind.get("use_atr", True))
                if "atr_n" not in payload: payload["atr_n"] = int(ind.get("atr_n", 14))
                if "use_rsi" not in payload: payload["use_rsi"] = bool(ind.get("use_rsi", True))
                if "rsi_n" not in payload: payload["rsi_n"] = int(ind.get("rsi_n", 14))
                if "use_bollinger" not in payload: payload["use_bollinger"] = bool(ind.get("use_bollinger", True))
                if "bb_n" not in payload: payload["bb_n"] = int(ind.get("bb_n", 20))
                if "bb_k" not in payload: payload["bb_k"] = int(ind.get("bb_k", 2))
                if "use_hurst" not in payload: payload["use_hurst"] = bool(ind.get("use_hurst", True))
                if "hurst_window" not in payload: payload["hurst_window"] = int(ind.get("hurst_window", 64))
            except Exception:
                pass
            logger.info("Forecast launching: models={}, type={}, symbol={}, tf={}", len(models), forecast_type, payload.get("symbol"), payload.get("timeframe"))
            self.signals.status.emit(f"Forecast: launching {len(models)} model(s)")

            # Decide whether to use parallel inference based on model count and settings
            use_parallel = len(models) > 1 and payload.get("use_parallel_inference", True)

            if use_parallel:
                # Use parallel inference for multiple models
                logger.info("Using parallel inference for {} models", len(models))
                self.signals.status.emit(f"Forecast: parallel inference with {len(models)} models")

                # Create a single parallel worker with all models
                pl = dict(payload)
                pl["model_paths"] = models  # Pass all models to parallel worker
                pl["forecast_type"] = forecast_type
                pl["advanced"] = (forecast_type == "advanced")
                pl["allowed_models"] = list(models)
                pl["source_label"] = f"parallel_{len(models)}_models"
                pl["name"] = f"Parallel Ensemble ({len(models)} models)"
                pl["parallel_inference"] = True

                logger.info("Starting parallel forecast for symbol={} tf={} models={}",
                           pl.get("symbol"), pl.get("timeframe"), len(models))

                fw = ForecastWorker(
                    engine_url=self.engine_url,
                    payload=pl,
                    market_service=self.market_service,
                    signals=self.signals
                )
                self._forecast_active += 1
                self.pool.start(fw)

            else:
                # Use sequential inference for single model or when parallel is disabled
                for mp in models:
                    pl = dict(payload)
                    pl["model_path"] = mp
                    pl["forecast_type"] = forecast_type
                    pl["advanced"] = (forecast_type == "advanced")
                    pl["allowed_models"] = list(models)
                    pl["parallel_inference"] = False
                    base_label = label_map.get(mp, os.path.splitext(os.path.basename(str(mp)))[0])
                    pl["source_label"] = base_label
                    pl["name"] = base_label
                    if pl.get("advanced") and self._forecast_active >= 1:
                        self.signals.status.emit(f"Forecast: advanced already running, skipping {base_label}.")
                        logger.info("Skipping advanced forecast for {}: another advanced job is active", base_label)
                        continue
                    self.signals.status.emit(f"Forecast starting for {pl.get('symbol')} {pl.get('timeframe')} [{base_label}]")
                    logger.info("Starting local forecast for file='{}' label='{}' symbol={} tf={} type={}", mp, base_label, pl.get("symbol"), pl.get("timeframe"), pl.get("forecast_type"))
                    fw = ForecastWorker(engine_url=self.engine_url, payload=pl, market_service=self.market_service, signals=self.signals)
                    self._forecast_active += 1
                    self.pool.start(fw)
        except Exception as e:
            logger.exception("handle_forecast_payload failed: {}", e)
            self.signals.error.emit(str(e))

    @Slot(object, object)
    def _on_forecast_finished(self, df, quantiles):
        try:
            self._forecast_active = max(0, self._forecast_active - 1)
            self.signals.status.emit("Forecast: done")
        except Exception:
            pass

    @Slot(str)
    def _on_forecast_failed(self, msg: str):
        try:
            self._forecast_active = max(0, self._forecast_active - 1)
        except Exception:
            pass

    # ---- Collega i segnali del menu alla UIController --------------------- #

    # ---- Collega i segnali del menu alla UIController --------------------- #
    def bind_menu_signals(self, menu_signals):
        """
        Collega in modo sicuro i segnali del MainMenuBar.
        Usa il nome dello slot come stringa per evitare AttributeError
        quando un handler non è presente.
        """

        def _safe_connect(obj, signal_name: str, owner, slot_name: str) -> bool:
            # prendi il segnale dal menu
            try:
                sig = getattr(obj, signal_name, None)
            except Exception:
                sig = None
            if sig is None:
                return False
            # prendi lo slot dalla UIController
            try:
                slot = getattr(owner, slot_name, None)
            except Exception:
                slot = None
            if slot is None:
                return False
            try:
                sig.connect(slot)
                return True
            except Exception:
                return False

        hooked = []
        if _safe_connect(menu_signals, "ingestRequested", self, "handle_ingest_requested"):
            hooked.append("ingestRequested")
        if _safe_connect(menu_signals, "trainRequested", self, "handle_train_requested"):
            hooked.append("trainRequested")
        if _safe_connect(menu_signals, "forecastRequested", self, "handle_forecast_requested"):
            hooked.append("forecastRequested")
        if _safe_connect(menu_signals, "calibrationRequested", self, "handle_calibration_requested"):
            hooked.append("calibrationRequested")
        if _safe_connect(menu_signals, "backtestRequested", self, "handle_backtest_requested"):
            hooked.append("backtestRequested")
        if _safe_connect(menu_signals, "realtimeToggled", self, "handle_realtime_toggled"):
            hooked.append("realtimeToggled")
        if _safe_connect(menu_signals, "configRequested", self, "handle_config_requested"):
            hooked.append("configRequested")
        if _safe_connect(menu_signals, "settingsRequested", self, "handle_prediction_settings_requested"):
            hooked.append("settingsRequested")

        try:
            from loguru import logger
            logger.info("bind_menu_signals: hooked {}", hooked)
        except Exception:
            pass


# ----------------------------- Training controller ------------------------ #
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

class TrainingControllerSignals(QObject):
    log = Signal(str)
    progress = Signal(int)   # 0..100; -1 means indeterminate
    finished = Signal(bool)  # True if rc==0

class TrainingController(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = TrainingControllerSignals()
        self.pool = QThreadPool.globalInstance()

    def start_training(self, args: list[str], cwd: str | None = None):
        """Spawn the external trainer (e.g., train_sklearn or lightning) and stream logs to the UI."""
        class _Runner(QRunnable):
            def __init__(self, outer, args, cwd):
                super().__init__()
                self.outer = outer
                self.args = args
                self.cwd = cwd

            def run(self):
                import subprocess
                ok = False
                try:
                    # indeterminate progress while running
                    self.outer.signals.progress.emit(-1)
                    p = subprocess.Popen(
                        self.args,
                        cwd=self.cwd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    for line in iter(p.stdout.readline, ""):
                        if not line:
                            break
                        self.outer.signals.log.emit(line.rstrip("\n"))
                    rc = p.wait()
                    ok = (rc == 0)
                except Exception as e:
                    self.outer.signals.log.emit(f"[error] {e}")
                    ok = False
                finally:
                    self.outer.signals.progress.emit(100 if ok else 0)
                    self.outer.signals.finished.emit(ok)

        self.pool.start(_Runner(self, args, cwd))

# optional: make sure it’s exported for star-imports (not strictly required)
__all__ = [
    "UIController",
    "ForecastWorker",
    "TrainingController",
    "TrainingControllerSignals",

]
