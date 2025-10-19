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

    def _tf_to_milliseconds(self, tf: str) -> int:
        """Convert timeframe string to milliseconds."""
        tf = str(tf).strip().lower()
        if tf.endswith("m"):
            return int(tf[:-1]) * 60_000
        elif tf.endswith("h"):
            return int(tf[:-1]) * 3_600_000
        elif tf.endswith("d"):
            return int(tf[:-1]) * 86_400_000
        elif tf.endswith("w"):
            return int(tf[:-1]) * 7 * 86_400_000
        else:
            return 60_000  # Default: 1 minute
    
    def _check_ldm4ts_enabled(self) -> bool:
        """Check if LDM4TS is enabled in settings."""
        try:
            from ..unified_prediction_settings_dialog import UnifiedPredictionSettingsDialog
            settings = UnifiedPredictionSettingsDialog.get_settings_from_file()
            return settings.get('ldm4ts_enabled', False)
        except Exception as e:
            logger.debug(f"Could not check LDM4TS settings: {e}")
            return False

    def _run_ldm4ts_inference(self) -> Optional[dict]:
        """Run LDM4TS inference if enabled."""
        try:
            from ...inference.ldm4ts_inference import LDM4TSInferenceService
            from ..unified_prediction_settings_dialog import UnifiedPredictionSettingsDialog
            
            # Load settings
            settings = UnifiedPredictionSettingsDialog.get_settings_from_file()
            
            checkpoint_path = settings.get('ldm4ts_checkpoint', '')
            if not checkpoint_path or not Path(checkpoint_path).exists():
                logger.warning("LDM4TS enabled but no valid checkpoint")
                return None
            
            # Get service
            service = LDM4TSInferenceService.get_instance(
                checkpoint_path=checkpoint_path
            )
            
            if not service._initialized:
                logger.warning("LDM4TS service not initialized")
                return None
            
            # Get OHLCV data
            symbol = self.payload['symbol']
            timeframe = self.payload['timeframe']
            window_size = settings.get('ldm4ts_window_size', 100)
            
            # Fetch from market service using direct SQL
            from sqlalchemy import text
            import pandas as pd
            
            # Check if contextual start time is provided (Ctrl+Click)
            contextual_start = self.payload.get('contextual_start_time')
            
            with self.market_service.engine.connect() as conn:
                if contextual_start:
                    # Contextual forecast: fetch data UP TO clicked timestamp
                    from dateutil.parser import parse
                    start_ts_ms = int(parse(contextual_start).timestamp() * 1000)
                    
                    query = text("""
                        SELECT ts_utc, open, high, low, close, volume
                        FROM market_data_candles
                        WHERE symbol = :symbol AND timeframe = :timeframe
                              AND ts_utc <= :start_ts
                        ORDER BY ts_utc DESC
                        LIMIT :window_size
                    """)
                    rows = conn.execute(query, {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "start_ts": start_ts_ms,
                        "window_size": window_size
                    }).fetchall()
                    logger.info(f"Contextual LDM4TS: fetching {window_size} bars up to {contextual_start}")
                else:
                    # Normal forecast: fetch most recent data
                    query = text("""
                        SELECT ts_utc, open, high, low, close, volume
                        FROM market_data_candles
                        WHERE symbol = :symbol AND timeframe = :timeframe
                        ORDER BY ts_utc DESC
                        LIMIT :window_size
                    """)
                    rows = conn.execute(query, {
                        "symbol": symbol,
                        "timeframe": timeframe,
                    "window_size": window_size
                }).fetchall()
            
            if not rows or len(rows) < window_size:
                logger.warning(f"Insufficient data for LDM4TS: {len(rows) if rows else 0} < {window_size}")
                return None
            
            # Convert to DataFrame (reverse to chronological order)
            df = pd.DataFrame([
                {
                    'timestamp': pd.to_datetime(r[0], unit='ms', utc=True),
                    'open': float(r[1]),
                    'high': float(r[2]),
                    'low': float(r[3]),
                    'close': float(r[4]),
                    'volume': float(r[5]) if r[5] is not None else 0.0
                }
                for r in reversed(rows)
            ])
            df = df.set_index('timestamp')
            
            # Run inference
            num_samples = settings.get('ldm4ts_num_samples', 50)
            prediction = service.predict(
                ohlcv_data=df,
                num_samples=num_samples
            )
            
            # Convert to quantiles format
            quantiles = prediction.to_quantiles_format()
            quantiles['source'] = 'ldm4ts'
            
            logger.info(f"LDM4TS inference completed: {len(prediction.horizons)} horizons, {prediction.inference_time_ms:.0f}ms")
            return quantiles
            
        except Exception as e:
            logger.error(f"LDM4TS inference failed: {e}", exc_info=True)
            return None

    def run(self):
        try:
            self.signals.status.emit("Forecast: running inference...")
            
            # Check forecast type from payload
            forecast_type = self.payload.get("forecast_type", "diffusion")
            
            if forecast_type == "ldm4ts":
                # LDM4TS forecast explicitly requested
                logger.info("LDM4TS forecast requested explicitly")
                quantiles = self._run_ldm4ts_inference()
                
                if quantiles:
                    # Emit LDM4TS results
                    self.signals.forecastReady.emit(pd.DataFrame(), quantiles)
                    self.signals.status.emit("LDM4TS forecast completed")
                    return
                else:
                    raise RuntimeError("LDM4TS inference failed")
            
            # Check if LDM4TS is enabled via settings (old behavior for compatibility)
            elif self._check_ldm4ts_enabled():
                logger.info("LDM4TS enabled via settings - running vision-enhanced forecast")
                quantiles = self._run_ldm4ts_inference()
                
                if quantiles:
                    # Emit LDM4TS results
                    self.signals.forecastReady.emit(pd.DataFrame(), quantiles)
                    self.signals.status.emit("LDM4TS forecast completed")
                    return
                else:
                    logger.warning("LDM4TS inference failed, falling back to standard models")
            
            # Standard diffusion models inference
            df, quantiles = self._parallel_infer()
            self.signals.status.emit("Forecast: ready")
            self.signals.forecastReady.emit(df, quantiles)
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
        """
        OPT-003: Lazy loading for inference using centralized data loader.
        Only fetches the minimum required bars instead of full history.
        """
        try:
            # Use centralized lazy loading function from data_loader
            from ...data.data_loader import fetch_candles_from_db_recent

            # If no end_ts specified, fetch recent bars directly
            if end_ts is None:
                df = fetch_candles_from_db_recent(
                    symbol=symbol,
                    timeframe=timeframe,
                    n_bars=n_bars,
                    engine_url=None  # Will use default MarketDataService
                )
                logger.debug(f"[OPT-003 Lazy Loading] Fetched {len(df)} recent bars for {symbol} {timeframe}")
                return df

            # For historical point (end_ts specified), use custom query
            from sqlalchemy import MetaData, select, text
            meta = MetaData()
            meta.reflect(bind=engine, only=["market_data_candles"])
            tbl = meta.tables.get("market_data_candles")
            if tbl is None:
                return pd.DataFrame()
            with engine.connect() as conn:
                q = text(
                    "SELECT ts_utc, timestamp, open, high, low, close, volume FROM market_data_candles "
                    "WHERE symbol = :symbol AND timeframe = :timeframe AND ts_utc <= :end_ts "
                    "ORDER BY ts_utc DESC LIMIT :limit"
                )
                rows = conn.execute(q, {"symbol": symbol, "timeframe": timeframe, "end_ts": int(end_ts), "limit": int(n_bars)}).fetchall()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=["ts_utc", "timestamp", "open", "high", "low", "close", "volume"])
                if "timestamp" in df.columns and df["timestamp"].notna().any():
                    df["ts_utc"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**6
                logger.debug(f"[OPT-003 Lazy Loading] Fetched {len(df)} bars up to ts={end_ts} for {symbol} {timeframe}")
                return df.sort_values("ts_utc").reset_index(drop=True)
        except Exception as e:
            logger.exception("Failed to fetch recent candles: {}", e)
            return pd.DataFrame()

    def _dict_to_candles(self, candles_list) -> pd.DataFrame:
        """Convert candles list to DataFrame."""
        import pandas as _pd
        return _pd.DataFrame(candles_list).copy()

    # --------------------------- parallel inference --------------------------- #

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

                # LEGACY SUPPORT: If no model_paths specified, try to resolve single model path
                if not model_paths:
                    model_path_input = self.payload.get("model_path")
                    if model_path_input:
                        resolved_path = self._resolve_model_path(model_path_input)
                        if resolved_path and resolved_path.exists():
                            model_paths = [str(resolved_path)]
                            logger.info(f"Legacy single model path resolved: {resolved_path.name}")

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

            # Normalize data - handle both timestamp index and ts_utc column
            if "ts_utc" not in df_candles.columns:
                # If timestamp is in index, reset it to column and convert to ts_utc
                if df_candles.index.name == 'timestamp' or isinstance(df_candles.index, pd.DatetimeIndex):
                    df_candles = df_candles.reset_index()
                    if 'timestamp' in df_candles.columns:
                        # Convert timestamp to ts_utc (milliseconds)
                        # Convert without pandas chaining warnings
                        if df_candles["timestamp"].dtype == 'int64':
                            df_candles["ts_utc"] = df_candles["timestamp"].astype('int64')
                        else:
                            df_candles["ts_utc"] = pd.to_datetime(df_candles["timestamp"]).astype('int64') // 10**6
                        df_candles = df_candles.drop(columns=['timestamp'])
                    else:
                        raise RuntimeError("DataFrame has no ts_utc or timestamp column")
                else:
                    raise RuntimeError("DataFrame has no ts_utc column and index is not timestamp")
            
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
            from ...features.feature_engineering import relative_ohlc, temporal_features, realized_volatility_feature
            from ...features.feature_utils import coerce_indicator_tfs
            from ...features.indicator_pipeline import compute_indicators  # ISSUE-001b: Use centralized indicator computation
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
                    feats_list.append(relative_ohlc(df_candles_full))

                if cache_config["use_temporal_features"]:
                    feats_list.append(temporal_features(df_candles_full))

                if cache_config["rv_window"] > 1:
                    feats_list.append(realized_volatility_feature(df_candles_full, cache_config["rv_window"]))

                # Multi-timeframe indicators - MUST use full dataframe for resampling
                indicator_tfs_raw = cache_config["indicator_tfs"]
                indicator_tfs = coerce_indicator_tfs(indicator_tfs_raw)

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
                        feats_list.append(compute_indicators(df_candles_full, ind_cfg, indicator_tfs, tf, symbol=symbol, days_history=days_history_est))

                if not feats_list:
                    # Fallback to basic features
                    feats_list.append(relative_ohlc(df_candles))
                    feats_list.append(temporal_features(df_candles))

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
            # Get GPU setting from payload
            use_gpu = self.payload.get("use_gpu_inference", False)

            # GPU inference limitation: cannot run multiple models in parallel on single GPU
            # If GPU is enabled, use only the first model for inference
            if use_gpu and len(model_paths) > 1:
                logger.warning(
                    f"GPU inference enabled with {len(model_paths)} models. "
                    f"Using only first model (GPU cannot run multiple models in parallel). "
                    f"Disable GPU to use all models in parallel on CPU."
                )
                parallel_settings["model_paths"] = [model_paths[0]]
                max_workers = 1

            # Get aggregation method from payload
            aggregation_method = self.payload.get("aggregation_method", "Mean")
            
            parallel_results = parallel_engine.run_parallel_inference(
                parallel_settings, feats_df, symbol, tf, horizons_raw, 
                use_gpu=use_gpu, candles_df=df_candles_full,
                aggregation_method=aggregation_method
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

            # === ENHANCED MULTI-HORIZON SYSTEM (Ripristinato) ===
            use_enhanced_scaling = self.payload.get("use_enhanced_scaling", True)
            scenario = self.payload.get("trading_scenario")
            scaling_mode = self.payload.get("scaling_mode", "smart_adaptive")

            # Scale to match horizon count
            if len(mean_returns) != len(horizons_bars):
                if len(mean_returns) == 1:
                    base_return = mean_returns[0]
                    base_std = std_returns[0]

                    # Apply Enhanced Multi-Horizon if enabled
                    if use_enhanced_scaling:
                        try:
                            from ...utils.horizon_converter import convert_single_to_multi_horizon

                            # Get recent market data for regime detection
                            market_data = df_candles.tail(100) if len(df_candles) >= 100 else df_candles

                            logger.info(f"[Enhanced Multi-Horizon] Converting single prediction using {scaling_mode} mode")

                            # Convert single prediction to multi-horizon using smart scaling
                            multi_horizon_results = convert_single_to_multi_horizon(
                                base_prediction=base_return,
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
                                    # Fallback: replicate base prediction
                                    scaled_preds.append(base_return)

                            mean_returns = np.array(scaled_preds)

                            # Store uncertainty data for later use
                            self.payload["enhanced_uncertainty"] = uncertainty_data

                            # For std, scale based on uncertainty bands
                            std_returns = np.array([
                                (uncertainty_data.get(h, {}).get("upper", base_return) -
                                 uncertainty_data.get(h, {}).get("lower", base_return)) / 3.29  # 90% CI → std
                                for h in horizons_time_labels
                            ])

                            logger.info(f"[Enhanced Multi-Horizon] Completed: {len(horizons_time_labels)} horizons using {scaling_mode}")

                        except Exception as e:
                            logger.warning(f"[Enhanced Multi-Horizon] Failed, using simple replication: {e}")
                            # Fallback: simple replication
                            mean_returns = np.full(len(horizons_bars), base_return)
                            std_returns = np.array([base_std * np.sqrt(i+1) for i in range(len(horizons_bars))])
                    else:
                        # Simple replication (NO Enhanced)
                        logger.debug(f"[Simple Replication] Single return {base_return:.6f} → {len(horizons_bars)} horizons")
                        mean_returns = np.full(len(horizons_bars), base_return)
                        std_returns = np.array([base_std * np.sqrt(i+1) for i in range(len(horizons_bars))])

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

            # Check if we have multi-horizon predictions
            has_multi_horizon = False
            predictions_dict = None
            future_ts_dict = None
            model_horizons = None
            
            # Try to extract multi-horizon data from individual results
            if individual_results and len(individual_results) > 0:
                first_result = individual_results[0]
                if first_result.get('is_multi_horizon') and first_result.get('predictions_dict'):
                    has_multi_horizon = True
                    predictions_dict = first_result['predictions_dict']
                    model_horizons = first_result.get('horizons', [])
                    
                    # Create future_ts_dict for each horizon
                    future_ts_dict = {}
                    for horizon in model_horizons:
                        # Each horizon gets a single timestamp at that offset
                        future_ts_dict[horizon] = [last_ts_ms + (horizon * self._tf_to_milliseconds(tf))]
                    
                    logger.info(f"Multi-horizon forecast detected: {len(model_horizons)} horizons")

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
                "anchor_price": self.payload.get("anchor_price"),  # Pass through Alt+Click Y coordinate
                
                # Multi-horizon support
                "is_multi_horizon": has_multi_horizon,
                "predictions_dict": predictions_dict,
                "future_ts_dict": future_ts_dict,
                "horizons": model_horizons
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