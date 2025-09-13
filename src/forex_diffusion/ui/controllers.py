# ui/controllers.py
# Controller to bind UI menu actions to background workers and services.
from __future__ import annotations

from typing import Optional, Tuple
from pathlib import Path

import httpx
import pandas as pd
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Slot
from loguru import logger

from ..services.marketdata import MarketDataService
from ..data import io as data_io
from .prediction_settings_dialog import PredictionSettingsDialog

import pickle
from pathlib import Path

# Import robusto della pipeline (assoluto → relativo come fallback)
try:
    from forex_diffusion.features.pipeline import pipeline_process
except ImportError:
    from ..features.pipeline import pipeline_process


def pickle_load_safe(p: Path):
    """
    Carica un file pickle (tipicamente sklearn) e lo normalizza in un dict.
    - Se il pickle è già un dict, lo ritorna.
    - Se contiene direttamente un modello/oggetto, lo incapsula come {"model": obj}.
    """
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj
    return {"model": obj}


class UIControllerSignals(QObject):
    """Signals emitted by the controller to update UI."""
    forecastReady = Signal(object, object)  # (pd.DataFrame, quantiles_dict)
    error = Signal(str)
    status = Signal(str)


class ForecastWorker(QRunnable):
    """
    Worker that calls the inference HTTP endpoint and fetches recent candles from DB,
    then emits the forecastReady signal with (df, quantiles).
    Implements detailed tracing and a local fallback inference path when remote call fails.
    """
    def __init__(self, engine_url: str, payload: dict, market_service: MarketDataService, signals: UIControllerSignals):
        super().__init__()
        self.engine_url = engine_url.rstrip("/")
        self.payload = payload
        self.market_service = market_service
        self.signals = signals

    def run(self):
        """
        Local-only forecast execution:
         - build features from latest candles in DB
         - load model from payload['model_path']
         - predict and emit forecastReady(df_candles, quantiles)
        """
        try:
            self.signals.status.emit("Forecast: running local inference...")
            # perform local inference
            df_local, quantiles_local = self._local_infer()
            # emit result for viewer/ChartTab
            self.signals.status.emit("Forecast: ready (local)")
            self.signals.forecastReady.emit(df_local, quantiles_local)
        except Exception as e:
            logger.exception("Forecast worker failed: {}", e)
            self.signals.error.emit(str(e))
            self.signals.status.emit("Forecast: failed")

    def _local_infer(self):
        """
        Fallback locale (Basic). Ritorna (df_candles, quantiles_dict).
        Flusso: dati → pipeline → ensure → fill mancanti con μ → z-score per colonna → inferenza → horizon → prezzi → quantili.
        """
        import pickle
        import numpy as np
        import pandas as pd
        from pathlib import Path

        try:
            from loguru import logger
        except Exception:
            class _L:
                def info(self, *a, **k): pass
                def debug(self, *a, **k): pass
                def warning(self, *a, **k): pass
                def error(self, *a, **k): pass
            logger = _L()

        # --- import robusti ---
        try:
            from forex_diffusion.features.pipeline import pipeline_process
        except Exception:
            from ..features.pipeline import pipeline_process  # type: ignore

        ensure_features_for_prediction = None
        try:
            from forex_diffusion.inference.prediction_config import ensure_features_for_prediction as _ens
            ensure_features_for_prediction = _ens
        except Exception:
            try:
                from ..inference.prediction_config import ensure_features_for_prediction as _ens  # type: ignore
                ensure_features_for_prediction = _ens
            except Exception:
                ensure_features_for_prediction = None

        def _pickle_load_safe(p: Path):
            with open(p, "rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, dict) else {"model": obj}

        # --- 1) Parametri payload ---
        model_path = self.payload.get("model_path") or self.payload.get("model")
        if not model_path:
            raise RuntimeError("No model_path provided for local fallback")

        limit = int(self.payload.get("limit_candles", 512))
        sym = self.payload.get("symbol")
        tf = (self.payload.get("timeframe") or "1m")
        horizons = self.payload.get("horizons", ["5m"])
        horizon_steps = int(pd.Timedelta(horizons[0]).total_seconds() / pd.Timedelta(tf).total_seconds()) if horizons else 5

        output_type = str(self.payload.get("output_type", "returns")).lower()

        # --- 2) Dati: ultime N candele (DEFINIZIONE SICURA DI df_candles) ---
        # If caller supplied candles_override (testing point), use that slice directly
        df_candles = None
        if self.payload.get("candles_override"):
            try:
                import pandas as pd
                df_candles = pd.DataFrame(self.payload.get("candles_override"))
                if "ts_utc" in df_candles.columns:
                    df_candles = df_candles.sort_values("ts_utc").reset_index(drop=True)
                else:
                    raise RuntimeError("candles_override missing ts_utc")
            except Exception as e:
                logger.exception("Failed to use candles_override: %s", e)
                df_candles = None

        if df_candles is None or df_candles.empty:
            # allow specifying an end timestamp (testing_point) so DB query returns up-to that moment
            end_ts = self.payload.get("testing_point_ts", None)
            df_candles = self._fetch_recent_candles(self.market_service.engine, sym, tf, n_bars=limit, end_ts=end_ts)
        if df_candles is None or df_candles.empty:
            raise RuntimeError("No candles available for local inference")
        df_candles = df_candles.sort_values("ts_utc").reset_index(drop=True)

        # --- 3) Carica modello PRIMA dell'ensure (serve features_list, μ, σ) ---
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        suffix = p.suffix.lower()
        if suffix in (".pt", ".pth", ".ptl"):
            try:
                import torch
                raw = torch.load(str(p), map_location="cpu")
                payload_obj = raw if isinstance(raw, dict) else {"model": raw}
            except Exception as e:
                raise RuntimeError(f"Failed to load torch model file: {e}")
        else:
            try:
                payload_obj = _pickle_load_safe(p)
            except Exception as e:
                raise RuntimeError(f"Failed to load pickle model file: {e}")

        model = payload_obj.get("model")
        features_list = payload_obj.get("features") or []
        mu = payload_obj.get("std_mu", {}) or {}
        sigma = payload_obj.get("std_sigma", {}) or {}

        if model is None:
            raise RuntimeError("Model payload missing 'model'")
        if not isinstance(features_list, (list, tuple)) or len(features_list) == 0:
            raise RuntimeError("Model payload missing 'features' list")

        # --- 4) Config Basic per pipeline ---
        features_config = {
            "warmup_bars": int(self.payload.get("warmup_bars", 16)),
            "indicators": {
                "atr": {"n": int(self.payload.get("atr_n", 14))},
                "rsi": {"n": int(self.payload.get("rsi_n", 14))},
                "bollinger": {"n": int(self.payload.get("bb_n", 20))},
                "hurst": {"window": int(self.payload.get("hurst_window", 64))},
            },
            "standardization": {"window_bars": int(self.payload.get("rv_window", 60))},
        }

        # --- 5) Pipeline ---
        feats, _ = pipeline_process(df_candles.copy(), timeframe=tf, features_config=features_config)
        if feats is None or feats.empty:
            raise RuntimeError("No features computed for local inference")

        # --- Fill missing features ---
        missing = [c for c in features_list if c not in feats.columns]
        if missing:
            logger.warning("Basic: missing features will be filled with mu: {}", missing)
            for col in missing:
                feats[col] = float(mu.get(col, 0.0))

        X = feats[features_list].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # --- 8) Standardizzazione per colonna ---
        for col in features_list:
            if col in mu and col in sigma and sigma[col] != 0:
                X[col] = (X[col] - mu[col]) / sigma[col]

        X_arr = X.to_numpy(dtype=float)

        # --- 9) Inferenza ---
        preds = None
        try:
            import torch
            if hasattr(model, "eval"): model.eval()
            with torch.no_grad():
                t_in = torch.tensor(X_arr, dtype=torch.float32)
                out = model(t_in)
                preds = out.detach().cpu().numpy()
        except Exception:
            if hasattr(model, "predict"): preds = np.asarray(model.predict(X_arr))
            else: raise RuntimeError("Unsupported model type for prediction")

        preds = np.squeeze(preds)
        if preds.size == 0: raise RuntimeError("Model returned empty prediction")

        # --- 10) Sequenza di lunghezza horizon ---
        seq = preds[-horizon_steps:] if preds.size >= horizon_steps else np.pad(preds, (0, horizon_steps - preds.size), mode='edge')

        # --- 11) Prezzi e quantili ---
        last_close = float(df_candles["close"].iat[-1])
        if output_type == "returns":
            prices = [last_close * (1.0 + r) for r in seq]
        else:
            prices = seq.tolist()
        
        forecast_prices = np.array(prices, dtype=float)

        quantiles = {
            "q50": forecast_prices.tolist(),
            "q05": (forecast_prices * 0.99).tolist(),
            "q95": (forecast_prices * 1.01).tolist(),
        }

        return df_candles, quantiles

    def _fetch_recent_candles(self, engine, symbol: str, timeframe: str, n_bars: int = 500, end_ts: Optional[int] = None) -> pd.DataFrame:
        """
        Query market_data_candles for the last n_bars for the given symbol/timeframe.
        If end_ts is provided (ms UTC) return the last n_bars with ts_utc <= end_ts (useful for testing point).
        """
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
                    # use parameterized query for end_ts limit
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


class UIController:
    """
    Binds menu signals to actions, runs background workers, exposes signals for UI updates.
    """
    def __init__(self, main_window, market_service: Optional[MarketDataService] = None, engine_url: str = "http://127.0.0.1:8000", db_writer: Optional["DBWriter"] = None):
        self.main_window = main_window
        self.market_service = market_service or MarketDataService()
        self.engine_url = engine_url
        self.signals = UIControllerSignals()
        self.pool = QThreadPool.globalInstance()
        self.db_writer = db_writer

    def bind_menu_signals(self, menu_signals):
        """
        Connect menu signals to controller handlers.
        """
        menu_signals.ingestRequested.connect(self.handle_ingest_requested)
        menu_signals.trainRequested.connect(self.handle_train_requested)
        menu_signals.forecastRequested.connect(self.handle_forecast_requested)
        menu_signals.calibrationRequested.connect(self.handle_calibration_requested)
        menu_signals.backtestRequested.connect(self.handle_backtest_requested)
        menu_signals.realtimeToggled.connect(self.handle_realtime_toggled)
        menu_signals.configRequested.connect(self.handle_config_requested)
        menu_signals.predictionSettingsRequested.connect(self.handle_prediction_settings_requested)

    @Slot()
    def handle_prediction_settings_requested(self):
        """Opens the Prediction Settings dialog."""
        dialog = PredictionSettingsDialog(self.main_window)
        dialog.exec()

    @Slot()
    def handle_ingest_requested(self):
        self.signals.status.emit("Ingest requested: launching backfill...")
        worker = _IngestWorker(self.market_service, self.signals)
        self.pool.start(worker)

    @Slot()
    def handle_train_requested(self):
        self.signals.status.emit("Train requested (not implemented).")

    @Slot()
    def handle_forecast_requested(self):
        settings = PredictionSettingsDialog.get_settings()
        if not settings or not settings.get("model_path"):
            self.signals.error.emit("Prediction settings not configured or model path is missing.")
            self.handle_prediction_settings_requested()
            return

        cfg = self.market_service.cfg if hasattr(self.market_service, "cfg") else None
        try:
            symbol = cfg.data.symbols[0] if (cfg and hasattr(cfg, "data") and hasattr(cfg.data, "symbols")) else "EUR/USD"
            timeframe = (cfg.timeframes.native[0] if (cfg and hasattr(cfg, "timeframes") and hasattr(cfg.timeframes, "native")) else "1m")
        except Exception:
            symbol = "EUR/USD"
            timeframe = "1m"

        # Merge extended settings into payload so worker has access to indicator params and adv flags
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_path": settings.get("model_path"),
            "horizons": settings.get("horizons", ["1m", "5m", "15m"]),
            "N_samples": settings.get("N_samples", 200),
            "apply_conformal": settings.get("apply_conformal", True),
            # extended settings (if present)
            "warmup_bars": settings.get("warmup_bars"),
            "atr_n": settings.get("atr_n"),
            "rsi_n": settings.get("rsi_n"),
            "bb_n": settings.get("bb_n"),
            "rv_window": settings.get("rv_window"),
            "ema_fast": settings.get("ema_fast"),
            "ema_slow": settings.get("ema_slow"),
            "don_n": settings.get("don_n"),
            "hurst_window": settings.get("hurst_window"),
            "keltner_k": settings.get("keltner_k"),
            "max_forecasts": settings.get("max_forecasts"),
            "auto_predict": settings.get("auto_predict"),
            "auto_interval_seconds": settings.get("auto_interval_seconds"),
        }

        # remove None values to keep payload tidy
        payload = {k: v for k, v in payload.items() if v is not None}

        self.signals.status.emit(f"Forecast requested for {symbol} {timeframe}")

        try:
            if getattr(self, "db_writer", None) is not None:
                self.db_writer.write_prediction_async(symbol=symbol, timeframe=timeframe, horizon="request", q05=0.0, q50=0.0, q95=0.0, meta={"event": "forecast_requested", "settings": settings})
        except Exception:
            pass

        fw = ForecastWorker(engine_url=self.engine_url, payload=payload, market_service=self.market_service, signals=self.signals)
        self.pool.start(fw)

    @Slot()
    def handle_calibration_requested(self):
        self.signals.status.emit("Calibration requested (not implemented).")

    @Slot()
    def handle_backtest_requested(self):
        self.signals.status.emit("Backtest requested (not implemented).")

    @Slot(bool)
    def handle_realtime_toggled(self, enabled: bool):
        self.signals.status.emit("Realtime toggled: {}".format("ON" if enabled else "OFF"))

    @Slot()
    def handle_config_requested(self):
        self.signals.status.emit("Config requested (not implemented).")


class _IngestWorker(QRunnable):
    """Worker that runs MarketDataService.ensure_startup_backfill in background."""
    def __init__(self, market_service: MarketDataService, signals: UIControllerSignals):
        super().__init__()
        self.market_service = market_service
        self.signals = signals

    def run(self):
        try:
            self.signals.status.emit("Backfill: running...")
            reports = self.market_service.ensure_startup_backfill()
            self.signals.status.emit(f"Backfill: completed ({len(reports)} reports)")
        except Exception as e:
            logger.exception("Backfill worker failed: {}", e)
            self.signals.error.emit(str(e))
            self.signals.status.emit("Backfill failed")
