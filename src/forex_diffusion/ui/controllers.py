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
        ftype = str(self.payload.get("forecast_type", "basic")).lower()
        model_path = self.payload.get("model_path") or self.payload.get("model")
        # baseline RW: non richiede modello
        if ftype == "rw":
            model_path = model_path  # ignore if missing
        if not model_path and ftype != "rw":
            raise RuntimeError("No model_path provided for local fallback")

        limit = int(self.payload.get("limit_candles", 512))
        sym = self.payload.get("symbol")
        tf = (self.payload.get("timeframe") or "1m")
        horizons = self.payload.get("horizons", ["5m"])
        # requested time (point 0 on the plot)
        from datetime import datetime, timezone
        requested_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # number of forecast points
        try:
            if isinstance(horizons, (list, tuple)) and len(horizons) > 0:
                horizon_steps = len(horizons)
            else:
                horizon_steps = int(pd.Timedelta(horizons[0]).total_seconds() / pd.Timedelta(tf).total_seconds()) if horizons else 5
        except Exception:
            horizon_steps = max(1, len(horizons) if isinstance(horizons, (list, tuple)) else 5)

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
        # reduce thread contention in BLAS/torch to keep UI responsive
        try:
            import os as _os_env
            _os_env.environ.setdefault("OMP_NUM_THREADS", "1")
            _os_env.environ.setdefault("MKL_NUM_THREADS", "1")
        except Exception:
            pass
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        suffix = p.suffix.lower()
        if suffix in (".pt", ".pth", ".ptl"):
            try:
                import torch
                # keep torch single-thread for UI responsiveness
                try:
                    torch.set_num_threads(1)
                except Exception:
                    pass
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
        if str(self.payload.get("forecast_type","")).lower() == "rw":
            # simple baseline: last_close * exp(sigma_1 * sqrt(h) * Z) approximated deterministically to a small drift
            import numpy as np
            sigma = float(np.std(np.log(df_candles["close"]).diff().fillna(0.0).tail(256)))
            # deterministic centerline as 0 drift (returns ~ 0)
            preds = np.zeros((horizon_steps,), dtype=float)
        else:
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
        if preds.size == 0:
            raise RuntimeError("Model returned empty prediction")
        # diagnostics
        try:
            tail = preds[-min(5, preds.size):].tolist() if preds.size > 0 else []
            logger.info("Forecast debug: preds.size={}, std={:.6g}, tail={}", int(preds.size), float(np.std(preds)), tail)
        except Exception:
            pass

        # --- 10) Sequenza di lunghezza horizon ---
        seq = preds[-horizon_steps:] if preds.size >= horizon_steps else np.pad(preds, (0, horizon_steps - preds.size), mode='edge')

        # --- 11) Prezzi e quantili ---
        last_close = float(df_candles["close"].iat[-1])
        if output_type == "returns":
            # interpret seq as per-step returns and compose cumulatively
            seq_arr = np.array(seq, dtype=float)
            cum = np.cumprod(1.0 + seq_arr)
            prices = (last_close * cum).tolist()
        else:
            prices = np.array(seq, dtype=float).tolist()

        forecast_prices = np.array(prices, dtype=float)

        # fallback: if prices are (near) constant, inject small drift estimated from recent returns
        try:
            if not np.any(np.isfinite(forecast_prices)) or np.allclose(forecast_prices, forecast_prices[0], rtol=0.0, atol=1e-12):
                # estimate drift from recent log returns
                logret = np.log(np.maximum(1e-12, df_candles["close"].astype(float))).diff().dropna()
                mu_drift = float(logret.tail(256).mean() if len(logret) > 0 else 0.0)
                step = np.exp(mu_drift) - 1.0
                drift_seq = np.cumprod(1.0 + np.full(int(horizon_steps), step, dtype=float))
                fallback_prices = last_close * drift_seq
                logger.info("Forecast fallback applied: step_drift={:.6g}, last_close={:.6g}", step, last_close)
                forecast_prices = fallback_prices.astype(float)
        except Exception:
            pass

        # Apply model weight (blend towards last_close): w in [0,1]
        try:
            w_pct = int(self.payload.get("model_weight_pct", 100))
            w = max(0.0, min(1.0, float(w_pct) / 100.0))
            forecast_prices = last_close + w * (forecast_prices - last_close)
        except Exception:
            pass

        # (rimosso: generazione future_ts da last_dt e relativo ensure)
        # Nota: future_ts viene generato più sotto a partire da requested_at_ms.
        # Ciò garantisce che il punto 0 corrisponda al momento della richiesta.

        # Build future_ts (UTC ms) from requested_at_ms and requested horizons
        future_ts_list: list[int] = []
        try:
            base = pd.to_datetime(requested_at_ms, unit="ms", utc=True)
            if isinstance(horizons, (list, tuple)) and len(horizons) > 0:
                for h in horizons:
                    try:
                        dt = base + pd.to_timedelta(str(h))
                    except Exception:
                        dt = base
                    future_ts_list.append(int(dt.value // 1_000_000))
            else:
                step = pd.to_timedelta(tf)
                future_ts_list = [int((base + step * (i + 1)).value // 1_000_000) for i in range(int(horizon_steps))]
        except Exception:
            base = pd.to_datetime(requested_at_ms, unit="ms", utc=True)
            step = pd.to_timedelta(tf) if tf else pd.to_timedelta("1m")
            future_ts_list = [int((base + step * (i + 1)).value // 1_000_000) for i in range(int(horizon_steps))]

        # Ensure sizes match (pad/trim)
        if forecast_prices.size != len(future_ts_list):
            if forecast_prices.size == 0:
                forecast_prices = np.array([last_close] * len(future_ts_list), dtype=float)
            elif forecast_prices.size < len(future_ts_list):
                pad = len(future_ts_list) - forecast_prices.size
                forecast_prices = np.concatenate([forecast_prices, np.repeat(forecast_prices[-1], pad)])
            else:
                forecast_prices = forecast_prices[: len(future_ts_list)]

        # Log series to help diagnose (time, price)
        try:
            times_iso = pd.to_datetime(future_ts_list, unit="ms", utc=True).tz_convert(None).astype(str).tolist()
            pairs = list(zip(times_iso, forecast_prices.astype(float).round(10).tolist()))
            logger.info("Forecast series ({} pts): {}", len(pairs), pairs)
        except Exception:
            pass

        # --- Quantiles q05/q95: bande da volatilità (meta residual_std -> fallback a log-returns) ---
        try:
            meta = {}
            try:
                # if model payload embeds meta
                meta = payload_obj.get("meta", {}) if isinstance(payload_obj, dict) else {}
            except Exception:
                meta = {}
            # base sigma: prefer residual_std/meta, else std dei log-returns recenti
            resid_std = meta.get("residual_std") or meta.get("resid_std") or None
            if resid_std is None:
                logret = np.log(np.maximum(1e-12, df_candles["close"].astype(float))).diff().dropna()
                resid_std = float(logret.tail(512).std() if len(logret) > 0 else 0.0)
            sigma_base = max(1e-8, float(resid_std))
            # z per ~90% (più conservativo di 1.0)
            apply_conf = bool(self.payload.get("apply_conformal", True))
            z = 1.645 if apply_conf else 1.0
            n = len(forecast_prices)
            k = np.arange(1, n + 1, dtype=float)
            # ampiezza percentuale cresce con sqrt(h)
            band_rel = z * sigma_base * np.sqrt(k)
            # limiti percentuali ragionevoli per evitare bande assurde
            band_rel = np.clip(band_rel, 1e-6, 0.2)  # max 20% a orizzonti lunghi
            q50_arr = forecast_prices.astype(float)
            q05_arr = q50_arr * (1.0 - band_rel)
            q95_arr = q50_arr * (1.0 + band_rel)
            # prezzi non negativi
            q05_arr = np.maximum(1e-12, q05_arr)
            q95_arr = np.maximum(1e-12, q95_arr)
            try:
                logger.info("Quantiles debug: z={}, sigma_base={:.6g}, q50_head={}, q05_head={}, q95_head={}",
                            z, sigma_base,
                            q50_arr[:3].round(8).tolist(),
                            q05_arr[:3].round(8).tolist(),
                            q95_arr[:3].round(8).tolist())
            except Exception:
                pass
            q05_list = q05_arr.tolist()
            q95_list = q95_arr.tolist()
        except Exception:
            # fallback minimo: ±1% come prima, se qualcosa fallisce
            q05_list = (forecast_prices * 0.99).tolist()
            q95_list = (forecast_prices * 1.01).tolist()

        quantiles = {
            "q50": forecast_prices.tolist(),
            "q05": q05_list,
            "q95": q95_list,
            "future_ts": future_ts_list,
            "requested_at_ms": int(requested_at_ms),
            "source": str(self.payload.get("source_label") or ("advanced" if self.payload.get("advanced") else "basic")),
            "label": str(self.payload.get("source_label") or "forecast"),
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
        # limit forecast concurrency to keep UI responsive
        try:
            self.pool.setMaxThreadCount(max(2, min(self.pool.maxThreadCount(), 4)))
        except Exception:
            pass
        self.db_writer = db_writer
        # track active forecasts (avoid overlapping advanced jobs)
        self._forecast_active = 0
        try:
            self.signals.forecastReady.connect(self._on_forecast_finished)
            self.signals.error.connect(self._on_forecast_failed)
        except Exception:
            pass

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
        # File->Settings
        try:
            menu_signals.settingsRequested.connect(self.handle_settings_requested)
        except Exception:
            pass
        # ensure completion/error signals are handled (decrement active)
        try:
            self.signals.forecastReady.connect(self._on_forecast_finished)
            self.signals.error.connect(self._on_forecast_failed)
        except Exception:
            pass

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
        # accetta anche multi-modello: se entrambi vuoti -> warning
        multi = [p for p in settings.get("model_paths", []) if p] if settings else []
        single = settings.get("model_path") if settings else None
        if not settings or (not multi and not single):
            self.signals.error.emit("Prediction settings not configured or model path(s) missing.")
            self.handle_prediction_settings_requested()
            return

        # Prefer symbol/timeframe from current ChartTab if available
        chart_tab = getattr(self, "chart_tab", None)
        if chart_tab and getattr(chart_tab, "symbol", None) and getattr(chart_tab, "timeframe", None):
            symbol = chart_tab.symbol
            timeframe = chart_tab.timeframe
        else:
            cfg = self.market_service.cfg if hasattr(self.market_service, "cfg") else None
            try:
                symbol = cfg.data.symbols[0] if (cfg and hasattr(cfg, "data") and hasattr(cfg.data, "symbols")) else "AUX/USD"
                timeframe = (cfg.timeframes.native[0] if (cfg and hasattr(cfg, "timeframes") and hasattr(cfg.timeframes, "native")) else "1m")
            except Exception:
                symbol = "AUX/USD"
                timeframe = "1m"

        # lista modelli effettiva
        models = multi if multi else [single]
        # tipi di previsione richiesti
        ftypes = settings.get("forecast_types", ["basic"]) or ["basic"]

        base_common = {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizons": settings.get("horizons", ["1m", "5m", "15m"]),
            "N_samples": settings.get("N_samples", 200),
            "apply_conformal": settings.get("apply_conformal", True),
            "model_weight_pct": settings.get("model_weight_pct"),
            "indicator_tfs": settings.get("indicator_tfs"),
            # pipeline params
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
        }
        # filtra None
        base_common = {k: v for k, v in base_common.items() if v is not None}

        self.signals.status.emit(f"Forecast requested for {symbol} {timeframe} (models={len(models)} types={','.join(ftypes)})")

        # schedula una combinazione per ogni (modello, tipo)
        import os
        for mp in models:
            name = os.path.basename(str(mp)) if mp else "model"
            for t in ftypes:
                payload = dict(base_common)
                payload["model_path"] = mp
                payload["advanced"] = (t.lower() == "advanced")
                payload["forecast_type"] = t.lower()
                payload["source_label"] = f"{name}:{t}"
                # limit candles to keep inference lightweight
                payload.setdefault("limit_candles", 512)
                # throttle advanced: allow only one at a time
                if payload["advanced"] and self._forecast_active >= 1:
                    self.signals.status.emit(f"Forecast skipped (advanced busy): {payload['source_label']}")
                    continue
                fw = ForecastWorker(engine_url=self.engine_url, payload=payload, market_service=self.market_service, signals=self.signals)
                self._forecast_active += 1
                self.pool.start(fw)

    @Slot(object, object)
    def _on_forecast_finished(self, df, quantiles):
        # one forecast completed
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

    @Slot(dict)
    def handle_forecast_payload(self, payload: dict):
        """
        Avvia una previsione usando il payload emesso dalla ChartTab (basic/advanced).
        Unisce le PredictionSettings correnti e imposta symbol/timeframe se mancanti.
        """
        try:
            settings = PredictionSettingsDialog.get_settings() or {}
            # symbol/timeframe dal chart se non presenti
            chart_tab = getattr(self, "chart_tab", None)
            if chart_tab:
                payload.setdefault("symbol", getattr(chart_tab, "symbol", None))
                payload.setdefault("timeframe", getattr(chart_tab, "timeframe", None))
            # modello: usa quello nel payload o quello delle settings
            if not payload.get("model_path"):
                if settings.get("model_path"):
                    payload["model_path"] = settings.get("model_path")
            # horizons e altri parametri di default
            payload.setdefault("horizons", settings.get("horizons", ["1m", "5m", "15m"]))
            payload.setdefault("N_samples", settings.get("N_samples", 200))
            payload.setdefault("apply_conformal", settings.get("apply_conformal", True))
            # tipo (basic/advanced) e label
            adv = bool(payload.get("advanced", False))
            payload.setdefault("forecast_type", "advanced" if adv else "basic")
            payload.setdefault("source_label", "advanced" if adv else "basic")

            # verifiche minime
            if not payload.get("symbol") or not payload.get("timeframe"):
                self.signals.error.emit("Missing symbol/timeframe for forecast.")
                return
            if not payload.get("model_path") and payload.get("forecast_type") != "rw":
                self.signals.error.emit("Missing model_path. Open Prediction Settings.")
                return

            # throttle: avoid overlapping advanced jobs
            if bool(payload.get("advanced", False)) and self._forecast_active >= 1:
                self.signals.status.emit("Forecast: advanced already running, skipping.")
                return

            self.signals.status.emit(f"Forecast (payload) for {payload.get('symbol')} {payload.get('timeframe')} [{payload.get('source_label')}]")
            payload.setdefault("limit_candles", 512)
            fw = ForecastWorker(engine_url=self.engine_url, payload=payload, market_service=self.market_service, signals=self.signals)
            self._forecast_active += 1
            self.pool.start(fw)
        except Exception as e:
            logger.exception("handle_forecast_payload failed: {}", e)
            self.signals.error.emit(str(e))

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

    @Slot()
    def handle_settings_requested(self):
        try:
            from .settings_dialog import SettingsDialog
            dlg = SettingsDialog(self.main_window)
            dlg.exec()
        except Exception as e:
            self.signals.error.emit(str(e))


class _IngestWorker(QRunnable):
    """Worker that runs backfill over configured symbols/timeframes with progress."""
    def __init__(self, market_service: MarketDataService, signals: UIControllerSignals):
        super().__init__()
        self.market_service = market_service
        self.signals = signals

    def run(self):
        try:
            self.signals.status.emit("Backfill: running...")
            # symbols from settings or safe defaults
            try:
                from ..utils.user_settings import get_setting
                symbols = get_setting("user_symbols", []) or ["EUR/USD","GBP/USD","AUX/USD","GBP/NZD","AUD/JPY","GBP/EUR","GBP/AUD"]
            except Exception:
                symbols = ["EUR/USD"]
            for sym in symbols:
                def _cb(pct: int, s=sym):
                    try: self.signals.status.emit(f"Backfill {s}: {pct}%")
                    except Exception: pass
                # abilita REST per la durata del backfill di questo simbolo
                try:
                    setattr(self.market_service, "rest_enabled", True)
                except Exception:
                    pass
                try:
                    # use '1d' to process all lower TFs up to daily
                    self.market_service.backfill_symbol_timeframe(sym, "1d", force_full=False, progress_cb=_cb)
                finally:
                    try:
                        setattr(self.market_service, "rest_enabled", False)
                    except Exception:
                        pass
            self.signals.status.emit("Backfill: completed")
        except Exception as e:
            logger.exception("Backfill worker failed: {}", e)
            self.signals.error.emit(str(e))
            self.signals.status.emit("Backfill failed")


# --- Training controller (async subprocess with logging/progress) ---

class TrainingControllerSignals(QObject):
    log = Signal(str)
    progress = Signal(int)  # 0..100; -1 for indeterminate
    finished = Signal(bool)  # ok

class TrainingController(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = TrainingControllerSignals()
        self.pool = QThreadPool.globalInstance()

    def start_training(self, args: list[str], cwd: Optional[str] = None):
        """Launch training subprocess asynchronously and stream logs to signals."""
        class _Runner(QRunnable):
            def __init__(self, outer, args, cwd):
                super().__init__()
                self.outer = outer
                self.args = args
                self.cwd = cwd

            def run(self):
                import subprocess, sys, time
                ok = False
                try:
                    self.outer.signals.progress.emit(-1)  # indeterminate
                    p = subprocess.Popen(self.args, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    best_r2 = None
                    for line in iter(p.stdout.readline, ''):
                        if not line:
                            break
                        self.outer.signals.log.emit(line.rstrip())
                        # heuristic progress by R2
                        try:
                            if "r2=" in line.lower():
                                import re
                                m = re.search(r"R2=([\-0-9\.eE]+)", line)
                                if m:
                                    r2 = float(m.group(1))
                                    best_r2 = r2 if (best_r2 is None or r2 > best_r2) else best_r2
                                    self.outer.signals.log.emit(f"[metric] R2={r2:.6f}")
                        except Exception:
                            pass
                    rc = p.wait()
                    ok = (rc == 0)
                except Exception as e:
                    self.outer.signals.log.emit(f"[error] {e}")
                    ok = False
                finally:
                    self.outer.signals.progress.emit(100 if ok else 0)
                    self.outer.signals.finished.emit(ok)
        self.pool.start(_Runner(self, args, cwd))

    def start_training_ga(self, base_args: list[str], cwd: Optional[str], strategy: str = "genetic-basic", generations: int = 5, pop_size: int = 8):
        """
        Genetic optimization loop orchestrating multiple training runs.
        - strategy:
            * 'genetic-basic': single-obiettivo (massimizza R2)
            * 'nsga2': multi-obiettivo con ordinamento non-dominato e crowding distance (minimizza [-R2, MAE])
        Progress determinato su (generations * pop_size), valutando ogni individuo generato.
        """
        class _GAJob(QRunnable):
            def __init__(self, outer, base_args, cwd, strategy, gens, pop):
                super().__init__()
                self.outer = outer
                self.base_args = base_args
                self.cwd = cwd
                self.strategy = strategy
                self.gens = max(1, int(gens))
                self.pop = max(2, int(pop))

            def _spawn_and_eval_objs(self, args) -> tuple[float, float]:
                """
                Esegue un run di training e ritorna (obj1, obj2) dove:
                 - obj1 = -R2 (da minimizzare)
                 - obj2 = MAE (da minimizzare)
                Penalizza run falliti con valori molto alti.
                """
                import subprocess, re
                r2_val = None
                mae_val = None
                p = subprocess.Popen(args, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in iter(p.stdout.readline, ''):
                    if not line:
                        break
                    self.outer.signals.log.emit(line.rstrip())
                    try:
                        m1 = re.search(r"R2=([\-0-9\.eE]+)", line)
                        if m1: r2_val = float(m1.group(1))
                        m2 = re.search(r"MAE=([\-0-9\.eE]+)", line)
                        if m2: mae_val = float(m2.group(1))
                    except Exception:
                        pass
                rc = p.wait()
                if rc != 0 or r2_val is None or mae_val is None:
                    self.outer.signals.log.emit("[warn] training run failed or metrics missing; penalizing fitness")
                    return (1e9, 1e9)
                # obj1 = -R2 to minimize, obj2 = MAE
                return (-float(r2_val), float(mae_val))

            def _spawn_and_eval_r2(self, args) -> float:
                import subprocess, re
                best_r2 = None
                p = subprocess.Popen(args, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in iter(p.stdout.readline, ''):
                    if not line:
                        break
                    self.outer.signals.log.emit(line.rstrip())
                    m = re.search(r"R2=([\-0-9\.eE]+)", line)
                    if m:
                        try:
                            r2 = float(m.group(1)); best_r2 = r2 if (best_r2 is None or r2 > best_r2) else best_r2
                        except Exception:
                            pass
                rc = p.wait()
                if rc != 0:
                    self.outer.signals.log.emit("[warn] training run failed; penalizing fitness")
                    return -1e9
                return best_r2 if best_r2 is not None else -1e9

            def _mutate(self, conf: dict) -> dict:
                import random
                c = conf.copy()
                if c.get("model") in ("ridge","lasso","elasticnet"):
                    c["alpha"] = max(1e-6, c.get("alpha", 1.0) * (10 ** random.uniform(-0.5, 0.5)))
                    if c["model"] == "elasticnet":
                        c["l1_ratio"] = min(0.99, max(0.01, c.get("l1_ratio", 0.5) + random.uniform(-0.1, 0.1)))
                elif c.get("model") == "rf":
                    c["n_estimators"] = max(50, min(1000, int(c.get("n_estimators", 200) + random.randint(-50, 50))))
                    md = c.get("max_depth", None)
                    if md is None:
                        md = 10
                    c["max_depth"] = max(3, min(64, int(md + random.randint(-3, 3))))
                if c.get("encoder") == "pca":
                    c["encoder_dim"] = max(8, min(256, int(c.get("encoder_dim", 64) + (random.randint(-8, 8)))))
                return c

            def _crossover(self, a: dict, b: dict) -> dict:
                import random
                child = {}
                keys = set(a.keys()).union(b.keys())
                for k in keys:
                    child[k] = a.get(k) if random.random() < 0.5 else b.get(k)
                return child

            def _args_from_conf_base(self, base: list[str], c: dict) -> list[str]:
                args = list(base)
                if "--model" in args:
                    i = args.index("--model"); args[i+1] = c.get("model", args[i+1])
                else:
                    args += ["--model", c.get("model","ridge")]
                if c.get("model") in ("ridge","lasso","elasticnet"):
                    if "--alpha" in args:
                        j = args.index("--alpha"); args[j+1] = str(c.get("alpha", 1.0))
                    else:
                        args += ["--alpha", str(c.get("alpha", 1.0))]
                    if c.get("model") == "elasticnet":
                        if "--l1_ratio" in args:
                            k = args.index("--l1_ratio"); args[k+1] = str(c.get("l1_ratio", 0.5))
                        else:
                            args += ["--l1_ratio", str(c.get("l1_ratio", 0.5))]
                elif c.get("model") == "rf":
                    args += ["--n_estimators", str(c.get("n_estimators", 200))]
                    if c.get("max_depth") is not None:
                        args += ["--max_depth", str(c.get("max_depth"))]
                if "--encoder" in args:
                    i = args.index("--encoder"); args[i+1] = c.get("encoder", args[i+1])
                else:
                    args += ["--encoder", c.get("encoder","none")]
                if c.get("encoder") == "pca":
                    args += ["--encoder_dim", str(c.get("encoder_dim", 64))]
                return args

            def _non_dominated_sort(self, objs: list[tuple[float,float]]) -> list[list[int]]:
                """
                Fast non-dominated sorting (O(n^2)).
                Returns list of fronts (list of indices).
                """
                n = len(objs)
                S = [set() for _ in range(n)]
                n_dom = [0]*n
                fronts: list[list[int]] = [[]]
                for p in range(n):
                    for q in range(n):
                        if p == q: 
                            continue
                        op = objs[p]; oq = objs[q]
                        # p dominates q if op <= oq for all, and < for at least one
                        if (op[0] <= oq[0] and op[1] <= oq[1]) and (op[0] < oq[0] or op[1] < oq[1]):
                            S[p].add(q)
                        elif (oq[0] <= op[0] and oq[1] <= op[1]) and (oq[0] < op[0] or oq[1] < op[1]):
                            n_dom[p] += 1
                    if n_dom[p] == 0:
                        fronts[0].append(p)
                i = 0
                while i < len(fronts) and fronts[i]:
                    next_front: list[int] = []
                    for p in fronts[i]:
                        for q in S[p]:
                            n_dom[q] -= 1
                            if n_dom[q] == 0:
                                next_front.append(q)
                    i += 1
                    if next_front:
                        fronts.append(next_front)
                return fronts

            def _crowding_distance(self, objs: list[tuple[float,float]], front: list[int]) -> dict[int, float]:
                """
                Crowding distance per front. Obj sono da minimizzare.
                """
                import math
                if not front: 
                    return {}
                distances = {i: 0.0 for i in front}
                for m in range(2):
                    sorted_idx = sorted(front, key=lambda i: objs[i][m])
                    fmin = objs[sorted_idx[0]][m]
                    fmax = objs[sorted_idx[-1]][m]
                    distances[sorted_idx[0]] = distances[sorted_idx[-1]] = math.inf
                    if fmax == fmin:
                        continue
                    for k in range(1, len(sorted_idx)-1):
                        prev = objs[sorted_idx[k-1]][m]
                        nextv = objs[sorted_idx[k+1]][m]
                        distances[sorted_idx[k]] += (nextv - prev) / (fmax - fmin)
                return distances

            def run(self):
                import random, copy
                total = self.gens * self.pop
                done = 0
                self.outer.signals.progress.emit(0)

                # init population
                pop: list[dict] = []
                for i in range(self.pop):
                    conf = {
                        "model": random.choice(["ridge","lasso","elasticnet","rf"]),
                        "alpha": 10 ** random.uniform(-3, 1),
                        "l1_ratio": random.uniform(0.1, 0.9),
                        "n_estimators": random.randint(100, 500),
                        "max_depth": random.choice([None, 8, 12, 16, 20]),
                        "encoder": random.choice(["none","pca"]),
                        "encoder_dim": random.choice([32, 64, 96, 128]),
                    }
                    pop.append(conf)

                root = None  # not used outside; args build uses base self.base_args

                if self.strategy == "genetic-basic":
                    best_conf = None
                    best_fit = -1e12
                    for g in range(self.gens):
                        fits = []
                        for i, conf in enumerate(pop):
                            args = self._args_from_conf_base(self.base_args, conf)
                            self.outer.signals.log.emit(f"[GA] Gen {g+1}/{self.gens} Ind {i+1}/{self.pop} -> {conf}")
                            fit = self._spawn_and_eval_r2(args)
                            fits.append(fit)
                            done += 1
                            pct = int(min(100, max(0, (done / total) * 100)))
                            self.outer.signals.progress.emit(pct)
                            if fit > best_fit:
                                best_fit = fit; best_conf = copy.deepcopy(conf)
                        ranked = sorted(zip(pop, fits), key=lambda x: x[1], reverse=True)
                        survivors = [c for c, f in ranked[: max(2, self.pop // 2)]]
                        next_pop = survivors[:]
                        while len(next_pop) < self.pop:
                            a = random.choice(survivors); b = random.choice(survivors)
                            child = self._crossover(a, b)
                            child = self._mutate(child)
                            next_pop.append(child)
                        pop = next_pop
                    if best_conf is not None:
                        args = self._args_from_conf_base(self.base_args, best_conf)
                        self.outer.signals.log.emit(f"[GA] Best config: {best_conf} -> running final training")
                        _ = self._spawn_and_eval_r2(args)
                    self.outer.signals.finished.emit(True)
                    return

                # NSGA-II
                # evaluate initial population
                objs: list[tuple[float,float]] = []
                for i, conf in enumerate(pop):
                    args = self._args_from_conf_base(self.base_args, conf)
                    self.outer.signals.log.emit(f"[NSGA2] Init Ind {i+1}/{self.pop} -> {conf}")
                    o = self._spawn_and_eval_objs(args)
                    objs.append(o)
                    # non contiamo le init nel progresso per semplicità (oppure includere pop iniziale)
                # start generations
                for g in range(self.gens):
                    # selection via crowded-comparison operator: torneo binario
                    def _tournament_select(indices: list[int], fronts: list[list[int]], crowd: dict[int,float]) -> int:
                        import random
                        a, b = random.sample(indices, 2)
                        # rank by front index
                        def _rank(i):
                            for r, fr in enumerate(fronts):
                                if i in fr: return r
                            return 1e9
                        ra, rb = _rank(a), _rank(b)
                        if ra < rb: return a
                        if rb < ra: return b
                        # tie-breaker: higher crowding wins
                        ca, cb = crowd.get(a, 0.0), crowd.get(b, 0.0)
                        return a if ca >= cb else b

                    # build mating pool
                    fronts = self._non_dominated_sort(objs)
                    # compute crowding per front
                    crowd_all: dict[int,float] = {}
                    for fr in fronts:
                        crowd = self._crowding_distance(objs, fr)
                        crowd_all.update(crowd)
                    idx_all = list(range(len(pop)))
                    mating = []
                    for _ in range(self.pop):
                        sel = _tournament_select(idx_all, fronts, crowd_all)
                        mating.append(pop[sel])

                    # create offspring via crossover/mutation
                    offspring: list[dict] = []
                    while len(offspring) < self.pop:
                        a = random.choice(mating); b = random.choice(mating)
                        child = self._crossover(a, b)
                        child = self._mutate(child)
                        offspring.append(child)

                    # evaluate offspring
                    off_objs: list[tuple[float,float]] = []
                    for i, conf in enumerate(offspring):
                        args = self._args_from_conf_base(self.base_args, conf)
                        self.outer.signals.log.emit(f"[NSGA2] Gen {g+1}/{self.gens} Off {i+1}/{self.pop} -> {conf}")
                        o = self._spawn_and_eval_objs(args)
                        off_objs.append(o)
                        done += 1
                        pct = int(min(100, max(0, (done / total) * 100)))
                        self.outer.signals.progress.emit(pct)

                    # combine population and perform non-dominated sorting
                    union_pop = pop + offspring
                    union_objs = objs + off_objs
                    fronts = self._non_dominated_sort(union_objs)

                    # fill next population
                    new_pop: list[dict] = []
                    new_objs: list[tuple[float,float]] = []
                    for fr in fronts:
                        if len(new_pop) + len(fr) <= self.pop:
                            for i in fr:
                                new_pop.append(union_pop[i])
                                new_objs.append(union_objs[i])
                        else:
                            # need to select by crowding distance
                            crowd = self._crowding_distance(union_objs, fr)
                            # sort fr by descending crowding
                            fr_sorted = sorted(fr, key=lambda i: crowd.get(i, 0.0), reverse=True)
                            needed = self.pop - len(new_pop)
                            for i in fr_sorted[:needed]:
                                new_pop.append(union_pop[i])
                                new_objs.append(union_objs[i])
                            break
                    pop, objs = new_pop, new_objs

                    # log current best front summary
                    try:
                        best_front = fronts[0] if fronts and fronts[0] else []
                        if best_front:
                            bf = [(union_objs[i][0], union_objs[i][1]) for i in best_front]
                            # remember obj1=-R2 => R2 = -obj1
                            summary = ", ".join([f"(R2={-o1:.4f}, MAE={o2:.4e})" for o1, o2 in bf[:min(5,len(bf))]])
                            self.outer.signals.log.emit(f"[NSGA2] Gen {g+1} best front: {summary}")
                    except Exception:
                        pass

                # optional: final best according to a scalarization (e.g., minimize obj2 while obj1 near best)
                self.outer.signals.finished.emit(True)

        self.pool.start(_GAJob(self, base_args, cwd, strategy, generations, pop_size))
