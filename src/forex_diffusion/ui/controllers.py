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
            try:
                logger.info("ForecastWorker: local forecast ready, df_rows=%s, q50_len=%s",
                            len(df_local) if df_local is not None else 0,
                            len(quantiles_local.get("q50", [])) if isinstance(quantiles_local, dict) else "n/a")
            except Exception:
                pass
            self.signals.forecastReady.emit(df_local, quantiles_local)
        except Exception as e:
            logger.exception("Forecast worker failed: {}", e)
            self.signals.error.emit(str(e))
            self.signals.status.emit("Forecast: failed")

    def _local_infer(self):
        """
        Fallback local inference using model file specified in payload['model_path'].
        Returns (df_candles, quantiles_dict).
        """
        import pickle, traceback, numpy as np
        try:
            model_path = self.payload.get("model_path") or self.payload.get("model")
            try:
                logger.debug("ForecastWorker._local_infer start: model_path=%s payload_keys=%s", model_path, list(self.payload.keys()))
            except Exception:
                pass
            if not model_path:
                raise RuntimeError("No model_path provided for local fallback")

            # load recent candles from DB (limit from payload or default)
            limit = int(self.payload.get("limit_candles", 256))
            sym = self.payload.get("symbol", None)
            tf = self.payload.get("timeframe", None)
            df_candles = self._fetch_recent_candles(self.market_service.engine, sym, tf, n_bars=limit)
            if df_candles is None or df_candles.empty:
                raise RuntimeError("No candles available for local inference")

            # compute features using pipeline_process
            try:
                from forex_diffusion.features.pipeline import pipeline_process
            except Exception as e:
                raise RuntimeError(f"pipeline_process import failed: {e}")

            features_config = {
                "warmup_bars": int(self.payload.get("warmup_bars", 16)),
                "indicators": {
                    "atr": {"n": int(self.payload.get("atr_n", 14))},
                    "rsi": {"n": int(self.payload.get("rsi_n", 14))},
                    "bollinger": {"n": int(self.payload.get("bb_n", 20))},
                    "hurst": {"window": int(self.payload.get("hurst_window", 64))},
                },
                "standardization": {"window_bars": int(self.payload.get("rv_window", 60))}
            }
            feats, _ = pipeline_process(df_candles.copy(), timeframe=tf or "1m", features_config=features_config)
            if feats.empty:
                raise RuntimeError("No features computed for local inference")

            # apply advanced ensure_features if provided in payload (adds missing engineered features)
            ensure_cfg = self.payload.get("ensure_cfg") or self.payload.get("advanced_cfg") or None
            if ensure_cfg:
                try:
                    from forex_diffusion.inference.prediction_config import ensure_features_for_prediction
                    # features_list may not be known yet; attempt to pass union of payload.features if present else empty
                    feats = ensure_features_for_prediction(feats, timeframe=tf or "1m", features_list=payload_obj.get("features") or [], adv_cfg=ensure_cfg)
                except Exception:
                    # non-fatal: continue with pipeline-produced feats
                    try:
                        logger.debug("ForecastWorker: ensure_features_for_prediction failed or not available")
                    except Exception:
                        pass

            # load model payload (pickle or torch)
            payload_obj = None
            p = Path(model_path)
            try:
                payload_obj = pickle.loads(p.read_bytes())
            except Exception:
                try:
                    import torch
                    raw = torch.load(str(p))
                    payload_obj = raw if isinstance(raw, dict) else {"model": raw}
                except Exception as e:
                    raise RuntimeError(f"Failed to load model file: {e}")

            model = payload_obj.get("model")
            features_list = payload_obj.get("features") or []
            mu = payload_obj.get("std_mu", {}) or {}
            sigma = payload_obj.get("std_sigma", {}) or {}

            # Build X from last rows of feats
            missing = [f for f in features_list if f not in feats.columns]
            if missing:
                try:
                    logger.warning("ForecastWorker._local_infer: missing features, synthesizing with mu/0.0 -> %s", missing)
                except Exception:
                    pass
                for col in missing:
                    try:
                        fill_val = float(mu.get(col, 0.0)) if isinstance(mu, dict) else 0.0
                    except Exception:
                        fill_val = 0.0
                    # create constant column with same length as feats so that after standardization it tends to zero if mu provided
                    feats[col] = fill_val
            # ensure column order matches features_list
            X = feats[features_list].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # apply mu/sigma if present
            for col in features_list:
                mu_c = mu.get(col)
                sig_c = sigma.get(col)
                if mu_c is not None and sig_c is not None:
                    try:
                        denom = float(sig_c) if float(sig_c) != 0.0 else 1.0
                        X[col] = (X[col].astype(float) - float(mu_c)) / denom
                    except Exception:
                        X[col] = X[col].astype(float).fillna(0.0)

            X_arr = X.to_numpy(dtype=float)

            # predict using sklearn-like or torch
            preds = None
            try:
                import numpy as _np
                try:
                    import torch
                    TORCH = True
                except Exception:
                    TORCH = False
                if TORCH and (hasattr(model, "forward") or isinstance(model, __import__("torch").nn.Module)):
                    model.eval()
                    with __import__("torch").no_grad():
                        t_in = __import__("torch").tensor(X_arr, dtype=__import__("torch").float32)
                        out = model(t_in)
                        preds = out.cpu().numpy().squeeze()
                else:
                    preds = np.asarray(model.predict(X_arr)).squeeze()
            except Exception as e:
                raise RuntimeError(f"Model predict failed: {e}")

            # build horizon sequence
            horizon = int(self.payload.get("horizon", 5))
            output_type = self.payload.get("output_type", "returns")
            if getattr(preds, "ndim", 0) == 0:
                seq = np.repeat(float(preds), horizon)
            elif preds.ndim == 1:
                seq = preds[-horizon:] if len(preds) >= horizon else np.concatenate([preds, np.repeat(preds[-1], horizon - len(preds))])
            else:
                try:
                    seq = preds[-1].reshape(-1)[:horizon]
                    if seq.size < horizon:
                        seq = np.pad(seq, (0, horizon - seq.size), mode='edge')
                except Exception:
                    seq = np.repeat(float(np.ravel(preds)[-1]), horizon)

            last_close = float(df_candles["close"].iat[-1])
            if output_type == "returns":
                prices = [last_close]
                for r in seq:
                    prices.append(prices[-1] * (1.0 + float(r)))
                forecast_prices = np.array(prices[1:], dtype=float)
            else:
                forecast_prices = np.array(seq, dtype=float)

            # build quantiles placeholder (simple deterministic offered as q50)
            quantiles = {"q50": forecast_prices.tolist(), "q05": (forecast_prices * 0.99).tolist(), "q95": (forecast_prices * 1.01).tolist()}

            return df_candles, quantiles
        except Exception as e:
            logger.exception("Local inference failed: %s", e)
            raise

    def _fetch_recent_candles(self, engine, symbol: str, timeframe: str, n_bars: int = 500) -> pd.DataFrame:
        """
        Query market_data_candles for the last n_bars for the given symbol/timeframe.
        Returns DataFrame sorted ascending by ts_utc.

        Uses SQLAlchemy reflection to locate the table safely.
        """
        try:
            from sqlalchemy import MetaData, select
            meta = MetaData()
            meta.reflect(bind=engine, only=["market_data_candles"])
            tbl = meta.tables.get("market_data_candles")
            if tbl is None:
                return pd.DataFrame()
            with engine.connect() as conn:
                stmt = (
                    select(
                        tbl.c.ts_utc,
                        tbl.c.open,
                        tbl.c.high,
                        tbl.c.low,
                        tbl.c.close,
                        tbl.c.volume,
                    )
                    .where(tbl.c.symbol == symbol)
                    .where(tbl.c.timeframe == timeframe)
                    .order_by(tbl.c.ts_utc.desc())
                    .limit(n_bars)
                )
                rows = conn.execute(stmt).fetchall()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=["ts_utc", "open", "high", "low", "close", "volume"])
                df = df.sort_values("ts_utc").reset_index(drop=True)
                return df
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
        self.db_writer = db_writer  # optional background writer for async persistence

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

    @Slot()
    def handle_ingest_requested(self):
        self.signals.status.emit("Ingest requested: launching backfill...")
        # run backfill in background via MarketDataService.ensure_startup_backfill
        worker = _IngestWorker(self.market_service, self.signals)
        self.pool.start(worker)

    @Slot()
    def handle_train_requested(self):
        self.signals.status.emit("Train requested (not implemented).")

    @Slot()
    def handle_forecast_requested(self):
        # Prepare simple payload using default symbol/timeframe from config if available
        cfg = self.market_service.cfg if hasattr(self.market_service, "cfg") else None
        try:
            symbol = cfg.data.symbols[0] if (cfg and hasattr(cfg, "data") and hasattr(cfg.data, "symbols")) else (cfg.data.get("symbols", [])[0] if isinstance(cfg.data, dict) else "EUR/USD")
            timeframe = (cfg.timeframes.native[0] if (cfg and hasattr(cfg, "timeframes") and hasattr(cfg.timeframes, "native")) else "1m")
        except Exception:
            symbol = "EUR/USD"
            timeframe = "1m"
        payload = {"symbol": symbol, "timeframe": timeframe, "horizons": ["1m", "5m", "15m"], "N_samples": 200, "apply_conformal": True}
        self.signals.status.emit(f"Forecast requested for {symbol} {timeframe}")

        # Log the forecast request asynchronously via DBWriter if available (lightweight audit)
        try:
            if getattr(self, "db_writer", None) is not None:
                self.db_writer.write_prediction_async(symbol=symbol, timeframe=timeframe, horizon="request", q05=0.0, q50=0.0, q95=0.0, meta={"event": "forecast_requested"})
        except Exception:
            # non-fatal: ignore logging errors from UI
            pass

        fw = ForecastWorker(engine_url=self.engine_url, payload=payload, market_service=self.market_service, signals=self.signals)
        self.pool.start(fw)

    def request_forecast(self, payload: dict) -> None:
        """
        Start a ForecastWorker using the provided payload (called from UI, e.g. ChartTab).
        Adds debug logging and emits status updates.
        """
        try:
            # Basic validation/logging
            symbol = payload.get("symbol", "unknown")
            tf = payload.get("timeframe", "unknown")
            try:
                logger.info("UIController.request_forecast called for %s %s payload=%s", symbol, tf, payload)
            except Exception:
                pass
            self.signals.status.emit(f"Forecast (UI) requested for {symbol} {tf}")

            # audit/log via db_writer if present (best-effort)
            try:
                if getattr(self, "db_writer", None) is not None:
                    self.db_writer.write_prediction_async(symbol=symbol, timeframe=tf, horizon="request", q05=0.0, q50=0.0, q95=0.0, meta={"event": "ui_forecast_requested"})
            except Exception:
                pass

            fw = ForecastWorker(engine_url=self.engine_url, payload=payload, market_service=self.market_service, signals=self.signals)
            self.pool.start(fw)
        except Exception as e:
            try:
                logger.exception("request_forecast failed: %s", e)
            except Exception:
                pass
            self.signals.error.emit(str(e))
            self.signals.status.emit("Forecast request failed")

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
