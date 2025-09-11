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
            try:
                logger.info("ForecastWorker: local forecast ready, df_rows={}, q50_len={}", len(df_local),
                            len(quantiles_local["q50"]))
                logger.info(f"on_forecast_ready: detected q50 as returns -> converting to prices (last_close={last_close})")


            except Exception:
                pass
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

        # ensure (se presente nel progetto)
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

        # loader pickle sicuro
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
        horizon = int(self.payload.get("horizon", 5))
        output_type = str(self.payload.get("output_type", "returns")).lower()

        # --- 2) Dati: ultime N candele (DEFINIZIONE SICURA DI df_candles) ---
        df_candles = self._fetch_recent_candles(self.market_service.engine, sym, tf, n_bars=limit)
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

        # --- LOG parametri Basic ---
        try:
            ind_dict = features_config.get("indicators", {})
            logger.info(
                "Basic forecast params: model={} symbol={} timeframe={} limit={} horizon={} "
                "indicators_keys={} standardization.window_bars={}",
                model_path, sym, tf, limit, horizon, list(ind_dict.keys()),
                features_config.get("standardization", {}).get("window_bars")
            )
            logger.debug("Basic features_config.indicators={}", ind_dict)
        except Exception as e:
            logger.debug("Basic forecast params log skipped: {}", e)

        # --- 5) Pipeline ---
        feats, _ = pipeline_process(df_candles.copy(), timeframe=tf, features_config=features_config)
        if feats is None or feats.empty:
            raise RuntimeError("No features computed for local inference")

        # --- (A) Ritorni e rolling std coerenti col modello ---
        # r: log-returns 1-step, se non già presenti
        if "r" not in feats.columns:
            c = feats["close"].astype(float)
            r = np.log(c).diff()
            feats["r"] = r

        # r_std_100: rolling std dei ritorni (default 100; consenti override via payload)
        std_n = int(self.payload.get("std_window", 100))  # il modello chiede r_std_100 → 100 è lo standard
        if f"r_std_{std_n}" not in feats.columns:
            feats[f"r_std_{std_n}"] = feats["r"].rolling(std_n, min_periods=std_n // 2).std()

        # Se il modello vuole ESATTAMENTE 'r_std_100' ma std_n ≠ 100, duplica/alias
        if "r_std_100" in features_list and f"r_std_{std_n}" in feats.columns and std_n != 100:
            feats["r_std_100"] = feats[f"r_std_{std_n}"]

        # --- (B) MACD (se mancante), usando parametri ragionevoli (o payload se presenti) ---
        if "macd" not in feats.columns:
            ema_fast = int(self.payload.get("ema_fast", 12))
            ema_slow = int(self.payload.get("ema_slow", 26))
            ema_sig = int(self.payload.get("ema_signal", 9))
            px = feats["close"].astype(float)
            ema_f = px.ewm(span=ema_fast, adjust=False).mean()
            ema_s = px.ewm(span=ema_slow, adjust=False).mean()
            macd_line = ema_f - ema_s
            macd_signal = macd_line.ewm(span=ema_sig, adjust=False).mean()
            feats["macd"] = macd_line  # se il modello ha 'macd' come linea, questa è coerente

        # --- (C) Time encodings (hour_sin/hour_cos) se mancanti ---
        if ("hour_sin" not in feats.columns) or ("hour_cos" not in feats.columns):
            # ts_utc in ms → datetime UTC
            ts = pd.to_datetime(feats["ts_utc"].astype("int64"), unit="ms", utc=True)
            minutes = ts.dt.hour * 60 + ts.dt.minute
            theta = 2 * np.pi * (minutes / (24 * 60))
            feats["hour_sin"] = np.sin(theta)
            feats["hour_cos"] = np.cos(theta)

        # --- (D) Session flags (Tokyo/London/NY) se mancanti ---
        need_sessions = any(k in features_list for k in ["session_tokyo", "session_london", "session_ny"])
        if need_sessions and not all(col in feats.columns for col in ["session_tokyo", "session_london", "session_ny"]):
            ts = pd.to_datetime(feats["ts_utc"].astype("int64"), unit="ms", utc=True)
            h = ts.dt.hour  # UTC; adatta se le tue sessioni sono mappate in altra tz
            # finestre approssimative in UTC: Tokyo ~ 00–09, London ~ 07–16, NY ~ 12–21
            feats["session_tokyo"] = ((h >= 0) & (h < 9)).astype(int)
            feats["session_london"] = ((h >= 7) & (h < 16)).astype(int)
            feats["session_ny"] = ((h >= 12) & (h < 21)).astype(int)

        # --- 6) Ensure (se disponibile) ---
        ensure_cfg = self.payload.get("ensure_cfg") or self.payload.get("advanced_cfg") or None
        if ensure_cfg and ensure_features_for_prediction is not None:
            try:
                feats = ensure_features_for_prediction(
                    feats, timeframe=tf, features_list=features_list, adv_cfg=ensure_cfg
                )
            except Exception as e:
                logger.debug("ensure_features_for_prediction failed: {}", e)

        # --- 7) Colonne mancanti → fill μ; ordine colonne; sanitizzazione ---
        missing = [c for c in features_list if c not in feats.columns]
        if missing:
            logger.warning("Basic: missing features will be filled with mu: {}", missing)
            for col in missing:
                try:
                    fill_val = float(mu.get(col, 0.0))
                except Exception:
                    fill_val = 0.0
                feats[col] = fill_val

        X = (
            feats[features_list]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # --- 8) Standardizzazione per colonna ---
        for col in features_list:
            mu_c = mu.get(col)
            sig_c = sigma.get(col)
            if mu_c is not None and sig_c is not None:
                denom = float(sig_c) if float(sig_c) != 0.0 else 1.0
                X[col] = (X[col] - float(mu_c)) / denom

        X_arr = X.to_numpy(dtype=float)

        # --- LOG diagnostico ---
        try:
            import hashlib
            logger.info("Basic model features: count={} first10={}", len(features_list), features_list[:10])
            logger.debug("Basic X shape={} hash={}", X_arr.shape, hashlib.md5(X_arr.tobytes()).hexdigest())
        except Exception as e:
            logger.debug("Basic diagnostics log skipped: {}", e)

        # --- 9) Inferenza ---
        preds = None
        try:
            import torch
            if hasattr(model, "eval"):
                model.eval()
            with torch.no_grad():
                t_in = torch.tensor(X_arr, dtype=torch.float32)
                out = model(t_in)
                preds = out.detach().cpu().numpy()
        except Exception:
            if hasattr(model, "predict"):
                preds = np.asarray(model.predict(X_arr))
            else:
                try:
                    preds = np.asarray([float(model)])
                except Exception as e:
                    raise RuntimeError(f"Unsupported model type for prediction: {e}")

        preds = np.squeeze(preds)
        if preds.size == 0:
            raise RuntimeError("Model returned empty prediction")

        # --- 10) Sequenza di lunghezza horizon ---
        if preds.ndim == 0:
            seq = np.repeat(float(preds), horizon)
        elif preds.ndim == 1:
            seq = preds[-horizon:] if preds.size >= horizon else np.concatenate(
                [preds, np.repeat(preds[-1], horizon - preds.size)]
            )
        else:
            last = preds[-1].reshape(-1)
            seq = last[:horizon] if last.size >= horizon else np.pad(last, (0, horizon - last.size), mode="edge")

        # --- 11) Prezzi e quantili ---
        last_close = float(df_candles["close"].iat[-1])
        if output_type == "returns":
            prices = [last_close]
            for r in seq:
                prices.append(prices[-1] * (1.0 + float(r)))
            forecast_prices = np.array(prices[1:], dtype=float)
        else:
            forecast_prices = np.array(seq, dtype=float)

        quantiles = {
            "q50": forecast_prices.tolist(),
            "q05": (forecast_prices * 0.99).tolist(),
            "q95": (forecast_prices * 1.01).tolist(),
        }

        return df_candles, quantiles

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

    def build_features_config_from_ensure(self, ensure_cfg: dict, *, atr_n_fallback: int = 14,
                                          warmup_bars_fallback: int = 16) -> dict:
        """
        Converte un ensure_cfg (Advanced) in un features_config coerente per pipeline_process.
        - Usa ATR(n) dal fallback (finché non metti lo slider ATR in Advanced).
        - Mappa rsi_n, bb_n/bb_k, don_n, ema_fast/ema_slow, keltner_k, hurst_window.
        - Usa rv_window per standardization.window_bars.
        """
        # fallback puliti
        atr_n = int(self._safe_get_setting("atr_n", atr_n_fallback)) if hasattr(self,
                                                                                "_safe_get_setting") else atr_n_fallback
        warmup_bars = int(self._safe_get_setting("warmup_bars", warmup_bars_fallback)) if hasattr(self,
                                                                                                  "_safe_get_setting") else warmup_bars_fallback

        # estrazioni sicure
        rsi_n = int(ensure_cfg.get("rsi_n", 14))
        bb_n = int(ensure_cfg.get("bb_n", 20))
        bb_k = float(ensure_cfg.get("bb_k", 2.0))
        don_n = int(ensure_cfg.get("don_n", 20))
        ema_fast = int(ensure_cfg.get("ema_fast", 12))
        ema_slow = int(ensure_cfg.get("ema_slow", 26))
        keltner_k = float(ensure_cfg.get("keltner_k", 1.5))
        hurst_window = int(ensure_cfg.get("hurst_window", 64))
        rv_window = int(ensure_cfg.get("rv_window", 60))

        return {
            "warmup_bars": warmup_bars,
            "indicators": {
                "atr": {"n": atr_n},
                "rsi": {"n": rsi_n},
                "bollinger": {"n": bb_n, "k": bb_k},
                "donchian": {"n": don_n},
                "ema": {"fast": ema_fast, "slow": ema_slow},
                "macd": {"fast": ema_fast, "slow": ema_slow, "signal": 9},
                "keltner": {"k": keltner_k},
                "hurst": {"window": hurst_window},
            },
            "standardization": {"window_bars": rv_window},
        }

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
