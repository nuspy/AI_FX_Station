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
        import numpy as np
        import pickle
        import hashlib

        sym = self.payload.get("symbol")
        tf = (self.payload.get("timeframe") or "1m")
        horizons = self.payload.get("horizons", ["5m"])
        limit = int(self.payload.get("limit_candles", 512))
        ftype = str(self.payload.get("forecast_type", "basic")).lower()

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

        # 2) carica modello (se non RW)
        used_model_path_str = ""
        model_sha16 = None
        payload_obj = {}
        model = None
        features_list: List[str] = []
        std_mu: Dict[str, float] = {}
        std_sigma: Dict[str, float] = {}
        if ftype != "rw":
            mp = self.payload.get("model_path") or self.payload.get("model")
            p = self._resolve_model_path(str(mp))
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            used_model_path_str = str(p)
            try:
                model_sha16 = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            except Exception:
                model_sha16 = None
            # payload pickle
            with open(p, "rb") as f:
                payload_obj = pickle.load(f)
            model = payload_obj.get("model")
            features_list = payload_obj.get("features") or []
            # supporto artifact legacy: 'std' senza 'std_sigma'
            std_mu = payload_obj.get("std_mu") or payload_obj.get("std") or {}
            std_sigma = payload_obj.get("std_sigma") or ({c: 1.0 for c in features_list} if "std" in payload_obj else {})
            if model is None or not features_list:
                raise RuntimeError("Model payload missing 'model' or 'features'")
        else:
            # RW baseline
            std_mu, std_sigma = {}, {}

        # 3) features via pipeline **senza** rifit dello standardizer
        no_std = Standardizer(cols=[], mu={}, sigma={})
        feats_cfg = {
            "warmup_bars": int(self.payload.get("warmup_bars", 16)),
            "indicators": {
                "atr": {"n": int(self.payload.get("atr_n", 14))},
                "rsi": {"n": int(self.payload.get("rsi_n", 14))},
                "bollinger": {"n": int(self.payload.get("bb_n", 20))},
                "hurst": {"window": int(self.payload.get("hurst_window", 64))},
            },
            # IMPORTANT: mai None; passa un dict
            "standardization": {"enabled": False},  # disattiva lo standardizer interno del pipeline
        }
        feats_df, _ = pipeline_process(df_candles.copy(), timeframe=tf, features_config=feats_cfg, standardizer=no_std)
        if feats_df is None or feats_df.empty:
            raise RuntimeError("No features computed for local inference")

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
            # torch?
            try:
                import torch
                if hasattr(model, "eval"):
                    model.eval()
                with torch.no_grad():
                    t_in = torch.tensor(X_arr[-max(1, len(horizons)):, :], dtype=torch.float32)
                    out = model(t_in)
                    preds = np.ravel(out.detach().cpu().numpy())
            except Exception:
                # sklearn?
                if hasattr(model, "predict"):
                    preds = np.ravel(model.predict(X_arr[-max(1, len(horizons)):, :]))
        if preds is None:
            # fallback: zero-returns
            preds = np.zeros((max(1, len(horizons)),), dtype=float)

        # 7) prezzi e quantili semplici (vol-based)
        last_close = float(df_candles["close"].iat[-1])
        seq = preds[-len(horizons):] if preds.size >= len(horizons) else \
              (np.pad(preds, (0, len(horizons) - preds.size), mode="edge") if preds.size > 0 else np.zeros(len(horizons)))
        prices = []
        p = last_close
        for r in seq:
            p *= (1.0 + float(r))
            prices.append(p)
        q50 = np.asarray(prices, dtype=float)

        # volatilità realizzata per bande
        logret = pd.Series(df_candles["close"], dtype=float).pipe(lambda s: np.log(s).diff()).dropna()
        sigma_base = float(logret.tail(512).std() if len(logret) else 0.0)
        z = 1.645 if bool(self.payload.get("apply_conformal", True)) else 1.0
        k = np.arange(1, len(q50) + 1, dtype=float)
        band_rel = np.clip(z * sigma_base * np.sqrt(k), 1e-6, 0.2)
        q05 = np.maximum(1e-12, q50 * (1.0 - band_rel))
        q95 = np.maximum(1e-12, q50 * (1.0 + band_rel))

        # 8) future_ts ancorati a last_ts + Δ(tf/horizon-label)
        base = pd.to_datetime(int(df_candles["ts_utc"].iat[-1]), unit="ms", utc=True)
        future_ts = []
        for h in horizons:
            try:
                future_ts.append(int((base + pd.to_timedelta(str(h))).value // 1_000_000))
            except Exception:
                future_ts.append(int((base + pd.to_timedelta("1m")).value // 1_000_000))

        display_name = str(self.payload.get("name") or self.payload.get("source_label") or "forecast")
        quantiles = {
            "q50": q50.tolist(),
            "q05": q05.tolist(),
            "q95": q95.tolist(),
            "future_ts": future_ts,
            "source": display_name,
            "label": display_name,
            "model_path_used": used_model_path_str,
            "model_sha16": model_sha16,
        }
        return df_candles, quantiles


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

        # raccogli TUTTE le sorgenti (unione)
        def _norm_paths(src) -> List[str]:
            try:
                if not src: return []
                if isinstance(src, (list, tuple, set)):
                    return [str(s).strip() for s in src if str(s).strip()]
                s = str(src).strip()
                import re
                if any(sep in s for sep in [",",";","\n"]):
                    return [t.strip() for t in re.split(r"[,\n;]+", s) if t.strip()]
                return [s] if s else []
            except Exception:
                return []

        models: List[str] = []
        models += _norm_paths(settings.get("model_paths"))
        models += _norm_paths(settings.get("models"))
        try:
            models += [p for p in (PredictionSettingsDialog.get_model_paths() or []) if p]
        except Exception:
            pass
        models += _norm_paths(settings.get("model_path") or settings.get("model"))

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

        # forza basic se multi-modello (evita advanced concorrenti)
        for mp in models:
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "horizons": settings.get("horizons", ["1m","5m","15m"]),
                "N_samples": settings.get("N_samples", 200),
                "apply_conformal": settings.get("apply_conformal", True),
                "limit_candles": settings.get("limit_candles", 512),
                "model_path": mp,
                "source_label": label_map[mp],
                "name": label_map[mp],
                "forecast_type": "basic",
                "advanced": False,
                "allowed_models": list(models),
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

            for mp in models:
                pl = dict(payload)
                pl["model_path"] = mp
                pl["forecast_type"] = forecast_type
                pl["advanced"] = (forecast_type == "advanced")
                pl["allowed_models"] = list(models)
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
