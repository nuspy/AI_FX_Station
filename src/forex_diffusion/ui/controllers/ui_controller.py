# ui/controllers/ui_controller.py
# Main UI Controller for ForexGPT - handles menu actions and forecast management
from __future__ import annotations

import os
import sys
from typing import Dict, List
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Slot
from loguru import logger

from ...services.marketdata import MarketDataService
from ..prediction_settings_dialog import PredictionSettingsDialog
from ..workers.forecast_worker import ForecastWorker
from ..handlers.signals import UIControllerSignals


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
            from ...utils.user_settings import get_setting
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
                        from ...utils.user_settings import get_setting
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
            from ..prediction_settings_dialog import PredictionSettingsDialog
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
            # tenta il dialog "ufficiale"
            from ..indicators_dialog import IndicatorsDialog  # deve esistere nel tuo repo
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
                from ...utils.user_settings import set_setting
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
        from ...models.model_path_resolver import ModelPathResolver

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