# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional, Dict, List
import pandas as pd
from pathlib import Path
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox,
    QDialog, QDialogButtonBox, QFormLayout, QSpinBox
)
from PySide6.QtCore import QTimer, Qt, Signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from loguru import logger

from ..utils.user_settings import get_setting, set_setting

class ChartTab(QWidget):
    """
    A comprehensive charting tab with real-time updates, indicators, and forecasting.
    Preserves zoom/pan when updating and supports multiple forecast overlays.
    """
    forecastRequested = Signal(dict)

    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self._main_window = parent
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.toolbar = NavigationToolbar(self.canvas, self)

        # --- Toolbar Setup ---
        topbar = QWidget()
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.toolbar)

        # Symbol selector
        from PySide6.QtWidgets import QComboBox
        self.symbol_combo = QComboBox()
        # Requested pairs
        self._symbols_supported = ["EUR/USD","GBP/USD","AUX/USD", "GBP/NZD", "AUD/JPY", "GBP/EUR", "GBP/AUD"]
        self.symbol_combo.addItems(self._symbols_supported)
        top_layout.addWidget(self.symbol_combo)

        # Basic Forecast Buttons
        self.forecast_settings_btn = QPushButton("Prediction Settings")
        self.forecast_btn = QPushButton("Make Prediction")
        top_layout.addWidget(self.forecast_settings_btn)
        top_layout.addWidget(self.forecast_btn)

        # Advanced Forecast Buttons (Restored)
        self.adv_settings_btn = QPushButton("Advanced Settings")
        self.adv_forecast_btn = QPushButton("Advanced Forecast")
        top_layout.addWidget(self.adv_settings_btn)
        top_layout.addWidget(self.adv_forecast_btn)

        # Backfill controls moved here (from History tab)
        self.backfill_btn = QPushButton("Backfill Missing")
        top_layout.addWidget(self.backfill_btn)
        from PySide6.QtWidgets import QProgressBar
        self.backfill_progress = QProgressBar()
        self.backfill_progress.setMaximumWidth(160)
        self.backfill_progress.setRange(0, 100)
        self.backfill_progress.setValue(0)
        self.backfill_progress.setTextVisible(False)
        top_layout.addWidget(self.backfill_progress)

        # Clear forecasts
        self.clear_forecasts_btn = QPushButton("Clear Forecasts")
        top_layout.addWidget(self.clear_forecasts_btn)

        top_layout.addStretch()
        self.bidask_label = QLabel("Bid: -    Ask: -")
        self.bidask_label.setStyleSheet("font-weight: bold;")
        top_layout.addWidget(self.bidask_label)
        self.layout.addWidget(topbar)
        self.layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.subplots()
        self._last_df = pd.DataFrame()
        # keep a list of forecast dicts: {id, created_at, quantiles, future_ts, artists:list, source}
        self._forecasts: List[Dict] = []
        self.max_forecasts = int(get_setting("max_forecasts", 5))

        # Auto-forecast timer
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(int(get_setting("auto_interval_seconds", 60) * 1000))
        self._auto_timer.timeout.connect(self._auto_forecast_tick)

        # Mouse interaction: Alt+Click => TestingPoint basic; Shift+Alt+Click => TestingPoint advanced
        # connect matplotlib canvas mouse press
        try:
            self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        except Exception:
            # if connection fails, continue without interactive testing feature
            logger.debug("Failed to connect mpl mouse event for testing point.")

        # --- Signal Connections ---
        self.forecast_settings_btn.clicked.connect(self._open_forecast_settings)
        self.forecast_btn.clicked.connect(self._on_forecast_clicked)
        self.adv_settings_btn.clicked.connect(self._open_adv_forecast_settings)
        self.adv_forecast_btn.clicked.connect(self._on_advanced_forecast_clicked)
        self.backfill_btn.clicked.connect(self._on_backfill_missing_clicked)
        self.clear_forecasts_btn.clicked.connect(self.clear_all_forecasts)
        # Symbol change
        self.symbol_combo.currentTextChanged.connect(self._on_symbol_changed)

        # try to auto-connect to controller signals if available
        try:
            controller = getattr(self._main_window, "controller", None)
            if controller and hasattr(controller, "signals"):
                controller.signals.forecastReady.connect(self.on_forecast_ready)
        except Exception:
            pass

    def _handle_tick(self, payload: dict):
        """Receives tick data, updates the internal DataFrame, and redraws the chart."""
        try:
            if not isinstance(payload, dict):
                return

            new_row = pd.DataFrame([payload])
            self._last_df = pd.concat([self._last_df, new_row], ignore_index=True)
            self._last_df.drop_duplicates(subset=['ts_utc'], keep='last', inplace=True)

            # Preserve zoom/pan: read limits before clear
            try:
                prev_xlim = self.ax.get_xlim()
                prev_ylim = self.ax.get_ylim()
            except Exception:
                prev_xlim = None
                prev_ylim = None

            # Crucial fix: Call update_plot to redraw the chart with the new data
            self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)

            if payload.get('bid') is not None and payload.get('ask') is not None:
                try:
                    self.bidask_label.setText(f"Bid: {payload['bid']:.5f}    Ask: {payload['ask']:.5f}")
                except Exception:
                    self.bidask_label.setText(f"Bid: {payload.get('bid')}    Ask: {payload.get('ask')}")

        except Exception as e:
            logger.exception(f"Failed to handle tick: {e}")

    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None, restore_xlim=None, restore_ylim=None):
        if df is None or df.empty:
            return

        self._last_df = df.copy()

        # preserve previous limits if provided
        try:
            prev_xlim = restore_xlim if restore_xlim is not None else self.ax.get_xlim()
            prev_ylim = restore_ylim if restore_ylim is not None else self.ax.get_ylim()
        except Exception:
            prev_xlim = None
            prev_ylim = None

        self.ax.clear()

        x_dt = pd.to_datetime(df["ts_utc"], unit="ms")

        # Use 'close' for candles, fallback to 'price' for ticks
        y_col = 'close' if 'close' in df.columns else 'price'
        self.ax.plot(x_dt, df[y_col], color="black", label="Price")

        if quantiles:
            self._plot_forecast_overlay(quantiles)

        self.ax.set_title(f"{getattr(self, 'symbol', '')} - {getattr(self, 'timeframe', '')}")
        self.ax.legend()
        self.ax.figure.autofmt_xdate()

        # restore limits to preserve zoom/pan
        try:
            if prev_xlim is not None:
                self.ax.set_xlim(prev_xlim)
            if prev_ylim is not None:
                self.ax.set_ylim(prev_ylim)
        except Exception:
            pass

        self.canvas.draw()

    def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
        """
        Plot quantiles on the chart. quantiles expected to have keys 'q50','q05','q95'
        Each value can be a list/array of floats. Optionally 'future_ts' can provide datetimes.
        """
        try:
            q50 = quantiles.get("q50")
            q05 = quantiles.get("q05") or quantiles.get("q10")
            q95 = quantiles.get("q95") or quantiles.get("q90")
            future_ts = quantiles.get("future_ts", None)
            label = quantiles.get("label", f"{source}")

            if q50 is None:
                return

            # x positions
            if future_ts:
                x_vals = pd.to_datetime(future_ts)
            else:
                if self._last_df is None or self._last_df.empty:
                    return
                last_ts = pd.to_datetime(self._last_df["ts_utc"].astype("int64"), unit="ms").iat[-1]
                td = self._tf_to_timedelta(getattr(self, "timeframe", "1m"))
                x_vals = [last_ts + td * (i + 1) for i in range(len(q50))]

            # deterministic color by source string
            def _color_for(src: str) -> str:
                palette = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
                if not src:
                    return "tab:orange"
                h = abs(hash(src)) % len(palette)
                return palette[h]
            color = _color_for(quantiles.get("source", source))

            line50, = self.ax.plot(x_vals, q50, color=color, linestyle='-', label=f"{label} (q50)")
            artists = [line50]

            if q05 is not None and q95 is not None:
                line05, = self.ax.plot(x_vals, q05, color=color, linestyle='--', alpha=0.8, label=None)
                line95, = self.ax.plot(x_vals, q95, color=color, linestyle='--', alpha=0.8, label=None)
                fill = self.ax.fill_between(x_vals, q05, q95, color=color, alpha=0.12)
                artists.extend([line05, line95, fill])

            fid = time.time()
            forecast = {
                "id": fid,
                "created_at": fid,
                "quantiles": quantiles,
                "future_ts": future_ts,
                "artists": artists,
                "source": quantiles.get("source", source)
            }
            self._forecasts.append(forecast)
            self._trim_forecasts()
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            logger.exception(f"Failed to plot forecast overlay: {e}")

    def _tf_to_timedelta(self, tf: str):
        """Convert timeframe string like '1m','5m','1h' into pandas Timedelta."""
        try:
            tf = str(tf).strip().lower()
            if tf.endswith("m"):
                return pd.to_timedelta(int(tf[:-1]), unit="m")
            if tf.endswith("h"):
                return pd.to_timedelta(int(tf[:-1]), unit="h")
            if tf.endswith("d"):
                return pd.to_timedelta(int(tf[:-1]), unit="d")
            # default 1 minute
            return pd.to_timedelta(1, unit="m")
        except Exception:
            return pd.to_timedelta(1, unit="m")

    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        self.db_service = db_service
        self.symbol = symbol
        self.timeframe = timeframe
        # sync combo if present
        try:
            if hasattr(self, "symbol_combo") and symbol:
                idx = self.symbol_combo.findText(symbol)
                if idx >= 0:
                    self.symbol_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def _on_symbol_changed(self, new_symbol: str):
        """Handle symbol change from combo: update context and reload candles from DB."""
        try:
            if not new_symbol:
                return
            self.symbol = new_symbol
            # reload last candles for this symbol/timeframe
            df = self._load_candles_from_db(new_symbol, getattr(self, "timeframe", "1m"), limit=3000)
            if df is not None and not df.empty:
                self.update_plot(df)
        except Exception as e:
            logger.exception("Failed to switch symbol: {}", e)

    def _open_forecast_settings(self):
        from .prediction_settings_dialog import PredictionSettingsDialog
        dialog = PredictionSettingsDialog(self)
        # execute and then apply relevant runtime settings (max_forecasts, auto)
        dialog.exec()
        try:
            settings = PredictionSettingsDialog.get_settings()
            self.max_forecasts = int(settings.get("max_forecasts", self.max_forecasts))
            auto = bool(settings.get("auto_predict", False))
            interval = int(settings.get("auto_interval_seconds", self._auto_timer.interval() // 1000))
            self._auto_timer.setInterval(max(1, interval) * 1000)
            if auto:
                self.start_auto_forecast()
            else:
                self.stop_auto_forecast()
        except Exception:
            pass

    def _on_canvas_click(self, event):
        """
        Handle mouse click on canvas:
        - Alt + Click => basic testing forecast from TestingPoint (same X coordinate)
        - Shift + Alt + Click => advanced testing forecast
        Builds candles_override (list of dicts) with the N previous candles up to TestingPoint and emits forecastRequested.
        """
        try:
            # require valid xdata and left mouse button (button==1)
            if event is None or event.xdata is None or getattr(event, "button", None) != 1:
                return

            # GUI event gives access to modifiers
            modifiers = None
            try:
                gui = getattr(event, "guiEvent", None)
                if gui is not None:
                    modifiers = gui.modifiers()
            except Exception:
                modifiers = None

            from PySide6.QtCore import Qt
            alt_pressed = False
            shift_pressed = False
            if modifiers is not None:
                try:
                    alt_pressed = bool(modifiers & Qt.AltModifier)
                    shift_pressed = bool(modifiers & Qt.ShiftModifier)
                except Exception:
                    alt_pressed = False
                    shift_pressed = False

            if not alt_pressed:
                return  # only interested in Alt+click combos

            # convert xdata (matplotlib float date) to utc ms
            try:
                import matplotlib.dates as mdates
                from datetime import timezone
                dt = mdates.num2date(event.xdata)
                # ensure timezone-aware UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                clicked_ms = int(dt.timestamp() * 1000)
            except Exception:
                logger.exception("Failed to convert click xdata to datetime")
                return

            if self._last_df is None or self._last_df.empty:
                QMessageBox.information(self, "No data", "No chart data available to create testing forecast.")
                return

            # find nearest ts index in _last_df by ts_utc
            try:
                arr = self._last_df["ts_utc"].astype("int64").to_numpy()
                # find index of nearest timestamp (<= clicked_ms preferred). choose the row with ts closest to clicked_ms
                import numpy as np
                idx = int(np.argmin(np.abs(arr - clicked_ms)))
                testing_ts = int(arr[idx])
            except Exception:
                logger.exception("Failed to locate testing point in data")
                return

            # load number of history bars from settings
            from .prediction_settings_dialog import PredictionSettingsDialog
            settings = PredictionSettingsDialog.get_settings() or {}
            n_bars = int(settings.get("test_history_bars", 128))

            # build candles_override: take last n_bars ending at testing_ts (inclusive)
            df = self._last_df.sort_values("ts_utc").reset_index(drop=True)
            pos = df.index[df["ts_utc"].astype("int64") == testing_ts]
            if len(pos) == 0:
                # if exact ts not found, fall back to nearest less-or-equal index
                pos = df.index[df["ts_utc"].astype("int64") <= testing_ts]
                if len(pos) == 0:
                    QMessageBox.information(self, "Testing point", "No suitable testing point found at clicked X.")
                    return
                idx_pos = int(pos[-1])
            else:
                idx_pos = int(pos[0])

            start_idx = max(0, idx_pos - n_bars + 1)
            df_slice = df.iloc[start_idx: idx_pos + 1].copy()
            if df_slice.empty:
                QMessageBox.information(self, "Testing point", "Not enough historical bars available for testing.")
                return

            # Build payload: include candles_override so worker can run local inference on this slice
            payload = {
                "symbol": getattr(self, "symbol", ""),
                "timeframe": getattr(self, "timeframe", ""),
                "testing_point_ts": int(testing_ts),
                "test_history_bars": int(n_bars),
                "candles_override": df_slice.to_dict(orient="records"),
            }
            if shift_pressed:
                payload["advanced"] = True
            else:
                payload["advanced"] = False

            # also include model_path and other settings to allow worker to run (use saved settings)
            saved = PredictionSettingsDialog.get_settings() or {}
            payload.update({
                "model_path": saved.get("model_path"),
                "horizons": saved.get("horizons", ["1m", "5m", "15m"]),
                "N_samples": saved.get("N_samples", 200),
                "apply_conformal": saved.get("apply_conformal", True),
            })

            # emit forecast request (will not remove existing forecasts on chart)
            self.forecastRequested.emit(payload)

            # visual feedback (optional): mark testing point with a vertical line
            try:
                line = self.ax.axvline(pd.to_datetime(testing_ts, unit="ms", utc=True), color="gray", linestyle=":", alpha=0.6)
                # remove marker after brief time (keeps chart clean); store temporarily
                self.canvas.draw()
                # keep marker as part of forecasts so it can be cleared with clear_all_forecasts
                self._forecasts.append({"id": time.time(), "created_at": time.time(), "quantiles": {}, "future_ts": None, "artists": [line], "source": "testing_marker"})
                self._trim_forecasts()
            except Exception:
                pass

        except Exception as e:
            logger.exception(f"Error in canvas click handler: {e}")

    def _on_forecast_clicked(self):
        from .prediction_settings_dialog import PredictionSettingsDialog
        settings = PredictionSettingsDialog.get_settings()
        if not settings.get("model_path"):
            QMessageBox.warning(self, "Missing Model", "Please select a model file.")
            return

        payload = {"symbol": self.symbol, "timeframe": self.timeframe, **settings}
        self.forecastRequested.emit(payload)

    def _open_adv_forecast_settings(self):
        # Reuse same dialog which now contains advanced options
        from .prediction_settings_dialog import PredictionSettingsDialog
        dialog = PredictionSettingsDialog(self)
        dialog.exec()

    def _on_advanced_forecast_clicked(self):
        # advanced forecast: use same settings but tag source
        from .prediction_settings_dialog import PredictionSettingsDialog
        settings = PredictionSettingsDialog.get_settings()
        if not settings.get("model_path"):
            QMessageBox.warning(self, "Missing Model", "Please select a model file.")
            return
        payload = {"symbol": self.symbol, "timeframe": self.timeframe, "advanced": True, **settings}
        self.forecastRequested.emit(payload)

    def on_forecast_ready(self, df: pd.DataFrame, quantiles: dict):
        """
        Slot to receive forecast results from controller/worker.
        Adds the forecast overlay without removing existing ones (trimming oldest if needed).
        """
        try:
            # Plot overlay and keep previous views
            self.update_plot(self._last_df, quantiles=None)  # redraw base chart preserving zoom
            # Plot forecast overlay, source label if present
            source = quantiles.get("source", "basic") if isinstance(quantiles, dict) else "basic"
            self._plot_forecast_overlay(quantiles, source=source)
        except Exception as e:
            logger.exception(f"Error handling forecast result: {e}")

    def clear_all_forecasts(self):
        """Remove all forecast artists from axes and clear internal list."""
        try:
            for f in self._forecasts:
                for art in f.get("artists", []):
                    try:
                        art.remove()
                    except Exception:
                        pass
            self._forecasts = []
            self.canvas.draw()
        except Exception as e:
            logger.exception(f"Failed to clear forecasts: {e}")

    def _trim_forecasts(self):
        """Enforce max_forecasts by removing oldest forecast artists."""
        try:
            maxf = int(get_setting("max_forecasts", self.max_forecasts))
            # enforce current runtime limit as well
            maxf = max(1, maxf)
            while len(self._forecasts) > maxf:
                old = self._forecasts.pop(0)
                for art in old.get("artists", []):
                    try:
                        art.remove()
                    except Exception:
                        pass
        except Exception as e:
            logger.exception(f"Failed trimming forecasts: {e}")

    def start_auto_forecast(self):
        if not self._auto_timer.isActive():
            self._auto_timer.start()
            logger.info("Auto-forecast started")

    def stop_auto_forecast(self):
        if self._auto_timer.isActive():
            self._auto_timer.stop()
            logger.info("Auto-forecast stopped")

    def _auto_forecast_tick(self):
        """Called by timer: trigger both basic and advanced forecasts (emit signals)."""
        try:
            # Use saved settings to build payloads; emit two requests: basic and advanced
            from .prediction_settings_dialog import PredictionSettingsDialog
            settings = PredictionSettingsDialog.get_settings() or {}
            # Basic
            payload_basic = {"symbol": self.symbol, "timeframe": self.timeframe, **settings}
            self.forecastRequested.emit(payload_basic)
            # Advanced
            payload_adv = {"symbol": self.symbol, "timeframe": self.timeframe, "advanced": True, **settings}
            self.forecastRequested.emit(payload_adv)
        except Exception as e:
            logger.exception(f"Auto forecast tick failed: {e}")

    def _on_backfill_missing_clicked(self):
        """Trigger backfill for current symbol/timeframe asynchronously with determinate progress."""
        controller = getattr(self._main_window, "controller", None)
        ms = getattr(controller, "market_service", None) if controller else None
        if ms is None:
            QMessageBox.warning(self, "Backfill", "MarketDataService non disponibile.")
            return
        sym = getattr(self, "symbol", None)
        tf = getattr(self, "timeframe", None)
        if not sym or not tf:
            QMessageBox.information(self, "Backfill", "Imposta prima symbol e timeframe.")
            return

        from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

        class BackfillSignals(QObject):
            progress = Signal(int)
            finished = Signal(bool)

        class BackfillJob(QRunnable):
            def __init__(self, svc, symbol, timeframe, signals):
                super().__init__()
                self.svc = svc
                self.symbol = symbol
                self.timeframe = timeframe
                self.signals = signals

            def run(self):
                ok = True
                try:
                    def _cb(pct: int):
                        try:
                            self.signals.progress.emit(int(pct))
                        except Exception:
                            pass
                    self.svc.backfill_symbol_timeframe(self.symbol, self.timeframe, force_full=False, progress_cb=_cb)
                except Exception as e:
                    ok = False
                finally:
                    try:
                        self.signals.finished.emit(ok)
                    except Exception:
                        pass

        self.setDisabled(True)
        self.backfill_progress.setRange(0, 100)
        self.backfill_progress.setValue(0)

        self._bf_signals = BackfillSignals(self)
        self._bf_signals.progress.connect(self.backfill_progress.setValue)

        def _on_done(ok: bool):
            try:
                # reload candles from DB
                df = self._load_candles_from_db(sym, tf, limit=3000)
                if df is not None and not df.empty:
                    self.update_plot(df)
                if ok:
                    QMessageBox.information(self, "Backfill", f"Backfill completato per {sym} {tf}.")
                else:
                    QMessageBox.warning(self, "Backfill", "Backfill fallito (vedi log).")
            finally:
                self.setDisabled(False)
                self.backfill_progress.setValue(100)

        self._bf_signals.finished.connect(_on_done)
        job = BackfillJob(ms, sym, tf, self._bf_signals)
        QThreadPool.globalInstance().start(job)

    def _load_candles_from_db(self, symbol: str, timeframe: str, limit: int = 5000):
        """Load candles from DB to refresh chart."""
        try:
            controller = getattr(self._main_window, "controller", None)
            eng = getattr(getattr(controller, "market_service", None), "engine", None) if controller else None
            if eng is None:
                return pd.DataFrame()
            from sqlalchemy import MetaData, select
            meta = MetaData()
            meta.reflect(bind=eng, only=["market_data_candles"])
            tbl = meta.tables.get("market_data_candles")
            if tbl is None:
                return pd.DataFrame()
            with eng.connect() as conn:
                stmt = select(tbl.c.ts_utc, tbl.c.open, tbl.c.high, tbl.c.low, tbl.c.close, tbl.c.volume)\
                    .where(tbl.c.symbol == symbol).where(tbl.c.timeframe == timeframe)\
                    .order_by(tbl.c.ts_utc.asc()).limit(limit)
                rows = conn.execute(stmt).fetchall()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=["ts_utc","open","high","low","close","volume"])
                return df
        except Exception as e:
            logger.exception("Load candles failed: {}", e)
            return pd.DataFrame()

