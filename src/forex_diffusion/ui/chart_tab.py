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
        self.clear_forecasts_btn.clicked.connect(self.clear_all_forecasts)

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

            if q50 is None:
                return

            # x positions
            if future_ts:
                x_vals = pd.to_datetime(future_ts)
            else:
                # build future timestamps from last available ts and timeframe
                if self._last_df is None or self._last_df.empty:
                    return
                last_ts = pd.to_datetime(self._last_df["ts_utc"].astype("int64"), unit="ms").iat[-1]
                td = self._tf_to_timedelta(getattr(self, "timeframe", "1m"))
                x_vals = [last_ts + td * (i + 1) for i in range(len(q50))]

            # prepare colors
            color = "tab:orange" if source == "basic" else "tab:purple"
            line50, = self.ax.plot(x_vals, q50, color=color, linestyle='-', label=f"Forecast q50 ({source})")
            artists = [line50]

            if q05 is not None and q95 is not None:
                line05, = self.ax.plot(x_vals, q05, color=color, linestyle='--', alpha=0.8, label=None)
                line95, = self.ax.plot(x_vals, q95, color=color, linestyle='--', alpha=0.8, label=None)
                fill = self.ax.fill_between(x_vals, q05, q95, color=color, alpha=0.12)
                artists.extend([line05, line95, fill])

            # store forecast
            fid = time.time()
            forecast = {
                "id": fid,
                "created_at": fid,
                "quantiles": quantiles,
                "future_ts": future_ts,
                "artists": artists,
                "source": source
            }
            self._forecasts.append(forecast)

            # trim old forecasts if needed
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

