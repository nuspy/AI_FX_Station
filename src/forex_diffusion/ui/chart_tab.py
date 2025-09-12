# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional, Dict
import pandas as pd
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import QTimer, Qt, Signal, QPoint
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sqlalchemy import MetaData, select
from loguru import logger

from ..features.indicators import sma, ema, bollinger, rsi, macd
from ..utils.user_settings import get_setting, set_setting


class ChartTab(QWidget):
    """
    Matplotlib-based chart tab with pan/zoom toolbar and forecast overlay capabilities.
    """
    forecastRequested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.toolbar = NavigationToolbar(self.canvas, self)

        topbar = QWidget()
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.toolbar)

        self.indicators_btn = QPushButton("Indicatori")
        top_layout.addWidget(self.indicators_btn)

        self.forecast_settings_btn = QPushButton("Setting previsione")
        self.forecast_settings_btn.clicked.connect(self._open_forecast_settings)
        top_layout.addWidget(self.forecast_settings_btn)

        self.forecast_btn = QPushButton("Fai previsione")
        self.forecast_btn.clicked.connect(self._on_forecast_clicked)
        top_layout.addWidget(self.forecast_btn)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        top_layout.addWidget(self.reset_view_btn)

        self.bidask_label = QLabel("Bid: -    Ask: -")
        self.bidask_label.setStyleSheet("font-weight: bold;")
        top_layout.addWidget(self.bidask_label)
        self.layout.addWidget(topbar)

        self.indicators_btn.clicked.connect(self._open_indicators_dialog)
        self.layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.subplots()
        self.ax.set_title("Historical Chart")
        self._last_df = None
        self._indicator_artists: dict[str, list] = {}
        self._forecast_artists: list = []
        self._aux_axes: dict[str, any] = {}

        self._connect_mpl_events()

    def _connect_mpl_events(self):
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)

    def _handle_tick(self, payload: dict):
        """Handles incoming tick data to update the chart."""
        try:
            if not isinstance(payload, dict):
                return

            # Update bid/ask label
            bid = payload.get('bid')
            ask = payload.get('ask')
            if bid and ask:
                self.bidask_label.setText(f"Bid: {bid:.5f}    Ask: {ask:.5f}")

            # Append tick to internal dataframe
            if self._last_df is not None:
                new_row = pd.DataFrame([payload])
                self._last_df = pd.concat([self._last_df, new_row], ignore_index=True)
                self._last_df.drop_duplicates(subset=['ts_utc'], keep='last', inplace=True)
                self.update_plot(self._last_df)

        except Exception as e:
            logger.exception(f"Failed to handle tick: {e}")

    def update_plot(self, df: pd.DataFrame, timeframe: Optional[str] = None, quantiles: Optional[dict] = None):
        try:
            if df is None or df.empty:
                self.ax.clear()
                self.ax.set_title("No data")
                self.canvas.draw_idle()
                return

            self._last_df = df
            self.ax.clear()

            x_dt = pd.to_datetime(df["ts_utc"].astype("int64"), unit="ms")
            x_num = mdates.date2num(x_dt)
            y = df["close"].astype(float)

            self.ax.plot(x_num, y, "-", color="black", linewidth=1.0, label="close")
            self.ax.set_title(f"Historical Close - {getattr(self, 'symbol', '')} ({getattr(self, 'timeframe', '')})")
            self.ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
            self.ax.figure.autofmt_xdate()

            if hasattr(self, '_indicator_cfg') and self._indicator_cfg:
                self._apply_indicators(self._indicator_cfg)

            if quantiles:
                self._plot_forecast_overlay(quantiles)

            self.ax.legend(loc="upper left", fontsize="small")
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()

        except Exception as e:
            logger.exception(f"Failed to update plot: {e}")

    def _plot_forecast_overlay(self, quantiles: dict):
        for artist in self._forecast_artists:
            try: artist.remove()
            except Exception: pass
        self._forecast_artists = []

        if not quantiles or self._last_df is None or self._last_df.empty:
            return

        try:
            q05, q50, q95 = quantiles.get("q05"), quantiles.get("q50"), quantiles.get("q95")
            if not all([q05, q50, q95]): return

            last_ts = pd.to_datetime(self._last_df["ts_utc"].iloc[-1], unit="ms")
            delta = pd.to_timedelta(self.timeframe)
            future_ts = [last_ts + delta * (i + 1) for i in range(len(q50))]
            x_num = mdates.date2num(future_ts)

            gray_color = '#AAAAAA'  # Approx 66.6% gray
            fill = self.ax.fill_between(x_num, q05, q95, color=gray_color, alpha=0.33, label='Forecast CI')
            line, = self.ax.plot(x_num, q50, color=gray_color, linestyle='--', linewidth=1.5, label='Forecast Median')
            self._forecast_artists.extend([fill, line])

        except Exception as e:
            logger.exception(f"Failed to plot forecast overlay: {e}")

    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        self.db_service = db_service
        self.symbol = symbol
        self.timeframe = timeframe

    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == "up" else base_scale
        cur_xlim = self.ax.get_xlim()
        xdata = event.xdata
        self.ax.set_xlim([xdata - (xdata - cur_xlim[0]) * scale_factor, xdata + (cur_xlim[1] - xdata) * scale_factor])
        self.canvas.draw_idle()

    def _on_button_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self._pan_start = (event.x, self.ax.get_xlim())
            self.canvas.setCursor(Qt.ClosedHandCursor)

    def _on_motion(self, event):
        if self._pan_start is None or event.inaxes != self.ax: return
        xpress, (x0, x1) = self._pan_start
        dx = event.x - xpress
        self.ax.set_xlim(x0 - dx, x1 - dx)
        self.canvas.draw_idle()

    def _on_button_release(self, event):
        if event.button == 1:
            self._pan_start = None
            self.canvas.setCursor(Qt.ArrowCursor)

    def reset_view(self):
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _open_forecast_settings(self):
        from .prediction_settings_dialog import PredictionSettingsDialog
        dialog = PredictionSettingsDialog(self)
        dialog.exec()

    def _on_forecast_clicked(self):
        from .prediction_settings_dialog import PredictionSettingsDialog
        settings = PredictionSettingsDialog.get_settings()
        if not settings.get("model_path"):
            QMessageBox.warning(self, "Missing Model", "Please select a model file in Prediction Settings.")
            self._open_forecast_settings()
            return
        
        payload = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            **settings
        }
        self.forecastRequested.emit(payload)

    def _open_indicators_dialog(self):
        # Placeholder for indicator dialog logic
        pass

    def _apply_indicators(self, cfg):
        # Placeholder for applying indicators
        pass
