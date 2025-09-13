# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional, Dict
import pandas as pd
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox, 
    QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QDoubleSpinBox
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

        top_layout.addStretch()
        self.bidask_label = QLabel("Bid: -    Ask: -")
        self.bidask_label.setStyleSheet("font-weight: bold;")
        top_layout.addWidget(self.bidask_label)
        self.layout.addWidget(topbar)
        self.layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.subplots()
        self._last_df = pd.DataFrame()
        self._forecast_artists = []

        # --- Signal Connections ---
        self.forecast_settings_btn.clicked.connect(self._open_forecast_settings)
        self.forecast_btn.clicked.connect(self._on_forecast_clicked)
        self.adv_settings_btn.clicked.connect(self._open_adv_forecast_settings)
        self.adv_forecast_btn.clicked.connect(self._on_advanced_forecast_clicked)

    def _handle_tick(self, payload: dict):
        """Receives tick data, updates the internal DataFrame, and redraws the chart."""
        try:
            if not isinstance(payload, dict):
                return

            new_row = pd.DataFrame([payload])
            self._last_df = pd.concat([self._last_df, new_row], ignore_index=True)
            self._last_df.drop_duplicates(subset=['ts_utc'], keep='last', inplace=True)
            
            # Crucial fix: Call update_plot to redraw the chart with the new data
            self.update_plot(self._last_df)

            if payload.get('bid') and payload.get('ask'):
                self.bidask_label.setText(f"Bid: {payload['bid']:.5f}    Ask: {payload['ask']:.5f}")

        except Exception as e:
            logger.exception(f"Failed to handle tick: {e}")

    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None):
        if df is None or df.empty:
            return

        self._last_df = df.copy()
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
        self.canvas.draw()

    def _plot_forecast_overlay(self, quantiles: dict):
        # Placeholder for forecast plotting logic
        pass

    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        self.db_service = db_service
        self.symbol = symbol
        self.timeframe = timeframe

    def _open_forecast_settings(self):
        from .prediction_settings_dialog import PredictionSettingsDialog
        dialog = PredictionSettingsDialog(self)
        dialog.exec()

    def _on_forecast_clicked(self):
        from .prediction_settings_dialog import PredictionSettingsDialog
        settings = PredictionSettingsDialog.get_settings()
        if not settings.get("model_path"):
            QMessageBox.warning(self, "Missing Model", "Please select a model file.")
            return
        
        payload = {"symbol": self.symbol, "timeframe": self.timeframe, **settings}
        self.forecastRequested.emit(payload)

    def _open_adv_forecast_settings(self):
        # Re-implementing the advanced settings dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced Forecast Settings")
        layout = QFormLayout(dialog)

        rsi_period_spin = QSpinBox()
        rsi_period_spin.setRange(2, 100)
        rsi_period_spin.setValue(get_setting("adv_rsi_period", 14))
        layout.addRow("RSI Period:", rsi_period_spin)

        bb_window_spin = QSpinBox()
        bb_window_spin.setRange(5, 200)
        bb_window_spin.setValue(get_setting("adv_bb_window", 20))
        layout.addRow("Bollinger Bands Window:", bb_window_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec():
            set_setting("adv_rsi_period", rsi_period_spin.value())
            set_setting("adv_bb_window", bb_window_spin.value())
            logger.info("Advanced forecast settings saved.")

    def _on_advanced_forecast_clicked(self):
        QMessageBox.information(self, "Not Implemented", "Advanced forecast logic to be connected here.")

