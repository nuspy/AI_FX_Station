# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional
import pandas as pd

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

class ChartTab(QWidget):
    """
    Simple matplotlib-based chart tab with pan/zoom toolbar.
    Exposes update_plot(df) where df is a DataFrame with ts_utc and close/open/high/low.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(6,4)))
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()
        self.ax.set_title("Historical chart")
        self._last_df = None

    def update_plot(self, df: pd.DataFrame, timeframe: Optional[str] = None):
        try:
            self.ax.clear()
            if df is None or df.empty:
                self.ax.set_title("No data")
                self.canvas.draw()
                return
            # assume df has ts_utc and close
            x = pd.to_datetime(df["ts_utc"].astype("int64"), unit="ms", utc=True)
            y = df["close"].astype(float)
            self.ax.plot(x, y, "-")
            self.ax.set_title("Historical close")
            self.ax.figure.autofmt_xdate()
            self.canvas.draw()
            self._last_df = df
        except Exception as e:
            try:
                self.ax.clear()
                self.ax.set_title(f"Plot error: {e}")
                self.canvas.draw()
            except Exception:
                pass
