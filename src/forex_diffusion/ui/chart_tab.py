# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional
import pandas as pd

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

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
        # indicators button
        try:
            from .indicators_dialog import IndicatorsDialog
            self.indicators_btn = QPushButton("Indicatori")
            self.indicators_btn.clicked.connect(self._open_indicators_dialog)
            # place toolbar and button in a horizontal layout
            topbar = QWidget()
            top_layout = QHBoxLayout(topbar)
            top_layout.setContentsMargins(0, 0, 0, 0)
            top_layout.addWidget(self.toolbar)
            top_layout.addWidget(self.indicators_btn)
            self.layout.addWidget(topbar)
        except Exception:
            # fallback: just add toolbar
            self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()
        self.ax.set_title("Historical chart")
        self._last_df = None

        # Interaction state for panning
        self._is_panning = False
        self._pan_start = None  # (xpress, ypress)
        # connect mouse events
        try:
            self.canvas.mpl_connect("scroll_event", self._on_scroll)
            self.canvas.mpl_connect("button_press_event", self._on_button_press)
            self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self.canvas.mpl_connect("button_release_event", self._on_button_release)
        except Exception:
            pass

    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        self.db_service = db_service
        self.symbol = symbol
        self.timeframe = timeframe

    # --- Mouse interaction handlers (wheel zoom on x axis, left-drag pan)
    def _on_scroll(self, event):
        try:
            if event.inaxes != self.ax:
                return
            # event.xdata is in matplotlib date floating point (if plotted with dates)
            import matplotlib.dates as mdates
            xdata = event.xdata
            if xdata is None:
                return
            cur_xlim = self.ax.get_xlim()
            # scale factor: zoom in/out by 20% per scroll step
            base_scale = 1.2
            if event.button == "up":
                scale_factor = 1 / base_scale
            elif event.button == "down":
                scale_factor = base_scale
            else:
                scale_factor = 1.0
            left = xdata - (xdata - cur_xlim[0]) * scale_factor
            right = xdata + (cur_xlim[1] - xdata) * scale_factor
            self.ax.set_xlim(left, right)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_button_press(self, event):
        try:
            if event.inaxes != self.ax:
                return
            # left button = pan
            if event.button == 1:
                self._is_panning = True
                self._pan_start = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
        except Exception:
            pass

    def _on_motion(self, event):
        try:
            if not self._is_panning or self._pan_start is None:
                return
            if event.inaxes != self.ax:
                return
            xpress, ypress, (x0, x1), (y0, y1) = self._pan_start
            dx = event.x - xpress
            import matplotlib.transforms as mtrans
            # compute shift in data coordinates using axis transform
            inv = self.ax.transData.inverted()
            p1 = inv.transform((xpress, 0))
            p2 = inv.transform((event.x, 0))
            dx_data = p2[0] - p1[0]
            self.ax.set_xlim(x0 - dx_data, x1 - dx_data)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_button_release(self, event):
        try:
            if event.button == 1:
                self._is_panning = False
                self._pan_start = None
        except Exception:
            pass

    def _open_indicators_dialog(self):
        try:
            from .indicators_dialog import IndicatorsDialog
            dlg = IndicatorsDialog(parent=self)
            if dlg.exec():
                cfg = dlg.result()
                self._apply_indicators(cfg)
        except Exception:
            pass

    def _apply_indicators(self, cfg: dict):
        """
        Compute indicators from current _last_df and overlay on axes.
        cfg example: {'sma': {'window':20}, 'rsi': {'period':14}}
        """
        if self._last_df is None or self._last_df.empty:
            return
        try:
            import pandas as pd
            from ..features.indicators import sma, ema, bollinger, rsi, macd
            df = self._last_df.copy()
            # ensure sorted ascending and close column present
            if "ts_utc" not in df.columns or "close" not in df.columns:
                return
            df = df.sort_values("ts_utc").reset_index(drop=True)
            x = pd.to_datetime(df["ts_utc"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
            y = df["close"].astype(float)
            # replot base
            try:
                self.ax.lines.clear()
            except Exception:
                pass
            self.ax.plot(x, y, color="black", linewidth=1.0, label="close")
            # compute/plot configured indicators
            if "sma" in cfg:
                w = int(cfg["sma"].get("window", 20))
                s = sma(df["close"], w)
                self.ax.plot(x, s, label=f"SMA({w})", alpha=0.9)
            if "ema" in cfg:
                sp = int(cfg["ema"].get("span", 20))
                e = ema(df["close"], sp)
                self.ax.plot(x, e, label=f"EMA({sp})", alpha=0.9)
            if "bollinger" in cfg:
                w = int(cfg["bollinger"].get("window", 20))
                nstd = float(cfg["bollinger"].get("n_std", 2.0))
                up, low = bollinger(df["close"], window=w, n_std=nstd)
                self.ax.plot(x, up, label=f"BB_up({w},{nstd})", color="green", alpha=0.6)
                self.ax.plot(x, low, label=f"BB_low({w},{nstd})", color="red", alpha=0.6)
            if "rsi" in cfg:
                p = int(cfg["rsi"].get("period", 14))
                r = rsi(df["close"], period=p)
                # plot RSI on secondary axis
                ax2 = self.ax.twinx()
                ax2.plot(x, r, label=f"RSI({p})", color="purple", alpha=0.8)
                ax2.set_ylabel("RSI")
            if "macd" in cfg:
                f = int(cfg["macd"].get("fast", 12))
                s = int(cfg["macd"].get("slow", 26))
                sg = int(cfg["macd"].get("signal", 9))
                m = macd(df["close"], fast=f, slow=s, signal=sg)
                ax2 = self.ax.twinx()
                ax2.plot(x, m["macd"], label=f"MACD({f},{s})", color="orange", alpha=0.9)
                ax2.plot(x, m["signal"], label=f"Signal({sg})", color="blue", alpha=0.6)
            try:
                self.ax.legend(loc="upper left", fontsize="small")
            except Exception:
                pass
            self.canvas.draw()
        except Exception:
            pass

    def _on_xlim_changed(self, ax):
        try:
            x0, x1 = ax.get_xlim()
            import matplotlib.dates as mdates
            # convert x0,x1 (float days) to timestamps in ms
            t0 = int(mdates.num2epoch(x0) * 1000)
            t1 = int(mdates.num2epoch(x1) * 1000)
            # expand by 50%
            span = t1 - t0
            ext0 = max(0, t0 - span // 2)
            ext1 = t1 + span // 2
            # fetch from DB
            if getattr(self, "db_service", None) is None or not hasattr(self, "symbol"):
                return
            meta = MetaData()
            meta.reflect(bind=self.db_service.engine, only=["market_data_candles"])
            tbl = meta.tables.get("market_data_candles")
            if tbl is None:
                return
            with self.db_service.engine.connect() as conn:
                stmt = select(tbl).where(tbl.c.symbol == self.symbol).where(tbl.c.timeframe == self.timeframe).where(tbl.c.ts_utc >= ext0).where(tbl.c.ts_utc <= ext1).order_by(tbl.c.ts_utc.asc())
                rows = conn.execute(stmt).fetchall()
                import pandas as pd
                if rows:
                    df = pd.DataFrame([dict(r) for r in rows])
                    self.update_plot(df, timeframe=self.timeframe)
        except Exception:
            pass

    def update_plot(self, df: pd.DataFrame, timeframe: Optional[str] = None):
        try:
            self.ax.clear()
            if df is None or df.empty:
                self.ax.set_title("No data")
                self.canvas.draw()
                return
            # ensure ts_utc column exists and convert to datetimes (local tz)
            try:
                x = pd.to_datetime(df["ts_utc"].astype("int64"), unit="ms", utc=True)
                # convert to local timezone naive datetimes for matplotlib
                x = x.dt.tz_convert(None)
            except Exception:
                # fallback: if ts_utc already datetime-like
                x = pd.to_datetime(df.get("ts_utc", df.index))
            y = df["close"].astype(float)
            self.ax.plot(x, y, "-")
            self.ax.set_title("Historical close")
            # format X axis with date+time
            try:
                self.ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
                self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            except Exception:
                pass
            self.ax.figure.autofmt_xdate()
            self.canvas.draw()
            self._last_df = df
            # connect xlim_changed handler once
            try:
                if not getattr(self, "_xlim_connected", False):
                    self.canvas.mpl_connect("xlim_changed", self._on_xlim_changed)
                    self._xlim_connected = True
            except Exception:
                pass
        except Exception as e:
            try:
                self.ax.clear()
                self.ax.set_title(f"Plot error: {e}")
                self.canvas.draw()
            except Exception:
                pass
