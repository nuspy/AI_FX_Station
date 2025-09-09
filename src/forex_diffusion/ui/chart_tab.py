# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional
import pandas as pd

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sqlalchemy import MetaData, select
from loguru import logger

from ..features.indicators import sma, ema, bollinger, rsi, macd


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
        # indicators button - always create the button (visible), enable only if dialog available
        self.indicators_btn = QPushButton("Indicatori")
        self.indicators_btn.setToolTip("Apri la finestra degli indicatori")
        # place toolbar and button in a horizontal layout; always add topbar so button is visible
        topbar = QWidget()
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.toolbar)
        top_layout.addWidget(self.indicators_btn)
        # real-time bid/ask label (initial placeholder)
        try:
            self.bidask_label = QLabel("Bid: -    Ask: -")
            self.bidask_label.setMinimumWidth(220)
            self.bidask_label.setStyleSheet("font-weight: bold;")
            top_layout.addWidget(self.bidask_label)
        except Exception:
            # fallback: ignore if QLabel not available
            self.bidask_label = None
        self.layout.addWidget(topbar)

        # Always connect the indicators button; if packaged dialog is missing, we'll show a dynamic fallback
        try:
            self.indicators_btn.clicked.connect(self._open_indicators_dialog)
            self.indicators_btn.setEnabled(True)
        except Exception:
            # best-effort: keep button visible but it will be inert (very unlikely)
            try:
                self.indicators_btn.setEnabled(False)
                self.indicators_btn.setToolTip("Cannot connect indicators handler")
            except Exception:
                pass
        self.layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()
        self.ax.set_title("Historical chart")
        self._last_df = None
        # track plotted artists per indicator name so we can remove/replace them instead of duplicating
        self._indicator_artists: dict[str, list] = {}

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

        # subscribe to in-process tick events for live updates
        try:
            from ..utils.event_bus import subscribe
            subscribe("tick", self._on_tick_event)
            logger.info("ChartTab subscribed to 'tick' events")
        except Exception as e:
            logger.exception("ChartTab failed to subscribe to 'tick' events: {}", e)

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
            # pan while left button pressed
            if not self._is_panning or self._pan_start is None:
                return
            if event.inaxes != self.ax:
                return
            xpress, ypress, (x0, x1), (y0, y1) = self._pan_start
            dx = event.x - xpress
            # compute shift in data coordinates using axis transform
            inv = self.ax.transData.inverted()
            p1 = inv.transform((xpress, 0))
            p2 = inv.transform((event.x, 0))
            dx_data = p2[0] - p1[0]
            self.ax.set_xlim(x0 - dx_data, x1 - dx_data)
            self.canvas.draw_idle()
        except Exception as e:
            logger.debug("ChartTab _on_motion exception: {}", e)

    def _on_button_release(self, event):
        try:
            if event.button == 1:
                self._is_panning = False
                self._pan_start = None
        except Exception:
            pass

    def _tick_noop(self, payload):
        """Fallback noop tick handler (UI-thread); logs payload for diagnostics."""
        try:
            logger.debug("Tick received but handler not ready: {}", payload)
            # attempt to update bid/ask label if payload contains simple fields
            try:
                bid = payload.get("bid") if isinstance(payload, dict) else None
                ask = payload.get("ask") if isinstance(payload, dict) else None
                if getattr(self, "bidask_label", None) is not None and bid is not None and ask is not None:
                    try:
                        self.bidask_label.setText(f"Bid: {float(bid):.5f}    Ask: {float(ask):.5f}")
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _on_tick_event(self, payload):
        """Schedule handling of incoming tick on UI thread."""
        try:
            QTimer.singleShot(0, lambda p=payload: self._handle_tick(p))
        except Exception as e:
            logger.debug("Failed to schedule tick handling: {}", e)

    def _handle_tick(self, payload):
        """
        UI-thread: accept payload (dict-like) and append to last_df, update bid/ask label and redraw.
        """
        try:
            # Normalize payload to dict
            if not isinstance(payload, dict):
                try:
                    payload = dict(payload)
                except Exception:
                    payload = {"price": payload}
            sym = payload.get("symbol")
            tf = payload.get("timeframe", "1m")
            if getattr(self, "symbol", None) is None or getattr(self, "timeframe", None) is None:
                # not configured yet
                return
            if sym != self.symbol or tf != self.timeframe:
                # different symbol/timeframe, ignore for display (DB still persists)
                return
            ts = int(payload.get("ts_utc", int(pd.Timestamp.utcnow().value // 1_000_000)))
            price = float(payload.get("price", 0.0))
            bid = payload.get("bid", None)
            ask = payload.get("ask", None)

            row = {"ts_utc": int(ts), "open": price, "high": price, "low": price, "close": price, "volume": None}
            import pandas as pd
            if getattr(self, "_last_df", None) is None or self._last_df.empty:
                self._last_df = pd.DataFrame([row])
            else:
                df_append = pd.DataFrame([row])
                df_combined = pd.concat([self._last_df, df_append], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=["ts_utc"], keep="last").sort_values("ts_utc").reset_index(drop=True)
                self._last_df = df_combined

            # update bid/ask label
            try:
                if getattr(self, "bidask_label", None) is not None:
                    bv = bid if bid is not None else price
                    av = ask if ask is not None else price
                    self.bidask_label.setText(f"Bid: {float(bv):.5f}    Ask: {float(av):.5f}")
            except Exception:
                pass

            # redraw
            try:
                self.update_plot(self._last_df, timeframe=self.timeframe)
            except Exception as e:
                logger.debug("Failed to update plot after tick: {}", e)
        except Exception as e:
            logger.exception("Error handling tick payload: {}", e)

    def _open_indicators_dialog(self):
        """
        Open indicators dialog (packaged if available) or fallback expanded dialog.
        Fallback includes SMA, EMA, Bollinger, RSI, MACD with color selectors.
        """
        cfg = None
        try:
            from .indicators_dialog import IndicatorsDialog  # type: ignore
            dlg = IndicatorsDialog(parent=self, initial=self._indicator_cfg if getattr(self, "_indicator_cfg", None) else None)
            if not dlg.exec():
                return
            cfg = dlg.result()
        except Exception as e:
            logger.debug("Packaged IndicatorsDialog not available or failed: {}", e)
            # build richer fallback dialog inline (SMA, EMA, Bollinger, RSI, MACD)
            try:
                from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox, QCheckBox, QLabel, QSpinBox, QLineEdit, QComboBox
                class FallbackDialog(QDialog):
                    def __init__(self, parent=None, initial=None):
                        super().__init__(parent)
                        self.setWindowTitle("Indicatori (fallback)")
                        self.layout = QVBoxLayout(self)
                        palette = [("blue","#1f77b4"),("red","#d62728"),("green","#2ca02c"),("orange","#ff7f0e"),("purple","#9467bd"),("black","#000000")]

                        # SMA
                        row = QHBoxLayout()
                        self.sma_cb = QCheckBox("SMA")
                        self.sma_w = QSpinBox(); self.sma_w.setRange(1,500); self.sma_w.setValue(20)
                        self.sma_col = QComboBox()
                        for n,h in palette: self.sma_col.addItem(f"{n} ({h})", h)
                        row.addWidget(self.sma_cb); row.addWidget(QLabel("w:")); row.addWidget(self.sma_w); row.addWidget(QLabel("color:")); row.addWidget(self.sma_col)
                        self.layout.addLayout(row)

                        # EMA
                        row = QHBoxLayout()
                        self.ema_cb = QCheckBox("EMA")
                        self.ema_span = QSpinBox(); self.ema_span.setRange(1,500); self.ema_span.setValue(20)
                        self.ema_col = QComboBox()
                        for n,h in palette: self.ema_col.addItem(f"{n} ({h})", h)
                        row.addWidget(self.ema_cb); row.addWidget(QLabel("span:")); row.addWidget(self.ema_span); row.addWidget(QLabel("color:")); row.addWidget(self.ema_col)
                        self.layout.addLayout(row)

                        # Bollinger
                        row = QHBoxLayout()
                        self.bb_cb = QCheckBox("Bollinger")
                        self.bb_n = QSpinBox(); self.bb_n.setRange(1,500); self.bb_n.setValue(20)
                        self.bb_k = QLineEdit("2.0")
                        self.bb_col_u = QComboBox(); self.bb_col_l = QComboBox()
                        for n,h in palette:
                            self.bb_col_u.addItem(f"{n} ({h})", h)
                            self.bb_col_l.addItem(f"{n} ({h})", h)
                        row.addWidget(self.bb_cb); row.addWidget(QLabel("n:")); row.addWidget(self.bb_n); row.addWidget(QLabel("k:")); row.addWidget(self.bb_k)
                        row.addWidget(QLabel("up col:")); row.addWidget(self.bb_col_u); row.addWidget(QLabel("low col:")); row.addWidget(self.bb_col_l)
                        self.layout.addLayout(row)

                        # RSI
                        row = QHBoxLayout()
                        self.rsi_cb = QCheckBox("RSI")
                        self.rsi_p = QSpinBox(); self.rsi_p.setRange(1,500); self.rsi_p.setValue(14)
                        self.rsi_col = QComboBox()
                        for n,h in palette: self.rsi_col.addItem(f"{n} ({h})", h)
                        row.addWidget(self.rsi_cb); row.addWidget(QLabel("period:")); row.addWidget(self.rsi_p); row.addWidget(QLabel("color:")); row.addWidget(self.rsi_col)
                        self.layout.addLayout(row)

                        # MACD
                        row = QHBoxLayout()
                        self.macd_cb = QCheckBox("MACD")
                        self.macd_fast = QSpinBox(); self.macd_fast.setRange(1,200); self.macd_fast.setValue(12)
                        self.macd_slow = QSpinBox(); self.macd_slow.setRange(1,500); self.macd_slow.setValue(26)
                        self.macd_sig = QSpinBox(); self.macd_sig.setRange(1,200); self.macd_sig.setValue(9)
                        self.macd_col = QComboBox()
                        for n,h in palette: self.macd_col.addItem(f"{n} ({h})", h)
                        row.addWidget(self.macd_cb); row.addWidget(QLabel("fast:")); row.addWidget(self.macd_fast)
                        row.addWidget(QLabel("slow:")); row.addWidget(self.macd_slow); row.addWidget(QLabel("sig:")); row.addWidget(self.macd_sig)
                        row.addWidget(QLabel("color:")); row.addWidget(self.macd_col)
                        self.layout.addLayout(row)

                        # Buttons
                        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
                        self.layout.addWidget(bb)

                        # load initial if provided (best-effort)
                        if initial:
                            try:
                                if "sma" in initial:
                                    self.sma_cb.setChecked(True)
                                    if "window" in initial["sma"]:
                                        self.sma_w.setValue(int(initial["sma"]["window"]))
                                    if "color" in initial["sma"]:
                                        col = initial["sma"]["color"]
                                        for idx in range(self.sma_col.count()):
                                            if self.sma_col.itemData(idx) == col:
                                                self.sma_col.setCurrentIndex(idx); break
                                if "ema" in initial:
                                    self.ema_cb.setChecked(True)
                                    if "span" in initial["ema"]:
                                        self.ema_span.setValue(int(initial["ema"]["span"]))
                                    if "color" in initial["ema"]:
                                        col = initial["ema"]["color"]
                                        for idx in range(self.ema_col.count()):
                                            if self.ema_col.itemData(idx) == col:
                                                self.ema_col.setCurrentIndex(idx); break
                            except Exception:
                                pass

                    def result(self):
                        out = {}
                        if self.sma_cb.isChecked():
                            out["sma"] = {"window": int(self.sma_w.value()), "color": self.sma_col.currentData()}
                        if self.ema_cb.isChecked():
                            out["ema"] = {"span": int(self.ema_span.value()), "color": self.ema_col.currentData()}
                        if self.bb_cb.isChecked():
                            out["bollinger"] = {"window": int(self.bb_n.value()), "n_std": float(self.bb_k.text()), "color_up": self.bb_col_u.currentData(), "color_low": self.bb_col_l.currentData()}
                        if self.rsi_cb.isChecked():
                            out["rsi"] = {"period": int(self.rsi_p.value()), "color": self.rsi_col.currentData()}
                        if self.macd_cb.isChecked():
                            out["macd"] = {"fast": int(self.macd_fast.value()), "slow": int(self.macd_slow.value()), "signal": int(self.macd_sig.value()), "color": self.macd_col.currentData()}
                        return out

                dlg = FallbackDialog(parent=self, initial=self._indicator_cfg if getattr(self, "_indicator_cfg", None) else None)
                if not dlg.exec():
                    return
                cfg = dlg.result()
            except Exception as ee:
                logger.exception("Failed to build fallback Indicators dialog: {}", ee)
                try:
                    QMessageBox.warning(self, "Indicators error", f"Cannot open indicators dialog: {ee}")
                except Exception:
                    pass
                return

        # persist and apply
        try:
            self._indicator_cfg = cfg
            from pathlib import Path
            import json
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / "indicators.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg_path.open("w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
        except Exception as e:
            logger.exception("Failed to persist indicators config: {}", e)

        try:
            self._apply_indicators(cfg)
        except Exception as e:
            logger.exception("Failed to apply indicators after dialog: {}", e)
