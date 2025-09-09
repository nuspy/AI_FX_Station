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
        # auxiliary axes for indicators that require their own y-axis (rsi, macd)
        self._aux_axes: dict[str, any] = {}

        # load persisted indicators config if present
        try:
            from pathlib import Path
            import json
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / "indicators.json"
            if cfg_path.exists():
                try:
                    with cfg_path.open("r", encoding="utf-8") as fh:
                        self._indicator_cfg = json.load(fh)
                        logger.info("Loaded indicator config from %s", str(cfg_path))
                except Exception:
                    self._indicator_cfg = None
            else:
                self._indicator_cfg = None
        except Exception:
            self._indicator_cfg = None

        # load persisted indicators config if present
        try:
            from pathlib import Path
            import json
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / "indicators.json"
            if cfg_path.exists():
                try:
                    with cfg_path.open("r", encoding="utf-8") as fh:
                        self._indicator_cfg = json.load(fh)
                        logger.info("Loaded indicator config from %s", str(cfg_path))
                except Exception:
                    self._indicator_cfg = None
            else:
                self._indicator_cfg = None
        except Exception:
            self._indicator_cfg = None

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

    # Public convenience methods (aliases/wrappers) to avoid AttributeError when other modules call them
    def apply_indicators(self, cfg: dict):
        """
        Public wrapper to apply indicators (safe entry point).
        """
        try:
            self._indicator_cfg = cfg
            # call internal implementation
            try:
                self._apply_indicators(cfg)
            except Exception as e:
                logger.exception("apply_indicators internal failure: {}", e)
        except Exception as e:
            logger.exception("apply_indicators failed: {}", e)

    def redraw(self):
        """
        Public redraw helper: replot last_df if available.
        """
        if getattr(self, "_last_df", None) is not None:
            self.update_plot(self._last_df, timeframe=getattr(self, "timeframe", None))

    def _apply_indicators(self, cfg: dict):
        """
        Internal: remove previous indicator artists and overlay configured indicators.
        Does NOT redraw base 'close' line (update_plot is responsible for that).
        Reuses auxiliary axes for indicators requiring separate y-axis (rsi, macd).
        """
        try:
            if self._last_df is None or self._last_df.empty:
                return
            import pandas as pd
            df = self._last_df.copy()
            if "ts_utc" not in df.columns or "close" not in df.columns:
                return
            df = df.sort_values("ts_utc").reset_index(drop=True)
            x = pd.to_datetime(df["ts_utc"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)

            # normalize cfg: support legacy format where keys exist only if enabled
            try:
                norm_cfg = {}
                for name in ["sma", "ema", "bollinger", "rsi", "macd"]:
                    val = cfg.get(name)
                    if isinstance(val, dict):
                        # if 'enabled' present, keep; else treat presence as enabled
                        enabled = bool(val.get("enabled", True)) if "enabled" in val else True
                        # ensure keys exist with defaults
                        if name == "sma":
                            norm_cfg[name] = {
                                "enabled": enabled,
                                "window": int(val.get("window", 20)),
                                "color": val.get("color")
                            }
                        elif name == "ema":
                            norm_cfg[name] = {
                                "enabled": enabled,
                                "span": int(val.get("span", 20)),
                                "color": val.get("color")
                            }
                        elif name == "bollinger":
                            norm_cfg[name] = {
                                "enabled": enabled,
                                "window": int(val.get("window", 20)),
                                "n_std": float(val.get("n_std", val.get("k", 2.0))),
                                "color_up": val.get("color_up"),
                                "color_low": val.get("color_low")
                            }
                        elif name == "rsi":
                            norm_cfg[name] = {
                                "enabled": enabled,
                                "period": int(val.get("period", 14)),
                                "color": val.get("color")
                            }
                        elif name == "macd":
                            norm_cfg[name] = {
                                "enabled": enabled,
                                "fast": int(val.get("fast", 12)),
                                "slow": int(val.get("slow", 26)),
                                "signal": int(val.get("signal", 9)),
                                "color": val.get("color")
                            }
                    else:
                        # missing -> disabled
                        if name == "sma":
                            norm_cfg[name] = {"enabled": False, "window": 20, "color": None}
                        elif name == "ema":
                            norm_cfg[name] = {"enabled": False, "span": 20, "color": None}
                        elif name == "bollinger":
                            norm_cfg[name] = {"enabled": False, "window": 20, "n_std": 2.0, "color_up": None, "color_low": None}
                        elif name == "rsi":
                            norm_cfg[name] = {"enabled": False, "period": 14, "color": None}
                        elif name == "macd":
                            norm_cfg[name] = {"enabled": False, "fast": 12, "slow": 26, "signal": 9, "color": None}
            except Exception:
                norm_cfg = cfg

            # remove existing indicator artists and clear aux axes if present
            try:
                for name, artists in list(self._indicator_artists.items()):
                    for art in artists:
                        try:
                            art.remove()
                        except Exception:
                            pass
                self._indicator_artists = {}
                # clear aux axes lines by hiding them (do not destroy)
                for k, ax_aux in list(self._aux_axes.items()):
                    try:
                        ax_aux.cla()
                    except Exception:
                        pass
            except Exception:
                pass

            def _get_color(entry, default):
                try:
                    if isinstance(entry, dict):
                        return entry.get("color") or entry.get("color_up") or entry.get("color_low") or default
                    return default
                except Exception:
                    return default

            def _register(name: str, arts):
                try:
                    if name not in self._indicator_artists:
                        self._indicator_artists[name] = []
                    self._indicator_artists[name].extend(arts if isinstance(arts, list) else [arts])
                except Exception:
                    pass

            # SMA
            if "sma" in cfg:
                try:
                    w = int(cfg["sma"].get("window", 20))
                    col = _get_color(cfg.get("sma"), "#1f77b4")
                    s = sma(df["close"], w)
                    ln, = self.ax.plot(x, s, label=f"SMA({w})", color=col, alpha=0.9)
                    _register(f"sma_{w}_{col}", ln)
                except Exception:
                    pass

            # EMA
            if "ema" in cfg:
                try:
                    sp = int(cfg["ema"].get("span", 20))
                    col = _get_color(cfg.get("ema"), "#ff7f0e")
                    e = ema(df["close"], sp)
                    ln, = self.ax.plot(x, e, label=f"EMA({sp})", color=col, alpha=0.9)
                    _register(f"ema_{sp}_{col}", ln)
                except Exception:
                    pass

            # Bollinger
            if "bollinger" in cfg:
                try:
                    w = int(cfg["bollinger"].get("window", 20))
                    nstd = float(cfg["bollinger"].get("n_std", 2.0))
                    col_up = cfg["bollinger"].get("color_up") or _get_color(cfg.get("bollinger"), "#2ca02c")
                    col_low = cfg["bollinger"].get("color_low") or _get_color(cfg.get("bollinger"), "#d62728")
                    up, low = bollinger(df["close"], window=w, n_std=nstd)
                    l1, = self.ax.plot(x, up, label=f"BB_up({w},{nstd})", color=col_up, alpha=0.6)
                    l2, = self.ax.plot(x, low, label=f"BB_low({w},{nstd})", color=col_low, alpha=0.6)
                    _register(f"bollinger_{w}_{nstd}", [l1, l2])
                except Exception:
                    pass

            # RSI (reuse/create aux axis)
            if "rsi" in cfg:
                try:
                    p = int(cfg["rsi"].get("period", 14))
                    col = _get_color(cfg.get("rsi"), "#9467bd")
                    r = rsi(df["close"], period=p)
                    ax_rsi = self._aux_axes.get("rsi")
                    if ax_rsi is None:
                        ax_rsi = self.ax.twinx()
                        # shift right if needed
                        try:
                            ax_rsi.spines["right"].set_position(("axes", 1.05))
                        except Exception:
                            pass
                        self._aux_axes["rsi"] = ax_rsi
                    # plot on ax_rsi
                    ln, = ax_rsi.plot(x, r, label=f"RSI({p})", color=col, alpha=0.8)
                    ax_rsi.set_ylabel("RSI")
                    _register(f"rsi_{p}_{col}", ln)
                except Exception:
                    pass

            # MACD (reuse/create aux axis)
            if "macd" in cfg:
                try:
                    f = int(cfg["macd"].get("fast", 12))
                    s = int(cfg["macd"].get("slow", 26))
                    sg = int(cfg["macd"].get("signal", 9))
                    col = _get_color(cfg.get("macd"), "#ff7f0e")
                    m = macd(df["close"], fast=f, slow=s, signal=sg)
                    ax_macd = self._aux_axes.get("macd")
                    if ax_macd is None:
                        ax_macd = self.ax.twinx()
                        try:
                            ax_macd.spines["right"].set_position(("axes", 1.10))
                        except Exception:
                            pass
                        self._aux_axes["macd"] = ax_macd
                    ln1, = ax_macd.plot(x, m["macd"], label=f"MACD({f},{s})", color=col, alpha=0.9)
                    ln2, = ax_macd.plot(x, m["signal"], label=f"Signal({sg})", color="#1f77b4", alpha=0.6)
                    _register(f"macd_{f}_{s}_{sg}", [ln1, ln2])
                except Exception:
                    pass

            # refresh legends: combine base and aux handles
            try:
                handles, labels = self.ax.get_legend_handles_labels()
                # include aux axes legends
                for name, ax_aux in self._aux_axes.items():
                    try:
                        h, l = ax_aux.get_legend_handles_labels()
                        handles += h
                        labels += l
                    except Exception:
                        pass
                if handles:
                    self.ax.legend(handles, labels, loc="upper left", fontsize="small")
            except Exception:
                pass

            self.canvas.draw()
        except Exception as e:
            logger.exception("Internal _apply_indicators failed: {}", e)

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
            import pandas as pd
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

                        # load initial if provided (best-effort) - populate all controls including Bollinger, RSI, MACD
                        if initial:
                            try:
                                # SMA
                                if "sma" in initial:
                                    try:
                                        self.sma_cb.setChecked(True)
                                        if "window" in initial["sma"]:
                                            self.sma_w.setValue(int(initial["sma"]["window"]))
                                        if "color" in initial["sma"]:
                                            col = initial["sma"]["color"]
                                            for idx in range(self.sma_col.count()):
                                                if self.sma_col.itemData(idx) == col:
                                                    self.sma_col.setCurrentIndex(idx)
                                                    break
                                    except Exception:
                                        pass
                                # EMA
                                if "ema" in initial:
                                    try:
                                        self.ema_cb.setChecked(True)
                                        if "span" in initial["ema"]:
                                            self.ema_span.setValue(int(initial["ema"]["span"]))
                                        if "color" in initial["ema"]:
                                            col = initial["ema"]["color"]
                                            for idx in range(self.ema_col.count()):
                                                if self.ema_col.itemData(idx) == col:
                                                    self.ema_col.setCurrentIndex(idx)
                                                    break
                                    except Exception:
                                        pass
                                # Bollinger
                                if "bollinger" in initial:
                                    try:
                                        self.bb_cb.setChecked(True)
                                        if "window" in initial["bollinger"]:
                                            self.bb_n.setValue(int(initial["bollinger"]["window"]))
                                        if "n_std" in initial["bollinger"]:
                                            self.bb_k.setText(str(initial["bollinger"]["n_std"]))
                                        if "color_up" in initial["bollinger"]:
                                            col = initial["bollinger"]["color_up"]
                                            for idx in range(self.bb_col_u.count()):
                                                if self.bb_col_u.itemData(idx) == col:
                                                    self.bb_col_u.setCurrentIndex(idx); break
                                        if "color_low" in initial["bollinger"]:
                                            col = initial["bollinger"]["color_low"]
                                            for idx in range(self.bb_col_l.count()):
                                                if self.bb_col_l.itemData(idx) == col:
                                                    self.bb_col_l.setCurrentIndex(idx); break
                                    except Exception:
                                        pass
                                # RSI
                                if "rsi" in initial:
                                    try:
                                        self.rsi_cb.setChecked(True)
                                        if "period" in initial["rsi"]:
                                            self.rsi_p.setValue(int(initial["rsi"]["period"]))
                                        if "color" in initial["rsi"]:
                                            col = initial["rsi"]["color"]
                                            for idx in range(self.rsi_col.count()):
                                                if self.rsi_col.itemData(idx) == col:
                                                    self.rsi_col.setCurrentIndex(idx); break
                                    except Exception:
                                        pass
                                # MACD
                                if "macd" in initial:
                                    try:
                                        self.macd_cb.setChecked(True)
                                        if "fast" in initial["macd"]:
                                            self.macd_fast.setValue(int(initial["macd"]["fast"]))
                                        if "slow" in initial["macd"]:
                                            self.macd_slow.setValue(int(initial["macd"]["slow"]))
                                        if "signal" in initial["macd"]:
                                            self.macd_sig.setValue(int(initial["macd"]["signal"]))
                                        if "color" in initial["macd"]:
                                            col = initial["macd"]["color"]
                                            for idx in range(self.macd_col.count()):
                                                if self.macd_col.itemData(idx) == col:
                                                    self.macd_col.setCurrentIndex(idx); break
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                    def result(self):
                        """
                        Return a complete config dict with enabled flags and parameters for all indicators.
                        Ensures persistence is stable and the format is consistent across sessions.
                        """
                        out = {}
                        # SMA
                        out["sma"] = {
                            "enabled": bool(self.sma_cb.isChecked()),
                            "window": int(self.sma_w.value()),
                            "color": self.sma_col.currentData()
                        }
                        # EMA
                        out["ema"] = {
                            "enabled": bool(self.ema_cb.isChecked()),
                            "span": int(self.ema_span.value()),
                            "color": self.ema_col.currentData()
                        }
                        # Bollinger
                        try:
                            nstd = float(self.bb_k.text())
                        except Exception:
                            nstd = 2.0
                        out["bollinger"] = {
                            "enabled": bool(self.bb_cb.isChecked()),
                            "window": int(self.bb_n.value()),
                            "n_std": nstd,
                            "color_up": self.bb_col_u.currentData(),
                            "color_low": self.bb_col_l.currentData()
                        }
                        # RSI
                        out["rsi"] = {
                            "enabled": bool(self.rsi_cb.isChecked()),
                            "period": int(self.rsi_p.value()),
                            "color": self.rsi_col.currentData()
                        }
                        # MACD
                        out["macd"] = {
                            "enabled": bool(self.macd_cb.isChecked()),
                            "fast": int(self.macd_fast.value()),
                            "slow": int(self.macd_slow.value()),
                            "signal": int(self.macd_sig.value()),
                            "color": self.macd_col.currentData()
                        }
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
            # persist config
            self._indicator_cfg = cfg
            from pathlib import Path
            import json
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / "indicators.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg_path.open("w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            try:
                logger.info(f"Saved indicator config to {cfg_path}")
            except Exception:
                logger.info("Saved indicator config")
        except Exception as e:
            logger.exception("Failed to persist indicators config: {}", e)

        try:
            # use public wrapper to apply indicators (ensures internal method exists)
            try:
                self.apply_indicators(cfg)
            except Exception as e:
                logger.exception("apply_indicators failed: {}", e)
        except Exception as e:
            logger.exception("Failed to apply indicators after dialog: {}", e)
