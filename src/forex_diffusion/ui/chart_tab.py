# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional, Dict, List
import pandas as pd
from pathlib import Path
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox,
    QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QTableWidgetItem,
    QSplitter, QListWidget, QListWidgetItem, QTableWidget
)
from PySide6.QtCore import QTimer, Qt, Signal, QSize
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from loguru import logger

from ..utils.user_settings import get_setting, set_setting
from ..services.brokers import get_broker_service

class ChartTab(QWidget):
    """
    A comprehensive charting tab with real-time updates, indicators, and forecasting.
    Preserves zoom/pan when updating and supports multiple forecast overlays.
    """
    forecastRequested = Signal(dict)
    tickArrived = Signal(dict)
    tickArrived = Signal(dict)

    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self._main_window = parent
        # drawing/tools state
        self._drawing_mode: Optional[str] = None
        self._pending_points: List = []
        self._trend_points: List = []
        self._rect_points: List = []
        self._fib_points: List = []
        self._orders_visible: bool = True
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.toolbar = NavigationToolbar(self.canvas, self)

        # --- Toolbar Setup ---
        topbar = QWidget()
        top_layout = QHBoxLayout(topbar)
        # margini/spacing compatti per ridurre l'altezza
        top_layout.setContentsMargins(4, 1, 4, 1)
        top_layout.setSpacing(4)
        # rendi la NavigationToolbar compatta
        try:
            self.toolbar.setIconSize(QSize(16, 16))
            self.toolbar.setStyleSheet("QToolBar{spacing:2px; padding:0px; margin:0px;}")
            self.toolbar.setMovable(False)
            self.toolbar.setFloatable(False)
            self.toolbar.setMaximumHeight(26)
        except Exception:
            pass
        # stile compatto solo per i widget dentro la topbar
        topbar.setStyleSheet("""
            QPushButton, QComboBox, QToolButton { padding: 2px 6px; min-height: 22px; }
            QLabel { padding: 0px; margin: 0px; }
        """)
        # altezza massima della barra poco sopra i bottoni
        topbar.setMaximumHeight(32)
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

        # Advanced Forecast Buttons
        self.adv_settings_btn = QPushButton("Advanced Settings")
        self.adv_forecast_btn = QPushButton("Advanced Forecast")
        top_layout.addWidget(self.adv_settings_btn)
        top_layout.addWidget(self.adv_forecast_btn)

        # Toggle per barra Drawing (icone sopra al grafico)
        from PySide6.QtWidgets import QToolButton
        self.toggle_drawbar_btn = QToolButton()
        self.toggle_drawbar_btn.setText("Draw")
        self.toggle_drawbar_btn.setCheckable(True)
        self.toggle_drawbar_btn.setChecked(True)
        self.toggle_drawbar_btn.setToolTip("Mostra/Nascondi barra di disegno")
        top_layout.addWidget(self.toggle_drawbar_btn)

        # Backfill range selectors (session-only; 0=full range)
        from PySide6.QtWidgets import QLabel, QComboBox
        top_layout.addWidget(QLabel("Years:"))
        self.years_combo = QComboBox()
        self.years_combo.addItems([str(x) for x in [0,1,2,3,4,5,10,15,20,30]])
        self.years_combo.setCurrentText("0")
        top_layout.addWidget(self.years_combo)

        top_layout.addWidget(QLabel("Months:"))
        self.months_combo = QComboBox()
        self.months_combo.addItems([str(x) for x in [0,1,2,3,4,5,6,7,8,9,10,11,12]])
        self.months_combo.setCurrentText("0")
        top_layout.addWidget(self.months_combo)

        # Backfill button + progress
        self.backfill_btn = QPushButton("Backfill")
        self.backfill_btn.setToolTip("Scarica storico per il range selezionato (Years/Months) per il simbolo corrente")
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
        # Trade button on toolbar
        self.trade_btn = QPushButton("Trade")
        top_layout.addWidget(self.trade_btn)

        # Theme switch
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText(get_setting("ui_theme", "Dark"))
        top_layout.addWidget(QLabel("Theme:"))
        top_layout.addWidget(self.theme_combo)
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        # apply at startup
        self._apply_theme(self.theme_combo.currentText())

        self.layout.addWidget(topbar)

        # Drawing bar (icone) subito sopra il grafico
        from PySide6.QtWidgets import QToolButton
        from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
        from PySide6.QtWidgets import QStyle
        drawbar = QWidget()
        dlay = QHBoxLayout(drawbar); dlay.setContentsMargins(4, 2, 4, 2)
        # Nav buttons
        self.tb_home = QToolButton(); self.tb_home.setToolTip("Reset view"); self.tb_home.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.tb_pan = QToolButton(); self.tb_pan.setToolTip("Pan"); self.tb_pan.setCheckable(True); self.tb_pan.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.tb_zoom = QToolButton(); self.tb_zoom.setToolTip("Zoom"); self.tb_zoom.setCheckable(True); self.tb_zoom.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        for b in (self.tb_home, self.tb_pan, self.tb_zoom):
            dlay.addWidget(b)
        dlay.addSpacing(12)
        # Drawing tools (icone provvisorie)
        self.tb_cross = QToolButton(); self.tb_cross.setToolTip("Cross"); self.tb_cross.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.tb_hline = QToolButton(); self.tb_hline.setToolTip("H-Line"); self.tb_hline.setIcon(self.style().standardIcon(QStyle.SP_TitleBarShadeButton))
        self.tb_trend = QToolButton(); self.tb_trend.setToolTip("Trend"); self.tb_trend.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.tb_rect = QToolButton(); self.tb_rect.setToolTip("Rect"); self.tb_rect.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.tb_fib = QToolButton(); self.tb_fib.setToolTip("Fib"); self.tb_fib.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.tb_label = QToolButton(); self.tb_label.setToolTip("Label"); self.tb_label.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        # Colors come tool sulla drawbar
        self.tb_colors = QToolButton(); self.tb_colors.setToolTip("Color/CSS Settings"); self.tb_colors.setIcon(self.style().standardIcon(QStyle.SP_DriveDVDIcon))
        self.tb_colors.clicked.connect(self._open_color_settings)
        # Toggle Orders visibilità
        self.tb_orders = QToolButton(); self.tb_orders.setToolTip("Show/Hide Orders"); self.tb_orders.setCheckable(True); self.tb_orders.setChecked(True)
        self.tb_orders.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        self.tb_orders.toggled.connect(self._toggle_orders)

        # Aggiunta ordinata dei tools (evita doppioni dei primi tre già inseriti)
        for b in (self.tb_cross, self.tb_hline, self.tb_trend, self.tb_rect, self.tb_fib, self.tb_label, self.tb_colors, self.tb_orders):
            dlay.addWidget(b)
        dlay.addStretch()
        # conserva la barra per inserirla nello splitter sopra al grafico
        self._drawbar = drawbar

        # Connessioni nav/drawing
        self.tb_home.clicked.connect(self._on_nav_home)
        self.tb_pan.clicked.connect(self._on_nav_pan)
        self.tb_zoom.clicked.connect(self._on_nav_zoom)
        self.tb_cross.clicked.connect(lambda: self._set_drawing_mode(None))
        self.tb_hline.clicked.connect(lambda: self._set_drawing_mode("hline"))
        self.tb_trend.clicked.connect(lambda: self._set_drawing_mode("trend"))
        self.tb_rect.clicked.connect(lambda: self._set_drawing_mode("rect"))
        self.tb_fib.clicked.connect(lambda: self._set_drawing_mode("fib"))
        self.tb_label.clicked.connect(lambda: self._set_drawing_mode("label"))

        # Toggle drawbar visibility
        self.toggle_drawbar_btn.toggled.connect(self._toggle_drawbar)

        # Splitter: Market Watch | (Chart + Orders)
        splitter = QSplitter(Qt.Horizontal)
        # left: market watch
        self.market_watch = QListWidget()
        for s in self._symbols_supported:
            QListWidgetItem(f"{s}  -", self.market_watch)
        splitter.addWidget(self.market_watch)
        # right: vertical with (drawbar+chart) and orders table
        right_vert = QSplitter(Qt.Vertical)
        # inner splitter: drawbar | chart canvas
        chart_area = QSplitter(Qt.Vertical)
        chart_area.setHandleWidth(4); chart_area.setOpaqueResize(True)
        self._chart_area = chart_area
        chart_area.addWidget(self._drawbar)
        chart_wrap = QWidget()
        cw_lay = QVBoxLayout(chart_wrap); cw_lay.setContentsMargins(0, 0, 0, 0)
        cw_lay.addWidget(self.canvas)
        chart_area.addWidget(chart_wrap)
        right_vert.addWidget(chart_area)
        # orders table
        self.orders_table = QTableWidget(0, 9)
        self.orders_table.setHorizontalHeaderLabels(["ID","Time","Symbol","Type","Volume","Price","SL","TP","Status"])
        right_vert.addWidget(self.orders_table)
        splitter.addWidget(right_vert)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        # Splitter handles/movimento per tutti
        splitter.setHandleWidth(6); splitter.setOpaqueResize(True)
        right_vert.setHandleWidth(6); right_vert.setOpaqueResize(True)
        # Splitter handles/movimento
        splitter.setHandleWidth(6); splitter.setOpaqueResize(True)
        right_vert.setHandleWidth(6); right_vert.setOpaqueResize(True)
        self.layout.addWidget(splitter)

        self.ax = self.canvas.figure.subplots()
        self._last_df = pd.DataFrame()
        # keep a list of forecast dicts: {id, created_at, quantiles, future_ts, artists:list, source}
        self._forecasts: List[Dict] = []
        self.max_forecasts = int(get_setting("max_forecasts", 5))

        # Broker (simulato)
        try:
            self.broker = get_broker_service()
        except Exception:
            self.broker = None

        # Auto-forecast timer
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(int(get_setting("auto_interval_seconds", 60) * 1000))
        self._auto_timer.timeout.connect(self._auto_forecast_tick)

        # Orders refresh timer
        self._orders_timer = QTimer(self)
        self._orders_timer.setInterval(1500)
        self._orders_timer.timeout.connect(self._refresh_orders)
        self._orders_timer.start()

        # Realtime throttling: batch ticks and redraw ~5 FPS
        self._rt_dirty = False
        self._rt_timer = QTimer(self)
        self._rt_timer.setInterval(200)
        self._rt_timer.timeout.connect(self._rt_flush)
        self._rt_timer.start()

        # Ensure tick handling runs on GUI thread
        try:
            self.tickArrived.connect(self._on_tick_main)
        except Exception:
            pass

        # Realtime redraw throttling (batch ticks and redraw ~5 fps)
        self._rt_dirty = False
        self._rt_timer = QTimer(self)
        self._rt_timer.setInterval(200)
        self._rt_timer.timeout.connect(self._rt_flush)
        self._rt_timer.start()

        # Pan (LMB) state
        self._lbtn_pan = False
        self._pan_last_xy = None  # (xdata, ydata)

        # Dynamic data cache state
        self._current_cache_tf: Optional[str] = None
        self._current_cache_range: Optional[tuple[int, int]] = None  # (start_ms, end_ms)
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(250)
        self._reload_timer.timeout.connect(self._reload_view_window)

        # Ensure tick handling runs on GUI thread
        try:
            self.tickArrived.connect(self._on_tick_main)
        except Exception:
            pass

        # Buttons signals
        self.trade_btn.clicked.connect(self._open_trade_dialog)
        self.backfill_btn.clicked.connect(self._on_backfill_missing_clicked)

        # Mouse interaction:
        # - Alt+Click => TestingPoint basic; Shift+Alt+Click => advanced (già gestito da _on_canvas_click)
        # - Wheel zoom centrato sul mouse
        # - Right button drag: orizzontale = zoom X (tempo); verticale = zoom Y (prezzo)
        self._rbtn_drag = False
        self._drag_last = None  # (x_px, y_px)
        self._drag_axis = None  # "x" | "y" determinata dal movimento prevalente

        try:
            # conserva il gestore esistente per strumenti/testing (left click)
            self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
            # nuovi handler UX (pan/zoom)
            self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
            self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
            self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
            self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        except Exception:
            logger.debug("Failed to connect mpl mouse events for zoom/pan.")

        # --- Signal Connections ---
        self.forecast_settings_btn.clicked.connect(self._open_forecast_settings)
        self.forecast_btn.clicked.connect(self._on_forecast_clicked)
        self.adv_settings_btn.clicked.connect(self._open_adv_forecast_settings)
        self.adv_forecast_btn.clicked.connect(self._on_advanced_forecast_clicked)
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
        """Thread-safe entrypoint: enqueue tick to GUI thread."""
        try:
            self.tickArrived.emit(payload)
        except Exception as e:
            logger.exception("Failed to emit tick: {}", e)

    def _on_tick_main(self, payload: dict):
        """GUI-thread handler: aggiorna Market Watch e buffer, delega il redraw al throttler."""
        try:
            if not isinstance(payload, dict):
                return
            sym = payload.get("symbol") or getattr(self, "symbol", None)
            # Market watch update
            try:
                px = payload.get("price")
                if px is not None and sym:
                    for i in range(self.market_watch.count()):
                        it = self.market_watch.item(i)
                        if it and it.text().split()[0] == sym:
                            it.setText(f"{sym}  {px:.5f}")
            except Exception:
                pass

            # Update chart buffer only for current symbol
            if sym and sym == getattr(self, "symbol", None):
                try:
                    new_row = pd.DataFrame([payload])
                    # normalize types
                    new_row["ts_utc"] = pd.to_numeric(new_row["ts_utc"], errors="coerce").astype("Int64")
                    self._last_df = pd.concat([self._last_df, new_row], ignore_index=True)
                    # drop NaN/dup, sort and trim buffer
                    self._last_df["ts_utc"] = pd.to_numeric(self._last_df["ts_utc"], errors="coerce").astype("Int64")
                    self._last_df.dropna(subset=["ts_utc"], inplace=True)
                    self._last_df["ts_utc"] = self._last_df["ts_utc"].astype("int64")
                    self._last_df.drop_duplicates(subset=["ts_utc"], keep="last", inplace=True)
                    self._last_df.sort_values("ts_utc", inplace=True)
                    # keep last N rows to avoid heavy redraws
                    N = 10000
                    if len(self._last_df) > N:
                        self._last_df = self._last_df.iloc[-N:].copy()
                except Exception:
                    pass

                # update bid/ask label
                try:
                    if payload.get('bid') is not None and payload.get('ask') is not None:
                        self.bidask_label.setText(f"Bid: {float(payload['bid']):.5f}    Ask: {float(payload['ask']):.5f}")
                except Exception:
                    try:
                        self.bidask_label.setText(f"Bid: {payload.get('bid')}    Ask: {payload.get('ask')}")
                    except Exception:
                        pass

                # mark dirty for throttled redraw
                self._rt_dirty = True
        except Exception as e:
            logger.exception("Failed to handle tick on GUI: {}", e)

    def _rt_flush(self):
        """Throttled redraw preserving zoom/pan."""
        try:
            if not getattr(self, "_rt_dirty", False):
                return
            self._rt_dirty = False
            # preserve current view
            try:
                prev_xlim = self.ax.get_xlim()
                prev_ylim = self.ax.get_ylim()
            except Exception:
                prev_xlim = prev_ylim = None
            # redraw base chart (quantiles overlay mantenuti da _forecasts)
            if self._last_df is not None and not self._last_df.empty:
                self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
        except Exception as e:
            logger.exception("Realtime flush failed: {}", e)

    def _set_drawing_mode(self, mode: Optional[str]):
        self._drawing_mode = mode
        self._pending_points.clear()

    def _on_canvas_click(self, event):
        # drawing tools
        if self._drawing_mode == "hline" and event and event.ydata is not None:
            y = event.ydata
            ln = self.ax.axhline(y, color="#9bdcff", linestyle="--", alpha=0.7)
            self.canvas.draw()
            self._pending_points.clear()
            return
        if self._drawing_mode == "trend" and event and event.xdata is not None and event.ydata is not None:
            self._pending_points.append((event.xdata, event.ydata))
            if len(self._pending_points) == 2:
                (x1, y1), (x2, y2) = self._pending_points
                self.ax.plot([mdates.num2date(x1), mdates.num2date(x2)], [y1, y2], color="#ff9bdc", linewidth=1.5)
                self.canvas.draw()
                self._pending_points.clear()
                return
        # testing point (Alt/Shift already implemented above in previous patch)
        # fall through to existing Alt/Shift logic
        try:
            return super()._on_canvas_click(event)  # type: ignore
        except Exception:
            # ignore if super implementation not present in MRO
            pass

    def _open_trade_dialog(self):
        try:
            from .trade_dialog import TradeDialog
            symbols = self._symbols_supported
            cur = getattr(self, "symbol", symbols[0])
            dlg = TradeDialog(self, symbols=symbols, current_symbol=cur)
            if dlg.exec() == QDialog.Accepted:
                order = dlg.get_order()
                # place order via broker
                try:
                    ok, oid = self.broker.place_order(order)
                    if ok:
                        QMessageBox.information(self, "Order", f"Order placed (id={oid})")
                    else:
                        QMessageBox.warning(self, "Order", f"Order rejected: {oid}")
                except Exception as e:
                    QMessageBox.warning(self, "Order", str(e))
        except Exception as e:
            logger.exception("Trade dialog failed: {}", e)

    def _refresh_orders(self):
        """Pull open orders from broker and refresh the table."""
        try:
            orders = self.broker.get_open_orders()
            self.orders_table.setRowCount(len(orders))
            for r, o in enumerate(orders):
                vals = [
                    str(o.get("id","")),
                    o.get("time",""),
                    o.get("symbol",""),
                    o.get("side","") + " " + o.get("type",""),
                    str(o.get("volume","")),
                    f"{o.get('price','')}",
                    f"{o.get('sl','')}",
                    f"{o.get('tp','')}",
                    o.get("status","")
                ]
                for c, v in enumerate(vals):
                    self.orders_table.setItem(r, c, QTableWidgetItem(str(v)))
        except Exception:
            pass

    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None, restore_xlim=None, restore_ylim=None):
        if df is None or df.empty:
            return

        self._last_df = df.copy()

        # preserve previous limits only if explicitly provided
        prev_xlim = restore_xlim if restore_xlim is not None else None
        prev_ylim = restore_ylim if restore_ylim is not None else None

        self.ax.clear()

        # Clean and normalize data for plotting (align x/y, drop NaN, sort by time)
        try:
            df2 = df.copy()
            y_col = 'close' if 'close' in df2.columns else 'price'
            df2["ts_utc"] = pd.to_numeric(df2["ts_utc"], errors="coerce")
            df2[y_col] = pd.to_numeric(df2[y_col], errors="coerce")
            df2 = df2.dropna(subset=["ts_utc", y_col]).reset_index(drop=True)
            df2["ts_utc"] = df2["ts_utc"].astype("int64")
            df2 = df2.sort_values("ts_utc").reset_index(drop=True)

            # build x (naive datetime for matplotlib) and y aligned
            x_dt = pd.to_datetime(df2["ts_utc"], unit="ms", utc=True)
            try:
                x_dt = x_dt.tz_localize(None)
            except Exception:
                pass
            y_vals = df2[y_col].astype(float).to_numpy()
        except Exception as e:
            logger.exception("Failed to normalize data for plotting: {}", e)
            return

        if len(x_dt) == 0 or len(y_vals) == 0:
            logger.info("Nothing to plot after cleaning ({} {}).", getattr(self, 'symbol', ''), getattr(self, 'timeframe', ''))
            return

        price_color = self._get_color("price_color", "#e0e0e0" if getattr(self, "_is_dark", True) else "#000000")
        self.ax.plot(x_dt, y_vals, color=price_color, label="Price")

        if quantiles:
            self._plot_forecast_overlay(quantiles)

        self.ax.set_title(f"{getattr(self, 'symbol', '')} - {getattr(self, 'timeframe', '')}")
        # assi/ticks colorati da settings
        axes_col = self._get_color("axes_color", "#cfd6e1")
        try:
            self.ax.tick_params(colors=axes_col)
            self.ax.xaxis.label.set_color(axes_col)
            self.ax.yaxis.label.set_color(axes_col)
            for spine in self.ax.spines.values():
                spine.set_color(axes_col)
        except Exception:
            pass
        self.ax.legend()
        self.ax.figure.autofmt_xdate()

        # restore limits (only if requested), else autoscale to data
        try:
            if prev_xlim is not None:
                self.ax.set_xlim(prev_xlim)
            if prev_ylim is not None:
                self.ax.set_ylim(prev_ylim)
            if prev_xlim is None and prev_ylim is None:
                self.ax.relim()
                self.ax.autoscale_view()
        except Exception:
            pass

        try:
            self.canvas.draw_idle()
        except Exception:
            self.canvas.draw()
        try:
            logger.info("Plotted {} points for {} {}", len(y_vals), getattr(self, 'symbol', ''), getattr(self, 'timeframe', ''))
        except Exception:
            pass

    # --- Theme helpers ---
    def _on_nav_home(self):
        try:
            self.toolbar.home()
        except Exception:
            pass

    def _on_nav_pan(self, checked: bool):
        try:
            # disattiva zoom se pan attivo
            if checked and hasattr(self, "tb_zoom"):
                self.tb_zoom.setChecked(False)
            self.toolbar.pan()
        except Exception:
            pass

    def _on_nav_zoom(self, checked: bool):
        try:
            # disattiva pan se zoom attivo
            if checked and hasattr(self, "tb_pan"):
                self.tb_pan.setChecked(False)
            self.toolbar.zoom()
        except Exception:
            pass

    def _apply_theme(self, theme: str):
        from PySide6.QtGui import QPalette, QColor
        from PySide6.QtWidgets import QApplication
        t = (theme or "Dark").lower()
        self._is_dark = (t == "dark")
        app = QApplication.instance()
        # Colori da settings (con fallback per dark/light)
        window_bg = self._get_color("window_bg", "#0f1115" if self._is_dark else "#f3f5f8")
        panel_bg = self._get_color("panel_bg", "#12151b" if self._is_dark else "#ffffff")
        text_color = self._get_color("text_color", "#e0e0e0" if self._is_dark else "#1a1e25")
        chart_bg = self._get_color("chart_bg", "#0f1115" if self._is_dark else "#ffffff")
        base_css = f"""
        QWidget {{ background-color: {window_bg}; color: {text_color}; }}
        QPushButton, QComboBox, QToolButton {{ background-color: {('#1c1f26' if self._is_dark else '#ffffff')}; color: {text_color}; border: 1px solid {('#2a2f3a' if self._is_dark else '#cfd6e1')}; padding: 4px 8px; border-radius: 4px; }}
        QPushButton:hover, QToolButton:hover {{ background-color: {('#242a35' if self._is_dark else '#eaeef4')}; }}
        QTableWidget, QListWidget {{ background-color: {panel_bg}; color: {text_color}; gridline-color: {('#2a2f3a' if self._is_dark else '#cfd6e1')}; }}
        QHeaderView::section {{ background-color: {('#1a1e25' if self._is_dark else '#e8edf4')}; color: {text_color}; border: 0px; }}
        """
        custom_qss = get_setting("custom_qss", "")
        if app:
            app.setStyleSheet(base_css + "\n" + (custom_qss or ""))
            pal = QPalette()
            pal.setColor(QPalette.Window, QColor(window_bg))
            pal.setColor(QPalette.WindowText, QColor(text_color))
            pal.setColor(QPalette.Base, QColor(panel_bg))
            pal.setColor(QPalette.AlternateBase, QColor(panel_bg))
            pal.setColor(QPalette.Text, QColor(text_color))
            pal.setColor(QPalette.Button, QColor(panel_bg))
            pal.setColor(QPalette.ButtonText, QColor(text_color))
            app.setPalette(pal)
        # colori figure
        try:
            self.canvas.figure.set_facecolor(chart_bg)
            self.ax.set_facecolor(chart_bg)
        except Exception:
            pass
        set_setting("ui_theme", "Dark" if self._is_dark else "Light")
        try: self.canvas.draw()
        except Exception: pass

    def _get_color(self, key: str, default: str) -> str:
        try:
            return str(get_setting(key, default))
        except Exception:
            return default

    def _open_color_settings(self):
        try:
            from .color_settings_dialog import ColorSettingsDialog
            dlg = ColorSettingsDialog(self)
            if dlg.exec():
                # re-draw to apply new colors
                if self._last_df is not None and not self._last_df.empty:
                    self.update_plot(self._last_df)
                # ri-applica il tema per aggiornare QSS/palette
                self._apply_theme(self.theme_combo.currentText())
        except Exception as e:
            QMessageBox.warning(self, "Colors", str(e))

    def _toggle_drawbar(self, visible: bool):
        try:
            if hasattr(self, "_drawbar") and self._drawbar is not None:
                self._drawbar.setVisible(bool(visible))
        except Exception:
            pass

    def _toggle_orders(self, visible: bool):
        try:
            self._orders_visible = bool(visible)
            if hasattr(self, "orders_table") and self.orders_table is not None:
                self.orders_table.setVisible(self._orders_visible)
        except Exception:
            pass

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
        # compute view range from UI (Years/Months) and load initial candles
        try:
            from datetime import datetime, timezone, timedelta
            yrs = int(self.years_combo.currentText() or "0") if hasattr(self, "years_combo") else 0
            mos = int(self.months_combo.currentText() or "0") if hasattr(self, "months_combo") else 0
            days = yrs * 365 + mos * 30
            start_ms_view = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000) if days > 0 else None
        except Exception:
            start_ms_view = None
        try:
            df = self._load_candles_from_db(self.symbol, self.timeframe, limit=3000, start_ms=start_ms_view)
            if df is not None and not df.empty:
                self.update_plot(df)
                logger.info("Plotted {} points for {} {}", len(df), self.symbol, self.timeframe)
            else:
                logger.info("No candles found in DB for {} {}", self.symbol, self.timeframe)
        except Exception as e:
            logger.exception("Initial load failed: {}", e)

    def _on_symbol_changed(self, new_symbol: str):
        """Handle symbol change from combo: update context and reload candles from DB."""
        try:
            if not new_symbol:
                return
            self.symbol = new_symbol
            # reload last candles for this symbol/timeframe (respect view range)
            from datetime import datetime, timezone, timedelta
            try:
                yrs = int(self.years_combo.currentText() or "0")
                mos = int(self.months_combo.currentText() or "0")
                days = yrs * 365 + mos * 30
                start_ms_view = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000) if days > 0 else None
            except Exception:
                start_ms_view = None
            df = self._load_candles_from_db(new_symbol, getattr(self, "timeframe", "1m"), limit=3000, start_ms=start_ms_view)
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
        Drawing tools + TestingPoint:
        - Se drawing_mode attivo: disegna H-Line, Trend, Rect, Fib, Label.
        - Con Alt+Click: TestingPoint basic; Shift+Alt+Click: advanced.
        """
        try:
            if event is None or getattr(event, "button", None) != 1:
                return

            # If drawing mode is active handle first
            if self._drawing_mode and event.xdata is not None and event.ydata is not None:
                import matplotlib.patches as patches
                if self._drawing_mode == "hline":
                    ln = self.ax.axhline(event.ydata, color=self._get_color("hline_color", "#9bdcff"), linestyle="--", alpha=0.8)
                    self.canvas.draw()
                    return
                if self._drawing_mode == "trend":
                    if not hasattr(self, "_trend_points"):
                        self._trend_points = []
                    self._trend_points.append((event.xdata, event.ydata))
                    if len(self._trend_points) == 2:
                        (x1, y1), (x2, y2) = self._trend_points
                        self.ax.plot([mdates.num2date(x1), mdates.num2date(x2)], [y1, y2], color=self._get_color("trend_color", "#ff9bdc"), linewidth=1.5)
                        self.canvas.draw()
                        self._trend_points = []
                    return
                if self._drawing_mode == "rect":
                    if not hasattr(self, "_rect_points"):
                        self._rect_points = []
                    self._rect_points.append((event.xdata, event.ydata))
                    if len(self._rect_points) == 2:
                        (x1, y1), (x2, y2) = self._rect_points
                        xmin, xmax = sorted([x1, x2]); ymin, ymax = sorted([y1, y2])
                        rect = patches.Rectangle((mdates.num2date(xmin), ymin), mdates.num2num(mdates.num2date(xmax)) - mdates.num2num(mdates.num2date(xmin)), ymax - ymin, fill=False, edgecolor=self._get_color("rect_color", "#f0c674"), linewidth=1.2)
                        self.ax.add_patch(rect); self.canvas.draw(); self._rect_points = []
                    return
                if self._drawing_mode == "fib":
                    if not hasattr(self, "_fib_points"):
                        self._fib_points = []
                    self._fib_points.append((event.xdata, event.ydata))
                    if len(self._fib_points) == 2:
                        (x1, y1), (x2, y2) = self._fib_points
                        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
                        a, b = (y1, y2) if y2 > y1 else (y2, y1)
                        for lv in levels:
                            y = a + lv * (b - a)
                            self.ax.axhline(y, color=self._get_color("fib_color", "#9fe6a0"), alpha=0.6, linestyle=":")
                        self.canvas.draw(); self._fib_points = []
                    return
                if self._drawing_mode == "label":
                    txt = self.ax.text(mdates.num2date(event.xdata), event.ydata, "Label", color=self._get_color("label_color", "#ffd479"))
                    try:
                        txt.set_draggable(True)
                    except Exception:
                        pass
                    # non bloccare il main loop: draw_idle
                    try:
                        self.canvas.draw_idle()
                    except Exception:
                        self.canvas.draw()
                    return

            # TestingPoint logic (requires Alt)
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
                # use global mdates imported at module level
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

    # --- Mouse UX: zoom con rotellina e tasto destro (drag) ---
    def _on_scroll_zoom(self, event):
        try:
            if event is None or event.inaxes != self.ax:
                return
            # fattore di zoom: up=in, down=out
            step = 0.85
            factor = step if getattr(event, "button", None) == "up" else (1.0 / step)
            cx = event.xdata
            cy = event.ydata
            self._zoom_axis("x", cx, factor)
            self._zoom_axis("y", cy, factor)
            try:
                self.canvas.draw_idle()
            except Exception:
                self.canvas.draw()
            # ricarica dati coerenti con il nuovo zoom
            self._schedule_view_reload()
        except Exception:
            pass

    def _on_mouse_press(self, event):
        try:
            if event is None or event.inaxes != self.ax:
                return
            # Left button: PAN (se nessun drawing tool attivo e nessun Alt)
            if getattr(event, "button", None) == 1:
                if not getattr(self, "_drawing_mode", None):
                    # evita conflitti coi testing point (Alt)
                    try:
                        if event.guiEvent and event.guiEvent.modifiers() & Qt.AltModifier:
                            return
                    except Exception:
                        pass
                    self._lbtn_pan = True
                    self._pan_last_xy = (event.xdata, event.ydata)
                    return
            # Right button: prepara zoom assiale
            if getattr(event, "button", None) == 3:
                self._rbtn_drag = True
                self._drag_last = (event.x, event.y)  # pixel coordinates
                self._drag_axis = None  # decideremo al primo movimento
        except Exception:
            pass

    def _on_mouse_move(self, event):
        try:
            if event is None or event.inaxes != self.ax:
                return
            # PAN con LMB: sposta i limiti in base allo spostamento del cursore in coordinate dati
            if self._lbtn_pan and self._pan_last_xy is not None:
                x_last, y_last = self._pan_last_xy
                if event.xdata is None or event.ydata is None or x_last is None or y_last is None:
                    return
                dx = event.xdata - x_last
                dy = event.ydata - y_last
                xmin, xmax = self.ax.get_xlim()
                ymin, ymax = self.ax.get_ylim()
                # sposta inverso del movimento (trascinando a destra "porti" i dati verso sinistra)
                self.ax.set_xlim(xmin - dx, xmax - dx)
                self.ax.set_ylim(ymin - dy, ymax - dy)
                self._pan_last_xy = (event.xdata, event.ydata)
                try:
                    self.canvas.draw_idle()
                except Exception:
                    self.canvas.draw()
                return

            # ZOOM con RMB (asse X o Y)
            if self._rbtn_drag and self._drag_last is not None:
                x0, y0 = self._drag_last
                dx = event.x - x0
                dy = event.y - y0
                # determina asse predominante al primo movimento significativo
                if self._drag_axis is None:
                    if abs(dx) > abs(dy) * 1.2:
                        self._drag_axis = "x"
                    elif abs(dy) > abs(dx) * 1.2:
                        self._drag_axis = "y"
                    else:
                        return
                # mappa delta pixel -> fattore di zoom
                if self._drag_axis == "x":
                    factor = (0.90 ** (dx / 20.0)) if dx != 0 else 1.0
                    cx = event.xdata if event.xdata is not None else sum(self.ax.get_xlim()) / 2.0
                    self._zoom_axis("x", cx, factor)
                else:
                    factor = (0.90 ** (-dy / 20.0)) if dy != 0 else 1.0
                    cy = event.ydata if event.ydata is not None else sum(self.ax.get_ylim()) / 2.0
                    self._zoom_axis("y", cy, factor)
                self._drag_last = (event.x, event.y)
                try:
                    self.canvas.draw_idle()
                except Exception:
                    self.canvas.draw()
        except Exception:
            pass

    def _on_mouse_release(self, event):
        try:
            btn = getattr(event, "button", None)
            if btn == 3:
                self._rbtn_drag = False
                self._drag_last = None
                self._drag_axis = None
                # reload dati dopo zoom
                self._schedule_view_reload()
            if btn == 1:
                self._lbtn_pan = False
                self._pan_last_xy = None
                # reload dati dopo pan
                self._schedule_view_reload()
        except Exception:
            pass

    def _zoom_axis(self, axis: str, center: float, factor: float):
        """Zoom helper: scala i limiti attorno a 'center' con 'factor' (factor<1=zoom-in)."""
        try:
            if axis == "x":
                xmin, xmax = self.ax.get_xlim()
                if center is None:
                    center = (xmin + xmax) * 0.5
                w = max(1e-9, (xmax - xmin))
                new_w = max(1e-9, w * float(factor))
                left = center - (center - xmin) * (new_w / w)
                right = center + (xmax - center) * (new_w / w)
                if right - left > 1e-12:
                    self.ax.set_xlim(left, right)
            elif axis == "y":
                ymin, ymax = self.ax.get_ylim()
                if center is None:
                    center = (ymin + ymax) * 0.5
                h = max(1e-12, (ymax - ymin))
                new_h = max(1e-12, h * float(factor))
                bottom = center - (center - ymin) * (new_h / h)
                top = center + (ymax - center) * (new_h / h)
                if top - bottom > 1e-12:
                    self.ax.set_ylim(bottom, top)
        except Exception:
            pass

    def _schedule_view_reload(self):
        """Throttle view-window reload after user interaction."""
        try:
            self._reload_timer.stop()
            self._reload_timer.start()
        except Exception:
            pass

    def _resolution_for_span(self, ms_span: int) -> str:
        """Pick best timeframe by visible span (ms)."""
        try:
            mins = max(1, int(ms_span / 60000))
            # mapping: <=30m -> 1m; <=5h -> 5m; <=24h -> 15m; <=7d -> 1h; >7d -> 4h
            if mins <= 30:
                return "1m"   # ticks storici non disponibili in DB: usiamo 1m
            if mins <= 5 * 60:
                return "5m"
            if mins <= 24 * 60:
                return "15m"
            if mins <= 7 * 24 * 60:
                return "1h"
            return "4h"
        except Exception:
            return "15m"

    def _reload_view_window(self):
        """Reload only data covering [view_left .. view_right] plus one span of history."""
        try:
            # get current view in data coordinates
            xlim = self.ax.get_xlim()
            if not xlim or xlim[0] >= xlim[1]:
                return
            # matplotlib date floats -> UTC ms
            import matplotlib.dates as mdates
            left_dt = mdates.num2date(xlim[0])
            right_dt = mdates.num2date(xlim[1])
            # ensure UTC ms
            from datetime import timezone
            if left_dt.tzinfo is None:
                left_dt = left_dt.replace(tzinfo=timezone.utc)
            else:
                left_dt = left_dt.astimezone(timezone.utc)
            if right_dt.tzinfo is None:
                right_dt = right_dt.replace(tzinfo=timezone.utc)
            else:
                right_dt = right_dt.astimezone(timezone.utc)
            left_ms = int(left_dt.timestamp() * 1000)
            right_ms = int(right_dt.timestamp() * 1000)
            span = max(1, right_ms - left_ms)
            start_ms = max(0, left_ms - span)  # una finestra “cache” a sinistra
            end_ms = right_ms

            tf_req = self._resolution_for_span(span)
            sym = getattr(self, "symbol", None)
            if not sym or not tf_req:
                return

            # se cache già copre bene, skip
            if self._current_cache_tf == tf_req and self._current_cache_range:
                c0, c1 = self._current_cache_range
                if start_ms >= c0 and end_ms <= c1:
                    return  # nulla da fare

            # carica da DB solo la finestra necessaria
            df = self._load_candles_from_db(sym, tf_req, limit=50000, start_ms=start_ms, end_ms=end_ms)
            if df is None or df.empty:
                return

            # aggiorna cache state
            self._current_cache_tf = tf_req
            self._current_cache_range = (int(df["ts_utc"].iat[0]), int(df["ts_utc"].iat[-1]))

            # ridisegna preservando i limiti attuali
            try:
                prev_xlim = self.ax.get_xlim()
                prev_ylim = self.ax.get_ylim()
            except Exception:
                prev_xlim = prev_ylim = None
            self.update_plot(df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
        except Exception as e:
            logger.exception("View-window reload failed: {}", e)

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

        # compute start override from UI years/months (if >0)
        try:
            yrs = int(self.years_combo.currentText() or "0")
            mos = int(self.months_combo.currentText() or "0")
        except Exception:
            yrs = 0; mos = 0
        start_override = None
        if yrs > 0 or mos > 0:
            from datetime import datetime, timezone, timedelta
            days = yrs * 365 + mos * 30
            start_dt = datetime.now(timezone.utc) - timedelta(days=days)
            start_override = int(start_dt.timestamp() * 1000)
            try:
                logger.info("Backfill requested via UI: Years={}, Months={} -> start={} (UTC)", yrs, mos, start_dt.isoformat())
            except Exception:
                pass
        else:
            try:
                logger.info("Backfill requested via UI: Years=0, Months=0 -> no override (service decides from last candle)")
            except Exception:
                pass

        from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

        class BackfillSignals(QObject):
            progress = Signal(int)
            finished = Signal(bool)

        class BackfillJob(QRunnable):
            def __init__(self, svc, symbol, timeframe, start_override, signals):
                super().__init__()
                self.svc = svc
                self.symbol = symbol
                self.timeframe = timeframe
                self.start_override = start_override
                self.signals = signals

            def run(self):
                ok = True
                try:
                    def _cb(pct: int):
                        try:
                            self.signals.progress.emit(int(pct))
                        except Exception:
                            pass
                    # Enable REST backfill only for this explicit request
                    try:
                        setattr(self.svc, "rest_enabled", True)
                    except Exception:
                        pass
                    self.svc.backfill_symbol_timeframe(self.symbol, self.timeframe, force_full=False, progress_cb=_cb, start_ms_override=self.start_override)
                except Exception as e:
                    ok = False
                finally:
                    # Always disable REST backfill after completion
                    try:
                        setattr(self.svc, "rest_enabled", False)
                    except Exception:
                        pass
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
                # reload last candles for the current view range
                from datetime import datetime, timezone, timedelta
                try:
                    yrs_v = int(self.years_combo.currentText() or "0")
                    mos_v = int(self.months_combo.currentText() or "0")
                    days_v = yrs_v * 365 + mos_v * 30
                    start_ms_view = int((datetime.now(timezone.utc) - timedelta(days=days_v)).timestamp() * 1000) if days_v > 0 else None
                except Exception:
                    start_ms_view = None
                df = self._load_candles_from_db(sym, tf, limit=3000, start_ms=start_ms_view)
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
        job = BackfillJob(ms, sym, tf, start_override, self._bf_signals)
        QThreadPool.globalInstance().start(job)

    def _load_candles_from_db(self, symbol: str, timeframe: str, limit: int = 5000, start_ms: Optional[int] = None, end_ms: Optional[int] = None):
        """Load candles from DB for symbol/timeframe, optionally constraining [start_ms, end_ms]."""
        try:
            controller = getattr(self._main_window, "controller", None)
            eng = getattr(getattr(controller, "market_service", None), "engine", None) if controller else None
            if eng is None:
                return pd.DataFrame()
            from sqlalchemy import MetaData, select, and_
            meta = MetaData()
            meta.reflect(bind=eng, only=["market_data_candles"])
            tbl = meta.tables.get("market_data_candles")
            if tbl is None:
                return pd.DataFrame()
            with eng.connect() as conn:
                conds = [tbl.c.symbol == symbol, tbl.c.timeframe == timeframe]
                if start_ms is not None:
                    conds.append(tbl.c.ts_utc >= int(start_ms))
                if end_ms is not None:
                    conds.append(tbl.c.ts_utc <= int(end_ms))
                cond = and_(*conds)
                # prendi le barre più recenti nel range e poi ordinale ASC per il plot
                stmt = select(tbl.c.ts_utc, tbl.c.open, tbl.c.high, tbl.c.low, tbl.c.close, tbl.c.volume)\
                    .where(cond).order_by(tbl.c.ts_utc.desc()).limit(limit)
                rows = conn.execute(stmt).fetchall()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=["ts_utc","open","high","low","close","volume"])
                # typing e ordinamento ASC
                try:
                    df["ts_utc"] = pd.to_numeric(df["ts_utc"], errors="coerce").astype("Int64")
                    for c in ["open","high","low","close","volume"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.dropna(subset=["ts_utc"]).reset_index(drop=True)
                    df["ts_utc"] = df["ts_utc"].astype("int64")
                    df = df.sort_values("ts_utc").reset_index(drop=True)
                    # drop duplicates on timestamp to avoid multi-insert artifacts
                    before = len(df)
                    df = df.drop_duplicates(subset=["ts_utc"], keep="last").reset_index(drop=True)
                    trimmed = before - len(df)
                    if trimmed > 0:
                        try:
                            logger.info("Trimmed {} duplicate bars for {} {}", trimmed, symbol, timeframe)
                        except Exception:
                            pass
                except Exception:
                    pass
                return df
        except Exception as e:
            logger.exception("Load candles failed: {}", e)
            return pd.DataFrame()

