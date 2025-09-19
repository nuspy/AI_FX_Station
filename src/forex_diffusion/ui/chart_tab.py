# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional, Dict, List
import pandas as pd

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox,
    QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QTableWidgetItem,
    QSplitter, QListWidget, QListWidgetItem, QTableWidget, QComboBox
)
from PySide6.QtCore import QTimer, Qt, Signal, QSize
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from loguru import logger

from ..utils.user_settings import get_setting, set_setting
from ..services.brokers import get_broker_service
from .chart_components.controllers.chart_controller import ChartTabController

class ChartTab(QWidget):
    """
    A comprehensive charting tab with real-time updates, indicators, and forecasting.
    Preserves zoom/pan when updating and supports multiple forecast overlays.
    """
    forecastRequested = Signal(dict)
    tickArrived = Signal(dict)

    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self._main_window = parent

        # attach controller immediately if present on parent
        self.controller = getattr(parent, "controller", None)
        try:
            if self.controller and hasattr(self.controller, "signals"):
                self.controller.signals.forecastReady.connect(self.on_forecast_ready)
        except Exception:
            pass

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
        self.chart_controller = ChartTabController(self, self.controller)

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
        self.symbol_combo = QComboBox()
        # Requested pairs
        self._symbols_supported = ["EUR/USD","GBP/USD","AUX/USD", "GBP/NZD", "AUD/JPY", "GBP/EUR", "GBP/AUD"]
        self.symbol_combo.addItems(self._symbols_supported)
        top_layout.addWidget(self.symbol_combo)

        # Timeframe selector (with 'auto')
        top_layout.addWidget(QLabel("TF:"))
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["auto","tick","1m","5m","15m","30m","1h","4h","1d"])
        self.tf_combo.setCurrentText("auto")
        self.tf_combo.setToolTip("Seleziona il timeframe di visualizzazione. 'auto' sceglie il TF in base allo zoom.")
        top_layout.addWidget(self.tf_combo)
        # Label di stato TF effettivo
        self.tf_used_label = QLabel("TF used: -")
        top_layout.addWidget(self.tf_used_label)
        self.tf_combo.currentTextChanged.connect(lambda _: self.chart_controller.schedule_view_reload())

        # Basic Forecast Buttons
        self.forecast_settings_btn = QPushButton("Prediction Settings")
        self.forecast_btn = QPushButton("Make Prediction")
        top_layout.addWidget(self.forecast_settings_btn)
        top_layout.addWidget(self.forecast_btn)

        # Forecast granularity selector
        top_layout.addWidget(QLabel("Pred step:"))
        self.pred_step_combo = QComboBox()
        self.pred_step_combo.addItems(["auto", "1m"])
        self.pred_step_combo.setCurrentText("auto")
        self.pred_step_combo.setToolTip("Granularità dei punti di previsione. 'auto' usa gli horizons, '1m' aggiunge punti a minuto.")
        top_layout.addWidget(self.pred_step_combo)

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

        # Indicators settings
        self.indicators_btn = QPushButton("Indicators")
        self.indicators_btn.setToolTip("Configura gli indicatori tecnici (ATR, RSI, Bollinger, Hurst)")
        top_layout.addWidget(self.indicators_btn)

        # Build Latents (PCA)
        self.build_latents_btn = QPushButton("Build Latents")
        self.build_latents_btn.setToolTip("Costruisci latents PCA dalle feature storiche per il simbolo/TF correnti")
        top_layout.addWidget(self.build_latents_btn)

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

        # Toggle prezzo: Candles/Line (manuale)
        from PySide6.QtWidgets import QToolButton
        self.mode_btn = QToolButton()
        self.mode_btn.setCheckable(True)
        self.mode_btn.setText("Candles")  # checked => candles, unchecked => line
        self.mode_btn.setToolTip("Commuta visualizzazione prezzo: Candles (candele OHLC) / Line (linea).")
        self.mode_btn.setChecked(True)
        self.mode_btn.toggled.connect(self.chart_controller.on_mode_toggled)
        self.mode_btn.setText("Linea" if self.mode_btn.isChecked() else "Candles")
        top_layout.addWidget(self.mode_btn)

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
        self.theme_combo.currentTextChanged.connect(self.chart_controller.apply_theme)
        # apply at startup
        self.chart_controller.apply_theme(self.theme_combo.currentText())

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
        self.tb_colors.clicked.connect(self.chart_controller.open_color_settings)
        # Toggle Orders visibilità
        self.tb_orders = QToolButton(); self.tb_orders.setToolTip("Show/Hide Orders"); self.tb_orders.setCheckable(True); self.tb_orders.setChecked(True)
        self.tb_orders.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        self.tb_orders.toggled.connect(self.chart_controller.toggle_orders)

        # Aggiunta ordinata dei tools (evita doppioni dei primi tre già inseriti)
        for b in (self.tb_cross, self.tb_hline, self.tb_trend, self.tb_rect, self.tb_fib, self.tb_label, self.tb_colors, self.tb_orders):
            dlay.addWidget(b)
        dlay.addStretch()
        # conserva la barra per inserirla nello splitter sopra al grafico
        self._drawbar = drawbar

        # Connessioni nav/drawing
        self.tb_home.clicked.connect(self.chart_controller.on_nav_home)
        self.tb_pan.clicked.connect(self.chart_controller.on_nav_pan)
        self.tb_zoom.clicked.connect(self.chart_controller.on_nav_zoom)
        self.tb_cross.clicked.connect(lambda: self.chart_controller.set_drawing_mode(None))
        self.tb_hline.clicked.connect(lambda: self.chart_controller.set_drawing_mode("hline"))
        self.tb_trend.clicked.connect(lambda: self.chart_controller.set_drawing_mode("trend"))
        self.tb_rect.clicked.connect(lambda: self.chart_controller.set_drawing_mode("rect"))
        self.tb_fib.clicked.connect(lambda: self.chart_controller.set_drawing_mode("fib"))
        self.tb_label.clicked.connect(lambda: self.chart_controller.set_drawing_mode("label"))

        # Toggle drawbar visibility
        self.toggle_drawbar_btn.toggled.connect(self.chart_controller.toggle_drawbar)

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
        try:
            # Minimize outer paddings around the plot area
            self.canvas.figure.set_constrained_layout(False)
            self.canvas.figure.subplots_adjust(left=0.04, right=0.995, top=0.97, bottom=0.08)
            # Reduce internal data margins (focus on data)
            self.ax.margins(x=0.001, y=0.05)
        except Exception:
            pass
        try:
            # Keep oscillator panel aligned on any x-limit change of main axis
            self._xlim_cid = self.ax.callbacks.connect('xlim_changed', self._on_main_xlim_changed)
        except Exception:
            pass

        self._ind_artists = {}  # dict[str, list[matplotlib.artist.Artist]]
        self._osc_ax = None  # asse “oscillatori” (RSI/MACD/ATR/Hurst)

        self._last_df = pd.DataFrame()
        # keep a list of forecast dicts: {id, created_at, quantiles, future_ts, artists:list, source}
        self._forecasts: List[Dict] = []
        # show multiple overlays without removing old ones (unless over limit)
        self.max_forecasts = int(get_setting("max_forecasts", 20))
        # legend registry: show each model only once in legend
        self._legend_once = set()

        # Broker (simulato)
        try:
            self.broker = get_broker_service()
        except Exception:
            self.broker = None

        # Auto-forecast timer
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(int(get_setting("auto_interval_seconds", 60) * 1000))
        self._auto_timer.timeout.connect(self.chart_controller.auto_forecast_tick)

        # Orders refresh timer
        self._orders_timer = QTimer(self)
        self._orders_timer.setInterval(1500)
        self._orders_timer.timeout.connect(self.chart_controller.refresh_orders)
        self._orders_timer.start()

        # Realtime throttling: batch ticks and redraw ~5 FPS
        self._rt_dirty = False
        self._rt_timer = QTimer(self)
        self._rt_timer.setInterval(200)
        self._rt_timer.timeout.connect(self.chart_controller.rt_flush)
        self._rt_timer.start()

        # Ensure tick handling runs on GUI thread
        try:
            self.tickArrived.connect(self.chart_controller.on_tick_main)
        except Exception:
            pass

        # Stato rendering prezzo e artist
        self._price_mode = "candles"   # "line" | "candles"
        self._price_line = None
        self._candle_artists: list = []

        # Pan (LMB) state
        self._lbtn_pan = False
        self._pan_last_xy = None  # (xdata, ydata)

        # Dynamic data cache state
        self._current_cache_tf: Optional[str] = None
        self._current_cache_range: Optional[tuple[int, int]] = None  # (start_ms, end_ms)
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(250)
        self._reload_timer.timeout.connect(self.chart_controller.reload_view_window)

        # Ensure tick handling runs on GUI thread
        try:
            self.tickArrived.connect(self.chart_controller.on_tick_main)
        except Exception:
            pass

        # Buttons signals
        self.trade_btn.clicked.connect(self.chart_controller.open_trade_dialog)
        self.backfill_btn.clicked.connect(self.chart_controller.on_backfill_missing_clicked)
        self.indicators_btn.clicked.connect(self.chart_controller.on_indicators_clicked)
        self.build_latents_btn.clicked.connect(self.chart_controller.on_build_latents_clicked)

        # Mouse interaction:
        # - Alt+Click => TestingPoint basic; Shift+Alt+Click => advanced (già gestito da _on_canvas_click)
        # - Wheel zoom centrato sul mouse
        # - Right button drag: orizzontale = zoom X (tempo); verticale = zoom Y (prezzo)
        self._rbtn_drag = False
        self._drag_last = None  # (x_px, y_px)
        self._drag_axis = None  # "x" | "y" determinata dal movimento prevalente

        try:
            # conserva il gestore esistente per strumenti/testing (left click)
            self.canvas.mpl_connect("button_press_event", self.chart_controller.on_canvas_click)
            # nuovi handler UX (pan/zoom)
            self.canvas.mpl_connect("button_press_event", self.chart_controller.on_mouse_press)
            self.canvas.mpl_connect("button_release_event", self.chart_controller.on_mouse_release)
            self.canvas.mpl_connect("motion_notify_event", self.chart_controller.on_mouse_move)
            self.canvas.mpl_connect("scroll_event", self.chart_controller.on_scroll_zoom)
        except Exception:
            logger.debug("Failed to connect mpl mouse events for zoom/pan.")
        # registry for adherence badges
        self._adh_badges: list = []

        # --- Signal Connections ---
        self.forecast_settings_btn.clicked.connect(self.chart_controller.open_forecast_settings)
        self.forecast_btn.clicked.connect(self.chart_controller.on_forecast_clicked)
        self.adv_settings_btn.clicked.connect(self.chart_controller.open_adv_forecast_settings)
        self.adv_forecast_btn.clicked.connect(self.chart_controller.on_advanced_forecast_clicked)
        self.clear_forecasts_btn.clicked.connect(self.chart_controller.clear_all_forecasts)
        # Symbol change
        self.symbol_combo.currentTextChanged.connect(self.chart_controller.on_symbol_changed)

        # try to auto-connect to controller signals if available
        try:
            controller = getattr(self._main_window, "controller", None)
            if controller and hasattr(controller, "signals"):
                controller.signals.forecastReady.connect(self.on_forecast_ready)
        except Exception:
            pass

    def _handle_tick(self, payload: dict):
        """Thread-safe entrypoint: enqueue tick to GUI thread."""
        return self.chart_controller.handle_tick(payload=payload)

    def _on_tick_main(self, payload: dict):
        """GUI-thread handler: aggiorna Market Watch e buffer, delega il redraw al throttler."""
        return self.chart_controller.on_tick_main(payload=payload)

    def _rt_flush(self):
        """Throttled redraw preserving zoom/pan."""
        return self.chart_controller.rt_flush()

    def _set_drawing_mode(self, mode: Optional[str]):
        return self.chart_controller.set_drawing_mode(mode=mode)

    def _on_canvas_click(self, event):
        return self.chart_controller.on_canvas_click(event=event)

    def _open_trade_dialog(self):
        return self.chart_controller.open_trade_dialog()

    def _on_indicators_clicked(self):
        """Chiede al controller di aprire il dialog; in fallback mostra un info box."""
        return self.chart_controller.on_indicators_clicked()

    def _get_indicator_settings(self) -> dict:
        """Prende la config indicatori dal controller (se c’è) o dalla sessione salvata."""
        return self.chart_controller.get_indicator_settings()

    def _ensure_osc_axis(self, need: bool):
        """Crea/mostra o nasconde l’asse “oscillatori” (inset sotto il main)."""
        return self.chart_controller.ensure_osc_axis(need=need)

    def _on_main_xlim_changed(self, ax):
        """Sync oscillator inset x-limits to main axis."""
        return self.chart_controller.on_main_xlim_changed(ax=ax)

    # ---- math helpers ----
    def _sma(self, x: pd.Series, n: int) -> pd.Series:
        return self.chart_controller.sma(x=x, n=n)

    def _ema(self, x: pd.Series, n: int) -> pd.Series:
        return self.chart_controller.ema(x=x, n=n)

    def _bollinger(self, x: pd.Series, n: int, k: float):
        return self.chart_controller.bollinger(x=x, n=n, k=k)

    def _donchian(self, high: pd.Series, low: pd.Series, n: int):
        return self.chart_controller.donchian(high=high, low=low, n=n)

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int):
        return self.chart_controller.atr(high=high, low=low, close=close, n=n)

    def _keltner(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int, k: float):
        return self.chart_controller.keltner(high=high, low=low, close=close, n=n, k=k)

    def _rsi(self, x: pd.Series, n: int):
        return self.chart_controller.rsi(x=x, n=n)

    def _macd(self, x: pd.Series, fast: int, slow: int, signal: int):
        return self.chart_controller.macd(x=x, fast=fast, slow=slow, signal=signal)

    def _hurst_roll(self, x: pd.Series, window: int):
        return self.chart_controller.hurst_roll(x=x, window=window)

    def _plot_indicators(self, df2: pd.DataFrame, x_dt: pd.Series):
        return self.chart_controller.plot_indicators(df2=df2, x_dt=x_dt)

    def _on_build_latents_clicked(self):
        """Prompt PCA dim and launch latents build via controller."""
        return self.chart_controller.on_build_latents_clicked()

    def _refresh_orders(self):
        """Pull open orders from broker and refresh the table."""
        return self.chart_controller.refresh_orders()

    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None, restore_xlim=None, restore_ylim=None):
        return self.chart_controller.update_plot(df=df, quantiles=quantiles, restore_xlim=restore_xlim, restore_ylim=restore_ylim)
    def _on_mode_toggled(self, checked: bool):
        """Manual switch placeholder (candles disabled)."""
        return self.chart_controller.on_mode_toggled(checked=checked)

    def _render_candles(self, df2: pd.DataFrame):
        """Disabled candle renderer (rollback)."""
        return self.chart_controller.render_candles(df2=df2)

    # --- Theme helpers ---
    def _on_nav_home(self):
        return self.chart_controller.on_nav_home()

    def _on_nav_pan(self, checked: bool):
        return self.chart_controller.on_nav_pan(checked=checked)

    def _on_nav_zoom(self, checked: bool):
        return self.chart_controller.on_nav_zoom(checked=checked)

    def _apply_theme(self, theme: str):
        return self.chart_controller.apply_theme(theme=theme)

    def _get_color(self, key: str, default: str) -> str:
        return self.chart_controller.get_color(key=key, default=default)

    def _open_color_settings(self):
        return self.chart_controller.open_color_settings()

    def _toggle_drawbar(self, visible: bool):
        return self.chart_controller.toggle_drawbar(visible=visible)

    def _toggle_orders(self, visible: bool):
        return self.chart_controller.toggle_orders(visible=visible)

    def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"):
        """Plot quantiles on the chart."""
        return self.chart_controller.plot_forecast_overlay(quantiles=quantiles, source=source)

    def _tf_to_timedelta(self, tf: str):
        """Convert timeframe string like '1m','5m','1h' into pandas Timedelta."""
        return self.chart_controller.tf_to_timedelta(tf=tf)

    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        return self.chart_controller.set_symbol_timeframe(db_service=db_service, symbol=symbol, timeframe=timeframe)

    def _on_symbol_changed(self, new_symbol: str):
        """Handle symbol change from combo: update context and reload candles from DB."""
        return self.chart_controller.on_symbol_changed(new_symbol=new_symbol)

    def _open_forecast_settings(self):
        return self.chart_controller.open_forecast_settings()

    def _on_canvas_click(self, event):
        """Gestisce strumenti di disegno e click di testing."""
        return self.chart_controller.on_canvas_click(event=event)

    # --- Mouse UX: zoom con rotellina e tasto destro (drag) ---
    def _on_scroll_zoom(self, event):
        return self.chart_controller.on_scroll_zoom(event=event)

    def _on_mouse_press(self, event):
        return self.chart_controller.on_mouse_press(event=event)

    def _on_mouse_move(self, event):
        return self.chart_controller.on_mouse_move(event=event)

    def _on_mouse_release(self, event):
        return self.chart_controller.on_mouse_release(event=event)

    def _zoom_axis(self, axis: str, center: float, factor: float):
        """Zoom helper: scala i limiti attorno a 'center' con 'factor' (factor<1=zoom-in)."""
        return self.chart_controller.zoom_axis(axis=axis, center=center, factor=factor)

    def _update_badge_visibility(self, event):
        """Hide badges when cursor is inside a rectangle centered on the badge with 2x width/height; show otherwise."""
        return self.chart_controller.update_badge_visibility(event=event)

    def _schedule_view_reload(self):
        """Throttle view-window reload after user interaction."""
        return self.chart_controller.schedule_view_reload()

    def _resolution_for_span(self, ms_span: int) -> str:
        """Pick best timeframe by visible span (ms)."""
        return self.chart_controller.resolution_for_span(ms_span=ms_span)

    def _reload_view_window(self):
        """Reload only data covering [view_left .. view_right] plus one span of history."""
        return self.chart_controller.reload_view_window()

    def _on_forecast_clicked(self):
        return self.chart_controller.on_forecast_clicked()

    def _open_adv_forecast_settings(self):
        return self.chart_controller.open_adv_forecast_settings()

    def _on_advanced_forecast_clicked(self):
        return self.chart_controller.on_advanced_forecast_clicked()

    def on_forecast_ready(self, df: pd.DataFrame, quantiles: dict):
        """
                Slot to receive forecast results from controller/worker.
                Adds the forecast overlay without removing existing ones (trimming oldest if needed).
        """
        return self.chart_controller.on_forecast_ready(df=df, quantiles=quantiles)

    def clear_all_forecasts(self):
        """Remove all forecast artists from axes and clear internal list."""
        return self.chart_controller.clear_all_forecasts()

    def _trim_forecasts(self):
        """Disabled: keep all forecast overlays visible."""
        return self.chart_controller.trim_forecasts()

    def start_auto_forecast(self):
        return self.chart_controller.start_auto_forecast()

    def stop_auto_forecast(self):
        return self.chart_controller.stop_auto_forecast()

    def _auto_forecast_tick(self):
        """Called by timer: trigger both basic and advanced forecasts (emit signals)."""
        return self.chart_controller.auto_forecast_tick()

    def _on_backfill_missing_clicked(self):
        """Trigger backfill for current symbol/timeframe asynchronously with determinate progress."""
        return self.chart_controller.on_backfill_missing_clicked()

    def _load_candles_from_db(self, symbol: str, timeframe: str, limit: int = 5000, start_ms: Optional[int] = None, end_ms: Optional[int] = None):
        """Load data from DB for symbol/timeframe, optionally constraining [start_ms, end_ms].
                - If timeframe == 'tick': read from market_data_ticks and map to (ts_utc, price).
                - Else: read candles from market_data_candles as before.
        """
        return self.chart_controller.load_candles_from_db(symbol=symbol, timeframe=timeframe, limit=limit, start_ms=start_ms, end_ms=end_ms)

