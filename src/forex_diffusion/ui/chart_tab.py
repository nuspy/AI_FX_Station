# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox,
    QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QTableWidgetItem,
    QSplitter, QListWidget, QListWidgetItem, QTableWidget, QComboBox
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QTimer, Qt, Signal, QSize, QFile
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
        self._price_mode = "candles"   # default mode
        self._price_line = None
        self._candle_artists: list = []
        self.chart_controller = ChartTabController(self, self.controller)

        self._build_ui()
        self._init_control_defaults()

        try:
            self.toggle_drawbar_btn.toggled.connect(self._toggle_drawbar)
        except Exception:
            pass
        try:
            self.tb_home.clicked.connect(self._on_nav_home)
            self.tb_pan.clicked.connect(self._on_nav_pan)
            self.tb_zoom.clicked.connect(self._on_nav_zoom)
        except Exception:
            pass
        try:
            self.tb_cross.clicked.connect(lambda: self._set_drawing_mode(None))
            self.tb_hline.clicked.connect(lambda: self._set_drawing_mode("hline"))
            self.tb_trend.clicked.connect(lambda: self._set_drawing_mode("trend"))
            self.tb_rect.clicked.connect(lambda: self._set_drawing_mode("rect"))
            self.tb_fib.clicked.connect(lambda: self._set_drawing_mode("fib"))
            self.tb_label.clicked.connect(lambda: self._set_drawing_mode("label"))
        except Exception:
            pass

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


    def _build_ui(self) -> None:
        "Use the original programmatic layout for the chart tab."
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.toolbar = NavigationToolbar(self.canvas, self)

        topbar = QWidget()
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(4, 1, 4, 1)
        top_layout.setSpacing(4)
        try:
            self.toolbar.setIconSize(QSize(16, 16))
            self.toolbar.setStyleSheet("QToolBar{spacing:2px; padding:0px; margin:0px;}")
            self.toolbar.setMovable(False)
            self.toolbar.setFloatable(False)
            self.toolbar.setMaximumHeight(26)
        except Exception:
            pass
        topbar.setStyleSheet("QPushButton, QComboBox, QToolButton { padding: 2px 6px; min-height: 22px; }; QLabel { padding: 0px; margin: 0px; }")
        topbar.setMaximumHeight(32)
        top_layout.addWidget(self.toolbar)

        self.symbol_combo = QComboBox()
        top_layout.addWidget(self.symbol_combo)

        top_layout.addWidget(QLabel("TF:"))
        self.tf_combo = QComboBox()
        top_layout.addWidget(self.tf_combo)
        self.tf_used_label = QLabel("TF used: -")
        top_layout.addWidget(self.tf_used_label)

        self.forecast_settings_btn = QPushButton("Prediction Settings")
        self.forecast_btn = QPushButton("Make Prediction")
        top_layout.addWidget(self.forecast_settings_btn)
        top_layout.addWidget(self.forecast_btn)

        top_layout.addWidget(QLabel("Pred step:"))
        self.pred_step_combo = QComboBox()
        top_layout.addWidget(self.pred_step_combo)

        self.adv_settings_btn = QPushButton("Advanced Settings")
        self.adv_forecast_btn = QPushButton("Advanced Forecast")
        top_layout.addWidget(self.adv_settings_btn)
        top_layout.addWidget(self.adv_forecast_btn)

        from PySide6.QtWidgets import QToolButton

        self.toggle_drawbar_btn = QToolButton()
        self.toggle_drawbar_btn.setText("Draw")
        self.toggle_drawbar_btn.setCheckable(True)
        self.toggle_drawbar_btn.setChecked(True)
        self.toggle_drawbar_btn.setToolTip("Mostra/Nascondi barra di disegno")
        top_layout.addWidget(self.toggle_drawbar_btn)

        top_layout.addWidget(QLabel("Years:"))
        self.years_combo = QComboBox()
        top_layout.addWidget(self.years_combo)
        top_layout.addWidget(QLabel("Months:"))
        self.months_combo = QComboBox()
        top_layout.addWidget(self.months_combo)

        self.backfill_btn = QPushButton("Backfill")
        self.backfill_btn.setToolTip(
            "Scarica storico per il range selezionato (Years/Months) per il simbolo corrente"
        )
        top_layout.addWidget(self.backfill_btn)

        self.indicators_btn = QPushButton("Indicators")
        top_layout.addWidget(self.indicators_btn)

        self.build_latents_btn = QPushButton("Build Latents")
        top_layout.addWidget(self.build_latents_btn)

        from PySide6.QtWidgets import QProgressBar

        self.backfill_progress = QProgressBar()
        self.backfill_progress.setMaximumWidth(160)
        self.backfill_progress.setRange(0, 100)
        self.backfill_progress.setValue(0)
        self.backfill_progress.setTextVisible(False)
        top_layout.addWidget(self.backfill_progress)

        self.clear_forecasts_btn = QPushButton("Clear Forecasts")
        top_layout.addWidget(self.clear_forecasts_btn)

        self.mode_btn = QToolButton()
        self.mode_btn.setCheckable(True)
        self.mode_btn.setText("Candles")
        self.mode_btn.setToolTip("Commuta visualizzazione prezzo: Candles (candele OHLC) / Line (linea).")
        self.mode_btn.setChecked(True)
        top_layout.addWidget(self.mode_btn)

        top_layout.addStretch()
        self.bidask_label = QLabel("Bid: -    Ask: -")
        self.bidask_label.setStyleSheet("font-weight: bold;")
        top_layout.addWidget(self.bidask_label)

        self.trade_btn = QPushButton("Trade")
        top_layout.addWidget(self.trade_btn)

        self.theme_combo = QComboBox()
        top_layout.addWidget(QLabel("Theme:"))
        top_layout.addWidget(self.theme_combo)

        self.layout.addWidget(topbar)

        self._drawbar = self._create_drawbar()

        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter
        self.market_watch = QListWidget()
        splitter.addWidget(self.market_watch)

        right_vert = QSplitter(Qt.Vertical)
        self.right_splitter = right_vert
        chart_area = QSplitter(Qt.Vertical)
        chart_area.setHandleWidth(4)
        chart_area.setOpaqueResize(True)
        self._chart_area = chart_area
        chart_area.addWidget(self._drawbar)

        chart_wrap = QWidget()
        chart_layout = QVBoxLayout(chart_wrap)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(self.canvas)
        chart_area.addWidget(chart_wrap)

        right_vert.addWidget(chart_area)
        self.orders_table = QTableWidget(0, 9)
        self.orders_table.setHorizontalHeaderLabels([
            "ID",
            "Time",
            "Symbol",
            "Type",
            "Volume",
            "Price",
            "SL",
            "TP",
            "Status",
        ])
        right_vert.addWidget(self.orders_table)
        splitter.addWidget(right_vert)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        splitter.setHandleWidth(6)
        splitter.setOpaqueResize(True)
        right_vert.setHandleWidth(6)
        right_vert.setOpaqueResize(True)

        self.layout.addWidget(splitter)


    def _init_control_defaults(self) -> None:
        """Populate combo boxes and default UI state shared across layouts."""
        self._symbols_supported = [
            "EUR/USD",
            "GBP/USD",
            "AUX/USD",
            "GBP/NZD",
            "AUD/JPY",
            "GBP/EUR",
            "GBP/AUD",
        ]
        try:
            self.symbol_combo.clear()
            self.symbol_combo.addItems(self._symbols_supported)
        except Exception:
            pass
    
        try:
            self.tf_combo.clear()
            self.tf_combo.addItems(["auto", "tick", "1m", "5m", "15m", "30m", "1h", "4h", "1d"])
            self.tf_combo.setCurrentText("auto")
            self.tf_combo.setToolTip("Seleziona il timeframe di visualizzazione. 'auto' sceglie il TF in base allo zoom.")
        except Exception:
            pass
    
        try:
            self.pred_step_combo.clear()
            self.pred_step_combo.addItems(["auto", "1m"])
            self.pred_step_combo.setCurrentText("auto")
            self.pred_step_combo.setToolTip(
                "Granularità dei punti di previsione. 'auto' usa gli horizons, '1m' aggiunge punti a minuto."
            )
        except Exception:
            pass
    
        try:
            self.years_combo.clear()
            self.years_combo.addItems([str(x) for x in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30]])
            self.years_combo.setCurrentText("0")
        except Exception:
            pass
    
        try:
            self.months_combo.clear()
            self.months_combo.addItems([str(x) for x in range(0, 13)])
            self.months_combo.setCurrentText("0")
        except Exception:
            pass
    
        try:
            self.theme_combo.clear()
            self.theme_combo.addItems(["Dark", "Light"])
            self.theme_combo.setCurrentText(get_setting("ui_theme", "Dark"))
        except Exception:
            pass
    
        try:
            self.market_watch.clear()
            for symbol in self._symbols_supported:
                QListWidgetItem(f"{symbol}  -", self.market_watch)
        except Exception:
            pass
    
        try:
            self.toggle_drawbar_btn.setChecked(True)
        except Exception:
            pass
    
        try:
            self.mode_btn.setText("Linea" if self.mode_btn.isChecked() else "Candles")
        except Exception:
            pass
    
        try:
            self.backfill_progress.setRange(0, 100)
            self.backfill_progress.setValue(0)
        except Exception:
            pass

    def _create_drawbar(self) -> QWidget:
        from PySide6.QtWidgets import QToolButton, QStyle

        drawbar = QWidget()
        dlay = QHBoxLayout(drawbar)
        dlay.setContentsMargins(4, 2, 4, 2)

        self.tb_home = QToolButton()
        self.tb_home.setToolTip("Reset view")
        self.tb_home.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.tb_pan = QToolButton()
        self.tb_pan.setToolTip("Pan")
        self.tb_pan.setCheckable(True)
        self.tb_pan.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.tb_zoom = QToolButton()
        self.tb_zoom.setToolTip("Zoom")
        self.tb_zoom.setCheckable(True)
        self.tb_zoom.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        for b in (self.tb_home, self.tb_pan, self.tb_zoom):
            dlay.addWidget(b)
        dlay.addSpacing(12)

        self.tb_cross = QToolButton()
        self.tb_cross.setToolTip("Cross")
        self.tb_cross.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.tb_hline = QToolButton()
        self.tb_hline.setToolTip("H-Line")
        self.tb_hline.setIcon(self.style().standardIcon(QStyle.SP_TitleBarShadeButton))
        self.tb_trend = QToolButton()
        self.tb_trend.setToolTip("Trend")
        self.tb_trend.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.tb_rect = QToolButton()
        self.tb_rect.setToolTip("Rect")
        self.tb_rect.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.tb_fib = QToolButton()
        self.tb_fib.setToolTip("Fib")
        self.tb_fib.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.tb_label = QToolButton()
        self.tb_label.setToolTip("Label")
        self.tb_label.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        self.tb_colors = QToolButton()
        self.tb_colors.setToolTip("Color/CSS Settings")
        self.tb_colors.setIcon(self.style().standardIcon(QStyle.SP_DriveDVDIcon))
        self.tb_colors.clicked.connect(self._open_color_settings)
        self.tb_orders = QToolButton()
        self.tb_orders.setToolTip("Show/Hide Orders")
        self.tb_orders.setCheckable(True)
        self.tb_orders.setChecked(True)
        self.tb_orders.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        self.tb_orders.toggled.connect(self._toggle_orders)

        for b in (
            self.tb_cross,
            self.tb_hline,
            self.tb_trend,
            self.tb_rect,
            self.tb_fib,
            self.tb_label,
            self.tb_colors,
            self.tb_orders,
        ):
            dlay.addWidget(b)
        dlay.addStretch()
        return drawbar
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




class _ChartTabUiLoader(QUiLoader):
    def __init__(self, baseinstance: QWidget) -> None:
        super().__init__()
        self._baseinstance = baseinstance

    def createWidget(self, className: str, parent: QWidget | None = None, name: str = "") -> QWidget:
        if parent is None and self._baseinstance is not None:
            return self._baseinstance
        widget = super().createWidget(className, parent, name)
        if self._baseinstance is not None and name:
            setattr(self._baseinstance, name, widget)
        return widget


class ChartTabUI(ChartTab):
    """Chart tab backed by a Qt Designer .ui layout."""

    UI_FILE = Path(__file__).with_name("chart_tab_ui.ui")

    def _build_ui(self) -> None:
        self._load_designer_ui()

        main_layout = self.layout()
        if main_layout is not None:
            self.layout = main_layout

        toolbar_container = getattr(self, "toolbar_placeholder", None)
        chart_container = getattr(self, "chart_container", None)
        drawbar_container = getattr(self, "drawbar_container", None)

        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        if chart_container is not None:
            chart_layout = chart_container.layout()
            if chart_layout is None:
                chart_layout = QVBoxLayout(chart_container)
            chart_layout.setContentsMargins(0, 0, 0, 0)
            chart_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        if toolbar_container is not None:
            toolbar_layout = toolbar_container.layout()
            if toolbar_layout is None:
                toolbar_layout = QHBoxLayout(toolbar_container)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)
            toolbar_layout.addWidget(self.toolbar)

        try:
            self.toolbar.setIconSize(QSize(16, 16))
            self.toolbar.setStyleSheet("QToolBar{spacing:2px; padding:0px; margin:0px;}")
            self.toolbar.setMovable(False)
            self.toolbar.setFloatable(False)
            self.toolbar.setMaximumHeight(26)
        except Exception:
            pass

        topbar = getattr(self, "topbar", None)
        if topbar is not None:
            topbar.setMaximumHeight(32)
            topbar.setStyleSheet("QPushButton, QComboBox, QToolButton { padding: 2px 6px; min-height: 22px; }; QLabel { padding: 0px; margin: 0px; }")

        self._drawbar = self._create_drawbar()
        if drawbar_container is not None:
            drawbar_layout = drawbar_container.layout()
            if drawbar_layout is None:
                drawbar_layout = QHBoxLayout(drawbar_container)
            while drawbar_layout.count():
                item = drawbar_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            drawbar_layout.addWidget(self._drawbar)

        self.main_splitter = getattr(self, "main_splitter", None)
        self.right_splitter = getattr(self, "right_splitter", None)
        self._chart_area = getattr(self, "chart_area_splitter", None)
        self.market_watch = getattr(self, "market_watch", None)
        self.orders_table = getattr(self, "orders_table", None)
        self.symbol_combo = getattr(self, "symbol_combo", None)
        self.tf_combo = getattr(self, "tf_combo", None)
        self.tf_used_label = getattr(self, "tf_used_label", None)
        self.forecast_settings_btn = getattr(self, "forecast_settings_btn", None)
        self.forecast_btn = getattr(self, "forecast_btn", None)
        self.pred_step_combo = getattr(self, "pred_step_combo", None)
        self.adv_settings_btn = getattr(self, "adv_settings_btn", None)
        self.adv_forecast_btn = getattr(self, "adv_forecast_btn", None)
        self.toggle_drawbar_btn = getattr(self, "toggle_drawbar_btn", None)
        self.years_combo = getattr(self, "years_combo", None)
        self.months_combo = getattr(self, "months_combo", None)
        self.backfill_btn = getattr(self, "backfill_btn", None)
        self.indicators_btn = getattr(self, "indicators_btn", None)
        self.build_latents_btn = getattr(self, "build_latents_btn", None)
        self.backfill_progress = getattr(self, "backfill_progress", None)
        self.clear_forecasts_btn = getattr(self, "clear_forecasts_btn", None)
        self.mode_btn = getattr(self, "mode_btn", None)
        self.bidask_label = getattr(self, "bidask_label", None)
        self.trade_btn = getattr(self, "trade_btn", None)
        self.theme_combo = getattr(self, "theme_combo", None)

        if self.main_splitter is not None:
            self.main_splitter.setStretchFactor(0, 1)
            self.main_splitter.setStretchFactor(1, 5)
            self.main_splitter.setHandleWidth(6)
            self.main_splitter.setOpaqueResize(True)
        if self.right_splitter is not None:
            self.right_splitter.setHandleWidth(6)
            self.right_splitter.setOpaqueResize(True)
        if self._chart_area is not None:
            self._chart_area.setHandleWidth(4)
            self._chart_area.setOpaqueResize(True)

    def _load_designer_ui(self) -> None:
        ui_path = self.UI_FILE
        if not ui_path.exists():
            raise FileNotFoundError(f"UI file not found: {ui_path}")
        loader = _ChartTabUiLoader(self)
        ui_file = QFile(str(ui_path))
        if not ui_file.open(QFile.ReadOnly):
            raise IOError(f"Unable to open UI file: {ui_path}")
        loader.load(ui_file, self)
        ui_file.close()
