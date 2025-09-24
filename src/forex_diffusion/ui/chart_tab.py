# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import time

import matplotlib.dates as mdates

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox,
    QSplitter, QTableWidget, QComboBox,
    QToolButton, QCheckBox, QTableWidgetItem, QAbstractItemView, QHeaderView
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QTimer, Qt, Signal, QSize, QFile, QSignalBlocker
from PySide6.QtGui import QBrush, QColor, QCursor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
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
        self.setObjectName("chartTab")
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
        self._connect_ui_signals()

        self._hover_legend = None
        self._hover_legend_text = None

        theme_combo = getattr(self, "theme_combo", None)
        if theme_combo is not None:
            self._apply_theme(theme_combo.currentText())

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
            self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
            # nuovi handler UX (pan/zoom)
            self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
            self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
            self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
            self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
            self.canvas.mpl_connect("figure_enter_event", self._on_figure_enter)
            self.canvas.mpl_connect("figure_leave_event", self._on_figure_leave)
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
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.toolbar = NavigationToolbar(self.canvas, self)

        topbar = QWidget()
        topbar.setObjectName("chartTopbar")
        self.topbar = topbar
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
        # Pattern toggles
        self.chart_patterns_checkbox = QCheckBox("Chart patterns")
        self.candle_patterns_checkbox = QCheckBox("Candlestick patterns")
        self.history_patterns_checkbox = QCheckBox("Patterns storici")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setText("Candles")
        self.mode_btn.setToolTip("Commuta visualizzazione prezzo: Candles (candele OHLC) / Line (linea).")
        self.mode_btn.setChecked(True)
        top_layout.addWidget(self.mode_btn)
        top_layout.addWidget(self.chart_patterns_checkbox)
        top_layout.addWidget(self.candle_patterns_checkbox)
        top_layout.addWidget(self.history_patterns_checkbox)

        self.follow_checkbox = QCheckBox("Segui")
        top_layout.addWidget(self.follow_checkbox)

        self.bidask_label = QLabel("Bid: -    Ask: -")
        self.bidask_label.setStyleSheet("font-weight: bold;")
        top_layout.addWidget(self.bidask_label)

        self.trade_btn = QPushButton("Trade")
        top_layout.addWidget(self.trade_btn)

        self.theme_combo = QComboBox()
        top_layout.addWidget(QLabel("Theme:"))
        top_layout.addWidget(self.theme_combo)

        self.settings_btn = QPushButton("Settings")
        top_layout.addWidget(self.settings_btn)

        top_layout.addStretch()

        self.layout.addWidget(topbar)

        self._drawbar = self._create_drawbar()

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("main_splitter")
        self.main_splitter = splitter

        self.market_watch = QTableWidget()
        self.market_watch.setObjectName("market_watch")
        self.market_watch.setColumnCount(3)
        self.market_watch.setHorizontalHeaderLabels(["Symbol", "Bid", "Offer"])
        self.market_watch.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.market_watch.setSelectionMode(QAbstractItemView.NoSelection)
        self.market_watch.setFocusPolicy(Qt.NoFocus)
        self.market_watch.setShowGrid(False)
        header = self.market_watch.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.market_watch.verticalHeader().setVisible(False)
        self.market_watch.setToolTip(
            "Offer/Bid: verde quando lo spread aumenta, rosso quando diminuisce, "
            "nero dopo 10 aggiornamenti senza variazione."
        )
        splitter.addWidget(self.market_watch)

        right_vert = QSplitter(Qt.Vertical)
        right_vert.setObjectName("right_splitter")
        self.right_splitter = right_vert
        chart_area = QSplitter(Qt.Vertical)
        chart_area.setObjectName("chart_area_splitter")
        chart_area.setHandleWidth(4)
        chart_area.setOpaqueResize(True)
        self._chart_area = chart_area
        chart_area.addWidget(self._drawbar)

        chart_wrap = QWidget()
        chart_wrap.setObjectName("chart_container")
        chart_layout = QVBoxLayout(chart_wrap)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)
        chart_layout.addWidget(self.canvas)
        chart_area.addWidget(chart_wrap)
        chart_area.setStretchFactor(0, 0)
        chart_area.setStretchFactor(1, 1)

        right_vert.addWidget(chart_area)
        right_vert.setStretchFactor(0, 6)

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
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        splitter.setHandleWidth(6)
        splitter.setOpaqueResize(True)
        right_vert.setHandleWidth(6)
        right_vert.setOpaqueResize(True)

        self.layout.addWidget(splitter)


    def _init_control_defaults(self) -> None:
        """Populate UI controls and restore persisted settings."""
        self._symbols_supported = [
            "EUR/USD",
            "GBP/USD",
            "AUX/USD",
            "GBP/NZD",
            "AUD/JPY",
            "GBP/EUR",
            "GBP/AUD",
        ]

        self._symbol_row_map: Dict[str, int] = {}
        self._spread_state: Dict[str, Dict[str, float]] = {}
        self._last_bidask: Dict[str, Dict[str, float]] = {}

        self._follow_suspend_seconds = float(get_setting('chart.follow_suspend_seconds', 30))
        self._follow_enabled = bool(get_setting('chart.follow_enabled', False))
        self._follow_suspend_until = 0.0

        follow_checkbox = getattr(self, 'follow_checkbox', None)
        if follow_checkbox is not None:
            blocker = QSignalBlocker(follow_checkbox)
            follow_checkbox.setChecked(self._follow_enabled)
            del blocker
        else:
            self._follow_enabled = False

        self._set_combo_with_items(
            getattr(self, "symbol_combo", None),
            self._symbols_supported,
            "chart.symbol",
            self._symbols_supported[0],
        )

        self._set_combo_with_items(
            getattr(self, "tf_combo", None),
            ["auto", "tick", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            "chart.timeframe",
            "auto",
        )

        self._set_combo_with_items(
            getattr(self, "pred_step_combo", None),
            ["auto", "1m"],
            "chart.pred_step",
            "auto",
        )

        self._set_combo_with_items(
            getattr(self, "years_combo", None),
            [str(x) for x in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30]],
            "chart.backfill_years",
            "0",
        )

        self._set_combo_with_items(
            getattr(self, "months_combo", None),
            [str(x) for x in range(0, 13)],
            "chart.backfill_months",
            "0",
        )

        theme_default = get_setting("ui_theme", "Dark")
        self._set_combo_with_items(
            getattr(self, "theme_combo", None),
            ["System", "Dark", "Light", "Custom"],
            "chart.theme",
            str(theme_default),
        )

        market_watch = getattr(self, "market_watch", None)
        if isinstance(market_watch, QTableWidget):
            blocker = QSignalBlocker(market_watch)
            market_watch.setRowCount(len(self._symbols_supported))
            for idx, sym in enumerate(self._symbols_supported):
                symbol_item = QTableWidgetItem(sym)
                symbol_item.setFlags(Qt.ItemIsEnabled)
                market_watch.setItem(idx, 0, symbol_item)

                bid_item = QTableWidgetItem("-")
                bid_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                market_watch.setItem(idx, 1, bid_item)

                ask_item = QTableWidgetItem("-")
                ask_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                market_watch.setItem(idx, 2, ask_item)

                self._symbol_row_map[sym] = idx
                self._spread_state[sym] = {
                    "prev_spread": None,
                    "color": "#000000",
                    "hold": 0,
                }
            del blocker

        toggle_drawbar = getattr(self, "toggle_drawbar_btn", None)
        if toggle_drawbar is not None:
            blocker = QSignalBlocker(toggle_drawbar)
            toggle_drawbar.setChecked(True)
            del blocker

        saved_mode = str(get_setting("chart.price_mode", "candles")).lower()
        if saved_mode not in ("candles", "line"):
            saved_mode = "candles"
        self._price_mode = saved_mode
        mode_btn = getattr(self, "mode_btn", None)
        if mode_btn is not None:
            blocker = QSignalBlocker(mode_btn)
            mode_btn.setChecked(self._price_mode == "candles")
            mode_btn.setText("Candles" if self._price_mode == "candles" else "Line")
            del blocker

        backfill_progress = getattr(self, "backfill_progress", None)
        if backfill_progress is not None:
            backfill_progress.setRange(0, 100)
            backfill_progress.setValue(0)

        self._restore_splitters()

        symbol_combo = getattr(self, "symbol_combo", None)
        self.symbol = symbol_combo.currentText() if symbol_combo is not None else self._symbols_supported[0]
        tf_combo = getattr(self, "tf_combo", None)
        self.timeframe = tf_combo.currentText() if tf_combo is not None else "auto"

    def _connect_ui_signals(self) -> None:
        toggle_drawbar = getattr(self, "toggle_drawbar_btn", None)
        if toggle_drawbar is not None:
            toggle_drawbar.toggled.connect(self._toggle_drawbar)

        mode_btn = getattr(self, "mode_btn", None)
        if mode_btn is not None:
            mode_btn.toggled.connect(self._on_price_mode_toggled)
        # Pattern toggles
        cpp = getattr(self, 'chart_patterns_checkbox', None)
        if cpp is not None:
            cpp.toggled.connect(lambda v: getattr(self.chart_controller, 'patterns_service').set_chart_enabled(v))
        cdp = getattr(self, 'candle_patterns_checkbox', None)
        if cdp is not None:
            cdp.toggled.connect(lambda v: getattr(self.chart_controller, 'patterns_service').set_candle_enabled(v))
        hsp = getattr(self, 'history_patterns_checkbox', None)
        if hsp is not None:
            hsp.toggled.connect(lambda v: getattr(self.chart_controller, 'patterns_service').set_history_enabled(v))

        follow_checkbox = getattr(self, "follow_checkbox", None)
        if follow_checkbox is not None:
            follow_checkbox.toggled.connect(self._on_follow_toggled)

        symbol_combo = getattr(self, "symbol_combo", None)
        if symbol_combo is not None:
            symbol_combo.currentTextChanged.connect(self._on_symbol_combo_changed)

        tf_combo = getattr(self, "tf_combo", None)
        if tf_combo is not None:
            tf_combo.currentTextChanged.connect(self._on_timeframe_changed)

        pred_combo = getattr(self, "pred_step_combo", None)
        if pred_combo is not None:
            pred_combo.currentTextChanged.connect(self._on_pred_step_changed)

        years_combo = getattr(self, "years_combo", None)
        if years_combo is not None:
            years_combo.currentTextChanged.connect(self._on_backfill_range_changed)

        months_combo = getattr(self, "months_combo", None)
        if months_combo is not None:
            months_combo.currentTextChanged.connect(self._on_backfill_range_changed)

        theme_combo = getattr(self, "theme_combo", None)
        if theme_combo is not None:
            theme_combo.currentTextChanged.connect(self._on_theme_changed)

        settings_btn = getattr(self, "settings_btn", None)
        if settings_btn is not None:
            settings_btn.clicked.connect(self._open_settings_dialog)

        nav_buttons = (
            (getattr(self, "tb_home", None), self._on_nav_home),
            (getattr(self, "tb_pan", None), self._on_nav_pan),
            (getattr(self, "tb_zoom", None), self._on_nav_zoom),
        )
        for button, handler in nav_buttons:
            if button is not None:
                button.clicked.connect(handler)

        draw_buttons = [
            (getattr(self, "tb_cross", None), None),
            (getattr(self, "tb_hline", None), "hline"),
            (getattr(self, "tb_trend", None), "trend"),
            (getattr(self, "tb_rect", None), "rect"),
            (getattr(self, "tb_fib", None), "fib"),
            (getattr(self, "tb_label", None), "label"),
        ]
        for button, mode in draw_buttons:
            if button is not None:
                button.clicked.connect(lambda checked=False, m=mode: self._set_drawing_mode(m))

        for key, splitter in (
            ('chart.splitter.main', getattr(self, "main_splitter", None)),
            ('chart.splitter.right', getattr(self, "right_splitter", None)),
            ('chart.splitter.chart', getattr(self, "_chart_area", None)),
        ):
            if isinstance(splitter, QSplitter):
                splitter.splitterMoved.connect(lambda _pos, _index, k=key, s=splitter: self._persist_splitter_positions(k, s))

    def _set_combo_with_items(self, combo: Optional[QComboBox], items: List[str], setting_key: str, default: str) -> str:
        if combo is None:
            return default
        saved = str(get_setting(setting_key, default)) if setting_key else default
        blocker = QSignalBlocker(combo)
        combo.clear()
        combo.addItems(items)
        if saved and combo.findText(saved) == -1:
            combo.addItem(saved)
        combo.setCurrentText(saved if saved else default)
        del blocker
        return saved if saved else default

    def _restore_splitters(self) -> None:
        for key, splitter in (
            ('chart.splitter.main', getattr(self, 'main_splitter', None)),
            ('chart.splitter.right', getattr(self, 'right_splitter', None)),
            ('chart.splitter.chart', getattr(self, '_chart_area', None)),
        ):
            if splitter is None:
                continue
            sizes = get_setting(key, None)
            if not isinstance(sizes, (list, tuple)):
                continue
            cleaned: List[int] = []
            for val in sizes:
                try:
                    cleaned.append(int(float(val)))
                except Exception:
                    continue
            if cleaned:
                try:
                    splitter.setSizes(cleaned)
                except Exception:
                    pass

    def _persist_splitter_positions(self, key: str, splitter: QSplitter) -> None:
        try:
            set_setting(key, splitter.sizes())
        except Exception:
            pass

    def _on_symbol_combo_changed(self, new_symbol: str) -> None:
        if not new_symbol:
            return
        set_setting('chart.symbol', new_symbol)
        self.symbol = new_symbol
        self.chart_controller.on_symbol_changed(new_symbol=new_symbol)

    def _on_timeframe_changed(self, value: str) -> None:
        set_setting('chart.timeframe', value)
        self.timeframe = value
        self._schedule_view_reload()

    def _on_pred_step_changed(self, value: str) -> None:
        set_setting('chart.pred_step', value)

    def _on_backfill_range_changed(self, _value: str) -> None:
        years_combo = getattr(self, 'years_combo', None)
        months_combo = getattr(self, 'months_combo', None)
        if years_combo is not None:
            set_setting('chart.backfill_years', years_combo.currentText())
        if months_combo is not None:
            set_setting('chart.backfill_months', months_combo.currentText())

    def _on_theme_changed(self, theme: str) -> None:
        set_setting('chart.theme', theme)
        self._apply_theme(theme)

    def _standardize_color(self, value: str, fallback: str) -> QColor:
        text = (value or fallback) if isinstance(value, str) else fallback
        color = QColor(text)
        if not color.isValid() and isinstance(text, str) and text.lower().startswith("rgba"):
            try:
                rgba = text[text.find("(") + 1:text.find(")")]
                parts = [float(p.strip()) for p in rgba.split(",") if p.strip()]
                if len(parts) == 4:
                    color = QColor(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
            except Exception:
                color = QColor(fallback)
        if not color.isValid():
            color = QColor(fallback)
        return color

    def _update_market_quote(self, symbol: str, bid: Optional[float], ask: Optional[float], ts_ms: Optional[int]) -> None:
        table = getattr(self, 'market_watch', None)
        if not isinstance(table, QTableWidget) or not symbol:
            return

        if symbol not in self._symbol_row_map:
            row = table.rowCount()
            table.insertRow(row)
            symbol_item = QTableWidgetItem(symbol)
            symbol_item.setFlags(Qt.ItemIsEnabled)
            table.setItem(row, 0, symbol_item)
            bid_item = QTableWidgetItem('-')
            bid_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            ask_item = QTableWidgetItem('-')
            ask_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 1, bid_item)
            table.setItem(row, 2, ask_item)
            self._symbol_row_map[symbol] = row
            self._spread_state[symbol] = {'prev_spread': None, 'color': '#000000', 'hold': 0}

        row = self._symbol_row_map[symbol]
        bid_item = table.item(row, 1)
        ask_item = table.item(row, 2)
        if bid_item is None:
            bid_item = QTableWidgetItem('-')
            bid_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 1, bid_item)
        if ask_item is None:
            ask_item = QTableWidgetItem('-')
            ask_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 2, ask_item)

        bid_val = float(bid) if bid is not None else None
        ask_val = float(ask) if ask is not None else None
        bid_item.setText('-' if bid_val is None else f"{bid_val:.5f}")
        ask_item.setText('-' if ask_val is None else f"{ask_val:.5f}")

        spread = None
        if bid_val is not None and ask_val is not None:
            spread = max(0.0, ask_val - bid_val)

        state = self._spread_state.setdefault(symbol, {'prev_spread': None, 'color': '#000000', 'hold': 0})
        up_color = self._standardize_color(str(get_setting('candle_up_color', '#2ecc71')), '#2ecc71')
        down_color = self._standardize_color(str(get_setting('candle_down_color', '#e74c3c')), '#e74c3c')
        neutral_color = QColor('#000000')

        if spread is None:
            state['color'] = '#000000'
            state['hold'] = 0
        else:
            prev_spread = state.get('prev_spread')
            tol = 1e-9
            if prev_spread is None:
                state['color'] = '#000000'
                state['hold'] = 0
            else:
                if spread > prev_spread + tol:
                    state['color'] = up_color.name(QColor.HexRgb)
                    state['hold'] = 0
                elif spread < prev_spread - tol:
                    state['color'] = down_color.name(QColor.HexRgb)
                    state['hold'] = 0
                else:
                    state['hold'] = state.get('hold', 0) + 1
                    if state['hold'] >= 10:
                        state['color'] = '#000000'
                        state['hold'] = 0
            state['prev_spread'] = spread

        qcolor = QColor(state.get('color', '#000000'))
        if not qcolor.isValid():
            qcolor = neutral_color
        brush = QBrush(qcolor)
        bid_item.setForeground(brush)
        ask_item.setForeground(brush)

        self._last_bidask[symbol] = {
            'bid': bid_val,
            'ask': ask_val,
            'ts': ts_ms,
        }

    def _ensure_hover_legend(self) -> None:
        if getattr(self, '_hover_legend', None) is not None:
            return
        dummy = Line2D([], [])
        dummy.set_visible(False)
        legend = Legend(self.ax, [dummy], ["Time: -\nBid: -\nOffer: -"], loc='upper left', frameon=True, fontsize=8)
        legend.set_title('Cursor')
        legend._legend_box.align = "left"
        legend.set_draggable(True)
        texts = legend.get_texts()
        if texts:
            texts[0].set_multialignment('left')
        # Apply unified legend/cursor text color from settings
        try:
            txt_col = self._get_color("legend_text_color", "#cfd6e1")
            for t in legend.get_texts() or []:
                try:
                    t.set_color(txt_col)
                except Exception:
                    pass
            ttl = legend.get_title()
            if ttl:
                try:
                    ttl.set_color(txt_col)
                except Exception:
                    pass
        except Exception:
            pass
        self.ax.add_artist(legend)
        self._hover_legend = legend
        self._hover_legend_text = texts[0] if texts else None
        try:
            self.canvas.draw_idle()
        except Exception:
            self.canvas.draw()

    def _reset_hover_info(self) -> None:
        if getattr(self, '_hover_legend_text', None) is not None:
            self._hover_legend_text.set_text("Time: -\nBid: -\nOffer: -")
            try:
                self.canvas.draw_idle()
            except Exception:
                self.canvas.draw()

    def _update_hover_info(self, event) -> None:
        if event is None or getattr(event, 'inaxes', None) != self.ax:
            return
        self._ensure_hover_legend()
        text_artist = getattr(self, '_hover_legend_text', None)
        if text_artist is None:
            return

        time_str = '-'
        bid_str = '-'
        ask_str = '-'

        if event.xdata is not None and self._last_df is not None and not self._last_df.empty:
            try:
                target_ts = int(mdates.num2date(event.xdata).timestamp() * 1000)
                ts_series = self._last_df['ts_utc'].astype('int64')
                ts_array = ts_series.to_numpy()
                if len(ts_array):
                    idx = int(np.argmin(np.abs(ts_array - target_ts)))
                    row = self._last_df.iloc[idx]
                    ts_val = int(row.get('ts_utc', target_ts))
                    dt = pd.to_datetime(ts_val, unit='ms', utc=True)
                    try:
                        dt = dt.tz_convert(None)
                    except Exception:
                        dt = dt.tz_localize(None) if dt.tzinfo is not None else dt
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    bid_val = row.get('bid', row.get('close', row.get('price')))
                    ask_val = row.get('ask', row.get('close', row.get('price')))
                    if bid_val is not None and not pd.isna(bid_val):
                        bid_str = f"{float(bid_val):.5f}"
                    if ask_val is not None and not pd.isna(ask_val):
                        ask_str = f"{float(ask_val):.5f}"
            except Exception:
                pass

        text_artist.set_text(f"Time: {time_str}\nBid: {bid_str}\nOffer: {ask_str}")
        try:
            self.canvas.draw_idle()
        except Exception:
            self.canvas.draw()

    def _open_settings_dialog(self) -> None:
        try:
            from .settings_dialog import SettingsDialog
        except Exception as exc:
            logger.warning("Unable to import SettingsDialog: {}", exc)
            QMessageBox.warning(self, "Settings", f"Unable to open settings dialog: {exc}")
            return

        dialog = SettingsDialog(self)
        if dialog.exec():
            # Refresh theme / palette in case colors changed
            theme_combo = getattr(self, 'theme_combo', None)
            if theme_combo is not None:
                self._apply_theme(theme_combo.currentText())

            # Refresh follow behaviour in case preferences changed
            try:
                self._follow_suspend_seconds = float(get_setting('chart.follow_suspend_seconds', self._follow_suspend_seconds))
            except Exception:
                self._follow_suspend_seconds = 30.0

            self._follow_enabled = bool(get_setting('chart.follow_enabled', self._follow_enabled))
            if self._follow_enabled:
                self._follow_suspend_until = 0.0

            follow_checkbox = getattr(self, 'follow_checkbox', None)
            if follow_checkbox is not None:
                blocker = QSignalBlocker(follow_checkbox)
                follow_checkbox.setChecked(self._follow_enabled)
                del blocker

            # Re-render plot to apply freshly chosen colors
            if getattr(self, '_last_df', None) is not None and not self._last_df.empty:
                prev_xlim = prev_ylim = None
                try:
                    prev_xlim = self.ax.get_xlim()
                    prev_ylim = self.ax.get_ylim()
                except Exception:
                    pass
                self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)

            self._follow_center_if_needed()

    def _on_price_mode_toggled(self, checked: bool) -> None:
        self._price_mode = 'candles' if checked else 'line'
        mode_btn = getattr(self, 'mode_btn', None)
        if mode_btn is not None:
            mode_btn.setText('Candles' if checked else 'Line')
        set_setting('chart.price_mode', self._price_mode)
        self.chart_controller.on_mode_toggled(checked=checked)

    def _on_follow_toggled(self, checked: bool) -> None:
        self._follow_enabled = bool(checked)
        set_setting('chart.follow_enabled', self._follow_enabled)
        if self._follow_enabled:
            self._follow_suspend_until = 0.0
            self._follow_center_if_needed()

    def _suspend_follow(self) -> None:
        if getattr(self, '_follow_enabled', False):
            duration = float(get_setting('chart.follow_suspend_seconds', getattr(self, '_follow_suspend_seconds', 30)))
            self._follow_suspend_seconds = duration
            self._follow_suspend_until = time.time() + max(duration, 1.0)

    def _follow_center_if_needed(self) -> None:
        if not getattr(self, '_follow_enabled', False):
            return
        if time.time() < getattr(self, '_follow_suspend_until', 0.0):
            return
        if self._last_df is None or self._last_df.empty:
            return
        ax = getattr(self, 'ax', None)
        if ax is None:
            return
        y_col = 'close' if 'close' in self._last_df.columns else 'price'
        try:
            last_row = self._last_df.iloc[-1]
            last_ts = float(last_row['ts_utc'])
            last_price = float(last_row.get(y_col, last_row.get('price')))
        except Exception:
            return
        try:
            last_dt = mdates.date2num(pd.to_datetime(last_ts, unit='ms', utc=True).tz_convert(None))
        except Exception:
            return
        try:
            xlim = ax.get_xlim()
            if xlim[1] > xlim[0]:
                span_x = xlim[1] - xlim[0]
                ax.set_xlim(last_dt - span_x / 2.0, last_dt + span_x / 2.0)
            ylim = ax.get_ylim()
            if ylim[1] > ylim[0]:
                span_y = ylim[1] - ylim[0]
                ax.set_ylim(last_price - span_y / 2.0, last_price + span_y / 2.0)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_figure_enter(self, _event):
        try:
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        except Exception:
            pass

    def _on_figure_leave(self, _event):
        try:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))
        except Exception:
            pass
        self._reset_hover_info()
    def _create_drawbar(self) -> QWidget:
        from PySide6.QtWidgets import QToolButton, QStyle

        drawbar = QWidget()
        drawbar.setObjectName("drawbar_container")
        dlay = QHBoxLayout(drawbar)
        dlay.setContentsMargins(4, 2, 4, 2)
        dlay.setSpacing(4)

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
        for button in (self.tb_home, self.tb_pan, self.tb_zoom):
            dlay.addWidget(button)
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

        for button in (
            self.tb_cross,
            self.tb_hline,
            self.tb_trend,
            self.tb_rect,
            self.tb_fib,
            self.tb_label,
            self.tb_colors,
            self.tb_orders,
        ):
            dlay.addWidget(button)
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
        result = self.chart_controller.rt_flush()
        self._follow_center_if_needed()
        return result

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
        symbol_combo = getattr(self, "symbol_combo", None)
        if symbol_combo is not None:
            blocker = QSignalBlocker(symbol_combo)
            if symbol_combo.findText(symbol) == -1:
                symbol_combo.addItem(symbol)
            symbol_combo.setCurrentText(symbol)
            del blocker
        tf_combo = getattr(self, "tf_combo", None)
        if tf_combo is not None:
            blocker = QSignalBlocker(tf_combo)
            if tf_combo.findText(timeframe) == -1:
                tf_combo.addItem(timeframe)
            tf_combo.setCurrentText(timeframe)
            del blocker
        self.symbol = symbol
        self.timeframe = timeframe
        set_setting('chart.symbol', symbol)
        set_setting('chart.timeframe', timeframe)
        return self.chart_controller.set_symbol_timeframe(db_service=db_service, symbol=symbol, timeframe=timeframe)

    def _on_symbol_changed(self, new_symbol: str):
        """Handle symbol change from combo: update context and reload candles from DB."""
        return self.chart_controller.on_symbol_changed(new_symbol=new_symbol)

    def _open_forecast_settings(self):
        return self.chart_controller.open_forecast_settings()

    # --- Mouse UX: zoom con rotellina e tasto destro (drag) ---
    def _on_scroll_zoom(self, event):
        if event is not None:
            self._suspend_follow()
        return self.chart_controller.on_scroll_zoom(event=event)

    def _on_mouse_press(self, event):
        if event is not None and getattr(event, "button", None) in (1, 3):
            self._suspend_follow()
        return self.chart_controller.on_mouse_press(event=event)

    def _on_mouse_move(self, event):
        if event is not None:
            btn = getattr(event, "button", None)
            gui_event = getattr(event, "guiEvent", None)
            buttons = getattr(gui_event, "buttons", lambda: 0)() if gui_event is not None else 0
            if btn in (1, 3) or buttons:
                self._suspend_follow()
        return self.chart_controller.on_mouse_move(event=event)

    def _on_mouse_release(self, event):
        if event is not None and getattr(event, "button", None) in (1, 3):
            self._suspend_follow()
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
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(4)
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
            chart_layout.setSpacing(0)
            chart_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        if toolbar_container is not None:
            toolbar_layout = toolbar_container.layout()
            if toolbar_layout is None:
                toolbar_layout = QHBoxLayout(toolbar_container)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)
            toolbar_layout.setSpacing(0)
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
            drawbar_layout.setContentsMargins(0, 0, 0, 0)
            drawbar_layout.setSpacing(0)
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
        self.follow_checkbox = getattr(self, "follow_checkbox", None)
        self.bidask_label = getattr(self, "bidask_label", None)
        self.trade_btn = getattr(self, "trade_btn", None)
        self.theme_combo = getattr(self, "theme_combo", None)
        self.settings_btn = getattr(self, "settings_btn", None)

        if self.main_splitter is not None:
            self.main_splitter.setStretchFactor(0, 2)
            self.main_splitter.setStretchFactor(1, 8)
            self.main_splitter.setHandleWidth(6)
            self.main_splitter.setOpaqueResize(True)
        if self.right_splitter is not None:
            self.right_splitter.setStretchFactor(0, 6)
            self.right_splitter.setHandleWidth(6)
            self.right_splitter.setOpaqueResize(True)
        if self._chart_area is not None:
            self._chart_area.setStretchFactor(0, 0)
            self._chart_area.setStretchFactor(1, 1)
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
