# src/forex_diffusion/ui/chart_tab_ui.py
from __future__ import annotations

from typing import Optional, Dict, List
import pandas as pd
import time

import numpy as np

import matplotlib.dates as mdates

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox,
    QSplitter, QListWidget, QListWidgetItem, QTableWidget, QComboBox,
    QToolButton, QCheckBox, QProgressBar, QScrollArea, QDialog
)
from PySide6.QtCore import QTimer, Qt, Signal, QSize, QSignalBlocker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from loguru import logger

# PATTERNS HOOK (import relativo con fallback assoluto, per evitare unresolved in IDE)
try:
    from .chart_components.services.patterns_hook import (
        get_patterns_service, set_patterns_toggle, call_patterns_detection
    )
except Exception:
    from src.forex_diffusion.ui.chart_components.services.patterns_hook import (
        get_patterns_service, set_patterns_toggle, call_patterns_detection
    )


from ..utils.user_settings import get_setting, set_setting
from ..services.brokers import get_broker_service
from .chart_components.controllers.chart_controller import ChartTabController




class DraggableOverlay(QLabel):
    """Small draggable label overlay used for legend and cursor values."""
    dragStarted = Signal()
    dragEnded = Signal()

    def __init__(self, text: str, parent: QWidget):
        super().__init__(parent)
        self.setText(text)
        self.setStyleSheet("QLabel { background: rgba(0,0,0,160); color: white; border: 1px solid rgba(255,255,255,80); border-radius: 4px; padding: 4px; }")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self._dragging = False
        self._drag_offset = None

    def mousePressEvent(self, event):
        if event and event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_offset = event.pos()
            self.dragStarted.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_offset is not None:
            # Move the label keeping the offset
            new_pos = self.mapToParent(event.pos() - self._drag_offset)
            # Keep inside parent bounds
            parent = self.parentWidget()
            if parent:
                x = max(0, min(new_pos.x(), parent.width() - self.width()))
                y = max(0, min(new_pos.y(), parent.height() - self.height()))
                self.move(x, y)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event and event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._drag_offset = None
            self.dragEnded.emit()
        super().mouseReleaseEvent(event)

class ChartTabUI(QWidget):
    """
    The primary chart tab, built entirely programmatically for stability and maintainability.
    """
    forecastRequested = Signal(dict)
    tickArrived = Signal(dict)

    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self.setObjectName("chartTabUI")
        self._main_window = parent

        self.controller = getattr(parent, "controller", None)
        if self.controller and hasattr(self.controller, "signals"):
            self.controller.signals.forecastReady.connect(self.on_forecast_ready)

        # Initialize core attributes
        self._drawing_mode: Optional[str] = None
        self._price_mode = "candles"
        self.chart_controller = ChartTabController(self, self.controller)

        # Build UI, then initialize plot axes and other components
        self._build_ui()

        self.ax = self.canvas.figure.subplots()
        self._osc_ax = None
        self._ind_artists = {}

        self.canvas.figure.set_constrained_layout(False)
        self.canvas.figure.subplots_adjust(left=0.04, right=0.995, top=0.97, bottom=0.08)
        self.ax.margins(x=0.001, y=0.05)
        self._xlim_cid = self.ax.callbacks.connect('xlim_changed', self._on_main_xlim_changed)

        self._init_control_defaults()
        self._connect_ui_signals()

        if theme_combo := getattr(self, "theme_combo", None):
            self._apply_theme(theme_combo.currentText())

        # Initialize state and timers
        self._last_df = pd.DataFrame()
        self._forecasts: List[Dict] = []
        self.max_forecasts = int(get_setting("max_forecasts", 20))
        self._legend_once = set()
        self.broker = get_broker_service()
        # dynamic data cache used by DataService._reload_view_window
        self._current_cache_tf = None
        self._current_cache_range = None
        # dynamic data cache state (used by DataService._reload_view_window)
        self._current_cache_tf = None
        self._current_cache_range = None
        self._setup_timers()

        self.tickArrived.connect(self.chart_controller.on_tick_main)

        # Mouse interaction state
        self._rbtn_drag = False
        self._drag_last = None
        self._drag_axis = None
        self._connect_mouse_events()

        # Overlays and grid setup
        self._overlay_dragging = False
        self._suppress_line_update = False
        self._x_cache_comp = None  # cached compressed X for nearest lookup
        self._init_overlays()
        self._apply_grid_style()

    def _build_ui(self) -> None:
        """Programmatically builds the entire UI with a two-row topbar."""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))

        # Create topbar and populate it with two rows
        topbar = QWidget()
        topbar.setObjectName("topbar")
        self._populate_topbar(topbar)
        self.layout().addWidget(topbar)

        # Create main content area with splitters
        main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = main_splitter

        market_panel = QWidget()
        market_layout = QVBoxLayout(market_panel)
        market_layout.setContentsMargins(0,0,0,0)
        self.market_watch = QListWidget()
        market_layout.addWidget(self.market_watch)
        main_splitter.addWidget(market_panel)

        right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter = right_splitter

        chart_area_splitter = QSplitter(Qt.Vertical)
        self._chart_area = chart_area_splitter

        drawbar_container = QWidget()
        drawbar_layout = QHBoxLayout(drawbar_container)
        drawbar_layout.setContentsMargins(0,0,0,0)
        self._drawbar = self._create_drawbar()
        drawbar_layout.addWidget(self._drawbar)
        chart_area_splitter.addWidget(drawbar_container)

        chart_container = QWidget()
        chart_container_layout = QVBoxLayout(chart_container)
        chart_container_layout.setContentsMargins(0,0,0,0)
        chart_container_layout.addWidget(self.canvas)
        self.chart_container = chart_container  # keep reference for overlays
        chart_area_splitter.addWidget(chart_container)

        right_splitter.addWidget(chart_area_splitter)

        self.orders_table = QTableWidget(0, 9)
        self.orders_table.setHorizontalHeaderLabels(["ID", "Time", "Symbol", "Type", "Volume", "Price", "SL", "TP", "Status"])
        right_splitter.addWidget(self.orders_table)

        main_splitter.addWidget(right_splitter)
        self.layout().addWidget(main_splitter)

        # Set stretch factors to make the chart area expand
        self.layout().setStretch(1, 1)
        main_splitter.setStretchFactor(1, 8)
        right_splitter.setStretchFactor(0, 6)
        chart_area_splitter.setStretchFactor(1, 1)

    def _populate_topbar(self, topbar: QWidget):
        """Creates and adds all widgets to the given topbar widget in two rows."""
        topbar_v_layout = QVBoxLayout(topbar)
        topbar_v_layout.setContentsMargins(4, 1, 4, 1)
        topbar_v_layout.setSpacing(4)
        topbar.setStyleSheet("QPushButton, QComboBox, QToolButton { padding: 2px 6px; min-height: 22px; }; QLabel { padding: 0px; margin: 0px; }")

        # --- Row 1 ---
        row1_widget = QWidget()
        row1_layout = QHBoxLayout(row1_widget)
        row1_layout.setContentsMargins(0,0,0,0); row1_layout.setSpacing(4)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setStyleSheet("QToolBar{spacing:2px; padding:0px; margin:0px;}")
        row1_layout.addWidget(self.toolbar)

        self.symbol_combo = QComboBox(); row1_layout.addWidget(self.symbol_combo)
        row1_layout.addWidget(QLabel("TF:"))
        self.tf_combo = QComboBox(); row1_layout.addWidget(self.tf_combo)
        self.tf_used_label = QLabel("TF used: -"); row1_layout.addWidget(self.tf_used_label)
        row1_layout.addWidget(QLabel("Years:"))
        self.years_combo = QComboBox(); row1_layout.addWidget(self.years_combo)
        row1_layout.addWidget(QLabel("Months:"))
        self.months_combo = QComboBox(); row1_layout.addWidget(self.months_combo)
        self.backfill_btn = QPushButton("Backfill"); row1_layout.addWidget(self.backfill_btn)
        self.backfill_progress = QProgressBar(); self.backfill_progress.setMaximumWidth(120); self.backfill_progress.setTextVisible(False); row1_layout.addWidget(self.backfill_progress)
        self.indicators_btn = QPushButton("Indicators"); row1_layout.addWidget(self.indicators_btn)
        self.build_latents_btn = QPushButton("Build Latents"); row1_layout.addWidget(self.build_latents_btn)
        row1_layout.addStretch()

        # --- Row 2 ---
        row2_widget = QWidget()
        row2_layout = QHBoxLayout(row2_widget)
        row2_layout.setContentsMargins(0,0,0,0); row2_layout.setSpacing(4)

        self.forecast_settings_btn = QPushButton("Prediction Settings"); row2_layout.addWidget(self.forecast_settings_btn)
        self.forecast_btn = QPushButton("Make Prediction"); row2_layout.addWidget(self.forecast_btn)
        row2_layout.addWidget(QLabel("Pred step:"))
        self.pred_step_combo = QComboBox(); row2_layout.addWidget(self.pred_step_combo)
        self.adv_settings_btn = QPushButton("Advanced Settings"); row2_layout.addWidget(self.adv_settings_btn)
        self.adv_forecast_btn = QPushButton("Advanced Forecast"); row2_layout.addWidget(self.adv_forecast_btn)
        self.clear_forecasts_btn = QPushButton("Clear Forecasts"); row2_layout.addWidget(self.clear_forecasts_btn)
        self.toggle_drawbar_btn = QToolButton(); self.toggle_drawbar_btn.setText("Draw"); self.toggle_drawbar_btn.setCheckable(True); row2_layout.addWidget(self.toggle_drawbar_btn)
        self.mode_btn = QToolButton(); self.mode_btn.setText("Candles"); self.mode_btn.setCheckable(True); row2_layout.addWidget(self.mode_btn)
        self.chart_patterns_checkbox = QCheckBox("Chart patterns"); row2_layout.addWidget(self.chart_patterns_checkbox)
        self.candle_patterns_checkbox = QCheckBox("Candlestick patterns"); row2_layout.addWidget(self.candle_patterns_checkbox)
        self.history_patterns_checkbox = QCheckBox("Patterns storici"); row2_layout.addWidget(self.history_patterns_checkbox)

        self.btn_scan_patterns = QToolButton()
        self.btn_scan_patterns.setText("Scansiona patterns")
        row2_layout.addWidget(self.btn_scan_patterns)



        # handler
        def _scan_patterns_now():
            try:
                ps = self.chart_controller.patterns_service
                df = None
                # prova a prendere tutto lo storico se disponibile
                ds = getattr(self.chart_controller, "data_service", None)
                if ds and hasattr(ds, "get_full_dataframe"):
                    df = ds.get_full_dataframe()
                # fallback: ultimo df plottato
                if df is None:
                    df = getattr(self.chart_controller.plot_service, "_last_df", None)
                if df is None or getattr(df, "empty", True):
                    logger.info("Scan patterns: df è vuoto o mancante")
                    return
                ps.on_update_plot(df)
            except Exception as e:
                logger.exception("Scan patterns failed: {}", e)

        self.btn_scan_patterns.clicked.connect(_scan_patterns_now)

        self.follow_checkbox = QCheckBox("Segui"); row2_layout.addWidget(self.follow_checkbox)
        self.bidask_label = QLabel("Bid: -    Ask: -"); row2_layout.addWidget(self.bidask_label)
        self.trade_btn = QPushButton("Trade"); row2_layout.addWidget(self.trade_btn)
        row2_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox(); row2_layout.addWidget(self.theme_combo)
        self.settings_btn = QPushButton("Settings"); row2_layout.addWidget(self.settings_btn)
        row2_layout.addStretch()

        topbar_v_layout.addWidget(row1_widget)
        topbar_v_layout.addWidget(row2_widget)

    def _setup_timers(self):
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(int(get_setting("auto_interval_seconds", 60) * 1000))
        self._auto_timer.timeout.connect(self.chart_controller.auto_forecast_tick)

        self._orders_timer = QTimer(self)
        self._orders_timer.setInterval(1500)
        self._orders_timer.timeout.connect(self.chart_controller.refresh_orders)
        self._orders_timer.start()

        self._rt_dirty = False
        self._rt_timer = QTimer(self)
        self._rt_timer.setInterval(200)
        self._rt_timer.timeout.connect(self.chart_controller.rt_flush)
        self._rt_timer.start()

        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(250)
        self._reload_timer.timeout.connect(self.chart_controller.reload_view_window)

    def _init_control_defaults(self) -> None:
        # Pattern toggles default states
        try:
            chart_on = bool(get_setting('patterns.chart_enabled', False))
            candle_on = bool(get_setting('patterns.candle_enabled', False))
            hist_on = bool(get_setting('patterns.history_enabled', False))
        except Exception:
            chart_on = False; candle_on = False; hist_on = False
        for attr, val in [('chart_patterns_checkbox', chart_on), ('candle_patterns_checkbox', candle_on), ('history_patterns_checkbox', hist_on)]:
            cb = getattr(self, attr, None)
            if cb:
                from PySide6.QtCore import QSignalBlocker as _SB
                with _SB(cb): cb.setChecked(val)
        """Populate UI controls and restore persisted settings."""
        self._symbols_supported = ["EUR/USD", "GBP/USD", "AUX/USD", "GBP/NZD", "AUD/JPY", "GBP/EUR", "GBP/AUD"]
        self._follow_suspend_seconds = float(get_setting('chart.follow_suspend_seconds', 30))
        self._follow_enabled = bool(get_setting('chart.follow_enabled', False))
        self._follow_suspend_until = 0.0

        if follow_checkbox := getattr(self, 'follow_checkbox', None):
            with QSignalBlocker(follow_checkbox): follow_checkbox.setChecked(self._follow_enabled)

        self._set_combo_with_items(getattr(self, "symbol_combo", None), self._symbols_supported, "chart.symbol", self._symbols_supported[0])
        self._set_combo_with_items(getattr(self, "tf_combo", None), ["auto", "tick", "1m", "5m", "15m", "30m", "1h", "4h", "1d"], "chart.timeframe", "auto")
        self._set_combo_with_items(getattr(self, "pred_step_combo", None), ["auto", "1m"], "chart.pred_step", "auto")
        self._set_combo_with_items(getattr(self, "years_combo", None), [str(x) for x in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30]], "chart.backfill_years", "0")
        self._set_combo_with_items(getattr(self, "months_combo", None), [str(x) for x in range(0, 13)], "chart.backfill_months", "0")
        self._set_combo_with_items(getattr(self, "theme_combo", None), ["System", "Dark", "Light", "Custom"], "chart.theme", str(get_setting("ui_theme", "Dark")))

        if market_watch := getattr(self, "market_watch", None):
            with QSignalBlocker(market_watch):
                market_watch.clear()
                for sym in self._symbols_supported: QListWidgetItem(f"{sym}  -", market_watch)

        if toggle_drawbar := getattr(self, "toggle_drawbar_btn", None):
            with QSignalBlocker(toggle_drawbar): toggle_drawbar.setChecked(True)

        self._price_mode = str(get_setting("chart.price_mode", "candles")).lower()
        if mode_btn := getattr(self, "mode_btn", None):
            with QSignalBlocker(mode_btn):
                mode_btn.setChecked(self._price_mode == "candles")
                mode_btn.setText("Candles" if self._price_mode == "candles" else "Line")

        if backfill_progress := getattr(self, "backfill_progress", None):
            backfill_progress.setRange(0, 100); backfill_progress.setValue(0)

        self._restore_splitters()

        self.symbol = self.symbol_combo.currentText() if self.symbol_combo else self._symbols_supported[0]
        self.timeframe = self.tf_combo.currentText() if self.tf_combo else "auto"

    def _connect_ui_signals(self) -> None:
        connections = {
            # Topbar
            getattr(self, "symbol_combo", None): ("currentTextChanged", self._on_symbol_combo_changed),
            getattr(self, "tf_combo", None): ("currentTextChanged", self._on_timeframe_changed),
            getattr(self, "pred_step_combo", None): ("currentTextChanged", self._on_pred_step_changed),
            getattr(self, "years_combo", None): ("currentTextChanged", self._on_backfill_range_changed),
            getattr(self, "months_combo", None): ("currentTextChanged", self._on_backfill_range_changed),
            getattr(self, "theme_combo", None): ("currentTextChanged", self._on_theme_changed),
            getattr(self, "settings_btn", None): ("clicked", self._open_settings_dialog),
            getattr(self, "backfill_btn", None): ("clicked", self.chart_controller.on_backfill_missing_clicked),
            getattr(self, "indicators_btn", None): ("clicked", self.chart_controller.on_indicators_clicked),
            getattr(self, "build_latents_btn", None): ("clicked", self.chart_controller.on_build_latents_clicked),
            getattr(self, "forecast_settings_btn", None): ("clicked", self.chart_controller.open_forecast_settings),
            getattr(self, "forecast_btn", None): ("clicked", self.chart_controller.on_forecast_clicked),
            getattr(self, "adv_settings_btn", None): ("clicked", self.chart_controller.open_adv_forecast_settings),
            getattr(self, "adv_forecast_btn", None): ("clicked", self.chart_controller.on_advanced_forecast_clicked),
            getattr(self, "clear_forecasts_btn", None): ("clicked", self.chart_controller.clear_all_forecasts),
            getattr(self, "toggle_drawbar_btn", None): ("toggled", self._toggle_drawbar),
            getattr(self, "mode_btn", None): ("toggled", self._on_price_mode_toggled),
            getattr(self, "follow_checkbox", None): ("toggled", self._on_follow_toggled),
            getattr(self, "trade_btn", None): ("clicked", self.chart_controller.open_trade_dialog),
            # Drawbar

            getattr(self, "tb_orders", None): ("toggled", self._toggle_orders),
        }
        for widget, (signal, handler) in connections.items():
            if widget: getattr(widget, signal).connect(handler)

        draw_buttons = {
            getattr(self, "tb_cross", None): None, getattr(self, "tb_hline", None): "hline",
            getattr(self, "tb_trend", None): "trend", getattr(self, "tb_rect", None): "rect",
            getattr(self, "tb_fib", None): "fib", getattr(self, "tb_label", None): "label",
        }
        for button, mode in draw_buttons.items():
            if button: button.clicked.connect(lambda checked=False, m=mode: self._set_drawing_mode(m))

        for key, splitter in self._get_splitters().items():
            if splitter: splitter.splitterMoved.connect(lambda _p, _i, k=key, s=splitter: self._persist_splitter_positions(k, s))
        # >>> PATTERNS: collega i tre checkbox già presenti e sincronizza subito lo stato <<<
        self._wire_pattern_checkboxes()

    def _wire_pattern_checkboxes(self) -> None:
        """
        Collega i tre checkbox già presenti nella topbar:
          - self.chart_patterns_checkbox  ("Chart patterns")
          - self.candle_patterns_checkbox ("Candlestick patterns")
          - self.history_patterns_checkbox ("Patterns storici")
        al PatternsService tramite l'hook (niente accessi diretti al controller).
        Esegue anche una sincronizzazione iniziale dello stato e un primo trigger.
        """
        from loguru import logger

        # Devono già esistere: se qualcuno manca, non facciamo nulla
        chart_cb = getattr(self, "chart_patterns_checkbox", None)
        candle_cb = getattr(self, "candle_patterns_checkbox", None)
        history_cb = getattr(self, "history_patterns_checkbox", None)
        if not (chart_cb or candle_cb or history_cb):
            logger.debug("Patterns wiring: nessun checkbox trovato in UI (chart/candle/history).")
            return

        ctrl = self.chart_controller
        # Istanzia lazy il service (registry interno, no setattr sul controller)
        get_patterns_service(ctrl, self, create=True)

        # Collega i toggle → set_*_enabled(...)
        if chart_cb:
            chart_cb.toggled.connect(lambda v, c=ctrl: set_patterns_toggle(c, self, chart=bool(v)))
        if candle_cb:
            candle_cb.toggled.connect(lambda v, c=ctrl: set_patterns_toggle(c, self, candle=bool(v)))
        if history_cb:
            history_cb.toggled.connect(lambda v, c=ctrl: set_patterns_toggle(c, self, history=bool(v)))

        # Sincronizza SUBITO lo stato dell’UI sul service (senza aspettare altri eventi)
        init_chart = bool(chart_cb.isChecked()) if chart_cb else False
        init_candle = bool(candle_cb.isChecked()) if candle_cb else False
        init_history = bool(history_cb.isChecked()) if history_cb else False
        set_patterns_toggle(ctrl, self, chart=init_chart, candle=init_candle, history=init_history)

        # Primo trigger: passa il DF corrente (se disponibile) per far partire la detection
        try:
            plot = getattr(ctrl, "plot_service", None)
            df_now = getattr(plot, "_last_df", None)
            call_patterns_detection(ctrl, self, df_now)
            logger.debug(
                f"Patterns wiring: sync iniziale → chart={init_chart}, candle={init_candle}, history={init_history}; "
                f"trigger con df={'ok' if df_now is not None else 'None'}"
            )
        except Exception as e:
            logger.debug(f"Patterns wiring: primo trigger fallito: {e}")


    def _connect_mouse_events(self):
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click) # For drawing

    def _get_splitters(self) -> Dict[str, QSplitter]:
        return {
            'chart.splitter.main': getattr(self, "main_splitter", None),
            'chart.splitter.right': getattr(self, "right_splitter", None),
            'chart.splitter.chart': getattr(self, "_chart_area", None),
        }

    def _set_combo_with_items(self, combo: Optional[QComboBox], items: List[str], setting_key: str, default: str) -> str:
        if not combo: return default
        saved = str(get_setting(setting_key, default))
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItems(items)
            if saved and combo.findText(saved) == -1: combo.addItem(saved)
            combo.setCurrentText(saved or default)
        return saved or default

    def _restore_splitters(self) -> None:
        for key, splitter in self._get_splitters().items():
            if not splitter: continue
            if sizes := get_setting(key, None):
                if isinstance(sizes, (list, tuple)):
                    cleaned = [int(v) for v in sizes if str(v).isdigit()]
                    if cleaned: splitter.setSizes(cleaned)

    def _persist_splitter_positions(self, key: str, splitter: QSplitter) -> None:
        set_setting(key, splitter.sizes())

    def _on_symbol_combo_changed(self, new_symbol: str) -> None:
        if not new_symbol: return
        set_setting('chart.symbol', new_symbol)
        self.symbol = new_symbol
        self.chart_controller.on_symbol_changed(new_symbol=new_symbol)

    def _on_timeframe_changed(self, value: str) -> None:
        set_setting('chart.timeframe', value)
        self.timeframe = value
        self._schedule_view_reload()

    def _on_pred_step_changed(self, value: str) -> None: set_setting('chart.pred_step', value)
    def _on_backfill_range_changed(self, _value: str) -> None:
        if (ys := getattr(self, 'years_combo', None)): set_setting('chart.backfill_years', ys.currentText())
        if (ms := getattr(self, 'months_combo', None)): set_setting('chart.backfill_months', ms.currentText())

    def _on_theme_changed(self, theme: str) -> None:
        set_setting('chart.theme', theme)
        self._apply_theme(theme)
        # Re-apply grid styling after theme change
        try:
            self._apply_grid_style()
        except Exception as e:
            logger.warning(f"Failed to apply grid style after theme change: {e}")

    def _open_settings_dialog(self) -> None:
        from .settings_dialog import SettingsDialog
        # Wrap settings dialog in a scrollable container to provide vertical scrollbar
        dialog = SettingsDialog(self)
        accepted = False
        try:
            wrapper = QDialog(self)
            wrapper.setWindowTitle(getattr(dialog, "windowTitle", lambda: "Settings")())
            area = QScrollArea(wrapper)
            area.setWidgetResizable(True)
            # Make inner dialog act as a widget
            dialog.setWindowFlags(Qt.Widget)
            area.setWidget(dialog)
            lay = QVBoxLayout(wrapper)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(area)
            accepted = bool(wrapper.exec())
        except Exception as e:
            logger.debug(f"Scrollable settings wrapper failed, fallback: {e}")
            accepted = bool(dialog.exec())

        if accepted:
            if (tc := getattr(self, 'theme_combo', None)): self._apply_theme(tc.currentText())
            # Apply grid style again to reflect unified grid color setting
            try:
                self._apply_grid_style()
            except Exception as e:
                logger.warning(f"Failed to apply grid style after settings: {e}")
            self._follow_suspend_seconds = float(get_setting('chart.follow_suspend_seconds', self._follow_suspend_seconds))
            self._follow_enabled = bool(get_setting('chart.follow_enabled', self._follow_enabled))
            if self._follow_enabled: self._follow_suspend_until = 0.0
            if (fc := getattr(self, 'follow_checkbox', None)): 
                with QSignalBlocker(fc): fc.setChecked(self._follow_enabled)
            if getattr(self, '_last_df', None) is not None and not self._last_df.empty:
                prev_xlim, prev_ylim = self.ax.get_xlim(), self.ax.get_ylim()
                self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
            self._follow_center_if_needed()

    def _on_price_mode_toggled(self, checked: bool) -> None:
        self._price_mode = 'candles' if checked else 'line'
        if (mb := getattr(self, 'mode_btn', None)): mb.setText('Candles' if checked else 'Line')
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
            self._follow_suspend_until = time.time() + max(duration, 1.0)

    def _follow_center_if_needed(self) -> None:
        if not (getattr(self, '_follow_enabled', False) and time.time() >= getattr(self, '_follow_suspend_until', 0.0) and (ldf := getattr(self, '_last_df', None)) is not None and not ldf.empty and (ax := getattr(self, 'ax', None)) is not None): return
        y_col = 'close' if 'close' in ldf.columns else 'price'
        last_row = ldf.iloc[-1]
        last_ts, last_price = float(last_row['ts_utc']), float(last_row.get(y_col, last_row.get('price')))
        # Convert to naive datetime (local) safely, then to mdates number
        try:
            ts_obj = pd.to_datetime(last_ts, unit='ms', utc=True).tz_convert('UTC').tz_localize(None)
            last_dt = mdates.date2num(ts_obj)
        except Exception:
            last_dt = mdates.date2num(pd.to_datetime(last_ts, unit='ms'))
        # map to compressed X if compression is active
        try:
            comp = getattr(self.chart_controller.plot_service, "_compress_real_x", None)
            last_dt_comp = comp(last_dt) if callable(comp) else last_dt
        except Exception:
            last_dt_comp = last_dt
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if xlim[1] > xlim[0]:
            span_x = xlim[1] - xlim[0]
            ax.set_xlim(last_dt_comp - span_x * 0.8, last_dt_comp + span_x * 0.2)
        if ylim[1] > ylim[0]:
            span_y = ylim[1] - ylim[0]
            try:
                import math
                if math.isfinite(span_y) and math.isfinite(last_price):
                    ax.set_ylim(last_price - span_y / 2.0, last_price + span_y / 2.0)
            except Exception:
                pass
        self.canvas.draw_idle()

    def _create_drawbar(self) -> QWidget:
        from PySide6.QtWidgets import QToolButton, QStyle
        drawbar = QWidget()
        drawbar.setObjectName("drawbar_container")
        dlay = QHBoxLayout(drawbar)
        dlay.setContentsMargins(4, 2, 4, 2)
        dlay.setSpacing(4)

        style = self.style()
        buttons_spec = {
            'tb_cross': (QStyle.SP_DialogYesButton, "Cross", True, lambda: self._set_drawing_mode(None)),
            'tb_hline': (QStyle.SP_TitleBarShadeButton, "H-Line", True, lambda: self._set_drawing_mode('hline')),
            'tb_trend': (QStyle.SP_ArrowRight, "Trend", True, lambda: self._set_drawing_mode('trend')),
            'tb_rect': (QStyle.SP_DirIcon, "Rect", True, lambda: self._set_drawing_mode('rect')),
            'tb_fib': (QStyle.SP_FileDialogDetailedView, "Fib", True, lambda: self._set_drawing_mode('fib')),
            'tb_label': (QStyle.SP_MessageBoxInformation, "Label", True, lambda: self._set_drawing_mode('label')),
            'tb_colors': (QStyle.SP_DriveDVDIcon, "Color/CSS Settings", False, self._open_color_settings),
            'tb_orders': (QStyle.SP_FileDialogListView, "Show/Hide Orders", True, self._toggle_orders),
        }

        for name, (icon_enum, tip, checkable, handler) in buttons_spec.items():
            btn = QToolButton()
            btn.setIcon(style.standardIcon(icon_enum))
            btn.setToolTip(tip)
            btn.setCheckable(checkable)
            if handler: btn.clicked.connect(handler)
            setattr(self, name, btn)
            dlay.addWidget(btn)
        if (orders_btn := getattr(self, 'tb_orders', None)): orders_btn.setChecked(True)
        if (cross_btn := getattr(self, 'tb_cross', None)): cross_btn.setChecked(True)

        dlay.addStretch()
        return drawbar

    # --- Controller Passthrough Methods ---
    def _handle_tick(self, payload: dict): return self.chart_controller.handle_tick(payload=payload)
    def _on_tick_main(self, payload: dict): return self.chart_controller.on_tick_main(payload=payload)
    def _rt_flush(self): self.chart_controller.rt_flush(); self._follow_center_if_needed()
    def _set_drawing_mode(self, mode: Optional[str]): return self.chart_controller.set_drawing_mode(mode=mode)
    def _on_canvas_click(self, event): return self.chart_controller.on_canvas_click(event=event)
    def _open_trade_dialog(self): return self.chart_controller.open_trade_dialog()
    def _on_indicators_clicked(self): return self.chart_controller.on_indicators_clicked()
    def _get_indicator_settings(self) -> dict: return self.chart_controller.get_indicator_settings()
    def _ensure_osc_axis(self, need: bool): return self.chart_controller.ensure_osc_axis(need=need)
    def _on_main_xlim_changed(self, ax): return self.chart_controller.on_main_xlim_changed(ax=ax)
    def _plot_indicators(self, df2: pd.DataFrame, x_dt: pd.Series): return self.chart_controller.plot_indicators(df2=df2, x_dt=x_dt)
    def _on_build_latents_clicked(self): return self.chart_controller.on_build_latents_clicked()
    def _refresh_orders(self): return self.chart_controller.refresh_orders()
    def update_plot(self, df: pd.DataFrame, quantiles: Optional[dict] = None, restore_xlim=None, restore_ylim=None):
        res = self.chart_controller.update_plot(df=df, quantiles=quantiles, restore_xlim=restore_xlim, restore_ylim=restore_ylim)
        # Keep reference for overlays and rebuild X cache
        try:
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                self._last_df = df
                self._rebuild_x_cache()
            self._apply_grid_style()
        except Exception as e:
            logger.debug(f"post-update_plot hooks failed: {e}")
        return res
    def _apply_theme(self, theme: str): return self.chart_controller.apply_theme(theme=theme)
    def _get_color(self, key: str, default: str) -> str: return self.chart_controller.get_color(key=key, default=default)
    def _open_color_settings(self): return self.chart_controller.open_color_settings()
    def _toggle_drawbar(self, visible: bool): return self.chart_controller.toggle_drawbar(visible=visible)
    def _toggle_orders(self, visible: bool): return self.chart_controller.toggle_orders(visible=visible)
    def _plot_forecast_overlay(self, quantiles: dict, source: str = "basic"): return self.chart_controller.plot_forecast_overlay(quantiles=quantiles, source=source)
    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        for combo, text in [(getattr(self, "symbol_combo", None), symbol), (getattr(self, "tf_combo", None), timeframe)]:
            if combo: 
                with QSignalBlocker(combo): 
                    if combo.findText(text) == -1: combo.addItem(text)
                    combo.setCurrentText(text)
        self.symbol, self.timeframe = symbol, timeframe
        set_setting('chart.symbol', symbol); set_setting('chart.timeframe', timeframe)
        return self.chart_controller.set_symbol_timeframe(db_service=db_service, symbol=symbol, timeframe=timeframe)
    def _on_symbol_changed(self, new_symbol: str): return self.chart_controller.on_symbol_changed(new_symbol=new_symbol)
    def _open_forecast_settings(self): return self.chart_controller.open_forecast_settings()
    def _on_scroll_zoom(self, event): 
        if event: self._suspend_follow()
        return self.chart_controller.on_scroll_zoom(event=event)
    def _on_mouse_press(self, event):
        if event and getattr(event, "button", None) in (1, 3): self._suspend_follow()
        return self.chart_controller.on_mouse_press(event=event)



    def _on_mouse_move(self, event):
        # 1) guard-rails: niente assi, niente x
        price_ax = getattr(self, "price_ax", None)
        if event is None or price_ax is None or event.inaxes is None:
            return
        if event.inaxes is not price_ax:
            return
        if event.xdata is None:
            return

        # 2) x → ms
        x_ms = self._x_to_ms(event.xdata)
        if x_ms is None:
            return

        # 3) trova riga valida
        idx = self._row_at_xms(x_ms)
        if idx is None:
            return

        # 4) throttle anti-rumore: se la riga non cambia, non fare nulla
        if getattr(self, "_last_cursor_idx", None) == idx:
            return
        self._last_cursor_idx = idx

        # 5) usa la riga in sicurezza
        df = self._get_active_df()
        try:
            row = df.iloc[idx]
        except Exception:
            return

        # 6) aggiorna overlays/cursori/tooltip con row
        try:
            self._update_cursor_overlays(row, idx)  # ← la tua funzione esistente
        except Exception as ex:
            # downgrade a DEBUG senza flood
            if getattr(self, "_mouse_err_once", True):
                self._mouse_err_once = False
                self._logger.debug("cursor overlay update skipped: %s", ex)




    def _on_mouse_release(self, event):
        if event and getattr(event, "button", None) in (1, 3): self._suspend_follow()
        return self.chart_controller.on_mouse_release(event=event)
    def _schedule_view_reload(self): return self.chart_controller.schedule_view_reload()
    def _reload_view_window(self): return self.chart_controller.reload_view_window()
    def _on_forecast_clicked(self): return self.chart_controller.on_forecast_clicked()
    def _open_adv_forecast_settings(self): return self.chart_controller.open_adv_forecast_settings()
    def _on_advanced_forecast_clicked(self): return self.chart_controller.on_advanced_forecast_clicked()
    def on_forecast_ready(self, df: pd.DataFrame, quantiles: dict): return self.chart_controller.on_forecast_ready(df=df, quantiles=quantiles)
    def clear_all_forecasts(self): return self.chart_controller.clear_all_forecasts()
    def start_auto_forecast(self): return self.chart_controller.start_auto_forecast()
    def stop_auto_forecast(self): return self.chart_controller.stop_auto_forecast()
    def _auto_forecast_tick(self): return self.chart_controller.auto_forecast_tick()
    def _on_backfill_missing_clicked(self): return self.chart_controller.on_backfill_missing_clicked()
    def _load_candles_from_db(self, symbol: str, timeframe: str, limit: int = 5000, start_ms: Optional[int] = None, end_ms: Optional[int] = None): return self.chart_controller.load_candles_from_db(symbol=symbol, timeframe=timeframe, limit=limit, start_ms=start_ms, end_ms=end_ms)

    # --- Helpers Mouse ---

    def _x_to_ms(self, x):
        """Converte l'ascissa matplotlib in ms epoch secondo la modalità usata nel plot."""
        # Se hai già standardizzato tutto a ms (consigliato), ritorna int(x)
        # Altrimenti rileva la modalità: 'ms_epoch' vs 'matplotlib_date'
        mode = getattr(self, "_x_mode", "ms_epoch")
        if x is None:
            return None
        if mode == "ms_epoch":
            try:
                return int(x)
            except (TypeError, ValueError):
                return None
        # fallback: ascissa in date-number matplotlib
        try:
            dt = mdates.num2date(x)  # timezone-naive
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    def _get_active_df(self):
        """Recupera il DataFrame attivo usato in chart (ts_ms, open, high, low, close)."""
        # Adatta questi getter al tuo wiring attuale:
        df = getattr(self.chart_controller, "last_df", None)
        if df is None:
            df = getattr(self.plot_service, "last_df", None) if hasattr(self, "plot_service") else None
        return df

    def _row_at_xms(self, x_ms):
        """Restituisce l'indice riga più vicino e valido per il timestamp in ms."""
        df = self._get_active_df()
        if df is None or df.empty:
            return None
        col = "ts_ms" if "ts_ms" in df.columns else "ts_utc"
        if col == "ts_utc" and df[col].dtype != np.int64 and df[col].dtype != np.int32:
            # Se ts_utc è datetime, converti in ms
            try:
                ts = df[col].view("int64") // 10 ** 6  # pandas datetime64[ns] → ms
            except Exception:
                ts = (df[col].astype("int64") // 10 ** 6)
        else:
            ts = df[col].to_numpy()

        # Cerca posizione e clamp
        i = int(np.searchsorted(ts, x_ms, side="left"))
        i = i - 1 if i > 0 else 0
        if i < 0 or i >= len(df):
            return None
        return i

    # --- Grid and Overlays helpers ---

    def _apply_grid_style(self) -> None:
        """Ensure Y axis values and grid lines are visible using unified grid color."""
        if not hasattr(self, "ax") or self.ax is None: return
        try:
            grid_color = str(get_setting("chart.grid_color", "#404040"))
        except Exception:
            grid_color = "#404040"
        ax = self.ax
        # Enable grid on both axes
        ax.grid(True, which='major', axis='both', color=grid_color, alpha=0.35, linewidth=0.6)
        ax.grid(True, which='minor', axis='y', color=grid_color, alpha=0.20, linewidth=0.4)
        # Ensure Y axis tick labels are visible
        ax.tick_params(axis='y', which='both', labelleft=True)
        # Draw minor ticks for more y-lines if formatter allows
        try:
            ax.minorticks_on()
        except Exception:
            pass
        self.canvas.draw_idle()

    def _init_overlays(self) -> None:
        """Create draggable overlays: legend-like and cursor values."""
        parent = getattr(self, "chart_container", None) or self
        # Legend-like overlay
        self.legend_overlay = DraggableOverlay("X: -\nY: -", parent)
        self.legend_overlay.resize(160, 56)
        self.legend_overlay.move(10, 10)
        # Cursor values overlay (mouse position)
        self.cursor_overlay = DraggableOverlay("Mouse: X -, Y -", parent)
        self.cursor_overlay.resize(240, 30)
        self.cursor_overlay.move(10, 70)

        # Connect drag events to suspend pan and line updates
        def _on_drag_start():
            self._overlay_dragging = True
            self._suppress_line_update = True
            # Disable pan if active
            try:
                prev = getattr(self.toolbar, "_active", None)
                self._prev_toolbar_active = prev
                if str(prev).upper() == "PAN":
                    # toggle off pan
                    self.toolbar.pan()
            except Exception:
                pass

        def _on_drag_end():
            # restore pan if it was active before dragging
            try:
                if getattr(self, "_prev_toolbar_active", None) and str(self._prev_toolbar_active).upper() == "PAN":
                    self.toolbar.pan()
            except Exception:
                pass
            self._overlay_dragging = False
            self._suppress_line_update = False

        for w in (self.legend_overlay, self.cursor_overlay):
            w.dragStarted.connect(_on_drag_start)
            w.dragEnded.connect(_on_drag_end)
            w.show()
            w.raise_()  # ensure overlays stay on top of the canvas

    def _update_cursor_overlays(self, event) -> None:
        """Update overlays content with X/Y info: mouse pos and nearest data value at cursor X."""
        if event is None:
            return
        xdata, ydata = getattr(event, "xdata", None), getattr(event, "ydata", None)
        if xdata is None:
            return

        # 1) Mouse position (use axis formatters)
        try:
            x_mouse_txt = self.ax.format_xdata(xdata) if hasattr(self, "ax") and self.ax else f"{xdata:.6f}"
        except Exception:
            x_mouse_txt = f"{xdata:.6f}"
        try:
            y_mouse_txt = self.ax.format_ydata(ydata) if (hasattr(self, "ax") and self.ax and ydata is not None) else ("-" if ydata is None else f"{ydata:.6f}")
        except Exception:
            y_mouse_txt = "-" if ydata is None else f"{ydata:.6f}"
        if hasattr(self, "cursor_overlay") and self.cursor_overlay:
            self.cursor_overlay.setText(f"Mouse: X {x_mouse_txt}, Y {y_mouse_txt}")
            self.cursor_overlay.adjustSize()

        # 2) Nearest data point at this X (compressed axis aware)
        ldf = getattr(self, "_last_df", None)
        if ldf is None or ldf.empty:
            return

        # Build/computed compressed X cache
        if getattr(self, "_x_cache_comp", None) is None:
            self._rebuild_x_cache()
        xcomp = self._x_cache_comp
        if xcomp is None or len(xcomp) == 0:
            return

        # Find nearest index on compressed X
        try:
            import numpy as np
            idx = int(np.searchsorted(xcomp, xdata))
            idx = max(0, min(idx, len(xcomp) - 1))
            # pick closer between idx and idx-1
            if idx > 0 and abs(xcomp[idx] - xdata) > abs(xcomp[idx - 1] - xdata):
                idx -= 1
        except Exception:
            idx = max(0, min(int(len(ldf) / 2), len(ldf) - 1))

        row = ldf.iloc[idx]
        ts_ms = float(row["ts_utc"])
        # Choose Y series based on current price mode
        if getattr(self, "_price_mode", "candles") == "line" and "price" in ldf.columns:
            y_col = "price"
        elif "close" in ldf.columns:
            y_col = "close"
        elif "price" in ldf.columns:
            y_col = "price"
        else:
            y_col = None
        try:
            y_val = float(row[y_col]) if (y_col and y_col in row) else (float(ydata) if ydata is not None else float("nan"))
        except Exception:
            y_val = float("nan")
        # Format date/time for the nearest candle timestamp (safe tz handling)
        try:
            ts_obj = pd.to_datetime(ts_ms, unit='ms', utc=True).tz_convert('UTC').tz_localize(None)
            dt_txt = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            dt_txt = str(ts_ms)

        # Update legend-like overlay (data values)
        if hasattr(self, "legend_overlay") and self.legend_overlay:
            self.legend_overlay.setText(f"X: {dt_txt}\nY: {y_val:.6f}")
            self.legend_overlay.adjustSize()

    def _rebuild_x_cache(self) -> None:
        """Rebuild compressed X cache from last_df timestamps."""
        ldf = getattr(self, "_last_df", None)
        if ldf is None or ldf.empty or "ts_utc" not in ldf.columns:
            self._x_cache_comp = None
            return
        try:
            x_ser = pd.to_datetime(ldf["ts_utc"], unit='ms', utc=True)
            # Convert to naive datetimes safely using Series.dt
            x_dt = x_ser.dt.tz_convert('UTC').dt.tz_localize(None)
            x_mpl = mdates.date2num(x_dt)
            comp = getattr(getattr(self.chart_controller, "plot_service", None), "_compress_real_x", None)
            if callable(comp):
                import numpy as np
                self._x_cache_comp = np.array([comp(float(v)) for v in x_mpl], dtype=float)
            else:
                self._x_cache_comp = x_mpl
        except Exception as e:
            logger.debug(f"_rebuild_x_cache failed: {e}")
            self._x_cache_comp = None

    # --- Internal Implementation Methods (Restored) ---
    def _on_mode_toggled(self, checked: bool):
        return self.chart_controller.on_mode_toggled(checked=checked)

    def _render_candles(self, df2: pd.DataFrame):
        return self.chart_controller.render_candles(df2=df2)

    def _tf_to_timedelta(self, tf: str):
        return self.chart_controller.tf_to_timedelta(tf=tf)

    def _zoom_axis(self, axis: str, center: float, factor: float):
        return self.chart_controller.zoom_axis(axis=axis, center=center, factor=factor)

    def _update_badge_visibility(self, event):
        return self.chart_controller.update_badge_visibility(event=event)

    def _resolution_for_span(self, ms_span: int) -> str:
        return self.chart_controller.resolution_for_span(ms_span=ms_span)

    def _trim_forecasts(self):
        return self.chart_controller.trim_forecasts()

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
