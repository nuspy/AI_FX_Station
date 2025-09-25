# src/forex_diffusion/ui/chart_tab_ui.py
from __future__ import annotations

from typing import Optional, Dict, List
import pandas as pd
import time

import matplotlib.dates as mdates

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox,
    QSplitter, QListWidget, QListWidgetItem, QTableWidget, QComboBox,
    QToolButton, QCheckBox, QProgressBar, QScrollArea, QDialog,
    QTabWidget, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox
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
        # Pattern overlays state/caches
        self._pattern_artists = []        # list of matplotlib artists for patterns
        self._artist_to_pattern = {}      # artist -> pattern dict
        self._patterns_cache = []         # last detected patterns (raw dicts)
        self._patterns_cache_map = {}     # key tuple -> canonical pattern
        self._patterns_scan_tf_hint = None  # tf hint when scanning sequentially

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
        
        self.btn_config_patterns = QToolButton()
        self.btn_config_patterns.setText("Configura Patterns")
        row2_layout.addWidget(self.btn_config_patterns)

        # handler
        self.btn_config_patterns.clicked.connect(self._open_patterns_config)
        def _scan_patterns_now():
            try:
                ps = self.chart_controller.patterns_service
                if ps is None:
                    return
                symbol = getattr(self, "symbol", None) or (self.symbol_combo.currentText() if hasattr(self, "symbol_combo") else None)
                if not symbol:
                    logger.info("Scan patterns: simbolo non impostato")
                    return
                # scan across multiple timeframes and merge into cache
                tfs_to_scan = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                for tf in tfs_to_scan:
                    try:
                        df_tf = self._load_candles_from_db(symbol, tf, limit=10000)
                        if df_tf is None or df_tf.empty:
                            continue
                        # tag hint so callback can attach tf to incoming results
                        self._patterns_scan_tf_hint = tf
                        ps.on_update_plot(df_tf)
                    except Exception:
                        continue
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
        # Prova ad agganciare un callback/signal per ricevere i pattern rilevati
        try:
            ps = get_patterns_service(ctrl, self, create=False)
            if ps is not None:
                # Preferisci un eventuale Signal Qt
                if hasattr(ps, "patternsDetected"):
                    try:
                        ps.patternsDetected.connect(self.on_patterns_ready)
                    except Exception:
                        pass
                # Fallback: esponi una callback attribuita dal service
                try:
                    setattr(ps, "ui_on_patterns_ready", self.on_patterns_ready)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Patterns wiring: unable to attach callback: {e}")

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
        # Enable pick events for pattern badges (dialogs on click)
        self.canvas.mpl_connect("pick_event", self._on_pick_pattern_artist)

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
        # reset pattern cache on symbol change
        try:
            self._clear_pattern_artists()
            self._patterns_cache = []
            self._patterns_cache_map = {}
        except Exception:
            pass
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
            # Redraw cached patterns without re-detection (UI legacy fallback)
            self._redraw_cached_patterns()
            # Ask PatternsService to repaint overlays from its cache (no detection)
            try:
                ps = get_patterns_service(self.chart_controller, self, create=False)
                if ps:
                    ps._repaint()
            except Exception:
                pass
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
        if event and (getattr(event, "button", None) in (1, 3) or (ge := getattr(event, "guiEvent", None)) and getattr(ge, "buttons", lambda: 0)()): self._suspend_follow()
        # Update overlays with cursor info unless dragging an overlay
        try:
            if not getattr(self, "_overlay_dragging", False):
                self._update_cursor_overlays(event)
        except Exception as e:
            logger.debug(f"_update_cursor_overlays failed: {e}")
        # Suppress line update during overlay drag
        if getattr(self, "_overlay_dragging", False) or getattr(self, "_suppress_line_update", False):
            return None
        return self.chart_controller.on_mouse_move(event=event)
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

    # -------- Patterns UI integration --------

    def on_patterns_ready(self, patterns: List[dict]) -> None:
        """
        Entry point for PatternsService: merge into canonical cache (by timestamps) and redraw.
        Supported keys from service: name/pattern, description/desc, idx_start/idx_end, start_ts/end_ts/ts_start/ts_end/start/end
        """
        try:
            if not isinstance(patterns, list) or len(patterns) == 0:
                return
            # ensure cache map exists
            if not isinstance(getattr(self, "_patterns_cache_map", None), dict):
                self._patterns_cache_map = {}
            symbol = getattr(self, "symbol", None)
            tf_hint = getattr(self, "_patterns_scan_tf_hint", None)
            # process and merge
            for raw in patterns:
                try:
                    p = dict(raw) if isinstance(raw, dict) else {}
                    name = str(p.get("name") or p.get("pattern") or "Pattern")
                    desc = p.get("description") or p.get("desc")
                    # resolve timestamps
                    ts_start = p.get("ts_start") or p.get("start_ts") or p.get("tsStart") or p.get("start")
                    ts_end = p.get("ts_end") or p.get("end_ts") or p.get("tsEnd") or p.get("end")
                    if (ts_start is None or ts_end is None) and getattr(self, "_last_df", None) is not None and not self._last_df.empty:
                        idx_start = p.get("idx_start")
                        idx_end = p.get("idx_end")
                        if idx_start is not None and 0 <= int(idx_start) < len(self._last_df):
                            ts_start = int(self._last_df["ts_utc"].iloc[int(idx_start)])
                        if idx_end is not None and 0 <= int(idx_end) < len(self._last_df):
                            ts_end = int(self._last_df["ts_utc"].iloc[int(idx_end)])
                    # still missing? skip this entry
                    if ts_start is None or ts_end is None:
                        continue
                    try:
                        ts_start = int(float(ts_start))
                        ts_end = int(float(ts_end))
                    except Exception:
                        continue
                    tf_val = p.get("tf") or p.get("timeframe") or tf_hint
                    # canonical pattern
                    canon = dict(p)
                    canon["name"] = name
                    if desc is not None:
                        canon["description"] = desc
                    canon["ts_start"] = ts_start
                    canon["ts_end"] = ts_end
                    if symbol:
                        canon["symbol"] = symbol
                    if tf_val:
                        canon["tf"] = str(tf_val)
                    # key: allow same name across TF as separate entries
                    key = (symbol or "-", name, ts_start, ts_end, canon.get("tf", "-"))
                    self._patterns_cache_map[key] = canon
                except Exception:
                    continue
            # refresh list cache and redraw
            self._patterns_cache = list(self._patterns_cache_map.values())
            self._redraw_cached_patterns()
        except Exception as e:
            logger.debug(f"on_patterns_ready failed: {e}")
        finally:
            # reset hint so future callbacks without explicit scan won't inherit it
            try:
                self._patterns_scan_tf_hint = None
            except Exception:
                pass

    def _redraw_cached_patterns(self) -> None:
        """Redraw previously detected patterns from cache."""
        if getattr(self, "_patterns_cache", None):
            self._draw_patterns(self._patterns_cache)

    def _clear_pattern_artists(self) -> None:
        """Remove previous pattern artists from axes."""
        try:
            for art in getattr(self, "_pattern_artists", []):
                try:
                    art.remove()
                except Exception:
                    pass
            self._pattern_artists = []
            self._artist_to_pattern = {}
        except Exception:
            self._pattern_artists = []
            self._artist_to_pattern = {}

    def _draw_patterns(self, patterns: List[dict]) -> None:
        """Draw badges and formation line for patterns."""
        if not hasattr(self, "ax") or self.ax is None:
            return
        if getattr(self, "_last_df", None) is None or self._last_df.empty:
            return
        # Clear previous drawings (we keep cache; no re-detection)
        self._clear_pattern_artists()

        for p in patterns:
            try:
                self._draw_single_pattern(p)
            except Exception as e:
                logger.debug(f"_draw_single_pattern error: {e}")

        self.canvas.draw_idle()

    def _draw_single_pattern(self, p: dict) -> None:
        """Draw one pattern: thick baseline under price + name badge; make badge pickable to open dialog."""
        df = self._last_df
        # Resolve pattern name and description
        name = str(p.get("name") or p.get("pattern") or "Pattern")
        desc = str(p.get("description") or p.get("desc") or "Nessuna descrizione disponibile.")
        tf_lbl = str(p.get("tf")) if p.get("tf") else None

        # Resolve range preferring absolute timestamps
        ts_start = p.get("ts_start") or p.get("start_ts") or p.get("tsStart") or p.get("start")
        ts_end = p.get("ts_end") or p.get("end_ts") or p.get("tsEnd") or p.get("end")
        idx_start = p.get("idx_start")
        idx_end = p.get("idx_end")
        if ts_start is not None and ts_end is not None:
            try:
                ts_start_f = float(ts_start); ts_end_f = float(ts_end)
                idx_start = self._find_index_by_ts(ts_start_f)
                idx_end = self._find_index_by_ts(ts_end_f)
            except Exception:
                pass
        if idx_start is None or idx_end is None:
            try:
                # choose last 20 bars as fallback
                idx_end = len(df) - 1 if idx_end is None else int(idx_end)
                idx_start = max(0, idx_end - 20) if idx_start is None else int(idx_start)
            except Exception:
                return  # give up if cannot resolve

        # Clamp and ensure start <= end
        idx_start = max(0, min(int(idx_start), len(df) - 1))
        idx_end = max(idx_start, min(int(idx_end), len(df) - 1))

        # X coordinates (prefer timestamps for stability across TF/zoom)
        try:
            if ts_start is not None and ts_end is not None:
                x_start = float(self._ts_to_x(float(ts_start)))
                x_end = float(self._ts_to_x(float(ts_end)))
            else:
                if getattr(self, "_x_cache_comp", None) is None or len(self._x_cache_comp) != len(df):
                    self._rebuild_x_cache()
                x_start = float(self._x_cache_comp[idx_start])
                x_end = float(self._x_cache_comp[idx_end])
        except Exception:
            x_start = float(idx_start)
            x_end = float(idx_end)

        # Y baseline: slightly under the local price line (use low if available)
        y_series = self._series_for_y(df)
        try:
            if "low" in df.columns:
                y_base = float(df["low"].iloc[idx_start:idx_end + 1].min())
            else:
                y_base = float(y_series.iloc[idx_start:idx_end + 1].min())
        except Exception:
            y_base = float(y_series.iloc[max(0, idx_end - 1)])

        # Slight padding below the min to make it "under" the line
        try:
            ymin, ymax = self.ax.get_ylim()
            pad = (ymax - ymin) * 0.01
            y_base = max(ymin + pad * 0.5, y_base - pad * 2.0)
        except Exception:
            pass

        # Color and styles
        try:
            color = self._get_color("pattern.color", "#FFA500")
        except Exception:
            color = "#FFA500"

        # Draw thick horizontal "formation" line (no vertical band)
        line_artists = self.ax.plot([x_start, x_end], [y_base, y_base],
                                    color=color, linewidth=4.0, alpha=0.9, solid_capstyle="butt", zorder=12)
        if line_artists:
            self._pattern_artists.extend(line_artists)
            for la in line_artists:
                self._artist_to_pattern[la] = p

        # Badge with pattern name (with TF if available) near the end of the segment
        try:
            # Place slightly above the baseline for readability
            y_text = y_base
            try:
                ymin, ymax = self.ax.get_ylim()
                y_text = min(y_base + (ymax - ymin) * 0.02, ymax)
            except Exception:
                pass
            label_text = f"{name} [{tf_lbl}]" if tf_lbl else name
            badge = self.ax.annotate(
                label_text,
                xy=(x_end, y_text),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=9,
                color="#ffffff",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="black", lw=0.6, alpha=0.95),
                zorder=13,
                picker=True  # enable pick to open dialog
            )
            self._pattern_artists.append(badge)
            # Store augmented info for dialog
            p_aug = dict(p)
            p_aug.setdefault("name", name)
            p_aug.setdefault("description", desc)
            p_aug["idx_start"] = idx_start
            p_aug["idx_end"] = idx_end
            if ts_start is not None and ts_end is not None:
                try:
                    p_aug["ts_start"] = int(float(ts_start))
                    p_aug["ts_end"] = int(float(ts_end))
                except Exception:
                    pass
            self._artist_to_pattern[badge] = p_aug
        except Exception as e:
            logger.debug(f"badge annotate failed: {e}")

    def _find_index_by_ts(self, ts_ms: float) -> int:
        """Find nearest index by timestamp in ms."""
        try:
            import numpy as np
            arr = self._last_df["ts_utc"].to_numpy(dtype="float64")
            idx = int(np.searchsorted(arr, ts_ms))
            idx = max(0, min(idx, len(arr) - 1))
            # choose closer neighbor
            if idx > 0 and abs(arr[idx] - ts_ms) > abs(arr[idx - 1] - ts_ms):
                idx -= 1
            return idx
        except Exception:
            return max(0, len(self._last_df) - 1)

    def _ts_to_x(self, ts_ms: float) -> float:
        """Convert timestamp (ms) to matplotlib X respecting compression."""
        try:
            import pandas as _pd
            import matplotlib.dates as _md
            dt = _pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert("UTC").tz_localize(None)
            x = _md.date2num(dt)
            comp = getattr(getattr(self.chart_controller, "plot_service", None), "_compress_real_x", None)
            return float(comp(x)) if callable(comp) else float(x)
        except Exception:
            return float(ts_ms)

    def _series_for_y(self, df: pd.DataFrame) -> pd.Series:
        """Choose best Y series for badges/lines."""
        if getattr(self, "_price_mode", "candles") == "line" and "price" in df.columns:
            return df["price"]
        if "close" in df.columns:
            return df["close"]
        if "price" in df.columns:
            return df["price"]
        # fallback: first numeric column
        try:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            return df[num_cols[0]] if num_cols else df.iloc[:, 0]
        except Exception:
            return df.iloc[:, 0]

    def _on_pick_pattern_artist(self, event):
        """Open dialog with pattern details when a badge (or related artist) is clicked."""
        try:
            art = getattr(event, "artist", None)
            if art is None:
                return
            p = self._artist_to_pattern.get(art)
            if p is None:
                # try to resolve through line artists (no picker)
                return
            self._open_pattern_dialog(p)
        except Exception as e:
            logger.debug(f"_on_pick_pattern_artist failed: {e}")

    def _open_pattern_dialog(self, p: dict) -> None:
        """Show a dialog with pattern name, period and description."""
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle(str(p.get("name", "Pattern")))
            lay = QVBoxLayout(dlg)
            # Compose textual info
            name = str(p.get("name", "Pattern"))
            desc = str(p.get("description") or p.get("desc") or "Nessuna descrizione disponibile.")
            idx_start = p.get("idx_start"); idx_end = p.get("idx_end")
            period_txt = ""
            try:
                if idx_start is not None and idx_end is not None and "ts_utc" in self._last_df.columns:
                    ts1 = pd.to_datetime(float(self._last_df["ts_utc"].iloc[int(idx_start)]), unit="ms", utc=True).tz_convert("UTC").tz_localize(None)
                    ts2 = pd.to_datetime(float(self._last_df["ts_utc"].iloc[int(idx_end)]), unit="ms", utc=True).tz_convert("UTC").tz_localize(None)
                    period_txt = f"Periodo: {ts1} → {ts2}"
            except Exception:
                pass
            txt = f"<b>{name}</b><br/>{period_txt}<br/><br/>{desc}"
            lbl = QLabel(txt, dlg)
            lbl.setTextFormat(Qt.RichText)
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
            btn = QPushButton("Chiudi", dlg)
            btn.clicked.connect(dlg.accept)
            lay.addWidget(btn, alignment=Qt.AlignRight)
            dlg.resize(420, 220)
            dlg.exec()
        except Exception as e:
            logger.debug(f"_open_pattern_dialog failed: {e}")



class PatternsConfigDialog(QDialog):
    """
    Dialog di configurazione con due tab: Chart e Candela.
    Carica dinamicamente le chiavi da configs/patterns.yaml e consente di attivare/disattivare,
    settare soglie, confidence e Target/Stop (altezza figura, flagpole, etc.).
    """
    def __init__(self, parent=None, app=None):
        super().__init__(parent)
        self.setWindowTitle("Configura Patterns")
        self.resize(720, 520)
        self.app = app
        self._load_config()

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        self.chart_tab = QWidget(); self.candle_tab = QWidget()
        self.tabs.addTab(self.chart_tab, "Patterns a Chart")
        self.tabs.addTab(self.candle_tab, "Patterns a Candela")

        self._build_tab(self.chart_tab, kind="chart")
        self._build_tab(self.candle_tab, kind="candle")

        # Buttons
        btns = QHBoxLayout()
        self.btn_save = QPushButton("Salva")
        self.btn_cancel = QPushButton("Annulla")
        btns.addStretch(1); btns.addWidget(self.btn_save); btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)

    def _load_config(self):
        from pathlib import Path
        import yaml, json
        base = Path(__file__).resolve().parents[2] / "configs" / "patterns.yaml"
        with open(base, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.pattern_info = {}
        info_path = Path(__file__).resolve().parents[2] / "configs" / "pattern_info.json"
        if info_path.exists():
            try:
                self.pattern_info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                self.pattern_info = {}

    def _save_config(self):
        from pathlib import Path
        import yaml
        base = Path(__file__).resolve().parents[2] / "configs" / "patterns.yaml"
        with open(base, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, allow_unicode=True, sort_keys=False)

    def _pattern_keys(self, kind: str):
        # Legge dinamicamente la lista da patterns.yaml
        keys = self.cfg.get("patterns",{}).get(f"{kind}_patterns",{}).get("keys_enabled",[]) or []
        return list(keys)

    def _build_tab(self, tab_widget: QWidget, kind: str):
        layout = QVBoxLayout(tab_widget)
        keys = self._pattern_keys(kind)
        for key in keys:
            group = QGroupBox(key, tab_widget)
            form = QFormLayout(group)
            # Switch attivo
            chk = QCheckBox("Attivo", group); chk.setChecked(True)
            form.addRow("Stato:", chk)

            # Confidence
            conf = QDoubleSpinBox(group); conf.setRange(0.0,1.0); conf.setSingleStep(0.05); conf.setValue(0.5)
            form.addRow("Confidence:", conf)

            # Soglie generiche (es. ATR multipliers, min_touches, spans)
            atr_mult = QDoubleSpinBox(group); atr_mult.setRange(0.0,10.0); atr_mult.setSingleStep(0.1); atr_mult.setValue(2.0)
            form.addRow("ATR Mult:", atr_mult)

            min_touches = QSpinBox(group); min_touches.setRange(0,20); min_touches.setValue(4)
            form.addRow("Min touches:", min_touches)

            min_span = QSpinBox(group); min_span.setRange(5,500); min_span.setValue(30)
            form.addRow("Min span:", min_span)

            max_span = QSpinBox(group); max_span.setRange(10,2000); max_span.setValue(180)
            form.addRow("Max span:", max_span)

            # Target/Stop selector
            target_box = QComboBox(group)
            target_box.addItems(["Altezza figura", "Flag pole", "Ampiezza canale", "Custom"])
            form.addRow("Target/Stop:", target_box)

            layout.addWidget(group)

        layout.addStretch(1)

    def _on_save(self):
        # TODO: raccogliere i valori UI e scriverli in cfg['patterns'][f'{kind}_patterns']['params'][key]...
        # Qui salviamo solo il file così da sbloccare il wiring di base; l'implementazione può essere estesa.
        self._save_config()
        self.accept()
