# src/forex_diffusion/ui/chart_tab_ui.py
from __future__ import annotations

from typing import Optional, Dict, List
import pandas as pd
import time

import matplotlib.dates as mdates

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox,
    QSplitter, QListWidget, QListWidgetItem, QTableWidget, QComboBox,
    QToolButton, QCheckBox, QProgressBar, QScrollArea, QDialog, QGroupBox,
    QTabWidget, QSpinBox, QDoubleSpinBox, QTextEdit, QFormLayout, QGridLayout,
    QSlider, QFrame, QButtonGroup, QRadioButton
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
        """Programmatically builds the entire UI with a two-row topbar and tab structure."""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))

        # Create topbar and populate it with two rows
        topbar = QWidget()
        topbar.setObjectName("topbar")
        self._populate_topbar(topbar)
        self.layout().addWidget(topbar)

        # Create main tab widget
        self.main_tabs = QTabWidget()
        self.layout().addWidget(self.main_tabs)

        # Chart tab (original functionality)
        self._create_chart_tab()

        # Training/Backtest tab (new functionality)
        self._create_training_tab()

        # Set stretch factors
        self.layout().setStretch(1, 1)

    def _create_chart_tab(self) -> None:
        """Create the original chart tab with existing functionality"""
        chart_tab = QWidget()

        # Create main content area with splitters (original structure)
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

        # Set up chart tab layout
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(main_splitter)

        # Set stretch factors to make the chart area expand
        main_splitter.setStretchFactor(1, 8)
        right_splitter.setStretchFactor(0, 6)
        chart_area_splitter.setStretchFactor(1, 1)

        self.main_tabs.addTab(chart_tab, "Chart")

    def _create_training_tab(self) -> None:
        """Create the new Training/Backtest tab"""
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        training_layout.setContentsMargins(10, 10, 10, 10)

        # Create scrollable area for all the controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # === Study Setup Section ===
        self._create_study_setup_section(scroll_layout)

        # === Dataset Configuration Section ===
        self._create_dataset_config_section(scroll_layout)

        # === Parameter Space Section ===
        self._create_parameter_space_section(scroll_layout)

        # === Optimization Configuration Section ===
        self._create_optimization_config_section(scroll_layout)

        # === Execution Control Section ===
        self._create_execution_control_section(scroll_layout)

        # === Results and Status Section ===
        self._create_results_section(scroll_layout)

        scroll.setWidget(scroll_content)
        training_layout.addWidget(scroll)

        self.main_tabs.addTab(training_tab, "Training/Backtest")

    def _create_study_setup_section(self, layout: QVBoxLayout) -> None:
        """Create study setup section"""
        group = QGroupBox("Study Setup")
        group_layout = QFormLayout(group)

        # Pattern selection
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems([
            "head_and_shoulders", "inverse_head_and_shoulders", "triangle_ascending",
            "triangle_descending", "triangle_symmetrical", "wedge_rising", "wedge_falling",
            "double_top", "double_bottom", "triple_top", "triple_bottom",
            "rectangle", "flag_bull", "flag_bear", "pennant", "diamond",
            "cup_and_handle", "rounding_bottom", "broadening_top"
        ])
        group_layout.addRow("Pattern:", self.pattern_combo)

        # Direction selection
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["bull", "bear"])
        group_layout.addRow("Direction:", self.direction_combo)

        # Asset selection (multiple)
        self.assets_edit = QTextEdit()
        self.assets_edit.setMaximumHeight(60)
        self.assets_edit.setPlainText("EUR/USD, GBP/USD, USD/JPY")
        group_layout.addRow("Assets (comma separated):", self.assets_edit)

        # Timeframe selection (multiple)
        self.timeframes_edit = QTextEdit()
        self.timeframes_edit.setMaximumHeight(60)
        self.timeframes_edit.setPlainText("1h, 4h, 1d")
        group_layout.addRow("Timeframes (comma separated):", self.timeframes_edit)

        # Regime selection
        self.regime_combo = QComboBox()
        self.regime_combo.addItems([
            "None", "nber_recession", "nber_expansion", "vix_low", "vix_medium", "vix_high",
            "pmi_above_50", "pmi_below_50", "epu_low", "epu_medium", "epu_high"
        ])
        group_layout.addRow("Regime Filter:", self.regime_combo)

        layout.addWidget(group)

    def _create_dataset_config_section(self, layout: QVBoxLayout) -> None:
        """Create dataset configuration section"""
        group = QGroupBox("Dataset Configuration")
        group_layout = QVBoxLayout(group)

        # Multi-objective toggle
        self.multi_objective_checkbox = QCheckBox("Enable Multi-Objective Optimization")
        self.multi_objective_checkbox.setChecked(True)
        group_layout.addWidget(self.multi_objective_checkbox)

        # Dataset 1 configuration
        d1_frame = QFrame()
        d1_frame.setFrameStyle(QFrame.StyledPanel)
        d1_layout = QFormLayout(d1_frame)

        self.d1_start_date = QTextEdit()
        self.d1_start_date.setMaximumHeight(30)
        self.d1_start_date.setPlainText("2020-01-01")
        d1_layout.addRow("D1 Start Date:", self.d1_start_date)

        self.d1_end_date = QTextEdit()
        self.d1_end_date.setMaximumHeight(30)
        self.d1_end_date.setPlainText("2022-12-31")
        d1_layout.addRow("D1 End Date:", self.d1_end_date)

        group_layout.addWidget(QLabel("Dataset 1 (Primary):"))
        group_layout.addWidget(d1_frame)

        # Dataset 2 configuration
        d2_frame = QFrame()
        d2_frame.setFrameStyle(QFrame.StyledPanel)
        d2_layout = QFormLayout(d2_frame)

        self.d2_start_date = QTextEdit()
        self.d2_start_date.setMaximumHeight(30)
        self.d2_start_date.setPlainText("2023-01-01")
        d2_layout.addRow("D2 Start Date:", self.d2_start_date)

        self.d2_end_date = QTextEdit()
        self.d2_end_date.setMaximumHeight(30)
        self.d2_end_date.setPlainText("2024-12-31")
        d2_layout.addRow("D2 End Date:", self.d2_end_date)

        group_layout.addWidget(QLabel("Dataset 2 (Secondary):"))
        group_layout.addWidget(d2_frame)

        # Objective weights (when not using pure multi-objective)
        weights_frame = QFrame()
        weights_layout = QFormLayout(weights_frame)

        self.d1_weight_slider = QSlider(Qt.Horizontal)
        self.d1_weight_slider.setMinimum(10)
        self.d1_weight_slider.setMaximum(90)
        self.d1_weight_slider.setValue(70)
        self.d1_weight_label = QLabel("70%")
        self.d1_weight_slider.valueChanged.connect(lambda v: self.d1_weight_label.setText(f"{v}%"))

        d1_weight_layout = QHBoxLayout()
        d1_weight_layout.addWidget(self.d1_weight_slider)
        d1_weight_layout.addWidget(self.d1_weight_label)
        weights_layout.addRow("D1 Weight:", d1_weight_layout)

        group_layout.addWidget(QLabel("Objective Weights (for single-objective mode):"))
        group_layout.addWidget(weights_frame)

        layout.addWidget(group)

    def _create_parameter_space_section(self, layout: QVBoxLayout) -> None:
        """Create parameter space configuration section"""
        group = QGroupBox("Parameter Space Configuration")
        group_layout = QVBoxLayout(group)

        # Form parameters
        form_group = QGroupBox("Form Parameters (Pattern Detection)")
        form_layout = QGridLayout(form_group)

        # Min touches
        form_layout.addWidget(QLabel("Min Touches:"), 0, 0)
        self.min_touches_min = QSpinBox()
        self.min_touches_min.setMinimum(2)
        self.min_touches_min.setMaximum(10)
        self.min_touches_min.setValue(3)
        form_layout.addWidget(self.min_touches_min, 0, 1)
        form_layout.addWidget(QLabel("to"), 0, 2)
        self.min_touches_max = QSpinBox()
        self.min_touches_max.setMinimum(2)
        self.min_touches_max.setMaximum(10)
        self.min_touches_max.setValue(6)
        form_layout.addWidget(self.min_touches_max, 0, 3)

        # Span range
        form_layout.addWidget(QLabel("Min Span (bars):"), 1, 0)
        self.min_span_min = QSpinBox()
        self.min_span_min.setMinimum(5)
        self.min_span_min.setMaximum(200)
        self.min_span_min.setValue(10)
        form_layout.addWidget(self.min_span_min, 1, 1)
        form_layout.addWidget(QLabel("to"), 1, 2)
        self.min_span_max = QSpinBox()
        self.min_span_max.setMinimum(5)
        self.min_span_max.setMaximum(200)
        self.min_span_max.setValue(50)
        form_layout.addWidget(self.min_span_max, 1, 3)

        form_layout.addWidget(QLabel("Max Span (bars):"), 2, 0)
        self.max_span_min = QSpinBox()
        self.max_span_min.setMinimum(20)
        self.max_span_min.setMaximum(500)
        self.max_span_min.setValue(50)
        form_layout.addWidget(self.max_span_min, 2, 1)
        form_layout.addWidget(QLabel("to"), 2, 2)
        self.max_span_max = QSpinBox()
        self.max_span_max.setMinimum(20)
        self.max_span_max.setMaximum(500)
        self.max_span_max.setValue(150)
        form_layout.addWidget(self.max_span_max, 2, 3)

        # Tolerance
        form_layout.addWidget(QLabel("Tolerance:"), 3, 0)
        self.tolerance_min = QDoubleSpinBox()
        self.tolerance_min.setMinimum(0.001)
        self.tolerance_min.setMaximum(0.1)
        self.tolerance_min.setValue(0.005)
        self.tolerance_min.setSingleStep(0.001)
        self.tolerance_min.setDecimals(4)
        form_layout.addWidget(self.tolerance_min, 3, 1)
        form_layout.addWidget(QLabel("to"), 3, 2)
        self.tolerance_max = QDoubleSpinBox()
        self.tolerance_max.setMinimum(0.001)
        self.tolerance_max.setMaximum(0.1)
        self.tolerance_max.setValue(0.05)
        self.tolerance_max.setSingleStep(0.001)
        self.tolerance_max.setDecimals(4)
        form_layout.addWidget(self.tolerance_max, 3, 3)

        group_layout.addWidget(form_group)

        # Action parameters
        action_group = QGroupBox("Action Parameters (Execution)")
        action_layout = QGridLayout(action_group)

        # Target mode
        action_layout.addWidget(QLabel("Target Modes:"), 0, 0)
        self.target_modes_edit = QTextEdit()
        self.target_modes_edit.setMaximumHeight(60)
        self.target_modes_edit.setPlainText("Altezza figura, Flag pole, Ampiezza canale")
        action_layout.addWidget(self.target_modes_edit, 0, 1, 1, 3)

        # Risk-reward ratio
        action_layout.addWidget(QLabel("Risk/Reward Ratio:"), 1, 0)
        self.rr_min = QDoubleSpinBox()
        self.rr_min.setMinimum(0.5)
        self.rr_min.setMaximum(10.0)
        self.rr_min.setValue(1.0)
        self.rr_min.setSingleStep(0.1)
        action_layout.addWidget(self.rr_min, 1, 1)
        action_layout.addWidget(QLabel("to"), 1, 2)
        self.rr_max = QDoubleSpinBox()
        self.rr_max.setMinimum(0.5)
        self.rr_max.setMaximum(10.0)
        self.rr_max.setValue(3.0)
        self.rr_max.setSingleStep(0.1)
        action_layout.addWidget(self.rr_max, 1, 3)

        # ATR buffer
        action_layout.addWidget(QLabel("ATR Buffer:"), 2, 0)
        self.atr_buffer_min = QDoubleSpinBox()
        self.atr_buffer_min.setMinimum(0.0)
        self.atr_buffer_min.setMaximum(5.0)
        self.atr_buffer_min.setValue(0.5)
        self.atr_buffer_min.setSingleStep(0.1)
        action_layout.addWidget(self.atr_buffer_min, 2, 1)
        action_layout.addWidget(QLabel("to"), 2, 2)
        self.atr_buffer_max = QDoubleSpinBox()
        self.atr_buffer_max.setMinimum(0.0)
        self.atr_buffer_max.setMaximum(5.0)
        self.atr_buffer_max.setValue(2.0)
        self.atr_buffer_max.setSingleStep(0.1)
        action_layout.addWidget(self.atr_buffer_max, 2, 3)

        group_layout.addWidget(action_group)

        # Suggested ranges button
        self.suggested_ranges_btn = QPushButton("Show Suggested Ranges for Selected Pattern")
        self.suggested_ranges_btn.clicked.connect(self._show_suggested_ranges)
        group_layout.addWidget(self.suggested_ranges_btn)

        layout.addWidget(group)

    def _create_optimization_config_section(self, layout: QVBoxLayout) -> None:
        """Create optimization configuration section"""
        group = QGroupBox("Optimization Configuration")
        group_layout = QFormLayout(group)

        # Max trials
        self.max_trials_spin = QSpinBox()
        self.max_trials_spin.setMinimum(10)
        self.max_trials_spin.setMaximum(10000)
        self.max_trials_spin.setValue(1000)
        group_layout.addRow("Max Trials:", self.max_trials_spin)

        # Max duration
        self.max_duration_spin = QDoubleSpinBox()
        self.max_duration_spin.setMinimum(0.1)
        self.max_duration_spin.setMaximum(168.0)  # 1 week
        self.max_duration_spin.setValue(24.0)
        self.max_duration_spin.setSuffix(" hours")
        group_layout.addRow("Max Duration:", self.max_duration_spin)

        # Parallel workers
        self.parallel_workers_spin = QSpinBox()
        self.parallel_workers_spin.setMinimum(1)
        self.parallel_workers_spin.setMaximum(64)
        self.parallel_workers_spin.setValue(32)
        group_layout.addRow("Parallel Workers:", self.parallel_workers_spin)

        # Early stopping alpha
        self.early_stopping_alpha_spin = QDoubleSpinBox()
        self.early_stopping_alpha_spin.setMinimum(0.1)
        self.early_stopping_alpha_spin.setMaximum(1.0)
        self.early_stopping_alpha_spin.setValue(0.8)
        self.early_stopping_alpha_spin.setSingleStep(0.05)
        group_layout.addRow("Early Stopping Alpha:", self.early_stopping_alpha_spin)

        # Invalidation rule parameters
        group_layout.addRow(QLabel(""), QLabel(""))  # Separator
        group_layout.addRow(QLabel("Invalidation Rules:"), QLabel(""))

        self.k_time_spin = QDoubleSpinBox()
        self.k_time_spin.setMinimum(1.0)
        self.k_time_spin.setMaximum(10.0)
        self.k_time_spin.setValue(4.0)
        self.k_time_spin.setSingleStep(0.5)
        group_layout.addRow("Time Multiplier (k_time):", self.k_time_spin)

        self.k_loss_spin = QDoubleSpinBox()
        self.k_loss_spin.setMinimum(1.0)
        self.k_loss_spin.setMaximum(10.0)
        self.k_loss_spin.setValue(4.0)
        self.k_loss_spin.setSingleStep(0.5)
        group_layout.addRow("Loss Multiplier (k_loss):", self.k_loss_spin)

        self.quantile_spin = QDoubleSpinBox()
        self.quantile_spin.setMinimum(0.5)
        self.quantile_spin.setMaximum(0.95)
        self.quantile_spin.setValue(0.75)
        self.quantile_spin.setSingleStep(0.05)
        group_layout.addRow("Quantile Threshold:", self.quantile_spin)

        layout.addWidget(group)

    def _create_execution_control_section(self, layout: QVBoxLayout) -> None:
        """Create execution control section"""
        group = QGroupBox("Execution Control")
        group_layout = QVBoxLayout(group)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.start_optimization_btn = QPushButton("Start Optimization")
        self.start_optimization_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.start_optimization_btn.clicked.connect(self._start_optimization)
        buttons_layout.addWidget(self.start_optimization_btn)

        self.pause_optimization_btn = QPushButton("Pause")
        self.pause_optimization_btn.clicked.connect(self._pause_optimization)
        self.pause_optimization_btn.setEnabled(False)
        buttons_layout.addWidget(self.pause_optimization_btn)

        self.resume_optimization_btn = QPushButton("Resume")
        self.resume_optimization_btn.clicked.connect(self._resume_optimization)
        self.resume_optimization_btn.setEnabled(False)
        buttons_layout.addWidget(self.resume_optimization_btn)

        self.stop_optimization_btn = QPushButton("Stop")
        self.stop_optimization_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_optimization_btn.clicked.connect(self._stop_optimization)
        self.stop_optimization_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_optimization_btn)

        group_layout.addLayout(buttons_layout)

        # Status display
        self.optimization_status_label = QLabel("Status: Ready")
        self.optimization_status_label.setStyleSheet("QLabel { font-weight: bold; }")
        group_layout.addWidget(self.optimization_status_label)

        # Progress bar
        self.optimization_progress = QProgressBar()
        self.optimization_progress.setVisible(False)
        group_layout.addWidget(self.optimization_progress)

        layout.addWidget(group)

    def _create_results_section(self, layout: QVBoxLayout) -> None:
        """Create results and status section"""
        group = QGroupBox("Results and Analysis")
        group_layout = QVBoxLayout(group)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Study status tab
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)

        self.study_status_text = QTextEdit()
        self.study_status_text.setMaximumHeight(150)
        self.study_status_text.setReadOnly(True)
        self.study_status_text.setPlainText("No optimization running...")
        status_layout.addWidget(self.study_status_text)

        self.refresh_status_btn = QPushButton("Refresh Status")
        self.refresh_status_btn.clicked.connect(self._refresh_status)
        status_layout.addWidget(self.refresh_status_btn)

        self.results_tabs.addTab(status_tab, "Status")

        # Best parameters tab
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)

        self.best_params_text = QTextEdit()
        self.best_params_text.setReadOnly(True)
        self.best_params_text.setPlainText("No results yet...")
        params_layout.addWidget(self.best_params_text)

        promote_layout = QHBoxLayout()
        self.promote_params_btn = QPushButton("Promote Parameters")
        self.promote_params_btn.clicked.connect(self._promote_parameters)
        self.promote_params_btn.setEnabled(False)
        promote_layout.addWidget(self.promote_params_btn)

        self.rollback_params_btn = QPushButton("Rollback Parameters")
        self.rollback_params_btn.clicked.connect(self._rollback_parameters)
        promote_layout.addWidget(self.rollback_params_btn)

        promote_layout.addStretch()
        params_layout.addLayout(promote_layout)

        self.results_tabs.addTab(params_tab, "Best Parameters")

        # Performance breakdown tab
        perf_tab = QWidget()
        perf_layout = QVBoxLayout(perf_tab)

        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        perf_layout.addWidget(self.performance_text)

        self.results_tabs.addTab(perf_tab, "Performance")

        group_layout.addWidget(self.results_tabs)

        layout.addWidget(group)

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
        self.btn_scan_historical = QToolButton()
        self.btn_scan_historical.setText("Scan Historical")
        self.btn_config_patterns = QToolButton()
        self.btn_config_patterns.setText("Configura Patterns")
        self.btn_scan_patterns.setText("Scansiona patterns")
        row2_layout.addWidget(self.btn_scan_patterns)
        row2_layout.addWidget(self.btn_scan_historical)
        row2_layout.addWidget(self.btn_config_patterns)



        # handler

        def _scan_historical():
            try:
                ps = self.chart_controller.patterns_service
                if ps is None: return
                symbol = getattr(self, "symbol", None) or (self.symbol_combo.currentText() if hasattr(self, "symbol_combo") else None)
                if not symbol: return
                view_df = getattr(self.chart_controller.plot_service, "_last_df", None)
                if view_df is None or view_df.empty: return
                # Scan on all tfs using current df snapshot
                tfs = ["1m","5m","15m","30m","1h","4h","1d"]
                for tf in tfs:
                    try:
                        self._patterns_scan_tf_hint = tf
                        ps.start_historical_scan(view_df)
                    except Exception:
                        continue
            except Exception as e:
                from loguru import logger
                logger.exception("Historical scan failed: {}", e)

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
        self.btn_scan_historical.clicked.connect(_scan_historical)
        self.btn_config_patterns.clicked.connect(self._open_patterns_config)

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


    def _collect_and_save(self, kind: str, container: QWidget):
        from pathlib import Path
        import yaml
        base = Path(__file__).resolve().parents[2] / "configs" / "patterns.yaml"
        with open(base, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kp = cfg.get("patterns", {}).get(f"{kind}_patterns", {})
        keys_enabled = set(kp.get("keys_enabled", []) or [])
        params = kp.get("params", {}) or {}
        for group in container.findChildren(QGroupBox):
            key = group.title()
            form = group.layout()
            chk = form.itemAt(1).widget(); conf = form.itemAt(3).widget(); atr_mult = form.itemAt(5).widget()
            min_touches = form.itemAt(7).widget(); min_span = form.itemAt(9).widget(); max_span = form.itemAt(11).widget()
            target_box = form.itemAt(13).widget()
            if chk.isChecked(): keys_enabled.add(key)
            else: keys_enabled.discard(key)
            p = params.get(key, {})
            p["confidence"] = float(conf.value()); p["atr_mult"] = float(atr_mult.value())
            p["min_touches"] = int(min_touches.value()); p["min_span"] = int(min_span.value()); p["max_span"] = int(max_span.value())
            p["target_mode"] = target_box.currentText()
            params[key] = p
        kp["keys_enabled"] = list(keys_enabled)
        kp["params"] = params
        cfg["patterns"][f"{kind}_patterns"] = kp
        with open(base, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    def _on_save(self):
        self._collect_and_save("chart", self.chart_tab)
        self._collect_and_save("candle", self.candle_tab)
        self.accept()

    # Training/Backtest tab handler methods
    def _show_suggested_ranges(self):
        """Show suggested parameter ranges based on historical data"""
        try:
            from ..training.optimization.parameter_space import ParameterSpace
            param_space = ParameterSpace()

            # Get current pattern type
            pattern_type = self.pattern_combo.currentText()
            if not pattern_type:
                self._show_message("Please select a pattern type first")
                return

            suggested = param_space.get_suggested_ranges(pattern_type)

            # Display in a dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Suggested Ranges for {pattern_type}")
            dialog.setModal(True)
            dialog.resize(400, 300)

            layout = QVBoxLayout(dialog)
            scroll = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)

            for param, ranges in suggested.items():
                group = QGroupBox(param)
                group_layout = QFormLayout()
                group_layout.addRow("Min:", QLabel(str(ranges.get('min', 'N/A'))))
                group_layout.addRow("Max:", QLabel(str(ranges.get('max', 'N/A'))))
                group_layout.addRow("Default:", QLabel(str(ranges.get('default', 'N/A'))))
                group.setLayout(group_layout)
                scroll_layout.addWidget(group)

            scroll.setWidget(scroll_widget)
            layout.addWidget(scroll)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            buttons.accepted.connect(dialog.accept)
            layout.addWidget(buttons)

            dialog.exec()

        except Exception as e:
            self._show_message(f"Error showing suggested ranges: {str(e)}")

    def _start_optimization(self):
        """Start optimization process"""
        try:
            config = self._collect_optimization_config()
            if not self._validate_config(config):
                return

            from ..training.optimization.engine import OptimizationEngine

            # Create and start optimization
            self.optimization_engine = OptimizationEngine()

            # Update UI state
            self._update_optimization_ui_state(running=True)

            # Start optimization in background
            def run_optimization():
                try:
                    self.optimization_engine.run_study(config)
                    self._update_optimization_ui_state(running=False)
                    self._show_message("Optimization completed successfully")
                except Exception as e:
                    self._update_optimization_ui_state(running=False)
                    self._show_message(f"Optimization failed: {str(e)}")

            from threading import Thread
            self.optimization_thread = Thread(target=run_optimization)
            self.optimization_thread.start()

        except Exception as e:
            self._show_message(f"Error starting optimization: {str(e)}")

    def _pause_optimization(self):
        """Pause optimization process"""
        try:
            if hasattr(self, 'optimization_engine') and self.optimization_engine:
                self.optimization_engine.pause_study()
                self._update_optimization_ui_state(paused=True)
                self._show_message("Optimization paused")
        except Exception as e:
            self._show_message(f"Error pausing optimization: {str(e)}")

    def _resume_optimization(self):
        """Resume optimization process"""
        try:
            if hasattr(self, 'optimization_engine') and self.optimization_engine:
                self.optimization_engine.resume_study()
                self._update_optimization_ui_state(paused=False)
                self._show_message("Optimization resumed")
        except Exception as e:
            self._show_message(f"Error resuming optimization: {str(e)}")

    def _stop_optimization(self):
        """Stop optimization process"""
        try:
            if hasattr(self, 'optimization_engine') and self.optimization_engine:
                self.optimization_engine.stop_study()
                self._update_optimization_ui_state(running=False)
                self._show_message("Optimization stopped")
        except Exception as e:
            self._show_message(f"Error stopping optimization: {str(e)}")

    def _refresh_status(self):
        """Refresh optimization status display"""
        try:
            if hasattr(self, 'optimization_engine') and self.optimization_engine:
                status = self.optimization_engine.get_study_status()

                # Update progress display
                if 'progress' in status:
                    progress = status['progress']
                    self.progress_bar.setValue(int(progress * 100))

                # Update status text
                status_text = f"Status: {status.get('status', 'Unknown')}\n"
                status_text += f"Trials: {status.get('completed_trials', 0)}/{status.get('total_trials', 0)}\n"
                status_text += f"Best Score: {status.get('best_score', 'N/A')}\n"
                status_text += f"Runtime: {status.get('runtime', 'N/A')}"

                # Find and update status label
                for child in self.training_tab.findChildren(QLabel):
                    if hasattr(child, 'objectName') and child.objectName() == 'status_label':
                        child.setText(status_text)
                        break

        except Exception as e:
            self._show_message(f"Error refreshing status: {str(e)}")

    def _promote_parameters(self):
        """Promote best parameters to production"""
        try:
            from ..training.optimization.task_manager import TaskManager

            task_manager = TaskManager()
            result = task_manager.promote_best_parameters()

            if result['success']:
                self._show_message(f"Parameters promoted successfully. Version: {result['version']}")
            else:
                self._show_message(f"Failed to promote parameters: {result['error']}")

        except Exception as e:
            self._show_message(f"Error promoting parameters: {str(e)}")

    def _rollback_parameters(self):
        """Rollback to previous parameter version"""
        try:
            from ..training.optimization.task_manager import TaskManager

            task_manager = TaskManager()
            result = task_manager.rollback_parameters()

            if result['success']:
                self._show_message(f"Parameters rolled back to version: {result['version']}")
            else:
                self._show_message(f"Failed to rollback parameters: {result['error']}")

        except Exception as e:
            self._show_message(f"Error rolling back parameters: {str(e)}")

    def _collect_optimization_config(self) -> dict:
        """Collect optimization configuration from UI"""
        config = {
            'study_name': '',
            'pattern_type': '',
            'datasets': [],
            'parameter_space': {},
            'optimization_config': {},
            'execution_config': {}
        }

        try:
            # Find widgets and collect values
            for child in self.training_tab.findChildren((QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox)):
                if hasattr(child, 'objectName') and child.objectName():
                    name = child.objectName()

                    if isinstance(child, QLineEdit):
                        value = child.text()
                    elif isinstance(child, (QSpinBox, QDoubleSpinBox)):
                        value = child.value()
                    elif isinstance(child, QComboBox):
                        value = child.currentText()
                    elif isinstance(child, QCheckBox):
                        value = child.isChecked()
                    else:
                        continue

                    # Organize by section
                    if name.startswith('study_'):
                        config['study_name'] = value if name == 'study_name' else config['study_name']
                        config['pattern_type'] = value if name == 'study_pattern_type' else config['pattern_type']
                    elif name.startswith('dataset_'):
                        # Handle dataset configuration
                        pass
                    elif name.startswith('param_'):
                        # Handle parameter space
                        pass
                    elif name.startswith('opt_'):
                        # Handle optimization config
                        pass
                    elif name.startswith('exec_'):
                        # Handle execution config
                        pass

        except Exception as e:
            logger.warning(f"Error collecting optimization config: {e}")

        return config

    def _validate_config(self, config: dict) -> bool:
        """Validate optimization configuration"""
        try:
            # Basic validation
            if not config.get('study_name'):
                self._show_message("Please enter a study name")
                return False

            if not config.get('pattern_type'):
                self._show_message("Please select a pattern type")
                return False

            # Add more validation as needed
            return True

        except Exception as e:
            self._show_message(f"Configuration validation error: {str(e)}")
            return False

    def _update_optimization_ui_state(self, running=False, paused=False):
        """Update UI state based on optimization status"""
        try:
            # Find control buttons and update their state
            for child in self.training_tab.findChildren(QPushButton):
                if hasattr(child, 'objectName') and child.objectName():
                    name = child.objectName()

                    if name == 'start_btn':
                        child.setEnabled(not running)
                    elif name == 'pause_btn':
                        child.setEnabled(running and not paused)
                    elif name == 'resume_btn':
                        child.setEnabled(running and paused)
                    elif name == 'stop_btn':
                        child.setEnabled(running)

        except Exception as e:
            logger.warning(f"Error updating UI state: {e}")

    def _show_message(self, message: str):
        """Show message to user"""
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Information", message)
        except Exception as e:
            logger.error(f"Error showing message: {e}")


    def _open_patterns_config(self):
        from .patterns_config_dialog import PatternsConfigDialog

        import os
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'configs', 'patterns.yaml')
        dlg = PatternsConfigDialog(self, yaml_path=yaml_path)
        dlg.exec()
