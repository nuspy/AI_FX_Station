"""
UI Builder Mixin for ChartTab - handles all UI construction methods.
"""
from __future__ import annotations

from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QSplitter, QListWidget, QTableWidget, QComboBox,
    QToolButton, QCheckBox, QProgressBar, QScrollArea, QGroupBox,
    QTabWidget, QSpinBox, QDoubleSpinBox, QTextEdit, QFormLayout, QGridLayout,
    QSlider, QFrame, QButtonGroup, QRadioButton, QDateEdit
)
from PySide6.QtCore import QTimer, Qt, QDate
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from ...utils.user_settings import get_setting, set_setting

# Try to import finplot
try:
    import finplot as fplt
    FINPLOT_AVAILABLE = True
except ImportError:
    FINPLOT_AVAILABLE = False
    fplt = None


class UIBuilderMixin:
    """Mixin containing all UI construction methods for ChartTab."""

    def _build_ui(self) -> None:
        """Programmatically builds the entire UI with a two-row topbar and tab structure."""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # Check chart system setting
        chart_system = get_setting("chart.system", "matplotlib")

        if chart_system in ["finplot", "finplot_enhanced"] and FINPLOT_AVAILABLE:
            # Use finplot - note: create_plot_widget expects 'master' not 'parent'
            self.canvas = fplt.create_plot_widget(master=self, init_zoom_periods=100)
            self.use_finplot = True
            # Initialize finplot axes (will be created by finplot dynamically)
            self.finplot_axes = []
        else:
            # Use matplotlib (default)
            self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
            self.use_finplot = False

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

        # Log Monitoring tab (for missing parameters and system monitoring)
        self._create_logs_tab()

        # Set stretch factors
        self.layout().setStretch(1, 1)

    def _create_chart_tab(self) -> None:
        """Create the original chart tab with existing functionality"""
        chart_tab = QWidget()

        # Create main content area with splitters (original structure)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = main_splitter

        market_panel = QWidget()
        market_layout = QVBoxLayout(market_panel)
        market_layout.setContentsMargins(0,0,0,0)
        self.market_watch = QListWidget()
        market_layout.addWidget(self.market_watch)
        main_splitter.addWidget(market_panel)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter = right_splitter

        chart_area_splitter = QSplitter(Qt.Orientation.Vertical)
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

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

    def _create_logs_tab(self) -> None:
        """Create the log monitoring tab for system monitoring and missing parameters"""
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        logs_layout.setContentsMargins(10, 10, 10, 10)

        # Header section
        header_layout = QHBoxLayout()

        # Title
        title_label = QLabel("🔍 System Logs & Monitoring")
        title_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #2c3e50; }")
        header_layout.addWidget(title_label)

        # Control buttons
        header_layout.addStretch()

        self.auto_refresh_logs = QCheckBox("Auto Refresh")
        self.auto_refresh_logs.setChecked(True)
        self.auto_refresh_logs.setToolTip("Automatically refresh logs every 30 seconds")
        header_layout.addWidget(self.auto_refresh_logs)

        clear_logs_btn = QPushButton("🗑️ Clear All")
        clear_logs_btn.setToolTip("Clear all log entries")
        clear_logs_btn.clicked.connect(self._clear_all_logs)
        header_layout.addWidget(clear_logs_btn)

        export_logs_btn = QPushButton("📁 Export")
        export_logs_btn.setToolTip("Export logs to file")
        export_logs_btn.clicked.connect(self._export_logs)
        header_layout.addWidget(export_logs_btn)

        logs_layout.addLayout(header_layout)

        # Log tabs widget
        self.logs_tab = QTabWidget()
        logs_layout.addWidget(self.logs_tab)

        # Create individual log tabs based on configuration
        self._create_log_subtabs()

        # Status bar
        status_layout = QHBoxLayout()

        self.log_status_label = QLabel("✅ Monitoring active")
        self.log_status_label.setStyleSheet("QLabel { color: #27ae60; }")
        status_layout.addWidget(self.log_status_label)

        status_layout.addStretch()

        # Statistics
        self.log_stats_label = QLabel("Entries: 0 | Warnings: 0 | Errors: 0")
        self.log_stats_label.setStyleSheet("QLabel { color: #7f8c8d; font-size: 11px; }")
        status_layout.addWidget(self.log_stats_label)

        logs_layout.addLayout(status_layout)

        # Auto-refresh timer
        self._log_auto_refresh_timer = QTimer()
        self._log_auto_refresh_timer.timeout.connect(self._refresh_logs)
        if self.auto_refresh_logs.isChecked():
            self._log_auto_refresh_timer.start(30000)  # 30 seconds

        self.auto_refresh_logs.toggled.connect(self._toggle_log_auto_refresh)

        self.main_tabs.addTab(logs_tab, "Logs")

    def _create_drawbar(self) -> QWidget:
        """Create the drawing toolbar."""
        drawbar = QWidget()
        drawbar.setObjectName("drawbar")
        drawbar_layout = QHBoxLayout(drawbar)
        drawbar_layout.setContentsMargins(2, 2, 2, 2)
        drawbar_layout.setSpacing(2)

        # Drawing tools
        tools = [
            ("🎯", "Select", None),
            ("📏", "Line", "line"),
            ("🔢", "Levels", "levels"),
            ("📐", "Trend", "trend"),
            ("📊", "Rectangle", "rectangle"),
            ("⭕", "Circle", "circle")
        ]

        self.draw_buttons = []
        for icon, name, mode in tools:
            btn = QToolButton()
            btn.setText(f"{icon} {name}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mode: self._set_drawing_mode(m if checked else None))
            drawbar_layout.addWidget(btn)
            self.draw_buttons.append(btn)

        drawbar_layout.addStretch()

        # Clear drawings button
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self._clear_drawings)
        drawbar_layout.addWidget(clear_btn)

        return drawbar

    def _populate_topbar(self, topbar: QWidget):
        """Populate the topbar with two rows of controls."""
        topbar_v_layout = QVBoxLayout(topbar)
        topbar_v_layout.setContentsMargins(5, 2, 5, 2)
        topbar_v_layout.setSpacing(2)

        # First row - main controls
        row1_widget = QWidget()
        row1_layout = QHBoxLayout(row1_widget)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(5)

        # Symbol selection
        row1_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.addItems(["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"])
        row1_layout.addWidget(self.symbol_combo)

        # Timeframe selection
        row1_layout.addWidget(QLabel("TF:"))
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"])
        row1_layout.addWidget(self.tf_combo)

        # Data range controls
        row1_layout.addWidget(QLabel("Range:"))
        self.months_combo = QComboBox()
        self.months_combo.addItems(["1", "3", "6", "12", "24", "All"])
        row1_layout.addWidget(self.months_combo)

        row1_layout.addStretch()

        # Chart controls
        self.backfill_btn = QPushButton("📊 Backfill")
        row1_layout.addWidget(self.backfill_btn)

        self.indicators_btn = QPushButton("📈 Indicators")
        row1_layout.addWidget(self.indicators_btn)

        self.build_latents_btn = QPushButton("🧠 Build Latents")
        row1_layout.addWidget(self.build_latents_btn)

        # Forecast controls
        self.forecast_settings_btn = QPushButton("⚙️ Settings")
        row1_layout.addWidget(self.forecast_settings_btn)

        self.forecast_btn = QPushButton("🔮 Forecast")
        row1_layout.addWidget(self.forecast_btn)

        self.adv_settings_btn = QPushButton("⚙️ Adv Settings")
        row1_layout.addWidget(self.adv_settings_btn)

        self.adv_forecast_btn = QPushButton("🔮 Adv Forecast")
        row1_layout.addWidget(self.adv_forecast_btn)

        self.clear_forecasts_btn = QPushButton("🗑️ Clear")
        row1_layout.addWidget(self.clear_forecasts_btn)

        # Second row - additional controls and toggles
        row2_widget = QWidget()
        row2_layout = QHBoxLayout(row2_widget)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(5)

        # Pattern controls
        row2_layout.addWidget(QLabel("Patterns:"))
        self.cb_chart_patterns = QCheckBox("Chart")
        self.cb_candle_patterns = QCheckBox("Candle")
        self.cb_history_patterns = QCheckBox("History")
        row2_layout.addWidget(self.cb_chart_patterns)
        row2_layout.addWidget(self.cb_candle_patterns)
        row2_layout.addWidget(self.cb_history_patterns)

        self.btn_scan_historical = QPushButton("🔍 Scan")
        self.btn_config_patterns = QPushButton("⚙️ Config")
        row2_layout.addWidget(self.btn_scan_historical)
        row2_layout.addWidget(self.btn_config_patterns)

        row2_layout.addStretch()

        # Display and interface controls
        self.toggle_drawbar_btn = QCheckBox("Drawbar")
        row2_layout.addWidget(self.toggle_drawbar_btn)

        self.mode_btn = QCheckBox("Candles")
        self.mode_btn.setChecked(True)
        row2_layout.addWidget(self.mode_btn)

        self.follow_checkbox = QCheckBox("Follow")
        row2_layout.addWidget(self.follow_checkbox)

        self.bidask_label = QLabel("Bid: -    Ask: -")
        row2_layout.addWidget(self.bidask_label)

        self.trade_btn = QPushButton("Trade")
        row2_layout.addWidget(self.trade_btn)

        row2_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light", "blue"])
        row2_layout.addWidget(self.theme_combo)

        self.settings_btn = QPushButton("Settings")
        row2_layout.addWidget(self.settings_btn)

        row2_layout.addStretch()

        topbar_v_layout.addWidget(row1_widget)
        topbar_v_layout.addWidget(row2_widget)

    # Large methods for creating complex sections - these would be implemented
    # by reading the remaining sections from the original file

    def _create_study_setup_section(self, layout: QVBoxLayout) -> None:
        """Create the study setup section - stub for now."""
        # This would contain the full implementation from the original file
        pass

    def _create_dataset_config_section(self, layout: QVBoxLayout) -> None:
        """Create the dataset configuration section - stub for now."""
        pass

    def _create_parameter_space_section(self, layout: QVBoxLayout) -> None:
        """Create the parameter space section - stub for now."""
        pass

    def _create_optimization_config_section(self, layout: QVBoxLayout) -> None:
        """Create the optimization configuration section - stub for now."""
        pass

    def _create_execution_control_section(self, layout: QVBoxLayout) -> None:
        """Create the execution control section - stub for now."""
        pass

    def _create_results_section(self, layout: QVBoxLayout) -> None:
        """Create the results section - stub for now."""
        pass

    def _create_log_subtabs(self) -> None:
        """Create individual log sub-tabs for different log categories."""
        pass

    # Utility methods for UI management
    def _clear_drawings(self):
        """Clear all drawings from the chart."""
        pass

    def _clear_all_logs(self):
        """Clear all log entries."""
        pass

    def _export_logs(self):
        """Export logs to file."""
        pass

    def _refresh_logs(self):
        """Refresh log display."""
        pass

    def _toggle_log_auto_refresh(self, checked: bool):
        """Toggle auto-refresh for logs."""
        if checked:
            self._log_auto_refresh_timer.start(30000)
        else:
            self._log_auto_refresh_timer.stop()