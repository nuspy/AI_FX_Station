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
from ...utils.user_settings import get_setting, set_setting

# Import finplot (required)
import finplot as fplt
from datetime import datetime
import pyqtgraph as pg


class DateAxisItem(pg.AxisItem):
    """Custom axis item for displaying timestamps as dates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.date_format = "YYYY-MM-DD"

    def set_date_format(self, fmt: str):
        """Set date format from settings (YYYY-MM-DD, DD-MM-YYYY, etc)."""
        self.date_format = fmt

    def tickStrings(self, values, scale, spacing):
        """Override to convert timestamps to formatted dates."""
        strings = []
        for v in values:
            try:
                dt = datetime.fromtimestamp(v)
                # Convert format from settings to strftime format
                if self.date_format == "YYYY-MM-DD":
                    formatted = dt.strftime("%Y-%m-%d %H:%M")
                elif self.date_format == "YYYY-DD-MM":
                    formatted = dt.strftime("%Y-%d-%m %H:%M")
                elif self.date_format == "DD-MM-YYYY":
                    formatted = dt.strftime("%d-%m-%Y %H:%M")
                elif self.date_format == "MM-DD-YYYY":
                    formatted = dt.strftime("%m-%d-%Y %H:%M")
                else:
                    formatted = dt.strftime("%Y-%m-%d %H:%M")
                strings.append(formatted)
            except:
                strings.append("")
        return strings


class UIBuilderMixin:
    """Mixin containing all UI construction methods for ChartTab."""

    def _build_ui(self) -> None:
        """Programmatically builds the entire UI with finplot as the primary chart widget."""
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # Create topbar and populate it with two rows
        topbar = QWidget()
        topbar.setObjectName("topbar")
        self._populate_topbar(topbar)
        self.layout().addWidget(topbar)

        # Create main tab widget
        self.main_tabs = QTabWidget()
        self.layout().addWidget(self.main_tabs)

        # Chart tab (finplot-based)
        self._create_chart_tab()

        # Training/Backtest tab (pattern training - restored from commit 11d3627)
        self._create_training_tab()

        # Signals tab
        self._create_signals_tab()

        # 3D Reports tab
        self._create_3d_reports_tab()

        # Log Monitoring tab moved to top-level (see logs_tab.py)

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

        # Create PyQtGraph widget for high-performance charting
        # PyQtGraph is what finplot uses internally - we use it directly for full control
        chart_container = QWidget()
        chart_container_layout = QVBoxLayout(chart_container)
        chart_container_layout.setContentsMargins(0,0,0,0)

        # Use PyQtGraph PlotWidget directly for embedding
        from pyqtgraph import PlotWidget, GraphicsLayoutWidget
        self.graphics_layout = GraphicsLayoutWidget()
        self.canvas = self.graphics_layout  # Keep canvas reference
        self.use_finplot = True  # Flag to use PyQtGraph-based plotting
        self.finplot_axes = []

        # Create custom date axis
        date_format = get_setting("chart.date_format", "YYYY-MM-DD")
        self.date_axis = DateAxisItem(orientation='bottom')
        self.date_axis.set_date_format(date_format)

        # Create initial plot with custom date axis
        self.main_plot = self.graphics_layout.addPlot(row=0, col=0, axisItems={'bottom': self.date_axis})
        self.finplot_axes.append(self.main_plot)

        # Add active provider label in top-right corner
        self.provider_label = pg.TextItem(text="Provider: ...", color=(200, 200, 200), anchor=(1, 0))
        self.main_plot.addItem(self.provider_label)
        # Position will be updated when data is loaded

        chart_container_layout.addWidget(self.graphics_layout)
        self.chart_container = chart_container  # keep reference for overlays
        chart_area_splitter.addWidget(chart_container)

        right_splitter.addWidget(chart_area_splitter)

        self.orders_table = QTableWidget(0, 9)
        self.orders_table.setHorizontalHeaderLabels(["ID", "Time", "Symbol", "Type", "Volume", "Price", "SL", "TP", "Status"])
        # Hide vertical header (row numbers) and minimize margins
        self.orders_table.verticalHeader().setVisible(False)
        self.orders_table.setContentsMargins(0, 0, 0, 0)
        right_splitter.addWidget(self.orders_table)
        # Minimize splitter handle width
        right_splitter.setHandleWidth(1)

        main_splitter.addWidget(right_splitter)

        # Set up chart tab layout
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)
        chart_layout.addWidget(main_splitter)

        # Set stretch factors to make the chart area expand
        main_splitter.setStretchFactor(1, 8)
        right_splitter.setStretchFactor(0, 6)
        chart_area_splitter.setStretchFactor(1, 1)

        self.main_tabs.addTab(chart_tab, "Chart")

    def _create_training_tab(self) -> None:
        """Create the pattern Training/Backtest tab (restored from commit 11d3627)"""
        from ..pattern_training_tab import PatternTrainingTab

        training_tab = PatternTrainingTab(self)
        self.pattern_training_tab = training_tab  # Keep reference

        self.main_tabs.addTab(training_tab, "Training/Backtest")

    def _create_signals_tab(self) -> None:
        """Create the Signals tab for trading signals analysis"""
        signals_tab = QWidget()
        signals_layout = QVBoxLayout(signals_tab)
        signals_layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header_label = QLabel("📊 Trading Signals")
        header_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #2c3e50; }")
        signals_layout.addWidget(header_label)

        # Placeholder content
        content_label = QLabel("Signals tab - Coming soon")
        content_label.setStyleSheet("QLabel { color: #7f8c8d; font-size: 14px; }")
        signals_layout.addWidget(content_label)

        signals_layout.addStretch()

        self.main_tabs.addTab(signals_tab, "Signals")

    def _create_3d_reports_tab(self) -> None:
        """Create the 3D Reports tab for advanced visualization"""
        reports_tab = QWidget()
        reports_layout = QVBoxLayout(reports_tab)
        reports_layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header_label = QLabel("📈 3D Reports & Visualization")
        header_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #2c3e50; }")
        reports_layout.addWidget(header_label)

        # Placeholder content
        content_label = QLabel("3D Reports tab - Coming soon")
        content_label.setStyleSheet("QLabel { color: #7f8c8d; font-size: 14px; }")
        reports_layout.addWidget(content_label)

        reports_layout.addStretch()

        self.main_tabs.addTab(reports_tab, "3D Reports")

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
        self.tf_combo.addItems(["auto", "tick", "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"])
        row1_layout.addWidget(self.tf_combo)

        # Data range controls
        row1_layout.addWidget(QLabel("Years:"))
        self.years_combo = QComboBox()
        self.years_combo.addItems(["0", "1", "2", "3", "5", "10", "20", "30"])
        row1_layout.addWidget(self.years_combo)

        row1_layout.addWidget(QLabel("Months:"))
        self.months_combo = QComboBox()
        self.months_combo.addItems(["0", "1", "3", "6", "9", "12"])
        row1_layout.addWidget(self.months_combo)

        row1_layout.addStretch()

        # Chart controls
        self.backfill_btn = QPushButton("📊 Backfill")
        row1_layout.addWidget(self.backfill_btn)

        # Backfill progress bar (initially hidden)
        self.backfill_progress = QProgressBar()
        self.backfill_progress.setRange(0, 100)
        self.backfill_progress.setValue(0)
        self.backfill_progress.setVisible(False)
        self.backfill_progress.setMaximumWidth(150)
        row1_layout.addWidget(self.backfill_progress)

        self.indicators_btn = QPushButton("📈 Indicators")
        row1_layout.addWidget(self.indicators_btn)

        self.build_latents_btn = QPushButton("🧠 Build Latents")
        row1_layout.addWidget(self.build_latents_btn)

        # Forecast controls (Settings buttons removed - now in Generative Forecast tab)
        self.forecast_btn = QPushButton("🔮 Forecast")
        row1_layout.addWidget(self.forecast_btn)

        self.adv_forecast_btn = QPushButton("🔮 Adv Forecast")
        row1_layout.addWidget(self.adv_forecast_btn)

        self.clear_forecasts_btn = QPushButton("🗑️ Clear")
        row1_layout.addWidget(self.clear_forecasts_btn)

        # Second row - display and interface controls
        row2_widget = QWidget()
        row2_layout = QHBoxLayout(row2_widget)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(5)

        # Pattern controls moved to DUE > Patterns tab

        row2_layout.addStretch()

        # Display and interface controls
        self.toggle_drawbar_btn = QCheckBox("Drawbar")
        row2_layout.addWidget(self.toggle_drawbar_btn)

        self.mode_btn = QCheckBox("Candles")
        self.mode_btn.setChecked(True)
        row2_layout.addWidget(self.mode_btn)

        self.follow_checkbox = QCheckBox("Follow")
        row2_layout.addWidget(self.follow_checkbox)

        # Pattern checkboxes (TASK 5)
        self.chart_patterns_checkbox = QCheckBox("Chart Patterns")
        self.chart_patterns_checkbox.setChecked(True)  # Default enabled
        row2_layout.addWidget(self.chart_patterns_checkbox)

        self.candle_patterns_checkbox = QCheckBox("Candle Patterns")
        self.candle_patterns_checkbox.setChecked(True)  # Default enabled
        row2_layout.addWidget(self.candle_patterns_checkbox)

        self.history_patterns_checkbox = QCheckBox("Historical")
        row2_layout.addWidget(self.history_patterns_checkbox)

        # Pattern action buttons (from DUE tab)
        self.scan_historical_btn = QPushButton("Scan Historical")
        row2_layout.addWidget(self.scan_historical_btn)

        self.setting_patterns_btn = QPushButton("Setting Patterns")
        row2_layout.addWidget(self.setting_patterns_btn)

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

    # Training tab methods moved to pattern_training_tab.py (commit 11d3627)

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