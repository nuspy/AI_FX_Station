"""
Logs Tab - System monitoring and log visualization
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QTabWidget
)
from PySide6.QtCore import QTimer
from loguru import logger


class LogsTab(QWidget):
    """
    Top-level tab for system logs and monitoring.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = None
        self._init_ui()

    def set_main_window(self, main_window):
        """Set reference to main window for data sources tab."""
        self._main_window = main_window
        if hasattr(self, 'data_sources_tab'):
            self.data_sources_tab.set_main_window(main_window)

    def _init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

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

        layout.addLayout(header_layout)

        # Log tabs widget
        self.logs_tab = QTabWidget()
        layout.addWidget(self.logs_tab)

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

        layout.addLayout(status_layout)

        # Auto-refresh timer
        self._log_auto_refresh_timer = QTimer()
        self._log_auto_refresh_timer.timeout.connect(self._refresh_logs)
        if self.auto_refresh_logs.isChecked():
            self._log_auto_refresh_timer.start(30000)  # 30 seconds

        self.auto_refresh_logs.toggled.connect(self._toggle_log_auto_refresh)

        logger.info("LogsTab initialized")

    def _create_log_subtabs(self) -> None:
        """Create individual log sub-tabs for different log categories."""
        # Data Sources tab
        from .data_sources_tab import DataSourcesTab
        self.data_sources_tab = DataSourcesTab()
        self.logs_tab.addTab(self.data_sources_tab, "Data Sources")

        # TODO: Implement other log sub-tabs (e.g., System, Training, Inference, Errors)
        placeholder = QLabel("Log monitoring will be implemented here")
        self.logs_tab.addTab(placeholder, "All Logs")

    def _clear_all_logs(self):
        """Clear all log entries."""
        logger.info("Clear all logs requested")
        # TODO: Implement log clearing
        self.log_stats_label.setText("Entries: 0 | Warnings: 0 | Errors: 0")

    def _export_logs(self):
        """Export logs to file."""
        logger.info("Export logs requested")
        # TODO: Implement log export
        pass

    def _refresh_logs(self):
        """Refresh log display."""
        # TODO: Implement log refresh
        pass

    def _toggle_log_auto_refresh(self, checked: bool):
        """Toggle auto-refresh for logs."""
        if checked:
            self._log_auto_refresh_timer.start(30000)
            logger.info("Auto-refresh enabled (30s interval)")
        else:
            self._log_auto_refresh_timer.stop()
            logger.info("Auto-refresh disabled")
