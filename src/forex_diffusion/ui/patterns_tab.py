"""
Patterns Tab - Pattern detection and configuration
Standalone version of pattern controls from chart_tab/patterns_mixin.py
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QFormLayout
)
from loguru import logger


class PatternsTab(QWidget):
    """
    Tab for pattern detection and configuration.
    Contains controls previously in chart toolbar (patterns_mixin).
    """

    def __init__(self, parent=None, chart_tab=None):
        super().__init__(parent)
        self.chart_tab = chart_tab  # Reference to chart for pattern display
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header
        title_label = QLabel("📊 Pattern Detection & Analysis")
        title_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #2c3e50; }")
        layout.addWidget(title_label)

        # Pattern Types Group - Now references Chart tab checkboxes
        types_group = QGroupBox("Pattern Types")
        types_layout = QVBoxLayout(types_group)

        # Create info label explaining the connection
        info_label = QLabel("⚠️ Pattern controls are now in the Chart tab toolbar.\nThese checkboxes mirror the Chart tab settings.")
        info_label.setStyleSheet("QLabel { color: #95a5a6; font-style: italic; font-size: 11px; }")
        info_label.setWordWrap(True)
        types_layout.addWidget(info_label)

        # Reference Chart tab checkboxes instead of creating new ones
        # These will be synced with Chart tab checkboxes after widget creation
        self.cb_chart_patterns = QCheckBox("Chart Patterns")
        self.cb_chart_patterns.setToolTip(
            "Enable chart pattern detection (Head & Shoulders, Double Top/Bottom, Triangles, etc.)\n"
            "⚠️ This mirrors the Chart tab checkbox"
        )
        types_layout.addWidget(self.cb_chart_patterns)

        self.cb_candle_patterns = QCheckBox("Candlestick Patterns")
        self.cb_candle_patterns.setToolTip(
            "Enable candlestick pattern detection (Doji, Hammer, Engulfing, etc.)\n"
            "⚠️ This mirrors the Chart tab checkbox"
        )
        types_layout.addWidget(self.cb_candle_patterns)

        self.cb_history_patterns = QCheckBox("Historical Patterns")
        self.cb_history_patterns.setToolTip(
            "Show historical pattern occurrences on the chart\n"
            "⚠️ This mirrors the Chart tab checkbox"
        )
        types_layout.addWidget(self.cb_history_patterns)

        layout.addWidget(types_group)

        # Sync with Chart tab checkboxes after initialization
        self._sync_with_chart_checkboxes()

        # Actions Group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        button_layout = QHBoxLayout()

        self.btn_scan_historical = QPushButton("🔍 Scan Historical")
        self.btn_scan_historical.setToolTip("Scan historical data for pattern occurrences")
        self.btn_scan_historical.clicked.connect(self._scan_historical)
        button_layout.addWidget(self.btn_scan_historical)

        self.btn_config_patterns = QPushButton("⚙️ Configure")
        self.btn_config_patterns.setToolTip("Open pattern configuration dialog")
        self.btn_config_patterns.clicked.connect(self._open_patterns_config)
        button_layout.addWidget(self.btn_config_patterns)

        actions_layout.addLayout(button_layout)

        layout.addWidget(actions_group)

        # Settings Group
        settings_group = QGroupBox("Detection Settings")
        settings_form = QFormLayout(settings_group)

        self.detection_info = QLabel("Configure detection parameters in the Configuration dialog")
        self.detection_info.setWordWrap(True)
        self.detection_info.setStyleSheet("QLabel { color: #7f8c8d; font-style: italic; }")
        settings_form.addRow(self.detection_info)

        layout.addWidget(settings_group)

        layout.addStretch()

        logger.info("PatternsTab initialized")

    def _sync_with_chart_checkboxes(self):
        """Sync DUE checkboxes with Chart tab checkboxes."""
        if not self.chart_tab:
            logger.warning("No chart_tab reference for checkbox sync")
            return

        try:
            # Get Chart tab checkboxes
            chart_patterns_cb = getattr(self.chart_tab, 'chart_patterns_checkbox', None)
            candle_patterns_cb = getattr(self.chart_tab, 'candle_patterns_checkbox', None)
            history_patterns_cb = getattr(self.chart_tab, 'history_patterns_checkbox', None)

            # Connect DUE checkboxes to Chart checkboxes bidirectionally
            # Use blockSignals to prevent infinite loops
            if chart_patterns_cb:
                # DUE -> Chart
                self.cb_chart_patterns.toggled.connect(
                    lambda checked: self._sync_checkbox(chart_patterns_cb, checked)
                )
                # Chart -> DUE
                chart_patterns_cb.toggled.connect(
                    lambda checked: self._sync_checkbox(self.cb_chart_patterns, checked)
                )
                # Initial sync
                self.cb_chart_patterns.blockSignals(True)
                self.cb_chart_patterns.setChecked(chart_patterns_cb.isChecked())
                self.cb_chart_patterns.blockSignals(False)

            if candle_patterns_cb:
                # DUE -> Chart
                self.cb_candle_patterns.toggled.connect(
                    lambda checked: self._sync_checkbox(candle_patterns_cb, checked)
                )
                # Chart -> DUE
                candle_patterns_cb.toggled.connect(
                    lambda checked: self._sync_checkbox(self.cb_candle_patterns, checked)
                )
                # Initial sync
                self.cb_candle_patterns.blockSignals(True)
                self.cb_candle_patterns.setChecked(candle_patterns_cb.isChecked())
                self.cb_candle_patterns.blockSignals(False)

            if history_patterns_cb:
                # DUE -> Chart
                self.cb_history_patterns.toggled.connect(
                    lambda checked: self._sync_checkbox(history_patterns_cb, checked)
                )
                # Chart -> DUE
                history_patterns_cb.toggled.connect(
                    lambda checked: self._sync_checkbox(self.cb_history_patterns, checked)
                )
                # Initial sync
                self.cb_history_patterns.blockSignals(True)
                self.cb_history_patterns.setChecked(history_patterns_cb.isChecked())
                self.cb_history_patterns.blockSignals(False)

            logger.info("DUE checkboxes synced with Chart tab checkboxes")

        except Exception as e:
            logger.exception(f"Failed to sync checkboxes: {e}")

    def _sync_checkbox(self, target_checkbox, checked: bool):
        """Sync a checkbox state without triggering signals (prevent loops)."""
        target_checkbox.blockSignals(True)
        target_checkbox.setChecked(checked)
        target_checkbox.blockSignals(False)

    def _on_pattern_toggle(self, pattern_type: str, checked: bool):
        """Handle pattern type toggle - now redirects to Chart tab checkbox."""
        # This method is no longer used since checkboxes are directly connected
        # to Chart tab checkboxes in _sync_with_chart_checkboxes
        logger.debug(f"Pattern type '{pattern_type}' toggled: {checked} (handled by Chart tab)")

    def _scan_historical(self):
        """Scan historical data for patterns."""
        logger.info("Historical pattern scan requested")

        if not self.chart_tab:
            logger.warning("No chart_tab reference - cannot scan")
            return

        try:
            # Call pattern scan if chart_tab has the method
            if hasattr(self.chart_tab, '_scan_historical'):
                self.chart_tab._scan_historical()
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Historical Scan",
                    "Historical pattern scanning will be implemented soon."
                )

        except Exception as e:
            logger.exception(f"Historical scan failed: {e}")

    def _open_patterns_config(self):
        """Open pattern configuration dialog."""
        from .patterns_config_dialog import PatternsConfigDialog

        chart_controller = getattr(self.chart_tab, 'chart_controller', None) if self.chart_tab else None
        patterns_service = getattr(chart_controller, 'patterns_service', None) if chart_controller else None

        dialog = PatternsConfigDialog(
            parent=self,
            yaml_path="configs/patterns.yaml",
            patterns_service=patterns_service
        )

        if dialog.exec():
            # Refresh patterns if config was changed
            logger.info("Pattern configuration updated")
            if self.chart_tab and hasattr(self.chart_tab, '_refresh_patterns'):
                self.chart_tab._refresh_patterns()
