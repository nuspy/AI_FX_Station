"""
Patterns Tab - Contains Pattern Settings and Training functionality
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QSizePolicy
from loguru import logger


class PatternsTab(QWidget):
    """
    Tab containing Pattern Settings and Training functionality.
    Pattern detection controls are in the Chart tab.
    """

    def __init__(self, parent=None, chart_tab=None):
        super().__init__(parent)
        self.chart_tab = chart_tab
        self._init_ui()

    def _init_ui(self):
        """Initialize UI with nested tabs for Settings and Training"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create nested tab widget
        nested_tabs = QTabWidget()
        nested_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nested_tabs.setObjectName("level2_tabs_patterns")
        # Level 2 tabs: Medium contrast (consistent with other level 2 tabs)
        nested_tabs.setStyleSheet("""
            QTabWidget#level2_tabs_patterns::pane {
                border: 1px solid #444444;
                background: #2e2e2e;
            }
            QTabWidget#level2_tabs_patterns QTabBar::tab {
                background: #333333;
                color: #c0c0c0;
                padding: 6px 12px;
                margin-right: 1px;
                border: 1px solid #444444;
            }
            QTabWidget#level2_tabs_patterns QTabBar::tab:selected {
                background: #404040;
                color: #e0e0e0;
                border-bottom: 2px solid #0078d7;
            }
            QTabWidget#level2_tabs_patterns QTabBar::tab:hover {
                background: #3a3a3a;
            }
        """)

        # Create Settings tab (pattern configuration)
        from .patterns_settings_tab import PatternsSettingsTab
        settings_tab = PatternsSettingsTab(self)
        self.pattern_settings_tab = settings_tab  # Keep reference

        # Create Training tab (pattern training/backtest)
        from .pattern_training_tab import PatternTrainingTab
        training_tab = PatternTrainingTab(self)
        self.pattern_training_tab = training_tab  # Keep reference

        # Add nested tabs
        nested_tabs.addTab(settings_tab, "Settings")
        nested_tabs.addTab(training_tab, "Training")

        layout.addWidget(nested_tabs)

        logger.info("PatternsTab initialized with nested Settings and Training tabs")
