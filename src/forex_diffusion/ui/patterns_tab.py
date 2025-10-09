"""
Patterns Tab - Contains Pattern Training/Backtest functionality
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout
from loguru import logger


class PatternsTab(QWidget):
    """
    Tab containing Pattern Training/Backtest functionality.
    Pattern detection controls are in the Chart tab.
    """

    def __init__(self, parent=None, chart_tab=None):
        super().__init__(parent)
        self.chart_tab = chart_tab
        self._init_ui()

    def _init_ui(self):
        """Initialize UI with Pattern Training/Backtest tab"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Import and add Pattern Training/Backtest tab
        from .pattern_training_tab import PatternTrainingTab

        training_tab = PatternTrainingTab(self)
        self.pattern_training_tab = training_tab  # Keep reference

        layout.addWidget(training_tab)

        logger.info("PatternsTab initialized with Pattern Training/Backtest")
