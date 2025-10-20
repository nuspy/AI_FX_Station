"""
Forecast Settings Tab - contains Base Settings and Advanced Settings as nested tabs.
This is the embedded version of UnifiedPredictionSettingsDialog for use as a main tab.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Signal
from loguru import logger

from .unified_prediction_settings_dialog import UnifiedPredictionSettingsDialog


class ForecastSettingsTab(QWidget):
    """
    Tab containing forecast settings with Base and Advanced nested tabs.
    Reuses the UnifiedPredictionSettingsDialog content but as an embedded widget.
    """

    # Signal emitted when settings are changed
    settingsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create the settings dialog content (without dialog wrapper)
        self.settings_dialog = UnifiedPredictionSettingsDialog(self)

        # Remove dialog buttons (OK/Cancel) - they're not needed in a tab
        if hasattr(self.settings_dialog, 'button_box'):
            self.settings_dialog.button_box.setParent(None)
            self.settings_dialog.button_box.deleteLater()

        # Extract the main content (tabs) from the dialog
        # The dialog has a main layout containing: tabs, button_layout, button_box
        # We want to embed everything except button_box
        dialog_layout = self.settings_dialog.layout()

        # Add all items from dialog layout to our layout
        # This includes: tabs, Save/Load/Reset buttons
        while dialog_layout.count() > 0:
            item = dialog_layout.takeAt(0)
            if item.widget():
                layout.addWidget(item.widget())
            elif item.layout():
                layout.addLayout(item.layout())

        # Add Apply button at the bottom
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.apply_button = QPushButton("Apply Settings")
        self.apply_button.clicked.connect(self._on_apply)
        self.apply_button.setToolTip("Applica le impostazioni correnti (salva in memoria)")

        button_layout.addWidget(self.apply_button)
        layout.addLayout(button_layout)

        logger.info("ForecastSettingsTab initialized")

    def _on_apply(self):
        """Handle Apply button - save settings and emit signal"""
        self.settings_dialog.save_settings()
        self.settingsChanged.emit()
        logger.info("Forecast settings applied")

    def get_settings(self):
        """Get current settings from the embedded dialog"""
        return self.settings_dialog.get_settings()

    def set_settings(self, settings):
        """Set settings in the embedded dialog"""
        self.settings_dialog.set_settings(settings)

    def load_settings(self):
        """Load saved settings"""
        self.settings_dialog.load_settings()

    def save_settings(self):
        """Save current settings"""
        self.settings_dialog.save_settings()
