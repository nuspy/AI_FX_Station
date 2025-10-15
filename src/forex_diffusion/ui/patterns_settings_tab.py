"""
Patterns Settings Tab - Contains pattern configuration UI
"""
from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout
from .patterns_config_dialog import PatternsConfigDialog


class PatternsSettingsTab(QWidget):
    """
    Tab containing pattern configuration UI.
    This wraps PatternsConfigDialog content in a widget instead of a dialog.
    """

    def __init__(self, parent=None, yaml_path: str = "configs/patterns.yaml", patterns_service=None):
        super().__init__(parent)

        # Create the dialog content widget (without the dialog wrapper)
        # We'll instantiate the dialog but embed its content here
        self.config_content = PatternsConfigDialog(
            parent=self,
            yaml_path=yaml_path,
            patterns_service=patterns_service
        )

        # Remove window flags to make it behave like a widget
        self.config_content.setWindowFlags(self.config_content.windowFlags() & ~0x00000001)  # Remove Qt.Window flag

        # Embed the dialog content in this widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.config_content)
