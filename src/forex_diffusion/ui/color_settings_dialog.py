"""Standalone color editor dialog sharing the same keys as SettingsDialog."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QColorDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)
from PySide6.QtCore import Signal

from ..utils.user_settings import get_setting, set_setting
from .settings_dialog import COLOR_FIELDS, COLOR_DEFAULTS


class ColorSettingsDialog(QDialog):
    """Lightweight dialog to tweak theme colors quickly."""

    # Signal emitted when colors are saved
    themeChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Color Settings")
        self.resize(520, 360)

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        self.edits: dict[str, QLineEdit] = {}

        for row, (key, label) in enumerate(COLOR_FIELDS):
            grid.addWidget(QLabel(label), row, 0)
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setText(str(get_setting(key, COLOR_DEFAULTS.get(key, "#000000"))))
            btn = QPushButton("Change")
            btn.clicked.connect(lambda _, k=key: self._pick(k))
            grid.addWidget(edit, row, 1)
            grid.addWidget(btn, row, 2)
            self.edits[key] = edit

        layout.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        if hasattr(self, 'theme_combo'):
            apply_tooltip(self.theme_combo, "theme_selection", "color_themes")
        if hasattr(self, 'bullish_color_btn'):
            apply_tooltip(self.bullish_color_btn, "bullish_color", "color_themes")
        if hasattr(self, 'bearish_color_btn'):
            apply_tooltip(self.bearish_color_btn, "bearish_color", "color_themes")
        if hasattr(self, 'grid_opacity_spin'):
            apply_tooltip(self.grid_opacity_spin, "grid_opacity", "color_themes")
        if hasattr(self, 'custom_theme_import_btn'):
            apply_tooltip(self.custom_theme_import_btn, "custom_theme_import", "color_themes")
    

    def _pick(self, key: str) -> None:
        current = self.edits[key].text().strip() or COLOR_DEFAULTS.get(key, "#000000")
        color = QColor(current)
        dialog = QColorDialog(color, self)
        dialog.setOption(QColorDialog.ShowAlphaChannel, True)
        if dialog.exec():
            chosen = dialog.selectedColor()
            if chosen.isValid():
                self.edits[key].setText(chosen.name(QColor.HexArgb))

    def _save(self) -> None:
        for key, edit in self.edits.items():
            set_setting(key, edit.text().strip())
        # Emit signal to trigger theme refresh
        self.themeChanged.emit()
        self.accept()
