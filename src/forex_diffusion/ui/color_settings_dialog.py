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

from ..utils.user_settings import get_setting, set_setting
from .settings_dialog import COLOR_FIELDS, COLOR_DEFAULTS


class ColorSettingsDialog(QDialog):
    """Lightweight dialog to tweak theme colors quickly."""

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
        self.accept()
