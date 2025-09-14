# src/forex_diffusion/ui/color_settings_dialog.py
from __future__ import annotations
# src/forex_diffusion/ui/color_settings_dialog.py
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QColorDialog, QDialogButtonBox
from ..utils.user_settings import get_setting, set_setting

COLOR_KEYS = {
    "price_color": "#e0e0e0",
    "hline_color": "#9bdcff",
    "trend_color": "#ff9bdc",
    "rect_color": "#f0c674",
    "fib_color": "#9fe6a0",
    "label_color": "#ffd479",
}

class ColorSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Settings")
        self.resize(420, 260)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.edits = {}
        for key, default in COLOR_KEYS.items():
            e = QLineEdit(get_setting(key, default))
            btn = QPushButton("Pick")
            btn.clicked.connect(lambda _, ed=e: self._pick(ed))
            form.addRow(QLabel(key.replace("_"," ").title()), e)
            self.edits[key] = e
        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save); btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _pick(self, edit: QLineEdit):
        c = QColorDialog.getColor()
        if c.isValid():
            edit.setText(c.name())

    def _save(self):
        for k, e in self.edits.items():
            set_setting(k, e.text().strip())
        self.accept()
from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QColorDialog, QDialogButtonBox
from ..utils.user_settings import get_setting, set_setting

DEFAULTS = {
    "price_color": "#e0e0e0",
    "hline_color": "#8bdcff",
    "trend_color": "#ff9bdc",
    "rect_color": "#f0c674",
    "fib_color": "#9fe6a0",
    "label_color": "#ffd479",
}

class ColorSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Settings")
        self.resize(420, 240)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.edits = {}
        for k in DEFAULTS.keys():
            e = QLineEdit(get_setting(f"color_{k}", DEFAULTS[k]))
            btn = QPushButton("Pick")
            def _mk(e=e):
                return lambda: self._pick(e)
            btn.clicked.connect(_mk())
            row = QVBoxLayout()
            row.addWidget(e); row.addWidget(btn)
            form.addRow(QLabel(k.replace("_"," ").title()), e)
            self.edits[k] = e
        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save); btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _pick(self, edit: QLineEdit):
        c = QColorDialog.getColor()
        if c.isValid(): edit.setText(c.name())

    def _save(self):
        for k, e in self.edits.items():
            set_setting(f"color_{k}", e.text().strip())
        self.accept()
