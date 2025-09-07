"""
SettingsDialog: simple dialog to edit user settings:
 - alpha_vantage_api_key
 - admin tokens (comma-separated token:role)
Settings persisted to ~/.config/magicforex/settings.json via user_settings.
"""

from __future__ import annotations

from typing import Optional
import os

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import Qt

from ..utils.user_settings import get_setting, set_setting


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(520, 180)
        layout = QVBoxLayout(self)

        # AlphaVantage API Key
        layout.addWidget(QLabel("AlphaVantage API Key:"))
        self.alpha_input = QLineEdit()
        self.alpha_input.setEchoMode(QLineEdit.Normal)
        self.alpha_input.setPlaceholderText("Enter AlphaVantage API key or leave blank to use env/config")
        layout.addWidget(self.alpha_input)

        # Admin tokens
        layout.addWidget(QLabel("ADMIN_TOKENS (comma-separated token:role e.g. tok1:admin,tok2:operator):"))
        self.admin_input = QLineEdit()
        self.admin_input.setPlaceholderText("token1:admin,token2:operator")
        layout.addWidget(self.admin_input)

        # Buttons
        btn_h = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.cancel_btn = QPushButton("Cancel")
        btn_h.addWidget(self.save_btn)
        btn_h.addWidget(self.cancel_btn)
        layout.addLayout(btn_h)

        self.save_btn.clicked.connect(self.on_save)
        self.cancel_btn.clicked.connect(self.reject)

        # Load current settings
        self.load_values()

    def load_values(self):
        alpha = get_setting("alpha_vantage_api_key", os.environ.get("ALPHAVANTAGE_KEY", "") or "")
        admin = get_setting("admin_tokens", os.environ.get("ADMIN_TOKENS", "") or "")
        self.alpha_input.setText(str(alpha))
        self.admin_input.setText(str(admin))

    def on_save(self):
        alpha = self.alpha_input.text().strip()
        admin = self.admin_input.text().strip()
        try:
            set_setting("alpha_vantage_api_key", alpha)
            set_setting("admin_tokens", admin)
            QMessageBox.information(self, "Settings", "Settings saved")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
