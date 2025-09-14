"""
SettingsDialog: simple dialog to edit user settings:
 - alpha_vantage_api_key
 - admin tokens (comma-separated token:role)
Settings persisted to ~/.config/magicforex/settings.json via user_settings.
"""

from __future__ import annotations

from typing import Optional
import os

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox, QComboBox, QFormLayout
from PySide6.QtCore import Qt

from ..utils.user_settings import get_setting, set_setting


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(560, 540)
        layout = QVBoxLayout(self)

        # Provider/API keys
        form = QFormLayout()
        self.alpha_input = QLineEdit(); self.alpha_input.setPlaceholderText("AlphaVantage API key")
        self.tiingo_input = QLineEdit(); self.tiingo_input.setPlaceholderText("Tiingo API key (or set env TIINGO_APIKEY)")
        form.addRow(QLabel("AlphaVantage API Key:"), self.alpha_input)
        form.addRow(QLabel("Tiingo API Key:"), self.tiingo_input)

        # Admin tokens
        self.admin_input = QLineEdit(); self.admin_input.setPlaceholderText("token1:admin,token2:operator")
        form.addRow(QLabel("ADMIN_TOKENS:"), self.admin_input)

        # Broker mode
        self.broker_mode = QComboBox(); self.broker_mode.addItems(["paper","ib","mt4","mt5"])
        form.addRow(QLabel("Broker Mode:"), self.broker_mode)

        # IB credentials
        self.ib_host = QLineEdit(); self.ib_host.setPlaceholderText("127.0.0.1")
        self.ib_port = QLineEdit(); self.ib_port.setPlaceholderText("7497")
        self.ib_client = QLineEdit(); self.ib_client.setPlaceholderText("1")
        self.ib_user = QLineEdit(); self.ib_user.setPlaceholderText("username")
        self.ib_pass = QLineEdit(); self.ib_pass.setEchoMode(QLineEdit.Password); self.ib_pass.setPlaceholderText("password")
        form.addRow(QLabel("IB Host:"), self.ib_host)
        form.addRow(QLabel("IB Port:"), self.ib_port)
        form.addRow(QLabel("IB Client ID:"), self.ib_client)
        form.addRow(QLabel("IB Username:"), self.ib_user)
        form.addRow(QLabel("IB Password:"), self.ib_pass)

        # MT credentials
        self.mt_server = QLineEdit(); self.mt_server.setPlaceholderText("broker server")
        self.mt_login = QLineEdit(); self.mt_login.setPlaceholderText("login")
        self.mt_pass = QLineEdit(); self.mt_pass.setEchoMode(QLineEdit.Password); self.mt_pass.setPlaceholderText("password")
        form.addRow(QLabel("MT Server:"), self.mt_server)
        form.addRow(QLabel("MT Login:"), self.mt_login)
        form.addRow(QLabel("MT Password:"), self.mt_pass)

        layout.addLayout(form)

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
        tiingo = get_setting("tiingo_api_key", os.environ.get("TIINGO_APIKEY", "") or "")
        admin = get_setting("admin_tokens", os.environ.get("ADMIN_TOKENS", "") or "")
        self.alpha_input.setText(str(alpha))
        self.tiingo_input.setText(str(tiingo))
        self.admin_input.setText(str(admin))
        # broker
        self.broker_mode.setCurrentText(str(get_setting("broker_mode","paper")))
        self.ib_host.setText(str(get_setting("ib_host","127.0.0.1")))
        self.ib_port.setText(str(get_setting("ib_port","7497")))
        self.ib_client.setText(str(get_setting("ib_client_id","1")))
        self.ib_user.setText(str(get_setting("ib_username","")))
        self.ib_pass.setText(str(get_setting("ib_password","")))
        self.mt_server.setText(str(get_setting("mt_server","")))
        self.mt_login.setText(str(get_setting("mt_login","")))
        self.mt_pass.setText(str(get_setting("mt_password","")))

    def on_save(self):
        try:
            set_setting("alpha_vantage_api_key", self.alpha_input.text().strip())
            set_setting("tiingo_api_key", self.tiingo_input.text().strip())
            set_setting("admin_tokens", self.admin_input.text().strip())
            set_setting("broker_mode", self.broker_mode.currentText())
            set_setting("ib_host", self.ib_host.text().strip())
            set_setting("ib_port", self.ib_port.text().strip())
            set_setting("ib_client_id", self.ib_client.text().strip())
            set_setting("ib_username", self.ib_user.text().strip())
            set_setting("ib_password", self.ib_pass.text().strip())
            set_setting("mt_server", self.mt_server.text().strip())
            set_setting("mt_login", self.mt_login.text().strip())
            set_setting("mt_password", self.mt_pass.text().strip())
            QMessageBox.information(self, "Settings", "Settings saved")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
