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
        self.resize(600, 600)
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

        # Accounts (simulato)
        self.acc_name = QLineEdit(); self.acc_name.setPlaceholderText("default")
        self.acc_currency = QLineEdit(); self.acc_currency.setPlaceholderText("USD")
        self.acc_balance = QLineEdit(); self.acc_balance.setPlaceholderText("100000")
        self.acc_leverage = QLineEdit(); self.acc_leverage.setPlaceholderText("30")
        self.acc_tiingo = QLineEdit(); self.acc_tiingo.setPlaceholderText("Account Tiingo API key (optional)")
        form.addRow(QLabel("Active Account:"), self.acc_name)
        form.addRow(QLabel("Currency:"), self.acc_currency)
        form.addRow(QLabel("Balance:"), self.acc_balance)
        form.addRow(QLabel("Leverage:"), self.acc_leverage)
        form.addRow(QLabel("Account Tiingo Key:"), self.acc_tiingo)

        # Lista accounts + azioni
        from PySide6.QtWidgets import QListWidget, QHBoxLayout
        self.accounts_list = QListWidget()
        layout.addWidget(QLabel("Accounts:"))
        layout.addWidget(self.accounts_list)
        acc_btns = QHBoxLayout()
        self.btn_acc_add = QPushButton("Add/Update")
        self.btn_acc_remove = QPushButton("Remove")
        self.btn_acc_set_active = QPushButton("Set Active")
        acc_btns.addWidget(self.btn_acc_add); acc_btns.addWidget(self.btn_acc_remove); acc_btns.addWidget(self.btn_acc_set_active)
        layout.addLayout(acc_btns)
        self.btn_acc_add.clicked.connect(self._on_account_add_update)
        self.btn_acc_remove.clicked.connect(self._on_account_remove)
        self.btn_acc_set_active.clicked.connect(self._on_account_set_active)

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
        # account
        self.acc_name.setText(str(get_setting("active_account","default")))
        self.acc_currency.setText(str(get_setting("account_currency","USD")))
        self.acc_balance.setText(str(get_setting("account_balance","100000")))
        self.acc_leverage.setText(str(get_setting("account_leverage","30")))
        self.acc_tiingo.setText(str(get_setting("account_tiingo_api_key","")))
        # IB / MT
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
            # account
            set_setting("active_account", self.acc_name.text().strip() or "default")
            set_setting("account_currency", self.acc_currency.text().strip() or "USD")
            set_setting("account_balance", self.acc_balance.text().strip() or "100000")
            set_setting("account_leverage", self.acc_leverage.text().strip() or "30")
            set_setting("account_tiingo_api_key", self.acc_tiingo.text().strip())
            # IB / MT
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
