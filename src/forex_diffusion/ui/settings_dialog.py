"""Settings dialog exposing API credentials, broker accounts, chart behaviour, and theme colors."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from functools import partial  # added


from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout, QCheckBox,
)
from loguru import logger

from ..utils.user_settings import get_setting, set_setting

COLOR_FIELDS = [
    ("window_bg", "Sfondo finestra"),
    ("panel_bg", "Sfondo pannelli"),
    ("border_color", "Bordi finestra"),
    ("splitter_handle_color", "Divisori"),
    ("chart_bg", "Sfondo grafico"),
    ("mini_chart_bg_1", "Sfondo minigrafico 1"),
    ("mini_chart_bg_2", "Sfondo minigrafico 2"),
    ("price_line_color", "Linea prezzo"),
    ("axes_color", "Assi grafico"),
    ("title_bar_color", "Titolo grafico"),
    ("text_color", "Testo"),
    ("bidask_color", "Label Bid/Ask"),
    ("legend_text_color", "Testo legenda/cursore"),
    ("grid_color", "Griglia asse X"),
    ("tab_bg", "Sfondo tab selector"),
    ("tab_text_color", "Testo tab selector"),
    ("market_cut_color", "Linea taglio mercato"),
    ("candle_up_color", "Candela rialzista"),
    ("candle_down_color", "Candela ribassista"),
]

COLOR_DEFAULTS = {
    "window_bg": "#0f1115",
    "panel_bg": "#12151b",
    "border_color": "#2a2f3a",
    "splitter_handle_color": "#2f3541",
    "chart_bg": "#0f1115",
    "mini_chart_bg_1": "#14181f",
    "mini_chart_bg_2": "#1a1e25",
    "price_line_color": "#e0e0e0",
    "axes_color": "#cfd6e1",
    "title_bar_color": "#cfd6e1",
    "text_color": "#e0e0e0",
    "bidask_color": "#ffd479",
    "legend_text_color": "#cfd6e1",
    "grid_color": "#3a4250",
    "tab_bg": "#12151b",
    "tab_text_color": "#e0e0e0",
    "market_cut_color": "#7f8fa6",
    "candle_up_color": "#2ecc71",
    "candle_down_color": "#e74c3c",
}

GENERAL_KEYS = [
    "alpha_vantage_api_key",
    "tiingo_api_key",
    "admin_tokens",
    "broker_mode",
    "active_account",
    "account_currency",
    "account_balance",
    "account_leverage",
    "account_tiingo_api_key",
    "ib_host",
    "ib_port",
    "ib_client_id",
    "ib_username",
    "ib_password",
    "mt_server",
    "mt_login",
    "mt_password",
    # Multi-provider settings
    "primary_data_provider",
    "secondary_data_provider",
    "ctrader_client_id",
    "ctrader_client_secret",
    "ctrader_environment",
]


class SettingsDialog(QDialog):
    """Contributor-friendly settings dialog with persistence and JSON import/export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(760, 720)

        self._accounts: Dict[str, Dict[str, str]] = {}
        self.color_edits: Dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)

        # General provider/broker form
        form_group = QGroupBox("General Provider/Broker")
        form = QGridLayout(form_group)

        self.alpha_input = QLineEdit()
        self.alpha_input.setPlaceholderText("AlphaVantage API key")
        self.tiingo_input = QLineEdit()
        self.tiingo_input.setPlaceholderText("Tiingo API key (or set env TIINGO_APIKEY)")
        self.admin_input = QLineEdit()
        self.admin_input.setPlaceholderText("token1:admin,token2:operator")

        self.broker_mode = QComboBox()
        self.broker_mode.addItems(["paper", "ib", "mt4", "mt5", "ctrader"])

        self.acc_name = QLineEdit(); self.acc_name.setPlaceholderText("default")
        self.acc_currency = QLineEdit(); self.acc_currency.setPlaceholderText("USD")
        self.acc_balance = QLineEdit(); self.acc_balance.setPlaceholderText("100000")
        self.acc_leverage = QLineEdit(); self.acc_leverage.setPlaceholderText("30")
        self.acc_tiingo = QLineEdit(); self.acc_tiingo.setPlaceholderText("Account Tiingo API key (optional)")

        self.ib_host = QLineEdit(); self.ib_host.setPlaceholderText("127.0.0.1")
        self.ib_port = QLineEdit(); self.ib_port.setPlaceholderText("7497")
        self.ib_client = QLineEdit(); self.ib_client.setPlaceholderText("1")
        self.ib_user = QLineEdit(); self.ib_user.setPlaceholderText("username")
        self.ib_pass = QLineEdit(); self.ib_pass.setEchoMode(QLineEdit.Password); self.ib_pass.setPlaceholderText("password")

        self.mt_server = QLineEdit(); self.mt_server.setPlaceholderText("broker server")
        self.mt_login = QLineEdit(); self.mt_login.setPlaceholderText("login")
        self.mt_pass = QLineEdit(); self.mt_pass.setEchoMode(QLineEdit.Password); self.mt_pass.setPlaceholderText("password")

        # cTrader fields (used in multiple places)
        self.ctrader_client_id = QLineEdit(); self.ctrader_client_id.setPlaceholderText("cTrader Client ID")
        self.ctrader_client_secret = QLineEdit(); self.ctrader_client_secret.setPlaceholderText("cTrader Client Secret"); self.ctrader_client_secret.setEchoMode(QLineEdit.Password)
        self.ctrader_access_token = QLineEdit(); self.ctrader_access_token.setEchoMode(QLineEdit.Password); self.ctrader_access_token.setPlaceholderText("Access Token (optional - use if already have one)")
        self.ctrader_environment = QComboBox(); self.ctrader_environment.addItems(["demo", "live"])

        # Left column
        form.addWidget(QLabel("AlphaVantage API Key:"), 0, 0)
        form.addWidget(self.alpha_input, 0, 1)
        form.addWidget(QLabel("Tiingo API Key:"), 1, 0)
        form.addWidget(self.tiingo_input, 1, 1)
        form.addWidget(QLabel("ADMIN_TOKENS:"), 2, 0)
        form.addWidget(self.admin_input, 2, 1)
        form.addWidget(QLabel("Broker Mode:"), 3, 0)
        form.addWidget(self.broker_mode, 3, 1)
        form.addWidget(QLabel("Active Account:"), 4, 0)
        form.addWidget(self.acc_name, 4, 1)
        form.addWidget(QLabel("Currency:"), 5, 0)
        form.addWidget(self.acc_currency, 5, 1)
        form.addWidget(QLabel("Balance:"), 6, 0)
        form.addWidget(self.acc_balance, 6, 1)
        form.addWidget(QLabel("Leverage:"), 7, 0)
        form.addWidget(self.acc_leverage, 7, 1)

        # Right column
        form.addWidget(QLabel("IB Host:"), 0, 2)
        form.addWidget(self.ib_host, 0, 3)
        form.addWidget(QLabel("IB Port:"), 1, 2)
        form.addWidget(self.ib_port, 1, 3)
        form.addWidget(QLabel("IB Client ID:"), 2, 2)
        form.addWidget(self.ib_client, 2, 3)
        form.addWidget(QLabel("IB Username:"), 3, 2)
        form.addWidget(self.ib_user, 3, 3)
        form.addWidget(QLabel("IB Password:"), 4, 2)
        form.addWidget(self.ib_pass, 4, 3)
        form.addWidget(QLabel("MT Server:"), 5, 2)
        form.addWidget(self.mt_server, 5, 3)
        form.addWidget(QLabel("MT Login:"), 6, 2)
        form.addWidget(self.mt_login, 6, 3)
        form.addWidget(QLabel("MT Password:"), 7, 2)
        form.addWidget(self.mt_pass, 7, 3)

        # cTrader credentials in Trading Configuration
        form.addWidget(QLabel("cTrader Client ID:"), 8, 0)
        form.addWidget(self.ctrader_client_id, 8, 1)
        form.addWidget(QLabel("cTrader Secret:"), 8, 2)
        form.addWidget(self.ctrader_client_secret, 8, 3)
        form.addWidget(QLabel("cTrader Token:"), 9, 0)
        form.addWidget(self.ctrader_access_token, 9, 1, 1, 3)

        # Spanning field
        form.addWidget(QLabel("Account Tiingo Key:"), 10, 0)
        form.addWidget(self.acc_tiingo, 10, 1, 1, 3)

        layout.addWidget(form_group)

        # Multi-Provider Configuration
        provider_group = QGroupBox("Data Provider Configuration")
        provider_form = QGridLayout(provider_group)

        self.primary_provider = QComboBox()
        self.primary_provider.addItems(["tiingo", "ctrader", "alphavantage"])
        self.secondary_provider = QComboBox()
        self.secondary_provider.addItems(["none", "tiingo", "ctrader", "alphavantage"])

        self.btn_ctrader_oauth = QPushButton("Authorize cTrader (OAuth)")
        self.btn_ctrader_test = QPushButton("Test Connection")

        # cTrader enabled checkbox (enables real-time ticks, volumes, order book)
        self.ctrader_enabled_checkbox = QCheckBox("Enable cTrader Real-time Features")
        self.ctrader_enabled_checkbox.setToolTip("Enable cTrader real-time ticks, volumes, and order book streaming via WebSocket")

        provider_form.addWidget(QLabel("Primary Provider:"), 0, 0)
        provider_form.addWidget(self.primary_provider, 0, 1)
        provider_form.addWidget(QLabel("Fallback Provider:"), 0, 2)
        provider_form.addWidget(self.secondary_provider, 0, 3)

        provider_form.addWidget(QLabel("cTrader Client ID:"), 1, 0)
        provider_form.addWidget(self.ctrader_client_id, 1, 1)
        provider_form.addWidget(QLabel("cTrader Client Secret:"), 1, 2)
        provider_form.addWidget(self.ctrader_client_secret, 1, 3)

        provider_form.addWidget(QLabel("cTrader Environment:"), 2, 0)
        provider_form.addWidget(self.ctrader_environment, 2, 1)
        provider_form.addWidget(self.btn_ctrader_oauth, 2, 2)
        provider_form.addWidget(self.btn_ctrader_test, 2, 3)

        # Add cTrader enabled checkbox on a new row
        provider_form.addWidget(self.ctrader_enabled_checkbox, 3, 0, 1, 2)

        layout.addWidget(provider_group)

        # cTrader Accounts List (shown only when broker_mode is ctrader)
        self.ctrader_accounts_group = QGroupBox("cTrader Accounts")
        ctrader_acc_layout = QVBoxLayout(self.ctrader_accounts_group)

        self.ctrader_accounts_list = QListWidget()
        self.ctrader_accounts_list.setMaximumHeight(150)
        ctrader_acc_layout.addWidget(self.ctrader_accounts_list)

        ctrader_btns = QHBoxLayout()
        self.btn_refresh_ctrader_accounts = QPushButton("Refresh Accounts")
        self.btn_refresh_ctrader_accounts.clicked.connect(self._on_refresh_ctrader_accounts)
        ctrader_btns.addWidget(self.btn_refresh_ctrader_accounts)
        ctrader_btns.addStretch()
        ctrader_acc_layout.addLayout(ctrader_btns)

        # Add legend
        legend_label = QLabel("🔵 Demo Account | 🟢 Live Account")
        legend_label.setStyleSheet("color: gray; font-size: 11px;")
        ctrader_acc_layout.addWidget(legend_label)

        layout.addWidget(self.ctrader_accounts_group)

        # Initially hide cTrader accounts (shown only when broker_mode == "ctrader")
        self.ctrader_accounts_group.setVisible(False)

        # Connect broker_mode change to show/hide cTrader accounts
        self.broker_mode.currentTextChanged.connect(self._on_broker_mode_changed)

        # Accounts list + controls
        accounts_group = QGroupBox("Accounts")
        acc_layout = QVBoxLayout(accounts_group)
        self.accounts_list = QListWidget()
        acc_layout.addWidget(self.accounts_list)
        acc_btns = QHBoxLayout()
        self.btn_acc_add = QPushButton("Add/Update")
        self.btn_acc_remove = QPushButton("Remove")
        self.btn_acc_set_active = QPushButton("Set Active")
        acc_btns.addWidget(self.btn_acc_add)
        acc_btns.addWidget(self.btn_acc_remove)
        acc_btns.addWidget(self.btn_acc_set_active)
        acc_layout.addLayout(acc_btns)
        layout.addWidget(accounts_group)

        # Risk Profile Management
        from .risk_profile_settings_widget import RiskProfileSettingsWidget
        from forex_diffusion.services.risk_profile_loader import RiskProfileLoader
        from forex_diffusion.utils.user_settings import SETTINGS_DIR

        self.risk_profile_widget = RiskProfileSettingsWidget()

        # Initialize and set risk profile loader
        db_path = SETTINGS_DIR / "trading_pipeline.db"
        risk_loader = RiskProfileLoader(str(db_path))
        self.risk_profile_widget.set_risk_profile_loader(risk_loader)

        layout.addWidget(self.risk_profile_widget)

        # Chart behaviour
        chart_group = QGroupBox("Chart Behaviour")
        chart_form = QFormLayout(chart_group)
        self.follow_suspend_spin = QDoubleSpinBox()
        self.follow_suspend_spin.setRange(1.0, 600.0)
        self.follow_suspend_spin.setDecimals(1)
        self.follow_suspend_spin.setSingleStep(1.0)
        self.follow_suspend_spin.setSuffix(" s")
        chart_form.addRow(QLabel("Follow suspend (seconds):"), self.follow_suspend_spin)

        # Date format selector
        self.date_format_combo = QComboBox()
        self.date_format_combo.addItems(["YYYY-MM-DD", "YYYY-DD-MM", "DD-MM-YYYY", "MM-DD-YYYY"])
        chart_form.addRow(QLabel("Date Format:"), self.date_format_combo)

        layout.addWidget(chart_group)

        # Color configuration
        colors_group = QGroupBox("Theme Colors")
        colors_layout = QGridLayout(colors_group)
        num_colors = len(COLOR_FIELDS)
        num_rows = (num_colors + 1) // 2  # Ceiling division
        for i, (key, label) in enumerate(COLOR_FIELDS):
            row = i % num_rows
            col_offset = (i // num_rows) * 3
            colors_layout.addWidget(QLabel(label), row, col_offset)
            edit = QLineEdit()
            edit.setReadOnly(True)
            btn = QPushButton("Change")
            btn.clicked.connect(partial(self._pick_color, key))
            colors_layout.addWidget(edit, row, col_offset + 1)
            colors_layout.addWidget(btn, row, col_offset + 2)
            self.color_edits[key] = edit
        layout.addWidget(colors_group)

        # JSON import/export controls
        json_buttons = QHBoxLayout()
        self.btn_load_json = QPushButton("Load JSON")
        self.btn_save_json = QPushButton("Save JSON")
        json_buttons.addWidget(self.btn_load_json)
        json_buttons.addWidget(self.btn_save_json)
        layout.addLayout(json_buttons)

        # Save / cancel buttons
        controls = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.cancel_btn = QPushButton("Cancel")
        controls.addWidget(self.save_btn)
        controls.addWidget(self.cancel_btn)
        layout.addLayout(controls)

        # Connections
        self.save_btn.clicked.connect(self.on_save)
        self.cancel_btn.clicked.connect(self.reject)
        self.btn_acc_add.clicked.connect(self._on_account_add_update)
        self.btn_acc_remove.clicked.connect(self._on_account_remove)
        self.btn_acc_set_active.clicked.connect(self._on_account_set_active)
        self.accounts_list.currentItemChanged.connect(self._on_account_selected)
        self.btn_load_json.clicked.connect(self._load_from_json)
        self.btn_save_json.clicked.connect(self._save_to_json)
        self.btn_ctrader_oauth.clicked.connect(self._on_ctrader_oauth)
        self.btn_ctrader_test.clicked.connect(self._on_ctrader_test)

        self.load_values()

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def load_values(self) -> None:
        alpha = get_setting("alpha_vantage_api_key", os.environ.get("ALPHAVANTAGE_KEY", "") or "")
        tiingo = get_setting("tiingo_api_key", os.environ.get("TIINGO_APIKEY", "") or "")
        admin = get_setting("admin_tokens", os.environ.get("ADMIN_TOKENS", "") or "")
        self.alpha_input.setText(str(alpha))
        self.tiingo_input.setText(str(tiingo))
        self.admin_input.setText(str(admin))
        self.broker_mode.setCurrentText(str(get_setting("broker_mode", "paper")))

        self.acc_name.setText(str(get_setting("active_account", "default")))
        self.acc_currency.setText(str(get_setting("account_currency", "USD")))
        self.acc_balance.setText(str(get_setting("account_balance", "100000")))
        self.acc_leverage.setText(str(get_setting("account_leverage", "30")))
        self.acc_tiingo.setText(str(get_setting("account_tiingo_api_key", "")))

        self.ib_host.setText(str(get_setting("ib_host", "127.0.0.1")))
        self.ib_port.setText(str(get_setting("ib_port", "7497")))
        self.ib_client.setText(str(get_setting("ib_client_id", "1")))
        self.ib_user.setText(str(get_setting("ib_username", "")))
        self.ib_pass.setText(str(get_setting("ib_password", "")))
        self.mt_server.setText(str(get_setting("mt_server", "")))
        self.mt_login.setText(str(get_setting("mt_login", "")))
        self.mt_pass.setText(str(get_setting("mt_password", "")))

        self.follow_suspend_spin.setValue(float(get_setting("chart.follow_suspend_seconds", 30.0)))
        self.date_format_combo.setCurrentText(str(get_setting("chart.date_format", "YYYY-MM-DD")))

        for key, _ in COLOR_FIELDS:
            default = COLOR_DEFAULTS.get(key, "#000000")
            value = str(get_setting(key, default))
            self.color_edits[key].setText(value)

        # Load provider settings
        self.primary_provider.setCurrentText(str(get_setting("primary_data_provider", "tiingo")))
        self.secondary_provider.setCurrentText(str(get_setting("secondary_data_provider", "none")))

        # Load ALL provider-specific settings (always load cTrader fields)
        self.ctrader_client_id.setText(str(get_setting("provider.ctrader.client_id", "")))
        self.ctrader_client_secret.setText(str(get_setting("provider.ctrader.client_secret", "")))
        self.ctrader_access_token.setText(str(get_setting("provider.ctrader.access_token", "")))
        self.ctrader_environment.setCurrentText(str(get_setting("provider.ctrader.environment", "demo")))

        # Load cTrader enabled state (auto-enable if primary_provider is ctrader)
        primary = str(get_setting("primary_data_provider", "tiingo"))
        ctrader_enabled = get_setting("ctrader_enabled", primary.lower() == "ctrader")
        self.ctrader_enabled_checkbox.setChecked(ctrader_enabled)

        self._accounts = self._load_accounts_dict()
        self._populate_accounts_list()

    def _load_accounts_dict(self) -> Dict[str, Dict[str, str]]:
        data = get_setting("accounts_profiles", {})
        result: Dict[str, Dict[str, str]] = {}
        if isinstance(data, dict):
            for name, payload in data.items():
                if isinstance(payload, dict):
                    result[str(name)] = {
                        "currency": str(payload.get("currency", "")),
                        "balance": str(payload.get("balance", "")),
                        "leverage": str(payload.get("leverage", "")),
                        "tiingo_key": str(payload.get("tiingo_key", "")),
                    }
        return result

    def _save_accounts_dict(self) -> None:
        set_setting("accounts_profiles", self._accounts)

    def _load_provider_settings(self, provider: str) -> None:
        """Load provider-specific settings from user_settings"""
        if provider == "ctrader":
            self.ctrader_client_id.setText(str(get_setting(f"provider.{provider}.client_id", "")))
            self.ctrader_client_secret.setText(str(get_setting(f"provider.{provider}.client_secret", "")))
            self.ctrader_environment.setCurrentText(str(get_setting(f"provider.{provider}.environment", "demo")))
        elif provider == "tiingo":
            # Tiingo settings are handled separately in alpha_input/tiingo_input
            pass
        elif provider == "alphavantage":
            # Alpha Vantage settings are handled separately in alpha_input
            pass

    def _save_provider_settings(self, provider: str) -> None:
        """Save provider-specific settings to user_settings"""
        if provider == "ctrader":
            set_setting(f"provider.{provider}.client_id", self.ctrader_client_id.text().strip())
            set_setting(f"provider.{provider}.client_secret", self.ctrader_client_secret.text().strip())
            set_setting(f"provider.{provider}.environment", self.ctrader_environment.currentText())
        elif provider == "tiingo":
            # Tiingo settings saved via tiingo_input
            pass
        elif provider == "alphavantage":
            # Alpha Vantage settings saved via alpha_input
            pass

    def _populate_accounts_list(self, select: str | None = None) -> None:
        self.accounts_list.blockSignals(True)
        self.accounts_list.clear()
        if not self._accounts:
            self.accounts_list.blockSignals(False)
            return
        active = select or str(get_setting("active_account", "default"))
        for name in sorted(self._accounts.keys()):
            item = QListWidgetItem(name)
            if name == active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.accounts_list.addItem(item)
        matches = self.accounts_list.findItems(active, Qt.MatchExactly)
        if matches:
            self.accounts_list.setCurrentItem(matches[0])
        else:
            self.accounts_list.setCurrentRow(0)
        self.accounts_list.blockSignals(False)
        current = self.accounts_list.currentItem()
        if current is not None:
            self._load_account_into_fields(current)

    # ------------------------------------------------------------------
    # Accounts handlers
    # ------------------------------------------------------------------
    def _collect_account_from_fields(self) -> Dict[str, str]:
        name = self.acc_name.text().strip() or "default"
        return {
            "name": name,
            "currency": self.acc_currency.text().strip() or "USD",
            "balance": self.acc_balance.text().strip() or "100000",
            "leverage": self.acc_leverage.text().strip() or "30",
            "tiingo_key": self.acc_tiingo.text().strip(),
        }

    def _load_account_into_fields(self, item: QListWidgetItem) -> None:
        name = item.text()
        payload = self._accounts.get(name, {})
        self.acc_name.setText(name)
        self.acc_currency.setText(payload.get("currency", str(get_setting("account_currency", "USD"))))
        self.acc_balance.setText(payload.get("balance", str(get_setting("account_balance", "100000"))))
        self.acc_leverage.setText(payload.get("leverage", str(get_setting("account_leverage", "30"))))
        self.acc_tiingo.setText(payload.get("tiingo_key", str(get_setting("account_tiingo_api_key", ""))))

    def _on_account_selected(self, current: QListWidgetItem | None, _: QListWidgetItem | None) -> None:
        if current is not None:
            self._load_account_into_fields(current)

    def _on_account_add_update(self) -> None:
        data = self._collect_account_from_fields()
        name = data.pop("name")
        self._accounts[name] = data
        self._save_accounts_dict()
        self._populate_accounts_list(select=name)

    def _on_account_remove(self) -> None:
        current = self.accounts_list.currentItem()
        if current is None:
            return
        name = current.text()
        if name in self._accounts:
            del self._accounts[name]
            self._save_accounts_dict()
            self._populate_accounts_list()

    def _on_account_set_active(self) -> None:
        current = self.accounts_list.currentItem()
        if current is None:
            return
        account_name = current.text()
        set_setting("active_account", account_name)
        self._populate_accounts_list(select=account_name)

    # ------------------------------------------------------------------
    # Color helpers
    # ------------------------------------------------------------------
    def _pick_color(self, key: str) -> None:
        current = self.color_edits[key].text().strip() or COLOR_DEFAULTS.get(key, "#000000")
        color = QColor(current)
        dialog = QColorDialog(color, self)
        dialog.setOption(QColorDialog.ShowAlphaChannel, True)
        if dialog.exec():
            selected = dialog.selectedColor()
            if selected.isValid():
                self.color_edits[key].setText(selected.name(QColor.HexArgb))

    # ------------------------------------------------------------------
    # JSON import/export
    # ------------------------------------------------------------------
    def _collect_export_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key in GENERAL_KEYS:
            attr_name = self._field_name_for_key(key)
            widget = getattr(self, attr_name, None)
            if widget is None:
                continue
            if isinstance(widget, QComboBox):
                data[key] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                data[key] = widget.text().strip()
            else:
                data[key] = str(get_setting(key, ""))
        data["chart.follow_suspend_seconds"] = float(self.follow_suspend_spin.value())
        current_account = self._collect_account_from_fields()
        name = current_account.pop("name")
        accounts_copy = dict(self._accounts)
        accounts_copy[name] = current_account
        data["accounts_profiles"] = accounts_copy
        data["colors"] = {key: self.color_edits[key].text().strip() for key, _ in COLOR_FIELDS}
        return data

    def _apply_loaded_settings(self, payload: Dict[str, Any]) -> None:
        colors = payload.get("colors", {})
        if isinstance(colors, dict):
            for key, value in colors.items():
                if key in dict(COLOR_FIELDS):
                    set_setting(key, value)
        accounts = payload.get("accounts_profiles")
        if isinstance(accounts, dict):
            set_setting("accounts_profiles", accounts)
        follow_seconds = payload.get("chart.follow_suspend_seconds")
        if follow_seconds is not None:
            set_setting("chart.follow_suspend_seconds", follow_seconds)
        for key in GENERAL_KEYS:
            if key in payload:
                set_setting(key, payload[key])

    def _field_name_for_key(self, key: str) -> str:
        mapping = {
            "alpha_vantage_api_key": "alpha_input",
            "tiingo_api_key": "tiingo_input",
            "admin_tokens": "admin_input",
            "broker_mode": "broker_mode",
            "active_account": "acc_name",
            "account_currency": "acc_currency",
            "account_balance": "acc_balance",
            "account_leverage": "acc_leverage",
            "account_tiingo_api_key": "acc_tiingo",
            "ib_host": "ib_host",
            "ib_port": "ib_port",
            "ib_client_id": "ib_client",
            "ib_username": "ib_user",
            "ib_password": "ib_pass",
            "mt_server": "mt_server",
            "mt_login": "mt_login",
            "mt_password": "mt_pass",
        }
        return mapping.get(key, "")

    def _load_from_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object with settings")
            self._apply_loaded_settings(payload)
            self.load_values()
            QMessageBox.information(self, "Settings", "Settings loaded from file")
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))

    def _save_to_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "settings.json", "JSON Files (*.json)")
        if not path:
            return
        try:
            data = self._collect_export_data()
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            QMessageBox.information(self, "Settings", "Settings saved to file")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def on_save(self) -> None:
        try:
            set_setting("alpha_vantage_api_key", self.alpha_input.text().strip())
            set_setting("tiingo_api_key", self.tiingo_input.text().strip())
            set_setting("admin_tokens", self.admin_input.text().strip())
            set_setting("broker_mode", self.broker_mode.currentText())

            account = self._collect_account_from_fields()
            name = account.pop("name")
            self._accounts[name] = account
            set_setting("active_account", name)
            set_setting("account_currency", account["currency"])
            set_setting("account_balance", account["balance"])
            set_setting("account_leverage", account["leverage"])
            set_setting("account_tiingo_api_key", account["tiingo_key"])
            self._save_accounts_dict()
            self._populate_accounts_list(select=name)

            set_setting("ib_host", self.ib_host.text().strip())
            set_setting("ib_port", self.ib_port.text().strip())
            set_setting("ib_client_id", self.ib_client.text().strip())
            set_setting("ib_username", self.ib_user.text().strip())
            set_setting("ib_password", self.ib_pass.text().strip())
            set_setting("mt_server", self.mt_server.text().strip())
            set_setting("mt_login", self.mt_login.text().strip())
            set_setting("mt_password", self.mt_pass.text().strip())

            set_setting("chart.follow_suspend_seconds", float(self.follow_suspend_spin.value()))
            set_setting("chart.date_format", self.date_format_combo.currentText())

            for key, _ in COLOR_FIELDS:
                set_setting(key, self.color_edits[key].text().strip())

            # Save provider settings
            set_setting("primary_data_provider", self.primary_provider.currentText())
            set_setting("secondary_data_provider", self.secondary_provider.currentText())

            # Save ALL provider-specific settings (not just the selected one)
            # Save cTrader settings
            set_setting("provider.ctrader.client_id", self.ctrader_client_id.text().strip())
            set_setting("provider.ctrader.client_secret", self.ctrader_client_secret.text().strip())
            set_setting("provider.ctrader.access_token", self.ctrader_access_token.text().strip())
            set_setting("provider.ctrader.environment", self.ctrader_environment.currentText())
            set_setting("ctrader_enabled", self.ctrader_enabled_checkbox.isChecked())

            QMessageBox.information(self, "Settings", "Settings saved")
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    # ------------------------------------------------------------------
    # Provider-specific handlers
    # ------------------------------------------------------------------
    def _on_ctrader_oauth(self) -> None:
        """Run cTrader OAuth flow."""
        try:
            import asyncio
            from forex_diffusion.credentials import OAuth2Flow, CredentialsManager, ProviderCredentials

            client_id = self.ctrader_client_id.text().strip()
            client_secret = self.ctrader_client_secret.text().strip()

            if not client_id or not client_secret:
                QMessageBox.warning(
                    self,
                    "OAuth Error",
                    "Please enter cTrader Client ID and Client Secret first"
                )
                return

            # Run OAuth flow in event loop
            async def run_oauth():
                oauth = OAuth2Flow(client_id=client_id, client_secret=client_secret)
                token_data = await oauth.authorize()

                creds = ProviderCredentials(
                    provider_name='ctrader',
                    client_id=client_id,
                    client_secret=client_secret,
                    access_token=token_data['access_token'],
                    refresh_token=token_data.get('refresh_token'),
                    environment=self.ctrader_environment.currentText()
                )

                CredentialsManager().save(creds)
                return True

            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            success = loop.run_until_complete(run_oauth())

            if success:
                QMessageBox.information(
                    self,
                    "OAuth Success",
                    "cTrader authorization successful! Credentials saved securely."
                )

        except Exception as exc:
            QMessageBox.warning(self, "OAuth Failed", f"Authorization failed: {exc}")

    def _on_ctrader_test(self) -> None:
        """Test cTrader connection - tries access token first, then OAuth."""
        try:
            import asyncio
            from forex_diffusion.providers import get_provider_manager
            from forex_diffusion.credentials import CredentialsManager

            # Get current form values
            client_id = self.ctrader_client_id.text().strip()
            client_secret = self.ctrader_client_secret.text().strip()
            access_token = self.ctrader_access_token.text().strip()
            environment = self.ctrader_environment.currentText()

            # Check if we have access token - try direct connection first
            if access_token:
                # Try with access token (no OAuth)
                async def test_with_token():
                    manager = get_provider_manager()
                    provider = manager.create_provider('ctrader', config={
                        'client_id': client_id,
                        'client_secret': client_secret,
                        'access_token': access_token,
                        'environment': environment
                    })

                    connected = await provider.connect()
                    if connected:
                        price = await provider.get_current_price("EUR/USD")
                        await provider.disconnect()
                        return connected, price
                    return False, None

                # Run token test
                success, price = asyncio.run(test_with_token())
                if success:
                    QMessageBox.information(
                        self,
                        "Connection Test",
                        f"✓ Connected successfully with access token!\n\nEUR/USD: {price}"
                    )
                    return
                else:
                    # Token failed, try OAuth
                    QMessageBox.information(
                        self,
                        "Token Failed",
                        "Access token connection failed. Trying OAuth flow..."
                    )

            # Fallback to OAuth credentials
            creds_manager = CredentialsManager()
            creds = creds_manager.load('ctrader')

            if not creds:
                QMessageBox.warning(
                    self,
                    "Test Failed",
                    "No access token and no OAuth credentials found.\nPlease provide an access token or run OAuth authorization."
                )
                return

            # Create and test provider with OAuth
            async def test_connection():
                manager = get_provider_manager()
                provider = manager.create_provider('ctrader', config={
                    'client_id': creds.client_id,
                    'client_secret': creds.client_secret,
                    'access_token': creds.access_token,
                    'environment': creds.environment
                })

                connected = await provider.connect()
                if connected:
                    # Try to get a price quote
                    price = await provider.get_current_price("EUR/USD")
                    await provider.disconnect()
                    return connected, price
                return False, None

            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            connected, price = loop.run_until_complete(test_connection())

            if connected:
                msg = "Connection successful!"
                if price:
                    msg += f"\n\nTest quote for EUR/USD: {price.get('price', 'N/A')}"
                QMessageBox.information(self, "Test Success", msg)
            else:
                QMessageBox.warning(self, "Test Failed", "Failed to connect to cTrader")

        except Exception as exc:
            QMessageBox.warning(self, "Test Failed", f"Connection test failed: {exc}")

    def _on_broker_mode_changed(self, broker_mode: str):
        """Show/hide cTrader accounts section based on broker mode."""
        is_ctrader = broker_mode.lower() == "ctrader"
        self.ctrader_accounts_group.setVisible(is_ctrader)

        # Auto-refresh accounts when switching to cTrader
        if is_ctrader:
            self._on_refresh_ctrader_accounts()

    def _on_refresh_ctrader_accounts(self):
        """Refresh cTrader accounts list from API."""
        try:
            self.ctrader_accounts_list.clear()

            # Get credentials from settings
            client_id = self.ctrader_client_id_edit.text().strip()
            client_secret = self.ctrader_client_secret_edit.text().strip()
            access_token = self.ctrader_access_token_edit.text().strip()

            if not all([client_id, client_secret, access_token]):
                QMessageBox.warning(self, "Warning", "Please configure cTrader credentials first")
                return

            # Use standalone function to get accounts
            from ..broker.ctrader_broker import get_ctrader_accounts

            # Get accounts from API
            accounts = get_ctrader_accounts(
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                environment='demo'  # TODO: Get from settings
            )

            if not accounts:
                logger.warning("No accounts returned from cTrader API, using fallback")
                # Fallback to mock data for testing
                accounts = [
                    {"id": "1234567", "type": "demo", "balance": 100000, "currency": "USD"},
                    {"id": "7654321", "type": "live", "balance": 5000, "currency": "EUR"},
                ]

            for acc in accounts:
                is_demo = acc.get("type", "demo") == "demo"
                color_icon = "🔵" if is_demo else "🟢"
                acc_id = acc.get("id", "N/A")
                currency = acc.get("currency", "USD")
                balance = acc.get("balance", 0)

                text = f"{color_icon} {acc_id} - {currency} {balance:,.2f}"

                item = QListWidgetItem(text)
                # Set background color
                if is_demo:
                    item.setBackground(QBrush(QColor(220, 235, 255)))  # Light blue
                else:
                    item.setBackground(QBrush(QColor(220, 255, 220)))  # Light green

                self.ctrader_accounts_list.addItem(item)

            logger.info(f"Loaded {len(accounts)} cTrader accounts (fallback: {not accounts})")

        except Exception as e:
            logger.error(f"Failed to load cTrader accounts: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load cTrader accounts: {str(e)}\n\nPlease check your credentials.")
