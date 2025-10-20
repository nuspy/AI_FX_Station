"""
FxPro cTrader Credentials Manager Dialog

Manages OAuth2 credentials and authentication flow for FxPro cTrader broker.
"""
from __future__ import annotations

import webbrowser
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QTextEdit, QMessageBox, QFormLayout,
    QTabWidget, QWidget, QCheckBox
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from loguru import logger

try:
    from ..brokers.fxpro_ctrader import FxProCTraderBroker, create_fxpro_broker
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False
    logger.warning("FxPro broker module not available")


class FxProCredentialsDialog(QDialog):
    """
    Dialog for managing FxPro cTrader OAuth2 credentials.

    Features:
    - OAuth2 setup wizard
    - Credential storage
    - Connection testing
    - Account information display
    """

    credentials_updated = Signal(dict)  # Emits when credentials are updated

    def __init__(self, parent=None):
        super().__init__(parent)

        self.broker: Optional[FxProCTraderBroker] = None
        self.auth_code: Optional[str] = None

        self.setWindowTitle("FxPro cTrader - Credentials Manager")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        self._build_ui()
        self._load_existing_credentials()
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()

    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets"""
        from ..i18n.widget_helper import apply_tooltip
        
        if hasattr(self, 'fxpro_account_edit'):
            apply_tooltip(self.fxpro_account_edit, "fxpro_account", "fxpro_integration")
        if hasattr(self, 'fxpro_server_combo'):
            apply_tooltip(self.fxpro_server_combo, "fxpro_server", "fxpro_integration")
        if hasattr(self, 'auto_sync_check'):
            apply_tooltip(self.auto_sync_check, "auto_sync", "fxpro_integration")
        if hasattr(self, 'webhook_url_edit'):
            apply_tooltip(self.webhook_url_edit, "webhook_url", "fxpro_integration")
    

    def _build_ui(self):
        """Build the dialog UI"""
        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()

        # Tab 1: Setup
        setup_tab = self._build_setup_tab()
        tabs.addTab(setup_tab, "Setup & Authentication")

        # Tab 2: Account Info
        account_tab = self._build_account_tab()
        tabs.addTab(account_tab, "Account Information")

        # Tab 3: Advanced
        advanced_tab = self._build_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced Settings")

        layout.addWidget(tabs)

        # Footer buttons
        footer = self._build_footer()
        layout.addWidget(footer)

    def _build_setup_tab(self) -> QWidget:
        """Build setup tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Instructions
        instructions = QGroupBox("Getting Started")
        inst_layout = QVBoxLayout(instructions)

        inst_text = QLabel(
            "To connect to FxPro cTrader, you need OAuth2 credentials:\n\n"
            "1. Visit FxPro Developer Portal: https://connect.ctrader.com/\n"
            "2. Create a new application\n"
            "3. Set redirect URI to: http://localhost:8080/callback\n"
            "4. Copy your Client ID and Client Secret below\n"
            "5. Click 'Authorize' to complete OAuth2 flow\n\n"
            "Your credentials will be stored securely on your local machine."
        )
        inst_text.setWordWrap(True)
        inst_layout.addWidget(inst_text)

        layout.addWidget(instructions)

        # Credentials input
        creds_group = QGroupBox("OAuth2 Credentials")
        creds_layout = QFormLayout(creds_group)

        self.client_id_edit = QLineEdit()
        self.client_id_edit.setPlaceholderText("Enter Client ID from FxPro Developer Portal")
        creds_layout.addRow("Client ID:", self.client_id_edit)

        self.client_secret_edit = QLineEdit()
        self.client_secret_edit.setPlaceholderText("Enter Client Secret")
        self.client_secret_edit.setEchoMode(QLineEdit.Password)
        creds_layout.addRow("Client Secret:", self.client_secret_edit)

        self.show_secret_check = QCheckBox("Show secret")
        self.show_secret_check.toggled.connect(self._toggle_secret_visibility)
        creds_layout.addRow("", self.show_secret_check)

        self.redirect_uri_edit = QLineEdit("http://localhost:8080/callback")
        creds_layout.addRow("Redirect URI:", self.redirect_uri_edit)

        self.account_id_edit = QLineEdit()
        self.account_id_edit.setPlaceholderText("Optional: Specific account ID")
        creds_layout.addRow("Account ID:", self.account_id_edit)

        layout.addWidget(creds_group)

        # Authorization flow
        auth_group = QGroupBox("Authorization")
        auth_layout = QVBoxLayout(auth_group)

        auth_info = QLabel(
            "Click 'Authorize' to open browser and complete OAuth2 authentication.\n"
            "After authorizing, paste the authorization code below."
        )
        auth_info.setWordWrap(True)
        auth_layout.addWidget(auth_info)

        auth_buttons = QHBoxLayout()

        self.authorize_btn = QPushButton("1. Authorize in Browser")
        self.authorize_btn.clicked.connect(self._start_authorization)
        auth_buttons.addWidget(self.authorize_btn)

        auth_buttons.addStretch()
        auth_layout.addLayout(auth_buttons)

        self.auth_code_edit = QLineEdit()
        self.auth_code_edit.setPlaceholderText("Paste authorization code here")
        auth_layout.addWidget(QLabel("2. Authorization Code:"))
        auth_layout.addWidget(self.auth_code_edit)

        exchange_buttons = QHBoxLayout()

        self.exchange_btn = QPushButton("3. Complete Authorization")
        self.exchange_btn.clicked.connect(self._exchange_code)
        self.exchange_btn.setEnabled(False)
        exchange_buttons.addWidget(self.exchange_btn)

        self.test_connection_btn = QPushButton("4. Test Connection")
        self.test_connection_btn.clicked.connect(self._test_connection)
        self.test_connection_btn.setEnabled(False)
        exchange_buttons.addWidget(self.test_connection_btn)

        exchange_buttons.addStretch()
        auth_layout.addLayout(exchange_buttons)

        # Status
        self.status_label = QLabel("Status: Not connected")
        self.status_label.setStyleSheet("font-weight: bold;")
        auth_layout.addWidget(self.status_label)

        layout.addWidget(auth_group)

        # Enable exchange button when auth code is entered
        self.auth_code_edit.textChanged.connect(
            lambda text: self.exchange_btn.setEnabled(bool(text))
        )

        layout.addStretch()

        return widget

    def _build_account_tab(self) -> QWidget:
        """Build account information tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        account_group = QGroupBox("Account Details")
        account_layout = QFormLayout(account_group)

        self.account_id_label = QLabel("--")
        account_layout.addRow("Account ID:", self.account_id_label)

        self.balance_label = QLabel("--")
        account_layout.addRow("Balance:", self.balance_label)

        self.equity_label = QLabel("--")
        account_layout.addRow("Equity:", self.equity_label)

        self.margin_used_label = QLabel("--")
        account_layout.addRow("Margin Used:", self.margin_used_label)

        self.margin_free_label = QLabel("--")
        account_layout.addRow("Free Margin:", self.margin_free_label)

        self.leverage_label = QLabel("--")
        account_layout.addRow("Leverage:", self.leverage_label)

        self.unrealized_pnl_label = QLabel("--")
        account_layout.addRow("Unrealized P&L:", self.unrealized_pnl_label)

        layout.addWidget(account_group)

        # Refresh button
        refresh_btn = QPushButton("Refresh Account Info")
        refresh_btn.clicked.connect(self._refresh_account_info)
        layout.addWidget(refresh_btn)

        layout.addStretch()

        return widget

    def _build_advanced_tab(self) -> QWidget:
        """Build advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Token info
        token_group = QGroupBox("Token Information")
        token_layout = QVBoxLayout(token_group)

        self.token_info = QTextEdit()
        self.token_info.setReadOnly(True)
        self.token_info.setMaximumHeight(150)
        self.token_info.setFont(QFont("Courier", 9))
        token_layout.addWidget(self.token_info)

        refresh_token_btn = QPushButton("Refresh Access Token")
        refresh_token_btn.clicked.connect(self._refresh_token)
        token_layout.addWidget(refresh_token_btn)

        layout.addWidget(token_group)

        # Danger zone
        danger_group = QGroupBox("Danger Zone")
        danger_layout = QVBoxLayout(danger_group)

        clear_creds_btn = QPushButton("Clear Saved Credentials")
        clear_creds_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        clear_creds_btn.clicked.connect(self._clear_credentials)
        danger_layout.addWidget(clear_creds_btn)

        layout.addWidget(danger_group)

        layout.addStretch()

        return widget

    def _build_footer(self) -> QWidget:
        """Build footer buttons"""
        footer = QWidget()
        layout = QHBoxLayout(footer)

        self.save_btn = QPushButton("Save Credentials")
        self.save_btn.clicked.connect(self._save_credentials)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)

        layout.addStretch()
        layout.addWidget(self.save_btn)
        layout.addWidget(self.close_btn)

        return footer

    def _toggle_secret_visibility(self, checked: bool):
        """Toggle client secret visibility"""
        if checked:
            self.client_secret_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.client_secret_edit.setEchoMode(QLineEdit.Password)

    def _load_existing_credentials(self):
        """Load existing credentials if available"""
        if not BROKER_AVAILABLE:
            self.status_label.setText("Status: Broker module not available")
            return

        try:
            from ..utils.user_settings import get_setting

            client_id = get_setting('fxpro_client_id', '')
            client_secret = get_setting('fxpro_client_secret', '')
            account_id = get_setting('fxpro_account_id', '')

            if client_id:
                self.client_id_edit.setText(client_id)
            if client_secret:
                self.client_secret_edit.setText(client_secret)
            if account_id:
                self.account_id_edit.setText(account_id)

            # Try to load existing broker instance
            if client_id and client_secret:
                self.broker = create_fxpro_broker(client_id, client_secret, account_id or None)

                # Check if we have a valid token
                if self.broker.access_token:
                    self.status_label.setText("Status: Credentials loaded (not connected)")
                    self.test_connection_btn.setEnabled(True)
                    self._update_token_info()

        except Exception as e:
            logger.error(f"Failed to load existing credentials: {e}")

    def _start_authorization(self):
        """Start OAuth2 authorization flow"""
        if not BROKER_AVAILABLE:
            QMessageBox.warning(self, "Module Not Available", "FxPro broker module is not available")
            return

        client_id = self.client_id_edit.text().strip()
        client_secret = self.client_secret_edit.text().strip()

        if not client_id or not client_secret:
            QMessageBox.warning(
                self,
                "Missing Credentials",
                "Please enter Client ID and Client Secret first"
            )
            return

        # Create broker instance
        self.broker = FxProCTraderBroker(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=self.redirect_uri_edit.text(),
            account_id=self.account_id_edit.text() or None
        )

        # Get authorization URL
        auth_url = self.broker.get_authorization_url()

        # Open in browser
        webbrowser.open(auth_url)

        self.status_label.setText("Status: Waiting for authorization...")

        QMessageBox.information(
            self,
            "Authorization Started",
            f"Browser opened for authorization.\n\n"
            f"After authorizing:\n"
            f"1. You'll be redirected to: {self.redirect_uri_edit.text()}\n"
            f"2. Copy the 'code' parameter from the URL\n"
            f"3. Paste it in the Authorization Code field\n"
            f"4. Click 'Complete Authorization'"
        )

    def _exchange_code(self):
        """Exchange authorization code for access token"""
        if not self.broker:
            QMessageBox.warning(self, "Error", "Please authorize first")
            return

        code = self.auth_code_edit.text().strip()

        if not code:
            QMessageBox.warning(self, "Error", "Please enter authorization code")
            return

        # Exchange code
        success = self.broker.exchange_code_for_token(code)

        if success:
            self.status_label.setText("Status: Authorization successful!")
            self.test_connection_btn.setEnabled(True)
            self._update_token_info()

            QMessageBox.information(
                self,
                "Success",
                "Authorization completed successfully!\n\nClick 'Test Connection' to verify."
            )
        else:
            self.status_label.setText("Status: Authorization failed")
            QMessageBox.critical(
                self,
                "Authorization Failed",
                "Failed to exchange authorization code for access token.\n\n"
                "Please try again."
            )

    def _test_connection(self):
        """Test connection to FxPro API"""
        if not self.broker:
            QMessageBox.warning(self, "Error", "Please authorize first")
            return

        # Try to connect
        success = self.broker.connect()

        if success:
            self.status_label.setText("Status: Connected ✓")
            self.status_label.setStyleSheet("font-weight: bold; color: #27ae60;")

            # Refresh account info
            self._refresh_account_info()

            QMessageBox.information(
                self,
                "Connection Successful",
                "Successfully connected to FxPro cTrader!\n\n"
                "Your credentials are working correctly."
            )
        else:
            self.status_label.setText("Status: Connection failed ✗")
            self.status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")

            QMessageBox.critical(
                self,
                "Connection Failed",
                "Failed to connect to FxPro cTrader.\n\n"
                "Please check your credentials and try again."
            )

    def _refresh_account_info(self):
        """Refresh account information display"""
        if not self.broker or not self.broker.is_connected():
            return

        account_info = self.broker.get_account_info()

        if account_info:
            self.account_id_label.setText(account_info.account_id)
            self.balance_label.setText(f"{account_info.balance:.2f} {account_info.currency}")
            self.equity_label.setText(f"{account_info.equity:.2f} {account_info.currency}")
            self.margin_used_label.setText(f"{account_info.margin_used:.2f} {account_info.currency}")
            self.margin_free_label.setText(f"{account_info.margin_available:.2f} {account_info.currency}")
            self.leverage_label.setText(f"1:{account_info.leverage}")

            pnl_color = "#27ae60" if account_info.unrealized_pnl >= 0 else "#e74c3c"
            self.unrealized_pnl_label.setText(f"{account_info.unrealized_pnl:+.2f} {account_info.currency}")
            self.unrealized_pnl_label.setStyleSheet(f"color: {pnl_color}; font-weight: bold;")

    def _refresh_token(self):
        """Manually refresh access token"""
        if not self.broker:
            QMessageBox.warning(self, "Error", "No broker instance available")
            return

        success = self.broker.refresh_access_token()

        if success:
            self._update_token_info()
            QMessageBox.information(self, "Success", "Access token refreshed successfully")
        else:
            QMessageBox.critical(self, "Error", "Failed to refresh access token")

    def _update_token_info(self):
        """Update token information display"""
        if not self.broker:
            return

        info = f"Access Token: {self.broker.access_token[:20]}... (truncated)\n"

        if self.broker.token_expiry:
            info += f"Expires: {self.broker.token_expiry.strftime('%Y-%m-%d %H:%M:%S')}\n"

        if self.broker.refresh_token:
            info += f"Refresh Token: {self.broker.refresh_token[:20]}... (truncated)\n"

        self.token_info.setText(info)

    def _save_credentials(self):
        """Save credentials to settings"""
        try:
            from ..utils.user_settings import set_setting

            set_setting('fxpro_client_id', self.client_id_edit.text())
            set_setting('fxpro_client_secret', self.client_secret_edit.text())
            set_setting('fxpro_account_id', self.account_id_edit.text())

            # Emit credentials updated signal
            self.credentials_updated.emit({
                'client_id': self.client_id_edit.text(),
                'client_secret': self.client_secret_edit.text(),
                'account_id': self.account_id_edit.text(),
            })

            QMessageBox.information(self, "Success", "Credentials saved successfully")

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save credentials:\n{str(e)}")

    def _clear_credentials(self):
        """Clear saved credentials"""
        reply = QMessageBox.question(
            self,
            "Clear Credentials?",
            "This will delete all saved FxPro credentials and tokens.\n\n"
            "You will need to authorize again.\n\n"
            "Are you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                from ..utils.user_settings import set_setting

                set_setting('fxpro_client_id', '')
                set_setting('fxpro_client_secret', '')
                set_setting('fxpro_account_id', '')

                # Clear credentials file
                if self.broker and self.broker.credentials_file.exists():
                    self.broker.credentials_file.unlink()

                # Clear UI
                self.client_id_edit.clear()
                self.client_secret_edit.clear()
                self.account_id_edit.clear()
                self.auth_code_edit.clear()

                self.broker = None
                self.status_label.setText("Status: Credentials cleared")

                QMessageBox.information(self, "Success", "Credentials cleared successfully")

            except Exception as e:
                logger.error(f"Failed to clear credentials: {e}")
                QMessageBox.critical(self, "Error", f"Failed to clear credentials:\n{str(e)}")

    def get_broker(self) -> Optional[FxProCTraderBroker]:
        """Get the configured broker instance"""
        return self.broker
