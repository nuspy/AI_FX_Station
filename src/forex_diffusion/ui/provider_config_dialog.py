"""
Provider Configuration Dialog.

Allows users to configure credentials and settings for data providers
(cTrader, Tiingo, AlphaVantage, etc.) and select primary/fallback providers.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox,
    QLabel, QLineEdit, QComboBox, QPushButton, QFormLayout,
    QCheckBox, QMessageBox
)
from PySide6.QtCore import Qt
from loguru import logger


class ProviderConfigDialog(QDialog):
    """
    Dialog for configuring data provider credentials and settings.

    Features:
    - Tabs for each provider (cTrader, Tiingo, AlphaVantage)
    - Primary and fallback provider selection
    - Credential input with show/hide password
    - Test connection button
    - Save/Cancel buttons
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Provider Configuration")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        # Load current settings
        self._load_current_settings()

        # Setup UI
        self._setup_ui()

    def _load_current_settings(self):
        """Load current provider settings."""
        try:
            from ..utils.user_settings import get_setting

            # General settings
            self.primary_provider = get_setting('primary_data_provider', 'ctrader')
            self.fallback_provider = get_setting('fallback_data_provider', 'tiingo')

            # cTrader settings
            self.ctrader_client_id = get_setting('ctrader_client_id', '')
            self.ctrader_client_secret = get_setting('ctrader_client_secret', '')
            self.ctrader_access_token = get_setting('ctrader_access_token', '')
            self.ctrader_account_id = get_setting('ctrader_account_id', '')
            self.ctrader_environment = get_setting('ctrader_environment', 'demo')

            # Tiingo settings
            self.tiingo_api_key = get_setting('tiingo_api_key', '')

            # AlphaVantage settings
            self.alphavantage_api_key = get_setting('alphavantage_api_key', '')

        except Exception as e:
            logger.warning(f"Could not load provider settings: {e}")
            # Set defaults
            self.primary_provider = 'ctrader'
            self.fallback_provider = 'tiingo'
            self.ctrader_client_id = ''
            self.ctrader_client_secret = ''
            self.ctrader_access_token = ''
            self.ctrader_account_id = ''
            self.ctrader_environment = 'demo'
            self.tiingo_api_key = ''
            self.alphavantage_api_key = ''

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Provider selection section
        provider_group = QGroupBox("Provider Selection")
        provider_layout = QFormLayout(provider_group)

        self.primary_combo = QComboBox()
        self.primary_combo.addItems(['ctrader', 'tiingo', 'alphavantage'])
        self.primary_combo.setCurrentText(self.primary_provider)
        provider_layout.addRow("Primary Provider:", self.primary_combo)

        self.fallback_combo = QComboBox()
        self.fallback_combo.addItems(['none', 'ctrader', 'tiingo', 'alphavantage'])
        self.fallback_combo.setCurrentText(self.fallback_provider if self.fallback_provider else 'none')
        provider_layout.addRow("Fallback Provider:", self.fallback_combo)

        layout.addWidget(provider_group)

        # Tabs for each provider
        tabs = QTabWidget()

        # cTrader Tab
        tabs.addTab(self._create_ctrader_tab(), "cTrader")

        # Tiingo Tab
        tabs.addTab(self._create_tiingo_tab(), "Tiingo")

        # AlphaVantage Tab
        tabs.addTab(self._create_alphavantage_tab(), "AlphaVantage")

        layout.addWidget(tabs)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_connection)
        buttons_layout.addWidget(self.test_btn)

        buttons_layout.addStretch()

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save_settings)
        buttons_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)

        layout.addLayout(buttons_layout)

    def _create_ctrader_tab(self):
        """Create cTrader configuration tab."""
        widget = QGroupBox()
        layout = QFormLayout(widget)

        # Client ID
        self.ctrader_client_id_edit = QLineEdit(self.ctrader_client_id)
        layout.addRow("Client ID:", self.ctrader_client_id_edit)

        # Client Secret
        secret_layout = QHBoxLayout()
        self.ctrader_client_secret_edit = QLineEdit(self.ctrader_client_secret)
        self.ctrader_client_secret_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.ctrader_show_secret = QCheckBox("Show")
        self.ctrader_show_secret.toggled.connect(
            lambda checked: self.ctrader_client_secret_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        secret_layout.addWidget(self.ctrader_client_secret_edit)
        secret_layout.addWidget(self.ctrader_show_secret)
        layout.addRow("Client Secret:", secret_layout)

        # Access Token (optional)
        token_layout = QHBoxLayout()
        self.ctrader_access_token_edit = QLineEdit(self.ctrader_access_token)
        self.ctrader_access_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.ctrader_show_token = QCheckBox("Show")
        self.ctrader_show_token.toggled.connect(
            lambda checked: self.ctrader_access_token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        token_layout.addWidget(self.ctrader_access_token_edit)
        token_layout.addWidget(self.ctrader_show_token)
        layout.addRow("Access Token (optional):", token_layout)

        # Account ID
        self.ctrader_account_id_edit = QLineEdit(self.ctrader_account_id)
        layout.addRow("Account ID (optional):", self.ctrader_account_id_edit)

        # Environment
        self.ctrader_env_combo = QComboBox()
        self.ctrader_env_combo.addItems(['demo', 'live'])
        self.ctrader_env_combo.setCurrentText(self.ctrader_environment)
        layout.addRow("Environment:", self.ctrader_env_combo)

        # Help text
        help_label = QLabel(
            "Get cTrader credentials from your broker's cTrader developer portal.\n"
            "Client ID and Client Secret are required for historical data access."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addRow(help_label)

        return widget

    def _create_tiingo_tab(self):
        """Create Tiingo configuration tab."""
        widget = QGroupBox()
        main_layout = QVBoxLayout(widget)
        layout = QFormLayout()

        # API Key
        api_key_layout = QHBoxLayout()
        self.tiingo_api_key_edit = QLineEdit(self.tiingo_api_key)
        self.tiingo_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.tiingo_show_key = QCheckBox("Show")
        self.tiingo_show_key.toggled.connect(
            lambda checked: self.tiingo_api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        api_key_layout.addWidget(self.tiingo_api_key_edit)
        api_key_layout.addWidget(self.tiingo_show_key)
        layout.addRow("API Key:", api_key_layout)

        # Help text
        help_label = QLabel(
            "Get your Tiingo API key from https://www.tiingo.com/\n"
            "Free tier available for personal use."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addRow(help_label)

        main_layout.addLayout(layout)
        main_layout.addStretch()
        return widget

    def _create_alphavantage_tab(self):
        """Create AlphaVantage configuration tab."""
        widget = QGroupBox()
        main_layout = QVBoxLayout(widget)
        layout = QFormLayout()

        # API Key
        api_key_layout = QHBoxLayout()
        self.alphavantage_api_key_edit = QLineEdit(self.alphavantage_api_key)
        self.alphavantage_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.alphavantage_show_key = QCheckBox("Show")
        self.alphavantage_show_key.toggled.connect(
            lambda checked: self.alphavantage_api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        api_key_layout.addWidget(self.alphavantage_api_key_edit)
        api_key_layout.addWidget(self.alphavantage_show_key)
        layout.addRow("API Key:", api_key_layout)

        # Help text
        help_label = QLabel(
            "Get your AlphaVantage API key from https://www.alphavantage.co/\n"
            "Free tier: 5 API requests per minute, 500 per day."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addRow(help_label)

        main_layout.addLayout(layout)
        main_layout.addStretch()
        return widget

    def _test_connection(self):
        """Test connection to selected provider."""
        provider = self.primary_combo.currentText()

        # Show progress dialog
        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt, QThread, Signal

        progress = QProgressDialog(
            f"Testing connection to {provider}...",
            "Cancel",
            0, 0,  # Indeterminate progress
            self
        )
        progress.setWindowTitle("Testing Connection")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Test in background thread to avoid blocking UI
        class TestWorker(QThread):
            finished = Signal(bool, str)

            def __init__(self, provider_name, config):
                super().__init__()
                self.provider_name = provider_name
                self.config = config

            def run(self):
                try:
                    if self.provider_name == "ctrader":
                        success, message = self._test_ctrader()
                    elif self.provider_name == "tiingo":
                        success, message = self._test_tiingo()
                    elif self.provider_name == "alphavantage":
                        success, message = self._test_alphavantage()
                    else:
                        success, message = False, f"Unknown provider: {self.provider_name}"

                    self.finished.emit(success, message)
                except Exception as e:
                    self.finished.emit(False, f"Test error: {str(e)}")

            def _test_ctrader(self):
                """Test cTrader connection."""
                try:
                    import asyncio
                    from ...providers.ctrader_provider import CTraderProvider, CTraderAuthorizationError

                    # Get credentials from config
                    client_id = self.config.get('client_id', '')
                    client_secret = self.config.get('client_secret', '')
                    access_token = self.config.get('access_token', '')
                    environment = self.config.get('environment', 'demo')

                    if not client_id or not client_secret:
                        return False, "Client ID and Client Secret are required"

                    # Create provider instance
                    provider = CTraderProvider(
                        client_id=client_id,
                        client_secret=client_secret,
                        access_token=access_token if access_token else None,
                        environment=environment
                    )

                    # Try to connect
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        connected = loop.run_until_complete(provider.connect())
                        if connected:
                            # Disconnect after successful test
                            loop.run_until_complete(provider.disconnect())
                            return True, "Connection successful!"
                        else:
                            return False, "Connection failed - check credentials"
                    finally:
                        loop.close()

                except CTraderAuthorizationError as e:
                    return False, f"Authorization error: {str(e)}"
                except Exception as e:
                    return False, f"Connection error: {str(e)}"

            def _test_tiingo(self):
                """Test Tiingo connection."""
                try:
                    import httpx

                    api_key = self.config.get('api_key', '')
                    if not api_key:
                        return False, "API Key is required"

                    # Test with a simple API call
                    url = "https://api.tiingo.com/api/test"
                    headers = {"Authorization": f"Token {api_key}"}

                    with httpx.Client(timeout=10.0) as client:
                        response = client.get(url, headers=headers)
                        response.raise_for_status()

                        # Tiingo returns {"message": "You successfully sent a request"}
                        data = response.json()
                        if "message" in data:
                            return True, f"Connection successful! {data['message']}"
                        else:
                            return True, "Connection successful!"

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 401:
                        return False, "Invalid API key"
                    else:
                        return False, f"HTTP error {e.response.status_code}"
                except Exception as e:
                    return False, f"Connection error: {str(e)}"

            def _test_alphavantage(self):
                """Test AlphaVantage connection."""
                try:
                    import httpx

                    api_key = self.config.get('api_key', '')
                    if not api_key:
                        return False, "API Key is required"

                    # Test with a simple API call (FX rate)
                    url = "https://www.alphavantage.co/query"
                    params = {
                        "function": "CURRENCY_EXCHANGE_RATE",
                        "from_currency": "USD",
                        "to_currency": "EUR",
                        "apikey": api_key
                    }

                    with httpx.Client(timeout=10.0) as client:
                        response = client.get(url, params=params)
                        response.raise_for_status()

                        data = response.json()

                        # Check for error messages
                        if "Error Message" in data:
                            return False, data["Error Message"]
                        elif "Note" in data:
                            return False, f"API limit: {data['Note']}"
                        elif "Realtime Currency Exchange Rate" in data:
                            return True, "Connection successful!"
                        else:
                            return False, "Unexpected response from AlphaVantage"

                except Exception as e:
                    return False, f"Connection error: {str(e)}"

        # Prepare configuration for test
        config = {}
        if provider == "ctrader":
            config = {
                'client_id': self.ctrader_client_id_edit.text(),
                'client_secret': self.ctrader_client_secret_edit.text(),
                'access_token': self.ctrader_access_token_edit.text(),
                'environment': self.ctrader_env_combo.currentText()
            }
        elif provider == "tiingo":
            config = {
                'api_key': self.tiingo_api_key_edit.text()
            }
        elif provider == "alphavantage":
            config = {
                'api_key': self.alphavantage_api_key_edit.text()
            }

        # Create and start worker thread
        self.test_worker = TestWorker(provider, config)

        def on_test_finished(success, message):
            progress.close()

            if success:
                QMessageBox.information(
                    self,
                    "Test Connection",
                    f"✓ {provider.upper()} Test Successful\n\n{message}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Test Connection Failed",
                    f"✗ {provider.upper()} Test Failed\n\n{message}"
                )

        self.test_worker.finished.connect(on_test_finished)
        self.test_worker.start()

        progress.exec()

    def _save_settings(self):
        """Save provider settings."""
        try:
            from ..utils.user_settings import set_setting

            # General settings
            set_setting('primary_data_provider', self.primary_combo.currentText())
            fallback = self.fallback_combo.currentText()
            set_setting('fallback_data_provider', fallback if fallback != 'none' else '')

            # cTrader settings
            set_setting('ctrader_client_id', self.ctrader_client_id_edit.text())
            set_setting('ctrader_client_secret', self.ctrader_client_secret_edit.text())
            set_setting('ctrader_access_token', self.ctrader_access_token_edit.text())
            set_setting('ctrader_account_id', self.ctrader_account_id_edit.text())
            set_setting('ctrader_environment', self.ctrader_env_combo.currentText())

            # Tiingo settings
            set_setting('tiingo_api_key', self.tiingo_api_key_edit.text())

            # AlphaVantage settings
            set_setting('alphavantage_api_key', self.alphavantage_api_key_edit.text())

            logger.info("Provider settings saved successfully")

            QMessageBox.information(
                self,
                "Settings Saved",
                "Provider settings have been saved.\n\n"
                "Please restart the application for changes to take effect."
            )

            self.accept()

        except Exception as e:
            logger.error(f"Failed to save provider settings: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save settings:\n{str(e)}"
            )
