"""
Data Sources Tab - Monitor all data sources and their connection types (REST/WS)
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor
from loguru import logger


class DataSourcesTab(QWidget):
    """
    Monitor all data sources and their connection methods.
    Shows real-time status of:
    - Realtime prices (WS/REST)
    - Historical candles (REST)
    - cTrader data (volumes, order books, etc.)
    - Provider info
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = None
        self._last_update_times: Dict[str, datetime] = {}
        self._init_ui()

    def set_main_window(self, main_window):
        """Set reference to main window for accessing data services."""
        self._main_window = main_window

    def _init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("📡 Data Sources Monitor")
        title_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #2c3e50; }")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setToolTip("Refresh data sources status")
        refresh_btn.clicked.connect(self._refresh_data_sources)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Status overview
        self._create_status_overview(layout)

        # Data sources table
        self._create_data_sources_table(layout)

        # Provider details
        self._create_provider_details(layout)

        # Auto-refresh timer (every 2 seconds for realtime monitoring)
        self._auto_refresh_timer = QTimer()
        self._auto_refresh_timer.timeout.connect(self._refresh_data_sources)
        self._auto_refresh_timer.start(2000)  # 2 seconds

        logger.info("DataSourcesTab initialized")

    def _create_status_overview(self, parent_layout: QVBoxLayout):
        """Create status overview section."""
        group = QGroupBox("Connection Status")
        layout = QHBoxLayout()

        # Realtime status
        self.realtime_status_label = QLabel("⚪ Realtime: Unknown")
        self.realtime_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; }")
        layout.addWidget(self.realtime_status_label)

        # Historical status
        self.historical_status_label = QLabel("⚪ Historical: Unknown")
        self.historical_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; }")
        layout.addWidget(self.historical_status_label)

        # cTrader status
        self.ctrader_status_label = QLabel("⚪ cTrader: Unknown")
        self.ctrader_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; }")
        layout.addWidget(self.ctrader_status_label)

        layout.addStretch()
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _create_data_sources_table(self, parent_layout: QVBoxLayout):
        """Create data sources table."""
        group = QGroupBox("Data Sources")
        layout = QVBoxLayout()

        self.sources_table = QTableWidget()
        self.sources_table.setColumnCount(6)
        self.sources_table.setHorizontalHeaderLabels([
            "Data Type", "Provider", "Method", "Status", "Last Update", "Details"
        ])

        # Configure table
        header = self.sources_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self.sources_table.setAlternatingRowColors(True)
        self.sources_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f5f5f5;
                gridline-color: #d0d0d0;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)

        layout.addWidget(self.sources_table)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _create_provider_details(self, parent_layout: QVBoxLayout):
        """Create provider details section."""
        group = QGroupBox("Provider Configuration")
        layout = QVBoxLayout()

        # Provider info labels
        self.primary_provider_label = QLabel("Primary: Not configured")
        self.secondary_provider_label = QLabel("Secondary: Not configured")
        self.ws_enabled_label = QLabel("WebSocket: Unknown")

        for label in [self.primary_provider_label, self.secondary_provider_label, self.ws_enabled_label]:
            label.setStyleSheet("QLabel { font-size: 11px; color: #555; padding: 2px; }")
            layout.addWidget(label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _refresh_data_sources(self):
        """Refresh all data sources status."""
        try:
            # Get current status
            status = self._get_data_sources_status()

            # Update overview
            self._update_status_overview(status)

            # Update table
            self._update_sources_table(status)

            # Update provider details
            self._update_provider_details(status)

        except Exception as e:
            logger.error(f"Failed to refresh data sources: {e}")

    def _get_data_sources_status(self) -> Dict[str, Any]:
        """Get current status of all data sources."""
        status = {
            'realtime_ticks': {
                'provider': 'Unknown',
                'method': 'Unknown',
                'status': 'Unknown',
                'details': ''
            },
            'historical_candles': {
                'provider': 'Unknown',
                'method': 'REST',
                'status': 'Unknown',
                'details': ''
            },
            'ctrader_ticks': {
                'provider': 'cTrader',
                'method': 'Unknown',
                'status': 'Unknown',
                'details': ''
            },
            'ctrader_volumes': {
                'provider': 'cTrader',
                'method': 'Unknown',
                'status': 'Unknown',
                'details': ''
            },
            'ctrader_orderbook': {
                'provider': 'cTrader',
                'method': 'Unknown',
                'status': 'Unknown',
                'details': ''
            },
            'primary_provider': 'Unknown',
            'secondary_provider': 'None',
            'ws_enabled': False
        }

        # Check environment variable for WebSocket
        ws_env = os.environ.get("FOREX_ENABLE_WS", "1")
        status['ws_enabled'] = ws_env == "1"

        # Get settings if available
        try:
            from forex_diffusion.utils.user_settings import get_setting
            primary = get_setting("primary_data_provider", "tiingo")
            secondary = get_setting("secondary_data_provider", None)
            use_ws = get_setting("use_websocket_streaming", False)

            status['primary_provider'] = primary
            status['secondary_provider'] = secondary or 'None'
            status['ws_enabled'] = use_ws or status['ws_enabled']
        except Exception as e:
            logger.debug(f"Could not load settings: {e}")

        # Check Tiingo WebSocket status
        if status['ws_enabled']:
            status['realtime_ticks']['method'] = 'WebSocket'
            status['realtime_ticks']['provider'] = 'Tiingo'

            # Check if connector exists and is active
            if self._main_window and hasattr(self._main_window, 'tiingo_ws_connector'):
                connector = self._main_window.tiingo_ws_connector
                if connector and hasattr(connector, '_thread'):
                    if connector._thread and connector._thread.is_alive():
                        status['realtime_ticks']['status'] = 'Connected'
                        status['realtime_ticks']['details'] = f"Subscribed to: {', '.join(connector.tickers)}"
                    else:
                        status['realtime_ticks']['status'] = 'Disconnected'
                        status['realtime_ticks']['details'] = 'Thread not running'
                else:
                    status['realtime_ticks']['status'] = 'Not Started'
            else:
                status['realtime_ticks']['status'] = 'Enabled (no connector)'
        else:
            status['realtime_ticks']['method'] = 'REST (Polling)'
            status['realtime_ticks']['provider'] = status['primary_provider']
            status['realtime_ticks']['status'] = 'Fallback Mode'
            status['realtime_ticks']['details'] = 'WebSocket disabled, using REST polling'

        # Historical candles - check actual provider used by MarketDataService
        status['historical_candles']['provider'] = status['primary_provider']
        status['historical_candles']['status'] = 'Active'

        # Check if MarketDataService exists and get actual provider
        try:
            if self._main_window and hasattr(self._main_window, 'market_service'):
                market_service = self._main_window.market_service
                if market_service and hasattr(market_service, 'provider_name'):
                    provider_name = market_service.provider_name
                    status['historical_candles']['details'] = f'MarketDataService ({provider_name})'

                    # Check if fallback occurred
                    if 'fallback' in provider_name.lower():
                        status['historical_candles']['status'] = 'Fallback Mode'
                        status['historical_candles']['details'] += ' ⚠️ Using fallback provider'
                else:
                    status['historical_candles']['details'] = 'MarketDataService (TiingoClient - default)'
            else:
                status['historical_candles']['details'] = 'MarketDataService (TiingoClient - default)'
        except Exception as e:
            logger.debug(f"Could not check MarketDataService provider: {e}")
            status['historical_candles']['details'] = 'MarketDataService (Unknown)'

        # cTrader status - check if broker is configured
        try:
            ctrader_enabled = get_setting("ctrader_enabled", False)
            if ctrader_enabled:
                # Check if broker is connected
                if self._main_window and hasattr(self._main_window, 'controller'):
                    controller = self._main_window.controller
                    if hasattr(controller, 'broker') and controller.broker:
                        broker = controller.broker
                        is_connected = getattr(broker, '_connected', False)

                        if is_connected:
                            status['ctrader_ticks']['method'] = 'WebSocket'
                            status['ctrader_ticks']['status'] = 'Connected'
                            status['ctrader_ticks']['details'] = 'ProtoOA WebSocket'

                            status['ctrader_volumes']['method'] = 'WebSocket'
                            status['ctrader_volumes']['status'] = 'Connected'
                            status['ctrader_volumes']['details'] = 'Included in tick data'

                            status['ctrader_orderbook']['method'] = 'WebSocket'
                            status['ctrader_orderbook']['status'] = 'Available'
                            status['ctrader_orderbook']['details'] = 'Depth of Market (DoM) via ProtoOA'
                        else:
                            for key in ['ctrader_ticks', 'ctrader_volumes', 'ctrader_orderbook']:
                                status[key]['status'] = 'Disconnected'
                                status[key]['details'] = 'Broker not connected'
                    else:
                        for key in ['ctrader_ticks', 'ctrader_volumes', 'ctrader_orderbook']:
                            status[key]['status'] = 'Not Initialized'
                            status[key]['details'] = 'Broker not created'
            else:
                for key in ['ctrader_ticks', 'ctrader_volumes', 'ctrader_orderbook']:
                    status[key]['status'] = 'Disabled'
                    status[key]['details'] = 'cTrader not enabled in settings'
        except Exception as e:
            logger.debug(f"Could not check cTrader status: {e}")

        return status

    def _update_status_overview(self, status: Dict[str, Any]):
        """Update status overview labels."""
        # Realtime status
        rt_status = status['realtime_ticks']['status']
        rt_method = status['realtime_ticks']['method']
        if rt_status == 'Connected':
            self.realtime_status_label.setText(f"🟢 Realtime: {rt_method} (Connected)")
            self.realtime_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #27ae60; }")
        elif rt_status == 'Fallback Mode':
            self.realtime_status_label.setText(f"🟡 Realtime: {rt_method}")
            self.realtime_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #f39c12; }")
        else:
            self.realtime_status_label.setText(f"🔴 Realtime: {rt_method} ({rt_status})")
            self.realtime_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #e74c3c; }")

        # Historical status
        hist_status = status['historical_candles']['status']
        if hist_status == 'Active':
            self.historical_status_label.setText("🟢 Historical: REST (Active)")
            self.historical_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #27ae60; }")
        else:
            self.historical_status_label.setText(f"🔴 Historical: REST ({hist_status})")
            self.historical_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #e74c3c; }")

        # cTrader status
        ct_status = status['ctrader_ticks']['status']
        if ct_status == 'Connected':
            self.ctrader_status_label.setText("🟢 cTrader: WebSocket (Connected)")
            self.ctrader_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #27ae60; }")
        elif ct_status == 'Disabled':
            self.ctrader_status_label.setText("⚪ cTrader: Disabled")
            self.ctrader_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #95a5a6; }")
        else:
            self.ctrader_status_label.setText(f"🔴 cTrader: {ct_status}")
            self.ctrader_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #e74c3c; }")

    def _update_sources_table(self, status: Dict[str, Any]):
        """Update data sources table."""
        # Define data sources to display
        sources = [
            ('Realtime Ticks', 'realtime_ticks'),
            ('Historical Candles', 'historical_candles'),
            ('cTrader Ticks', 'ctrader_ticks'),
            ('cTrader Volumes', 'ctrader_volumes'),
            ('cTrader Order Book', 'ctrader_orderbook'),
        ]

        self.sources_table.setRowCount(len(sources))

        now = datetime.now()

        for row, (label, key) in enumerate(sources):
            source = status[key]

            # Data Type
            item = QTableWidgetItem(label)
            self.sources_table.setItem(row, 0, item)

            # Provider
            item = QTableWidgetItem(source['provider'])
            self.sources_table.setItem(row, 1, item)

            # Method (REST/WS)
            method_item = QTableWidgetItem(source['method'])
            if 'WebSocket' in source['method']:
                method_item.setBackground(QColor(46, 204, 113, 50))  # Green tint
            else:
                method_item.setBackground(QColor(52, 152, 219, 50))  # Blue tint
            self.sources_table.setItem(row, 2, method_item)

            # Status
            status_item = QTableWidgetItem(source['status'])
            if source['status'] in ['Connected', 'Active', 'Available']:
                status_item.setForeground(QColor(39, 174, 96))  # Green
            elif source['status'] in ['Fallback Mode', 'Enabled (no connector)']:
                status_item.setForeground(QColor(243, 156, 18))  # Orange
            elif source['status'] in ['Disabled']:
                status_item.setForeground(QColor(149, 165, 166))  # Gray
            else:
                status_item.setForeground(QColor(231, 76, 60))  # Red
            self.sources_table.setItem(row, 3, status_item)

            # Last Update
            if key in self._last_update_times:
                elapsed = (now - self._last_update_times[key]).total_seconds()
                if elapsed < 60:
                    time_str = f"{int(elapsed)}s ago"
                elif elapsed < 3600:
                    time_str = f"{int(elapsed/60)}m ago"
                else:
                    time_str = f"{int(elapsed/3600)}h ago"
            else:
                time_str = "Never"
            item = QTableWidgetItem(time_str)
            self.sources_table.setItem(row, 4, item)

            # Details
            item = QTableWidgetItem(source['details'])
            self.sources_table.setItem(row, 5, item)

    def _update_provider_details(self, status: Dict[str, Any]):
        """Update provider configuration details."""
        self.primary_provider_label.setText(f"Primary Provider: {status['primary_provider'].upper()}")
        self.secondary_provider_label.setText(f"Secondary Provider: {status['secondary_provider']}")

        ws_text = "Enabled" if status['ws_enabled'] else "Disabled"
        ws_color = "#27ae60" if status['ws_enabled'] else "#e74c3c"
        self.ws_enabled_label.setText(f"WebSocket Streaming: {ws_text}")
        self.ws_enabled_label.setStyleSheet(f"QLabel {{ font-size: 11px; color: {ws_color}; padding: 2px; font-weight: bold; }}")

    def mark_data_received(self, data_type: str):
        """
        Mark that data was received for a specific data type.
        This updates the 'Last Update' timestamp.

        Args:
            data_type: One of 'realtime_ticks', 'historical_candles', 'ctrader_ticks', etc.
        """
        self._last_update_times[data_type] = datetime.now()
