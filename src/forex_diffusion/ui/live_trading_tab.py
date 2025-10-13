"""
Live Trading Tab - cTrader Broker Integration

Provides interface for live trading with FxPro cTrader broker:
- Connect to broker (real or simulated)
- Monitor positions and orders
- Place market/limit orders
- Modify stop-loss and take-profit
- Real-time P&L tracking
"""
from __future__ import annotations

from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QTextEdit,
    QCheckBox, QTabWidget, QDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from loguru import logger

from ..broker import CTraderBroker, BrokerSimulator, OrderSide, Position


class LiveTradingTab(QWidget):
    """
    Live trading interface for cTrader broker integration.

    Features:
    - Connection management (real/simulated)
    - Position monitoring
    - Order execution
    - Real-time P&L updates
    - Risk management controls
    """

    def __init__(self, parent=None, dom_service=None, execution_optimizer=None):
        super().__init__(parent)
        self.broker: Optional[CTraderBroker] = None
        self.is_connected = False
        self.positions_cache: Dict[str, Position] = {}
        self.dom_service = dom_service
        self.execution_optimizer = execution_optimizer

        self._init_ui()
        self._setup_refresh_timer()

    def _init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Connection section
        self._create_connection_section(layout)

        # Trading controls
        self._create_trading_controls(layout)

        # Positions table
        self._create_positions_section(layout)

        # Log section
        self._create_log_section(layout)

    def _create_connection_section(self, layout: QVBoxLayout):
        """Create broker connection section"""
        group = QGroupBox("Broker Connection")
        group_layout = QFormLayout(group)

        # Connection mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simulated", "Live (cTrader)"])
        self.mode_combo.setToolTip("Select simulation mode for testing or live mode for real trading")
        group_layout.addRow("Mode:", self.mode_combo)

        # API credentials (only for live mode)
        self.client_id_edit = QLineEdit()
        self.client_id_edit.setPlaceholderText("Your cTrader Client ID")
        self.client_id_edit.setEchoMode(QLineEdit.Password)
        group_layout.addRow("Client ID:", self.client_id_edit)

        self.client_secret_edit = QLineEdit()
        self.client_secret_edit.setPlaceholderText("Your cTrader Client Secret")
        self.client_secret_edit.setEchoMode(QLineEdit.Password)
        group_layout.addRow("Client Secret:", self.client_secret_edit)

        # Connection controls
        controls_layout = QHBoxLayout()

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect_broker)
        self.connect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        controls_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._disconnect_broker)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        controls_layout.addWidget(self.disconnect_btn)

        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("QLabel { color: #f44336; font-weight: bold; }")
        controls_layout.addWidget(self.status_label)

        controls_layout.addStretch()
        group_layout.addRow(controls_layout)

        layout.addWidget(group)

    def _create_trading_controls(self, layout: QVBoxLayout):
        """Create trading control section"""
        group = QGroupBox("Place Order")
        group_layout = QFormLayout(group)

        # Symbol
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"])
        self.symbol_combo.setEditable(True)
        group_layout.addRow("Symbol:", self.symbol_combo)

        # Side
        self.side_combo = QComboBox()
        self.side_combo.addItems(["BUY", "SELL"])
        group_layout.addRow("Side:", self.side_combo)

        # Volume (in lots)
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0.01, 100.0)
        self.volume_spin.setValue(0.1)
        self.volume_spin.setSingleStep(0.01)
        self.volume_spin.setDecimals(2)
        self.volume_spin.setSuffix(" lots")
        group_layout.addRow("Volume:", self.volume_spin)

        # Stop Loss (optional)
        self.sl_check = QCheckBox("Set Stop Loss")
        self.sl_spin = QDoubleSpinBox()
        self.sl_spin.setRange(0.0, 10.0)
        self.sl_spin.setValue(0.001)
        self.sl_spin.setDecimals(5)
        self.sl_spin.setSuffix(" (price offset)")
        self.sl_spin.setEnabled(False)
        self.sl_check.toggled.connect(self.sl_spin.setEnabled)

        sl_layout = QHBoxLayout()
        sl_layout.addWidget(self.sl_check)
        sl_layout.addWidget(self.sl_spin)
        group_layout.addRow("Stop Loss:", sl_layout)

        # Take Profit (optional)
        self.tp_check = QCheckBox("Set Take Profit")
        self.tp_spin = QDoubleSpinBox()
        self.tp_spin.setRange(0.0, 10.0)
        self.tp_spin.setValue(0.002)
        self.tp_spin.setDecimals(5)
        self.tp_spin.setSuffix(" (price offset)")
        self.tp_spin.setEnabled(False)
        self.tp_check.toggled.connect(self.tp_spin.setEnabled)

        tp_layout = QHBoxLayout()
        tp_layout.addWidget(self.tp_check)
        tp_layout.addWidget(self.tp_spin)
        group_layout.addRow("Take Profit:", tp_layout)

        # Place order button
        self.place_order_btn = QPushButton("Place Market Order")
        self.place_order_btn.clicked.connect(self._place_market_order)
        self.place_order_btn.setEnabled(False)
        self.place_order_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        group_layout.addRow(self.place_order_btn)

        layout.addWidget(group)

    def _create_positions_section(self, layout: QVBoxLayout):
        """Create positions monitoring section"""
        group = QGroupBox("Open Positions")
        group_layout = QVBoxLayout(group)

        # Positions table
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(9)
        self.positions_table.setHorizontalHeaderLabels([
            "Position ID", "Symbol", "Side", "Volume", "Entry Price",
            "Current Price", "P&L", "Stop Loss", "Take Profit"
        ])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.positions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.positions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        group_layout.addWidget(self.positions_table)

        # Position controls
        controls_layout = QHBoxLayout()

        self.close_position_btn = QPushButton("Close Selected Position")
        self.close_position_btn.clicked.connect(self._close_selected_position)
        self.close_position_btn.setEnabled(False)
        controls_layout.addWidget(self.close_position_btn)

        self.modify_position_btn = QPushButton("Modify SL/TP")
        self.modify_position_btn.clicked.connect(self._modify_selected_position)
        self.modify_position_btn.setEnabled(False)
        controls_layout.addWidget(self.modify_position_btn)

        self.refresh_positions_btn = QPushButton("Refresh")
        self.refresh_positions_btn.clicked.connect(self._refresh_positions)
        self.refresh_positions_btn.setEnabled(False)
        controls_layout.addWidget(self.refresh_positions_btn)

        controls_layout.addStretch()
        group_layout.addLayout(controls_layout)

        layout.addWidget(group)

    def _create_log_section(self, layout: QVBoxLayout):
        """Create log section"""
        group = QGroupBox("Activity Log")
        group_layout = QVBoxLayout(group)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        group_layout.addWidget(self.log_view)

        layout.addWidget(group)

    def _setup_refresh_timer(self):
        """Setup auto-refresh timer for positions"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._auto_refresh_positions)
        # Will start when connected

    def _append_log(self, message: str):
        """Append message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")

    async def _connect_broker(self):
        """Connect to broker"""
        try:
            mode = self.mode_combo.currentText()

            if mode == "Simulated":
                # Use simulator
                self.broker = BrokerSimulator()
                self._append_log("Connected to simulated broker")

            else:
                # Use real cTrader API
                client_id = self.client_id_edit.text().strip()
                client_secret = self.client_secret_edit.text().strip()

                if not client_id or not client_secret:
                    QMessageBox.warning(self, "Missing Credentials", "Please enter Client ID and Client Secret")
                    return

                self.broker = CTraderBroker(
                    client_id=client_id,
                    client_secret=client_secret
                )

                # Connect
                await self.broker.connect()
                self._append_log("Connected to cTrader broker")

            # Update UI state
            self.is_connected = True
            self.status_label.setText("Status: Connected")
            self.status_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")

            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.place_order_btn.setEnabled(True)
            self.close_position_btn.setEnabled(True)
            self.modify_position_btn.setEnabled(True)
            self.refresh_positions_btn.setEnabled(True)

            # Start auto-refresh
            self.refresh_timer.start(5000)  # Refresh every 5 seconds

            # Initial refresh
            await self._refresh_positions()

        except Exception as e:
            logger.exception(f"Failed to connect: {e}")
            self._append_log(f"ERROR: {e}")
            QMessageBox.critical(self, "Connection Error", f"Failed to connect:\n{e}")

    async def _disconnect_broker(self):
        """Disconnect from broker"""
        try:
            if self.broker:
                if hasattr(self.broker, 'disconnect'):
                    await self.broker.disconnect()
                self.broker = None

            # Stop refresh timer
            self.refresh_timer.stop()

            # Update UI state
            self.is_connected = False
            self.status_label.setText("Status: Disconnected")
            self.status_label.setStyleSheet("QLabel { color: #f44336; font-weight: bold; }")

            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.place_order_btn.setEnabled(False)
            self.close_position_btn.setEnabled(False)
            self.modify_position_btn.setEnabled(False)
            self.refresh_positions_btn.setEnabled(False)

            # Clear positions
            self.positions_table.setRowCount(0)
            self.positions_cache.clear()

            self._append_log("Disconnected from broker")

        except Exception as e:
            logger.exception(f"Failed to disconnect: {e}")
            self._append_log(f"ERROR: {e}")

    async def _place_market_order(self):
        """Place market order with pre-trade validation"""
        if not self.broker:
            return

        try:
            symbol = self.symbol_combo.currentText()
            side_text = self.side_combo.currentText()
            side = OrderSide.BUY if side_text == "BUY" else OrderSide.SELL
            volume = self.volume_spin.value()

            # PRE-TRADE VALIDATION
            # Show validation dialog before executing order
            from .pre_trade_validation_dialog import PreTradeValidationDialog

            validation_dialog = PreTradeValidationDialog(
                parent=self,
                symbol=symbol.replace("/", ""),  # Convert EUR/USD to EURUSD
                side=side_text,
                volume=volume,
                dom_service=self.dom_service,
                execution_optimizer=self.execution_optimizer
            )

            # Show dialog and wait for user decision
            result = validation_dialog.exec()

            if result != QDialog.Accepted:
                self._append_log(f"Order cancelled by user after validation check")
                return

            # Log validation results
            if validation_dialog.validation_results:
                risks = ", ".join(
                    f"{k}={v}" for k, v in validation_dialog.validation_results.items()
                )
                self._append_log(f"Pre-trade validation: {risks}")

            # Calculate SL/TP prices (simplified - should use current market price)
            stop_loss = None
            take_profit = None

            if self.sl_check.isChecked():
                # In real implementation, get current price and calculate SL
                stop_loss = self.sl_spin.value()

            if self.tp_check.isChecked():
                # In real implementation, get current price and calculate TP
                take_profit = self.tp_spin.value()

            self._append_log(f"Placing {side.value} order: {volume} lots of {symbol}")

            order = await self.broker.place_market_order(
                symbol=symbol,
                side=side,
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            self._append_log(f"Order placed: {order.order_id}")

            # Refresh positions
            await self._refresh_positions()

            QMessageBox.information(self, "Order Placed", f"Order {order.order_id} placed successfully")

        except Exception as e:
            logger.exception(f"Failed to place order: {e}")
            self._append_log(f"ERROR: {e}")
            QMessageBox.critical(self, "Order Error", f"Failed to place order:\n{e}")

    async def _refresh_positions(self):
        """Refresh positions from broker"""
        if not self.broker:
            return

        try:
            positions = await self.broker.get_positions()

            # Update cache
            self.positions_cache = {p.position_id: p for p in positions}

            # Update table
            self.positions_table.setRowCount(len(positions))

            for row, position in enumerate(positions):
                self.positions_table.setItem(row, 0, QTableWidgetItem(position.position_id))
                self.positions_table.setItem(row, 1, QTableWidgetItem(position.symbol))
                self.positions_table.setItem(row, 2, QTableWidgetItem(position.side.value))
                self.positions_table.setItem(row, 3, QTableWidgetItem(f"{position.volume:.2f}"))
                self.positions_table.setItem(row, 4, QTableWidgetItem(f"{position.entry_price:.5f}"))

                # Current price (would come from broker in real implementation)
                current_price = position.entry_price  # Placeholder
                self.positions_table.setItem(row, 5, QTableWidgetItem(f"{current_price:.5f}"))

                # P&L
                pnl_item = QTableWidgetItem(f"{position.unrealized_pnl:.2f}")
                if position.unrealized_pnl > 0:
                    pnl_item.setForeground(QColor(76, 175, 80))  # Green
                elif position.unrealized_pnl < 0:
                    pnl_item.setForeground(QColor(244, 67, 54))  # Red
                self.positions_table.setItem(row, 6, pnl_item)

                # SL/TP
                sl_text = f"{position.stop_loss:.5f}" if position.stop_loss else "N/A"
                tp_text = f"{position.take_profit:.5f}" if position.take_profit else "N/A"
                self.positions_table.setItem(row, 7, QTableWidgetItem(sl_text))
                self.positions_table.setItem(row, 8, QTableWidgetItem(tp_text))

        except Exception as e:
            logger.exception(f"Failed to refresh positions: {e}")
            self._append_log(f"ERROR refreshing positions: {e}")

    def _auto_refresh_positions(self):
        """Auto-refresh positions (called by timer)"""
        import asyncio
        try:
            asyncio.create_task(self._refresh_positions())
        except Exception as e:
            logger.error(f"Auto-refresh failed: {e}")

    async def _close_selected_position(self):
        """Close selected position"""
        if not self.broker:
            return

        try:
            selected_rows = self.positions_table.selectedItems()
            if not selected_rows:
                QMessageBox.warning(self, "No Selection", "Please select a position to close")
                return

            row = self.positions_table.currentRow()
            position_id = self.positions_table.item(row, 0).text()

            reply = QMessageBox.question(
                self,
                "Confirm Close",
                f"Close position {position_id}?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self._append_log(f"Closing position {position_id}")
                success = await self.broker.close_position(position_id)

                if success:
                    self._append_log(f"Position {position_id} closed")
                    await self._refresh_positions()
                    QMessageBox.information(self, "Success", "Position closed successfully")
                else:
                    self._append_log(f"Failed to close position {position_id}")
                    QMessageBox.warning(self, "Error", "Failed to close position")

        except Exception as e:
            logger.exception(f"Failed to close position: {e}")
            self._append_log(f"ERROR: {e}")
            QMessageBox.critical(self, "Error", f"Failed to close position:\n{e}")

    async def _modify_selected_position(self):
        """Modify SL/TP for selected position"""
        if not self.broker:
            return

        try:
            selected_rows = self.positions_table.selectedItems()
            if not selected_rows:
                QMessageBox.warning(self, "No Selection", "Please select a position to modify")
                return

            row = self.positions_table.currentRow()
            position_id = self.positions_table.item(row, 0).text()

            # TODO: Create dialog for SL/TP modification
            QMessageBox.information(
                self,
                "Modify Position",
                f"Position modification dialog for {position_id}\n(To be implemented)"
            )

        except Exception as e:
            logger.exception(f"Failed to modify position: {e}")
            self._append_log(f"ERROR: {e}")
