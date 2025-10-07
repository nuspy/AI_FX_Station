"""
Order Flow Panel

Displays real-time order flow metrics including bid/ask spread, depth imbalance,
volume imbalance, and large order detection.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QGroupBox, QProgressBar, QComboBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from typing import Dict, List, Optional, Any
import numpy as np


class OrderFlowPanel(QWidget):
    """
    Order Flow monitoring panel showing:
    - Current bid/ask spread and depth
    - Volume imbalance metrics
    - Large order detection alerts
    - Order flow signals
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_metrics: Dict[str, Any] = {}
        self.signals_data: List[Dict[str, Any]] = []
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)

        # Title
        title_layout = QHBoxLayout()
        title = QLabel("Order Flow Analysis")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title_layout.addWidget(title)

        # Symbol selector
        title_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'])
        self.symbol_combo.currentTextChanged.connect(self.on_symbol_changed)
        title_layout.addWidget(self.symbol_combo)

        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Current Metrics Section
        metrics_group = self._create_metrics_section()
        layout.addWidget(metrics_group)

        # Imbalance Indicators
        imbalance_group = self._create_imbalance_section()
        layout.addWidget(imbalance_group)

        # Order Flow Signals Table
        signals_group = self._create_signals_section()
        layout.addWidget(signals_group)

        # Alerts Section
        alerts_group = self._create_alerts_section()
        layout.addWidget(alerts_group)

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(2000)  # Refresh every 2 seconds for order flow

    def _create_metrics_section(self) -> QGroupBox:
        """Create current metrics display"""
        group = QGroupBox("Current Order Flow Metrics")
        layout = QVBoxLayout()

        # Spread and Depth
        spread_layout = QHBoxLayout()

        # Bid/Ask Spread
        spread_layout.addWidget(QLabel("Spread:"))
        self.spread_label = QLabel("--")
        self.spread_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        spread_layout.addWidget(self.spread_label)

        # Spread Z-Score (anomaly detection)
        spread_layout.addWidget(QLabel("Z-Score:"))
        self.spread_zscore_label = QLabel("--")
        spread_layout.addWidget(self.spread_zscore_label)

        spread_layout.addStretch()
        layout.addLayout(spread_layout)

        # Depth Metrics
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Bid Depth:"))
        self.bid_depth_label = QLabel("--")
        depth_layout.addWidget(self.bid_depth_label)

        depth_layout.addWidget(QLabel("Ask Depth:"))
        self.ask_depth_label = QLabel("--")
        depth_layout.addWidget(self.ask_depth_label)

        depth_layout.addStretch()
        layout.addLayout(depth_layout)

        # Volume Metrics
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Buy Volume:"))
        self.buy_volume_label = QLabel("--")
        self.buy_volume_label.setStyleSheet("color: green; font-weight: bold;")
        volume_layout.addWidget(self.buy_volume_label)

        volume_layout.addWidget(QLabel("Sell Volume:"))
        self.sell_volume_label = QLabel("--")
        self.sell_volume_label.setStyleSheet("color: red; font-weight: bold;")
        volume_layout.addWidget(self.sell_volume_label)

        volume_layout.addStretch()
        layout.addLayout(volume_layout)

        group.setLayout(layout)
        return group

    def _create_imbalance_section(self) -> QGroupBox:
        """Create imbalance indicators"""
        group = QGroupBox("Imbalance Indicators")
        layout = QVBoxLayout()

        # Depth Imbalance
        depth_imb_layout = QHBoxLayout()
        depth_imb_layout.addWidget(QLabel("Depth Imbalance:"))

        self.depth_imbalance_bar = QProgressBar()
        self.depth_imbalance_bar.setRange(-100, 100)
        self.depth_imbalance_bar.setValue(0)
        self.depth_imbalance_bar.setTextVisible(True)
        self.depth_imbalance_bar.setFormat("%v%")
        depth_imb_layout.addWidget(self.depth_imbalance_bar)

        self.depth_imb_label = QLabel("Neutral")
        depth_imb_layout.addWidget(self.depth_imb_label)

        layout.addLayout(depth_imb_layout)

        # Volume Imbalance
        vol_imb_layout = QHBoxLayout()
        vol_imb_layout.addWidget(QLabel("Volume Imbalance:"))

        self.volume_imbalance_bar = QProgressBar()
        self.volume_imbalance_bar.setRange(-100, 100)
        self.volume_imbalance_bar.setValue(0)
        self.volume_imbalance_bar.setTextVisible(True)
        self.volume_imbalance_bar.setFormat("%v%")
        vol_imb_layout.addWidget(self.volume_imbalance_bar)

        self.vol_imb_label = QLabel("Neutral")
        vol_imb_layout.addWidget(self.vol_imb_label)

        layout.addLayout(vol_imb_layout)

        group.setLayout(layout)
        return group

    def _create_signals_section(self) -> QGroupBox:
        """Create order flow signals table"""
        group = QGroupBox("Order Flow Signals")
        layout = QVBoxLayout()

        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(6)
        self.signals_table.setHorizontalHeaderLabels([
            'Timestamp', 'Signal Type', 'Direction', 'Strength', 'Confidence', 'Status'
        ])
        self.signals_table.setAlternatingRowColors(True)
        self.signals_table.setMaximumHeight(150)

        layout.addWidget(self.signals_table)
        group.setLayout(layout)
        return group

    def _create_alerts_section(self) -> QGroupBox:
        """Create alerts section"""
        group = QGroupBox("Alerts")
        layout = QVBoxLayout()

        # Large Order Alert
        self.large_order_alert = QLabel("")
        self.large_order_alert.setStyleSheet(
            "background-color: #FFF3CD; color: #856404; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.large_order_alert.hide()
        layout.addWidget(self.large_order_alert)

        # Absorption Alert
        self.absorption_alert = QLabel("")
        self.absorption_alert.setStyleSheet(
            "background-color: #D1ECF1; color: #0C5460; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.absorption_alert.hide()
        layout.addWidget(self.absorption_alert)

        # Exhaustion Alert
        self.exhaustion_alert = QLabel("")
        self.exhaustion_alert.setStyleSheet(
            "background-color: #F8D7DA; color: #721C24; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.exhaustion_alert.hide()
        layout.addWidget(self.exhaustion_alert)

        group.setLayout(layout)
        return group

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update order flow metrics.

        Args:
            metrics: Dictionary with order flow metrics
        """
        self.current_metrics = metrics

        # Spread
        spread = metrics.get('spread', 0.0)
        spread_pips = spread * 10000  # Convert to pips for forex
        self.spread_label.setText(f"{spread_pips:.1f} pips")

        # Spread Z-Score
        spread_zscore = metrics.get('spread_zscore', 0.0)
        self.spread_zscore_label.setText(f"{spread_zscore:.2f}")
        if abs(spread_zscore) > 2.0:
            self.spread_zscore_label.setStyleSheet("color: red; font-weight: bold;")
        elif abs(spread_zscore) > 1.0:
            self.spread_zscore_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.spread_zscore_label.setStyleSheet("color: green;")

        # Depth
        bid_depth = metrics.get('bid_depth', 0.0)
        ask_depth = metrics.get('ask_depth', 0.0)
        self.bid_depth_label.setText(f"{bid_depth:,.0f}")
        self.ask_depth_label.setText(f"{ask_depth:,.0f}")

        # Volume
        buy_volume = metrics.get('buy_volume', 0.0)
        sell_volume = metrics.get('sell_volume', 0.0)
        self.buy_volume_label.setText(f"{buy_volume:,.0f}")
        self.sell_volume_label.setText(f"{sell_volume:,.0f}")

        # Depth Imbalance (-1 to +1)
        depth_imbalance = metrics.get('depth_imbalance', 0.0)
        depth_imb_pct = int(depth_imbalance * 100)
        self.depth_imbalance_bar.setValue(depth_imb_pct)

        if depth_imbalance > 0.3:
            self.depth_imb_label.setText("Bid Heavy")
            self.depth_imb_label.setStyleSheet("color: green; font-weight: bold;")
            self.depth_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        elif depth_imbalance < -0.3:
            self.depth_imb_label.setText("Ask Heavy")
            self.depth_imb_label.setStyleSheet("color: red; font-weight: bold;")
            self.depth_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        else:
            self.depth_imb_label.setText("Neutral")
            self.depth_imb_label.setStyleSheet("color: gray;")
            self.depth_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: gray; }")

        # Volume Imbalance
        volume_imbalance = metrics.get('volume_imbalance', 0.0)
        vol_imb_pct = int(volume_imbalance * 100)
        self.volume_imbalance_bar.setValue(vol_imb_pct)

        if volume_imbalance > 0.3:
            self.vol_imb_label.setText("Buy Pressure")
            self.vol_imb_label.setStyleSheet("color: green; font-weight: bold;")
            self.volume_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        elif volume_imbalance < -0.3:
            self.vol_imb_label.setText("Sell Pressure")
            self.vol_imb_label.setStyleSheet("color: red; font-weight: bold;")
            self.volume_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        else:
            self.vol_imb_label.setText("Neutral")
            self.vol_imb_label.setStyleSheet("color: gray;")
            self.volume_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: gray; }")

        # Alerts
        self._update_alerts(metrics)

    def _update_alerts(self, metrics: Dict[str, Any]):
        """Update alert labels"""
        # Large Order Detection
        large_order_detected = metrics.get('large_order_detected', False)
        if large_order_detected:
            direction = metrics.get('large_order_direction', 'unknown')
            self.large_order_alert.setText(f"⚠️ Large {direction.upper()} order detected!")
            self.large_order_alert.show()
        else:
            self.large_order_alert.hide()

        # Absorption
        absorption_detected = metrics.get('absorption_detected', False)
        if absorption_detected:
            self.absorption_alert.setText("🔵 Price absorption detected - Support/Resistance forming")
            self.absorption_alert.show()
        else:
            self.absorption_alert.hide()

        # Exhaustion
        exhaustion_detected = metrics.get('exhaustion_detected', False)
        if exhaustion_detected:
            self.exhaustion_alert.setText("🔴 Exhaustion detected - Potential reversal")
            self.exhaustion_alert.show()
        else:
            self.exhaustion_alert.hide()

    def update_signals(self, signals: List[Dict[str, Any]]):
        """
        Update order flow signals table.

        Args:
            signals: List of order flow signal dictionaries
        """
        self.signals_data = signals
        self._update_signals_table()

    def _update_signals_table(self):
        """Update signals table display"""
        self.signals_table.setRowCount(len(self.signals_data))

        for i, signal in enumerate(self.signals_data):
            # Timestamp
            timestamp = signal.get('timestamp', 0)
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp / 1000)
            self.signals_table.setItem(i, 0, QTableWidgetItem(dt.strftime('%H:%M:%S')))

            # Signal Type
            signal_type = signal.get('signal_type', 'Unknown')
            self.signals_table.setItem(i, 1, QTableWidgetItem(signal_type))

            # Direction
            direction = signal.get('direction', 'neutral')
            direction_item = QTableWidgetItem(direction.upper())
            if direction == 'bull':
                direction_item.setForeground(QColor('green'))
            elif direction == 'bear':
                direction_item.setForeground(QColor('red'))
            self.signals_table.setItem(i, 2, direction_item)

            # Strength
            strength = signal.get('strength', 0.0)
            self.signals_table.setItem(i, 3, QTableWidgetItem(f"{strength:.2f}"))

            # Confidence
            confidence = signal.get('confidence', 0.0)
            self.signals_table.setItem(i, 4, QTableWidgetItem(f"{confidence:.2f}"))

            # Status
            status = signal.get('status', 'active')
            status_item = QTableWidgetItem(status.upper())
            if status == 'active':
                status_item.setForeground(QColor('green'))
            elif status == 'closed':
                status_item.setForeground(QColor('gray'))
            self.signals_table.setItem(i, 5, status_item)

        self.signals_table.resizeColumnsToContents()

    def refresh_display(self):
        """Refresh display (called by timer)"""
        # This would be called by timer to refresh data
        # In production, would fetch latest metrics from backend
        pass

    def on_symbol_changed(self, symbol: str):
        """Handle symbol change"""
        # Clear current data and request new data for symbol
        self.current_metrics = {}
        self.signals_data = []
        # In production, would emit signal to request data for new symbol

    def clear_data(self):
        """Clear all data"""
        self.current_metrics = {}
        self.signals_data = []
        self._update_signals_table()
