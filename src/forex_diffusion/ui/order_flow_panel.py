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
from PySide6.QtGui import QColor
from typing import Dict, List, Any
from loguru import logger


class OrderFlowPanel(QWidget):
    """
    Order Flow monitoring panel showing:
    - Current bid/ask spread and depth
    - Volume imbalance metrics
    - Large order detection alerts
    - Order flow signals
    """

    def __init__(self, parent=None, dom_service=None, order_flow_analyzer=None):
        super().__init__(parent)
        self.current_metrics: Dict[str, Any] = {}
        self.signals_data: List[Dict[str, Any]] = []
        self.dom_service = dom_service
        self.order_flow_analyzer = order_flow_analyzer
        self.current_symbol = 'EURUSD'

        # Track historical data for volume calculations
        self.last_snapshot = None

        # Fixed number of rows for order book tables to avoid layout resizes
        self._order_book_rows = 10

        # Caches to avoid redundant UI churn (prevents fullscreen drop)
        self._metrics_cache: Dict[str, Any] = {
            "spread_text": None,
            "spread_style": None,
            "spread_zscore": None,
            "bid_depth": None,
            "ask_depth": None,
            "buy_volume": None,
            "sell_volume": None,
            "depth_bar_value": None,
            "depth_state": None,
            "volume_bar_value": None,
            "volume_state": None,
        }
        self._alerts_cache: Dict[str, Any] = {
            "large_order": None,
            "absorption": None,
            "exhaustion": None,
        }

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
        
        # Order Books Section
        order_books_group = self._create_order_books_section()
        layout.addWidget(order_books_group)

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

    def _create_order_books_section(self) -> QGroupBox:
        """Create order books display (bids and asks)"""
        group = QGroupBox("Order Books (Level 2)")
        layout = QHBoxLayout()
        
        # Bids (Buy orders) - Left side
        bids_layout = QVBoxLayout()
        bids_label = QLabel("Bids (Buy)")
        bids_label.setStyleSheet("font-weight: bold; color: green;")
        bids_layout.addWidget(bids_label)
        
        self.bids_table = QTableWidget()
        self.bids_table.setColumnCount(2)
        self.bids_table.setHorizontalHeaderLabels(['Price', 'Size'])
        self.bids_table.setAlternatingRowColors(True)
        self.bids_table.setMaximumHeight(200)
        bids_layout.addWidget(self.bids_table)
        
        # Asks (Sell orders) - Right side
        asks_layout = QVBoxLayout()
        asks_label = QLabel("Asks (Sell)")
        asks_label.setStyleSheet("font-weight: bold; color: red;")
        asks_layout.addWidget(asks_label)
        
        self.asks_table = QTableWidget()
        self.asks_table.setColumnCount(2)
        self.asks_table.setHorizontalHeaderLabels(['Price', 'Size'])
        self.asks_table.setAlternatingRowColors(True)
        self.asks_table.setMaximumHeight(200)
        asks_layout.addWidget(self.asks_table)

        # Initialize tables with fixed row count
        self.bids_table.setRowCount(self._order_book_rows)
        self.asks_table.setRowCount(self._order_book_rows)
        
        layout.addLayout(bids_layout)
        layout.addLayout(asks_layout)
        
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

        # Exhaustion Alert (kept visible to avoid layout thrash)
        self.exhaustion_alert = QLabel("")
        self._exhaustion_active_style = (
            "background-color: #F8D7DA; color: #721C24; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self._exhaustion_inactive_style = (
            "background-color: transparent; color: #721C24; "
            "padding: 5px; border-radius: 3px; font-weight: bold;"
        )
        self.exhaustion_alert.setStyleSheet(self._exhaustion_inactive_style)
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

        spread = metrics.get('spread', 0.0)
        spread_text = f"{spread * 10000:.1f} pips"
        if self._metrics_cache["spread_text"] != spread_text:
            self.spread_label.setText(spread_text)
            self._metrics_cache["spread_text"] = spread_text

        spread_zscore = metrics.get('spread_zscore', 0.0)
        spread_z_text = f"{spread_zscore:.2f}"
        if self._metrics_cache["spread_zscore"] != spread_z_text:
            self.spread_zscore_label.setText(spread_z_text)
            self._metrics_cache["spread_zscore"] = spread_z_text

        spread_style = "green"
        if abs(spread_zscore) > 2.0:
            spread_style = "red"
        elif abs(spread_zscore) > 1.0:
            spread_style = "orange"
        if self._metrics_cache["spread_style"] != spread_style:
            style_map = {
                "red": "color: red; font-weight: bold;",
                "orange": "color: orange; font-weight: bold;",
                "green": "color: green;",
            }
            self.spread_zscore_label.setStyleSheet(style_map[spread_style])
            self._metrics_cache["spread_style"] = spread_style

        bid_depth = metrics.get('bid_depth', 0.0)
        ask_depth = metrics.get('ask_depth', 0.0)
        bid_depth_text = f"{bid_depth:,.0f}"
        ask_depth_text = f"{ask_depth:,.0f}"
        if self._metrics_cache["bid_depth"] != bid_depth_text:
            self.bid_depth_label.setText(bid_depth_text)
            self._metrics_cache["bid_depth"] = bid_depth_text
        if self._metrics_cache["ask_depth"] != ask_depth_text:
            self.ask_depth_label.setText(ask_depth_text)
            self._metrics_cache["ask_depth"] = ask_depth_text

        buy_volume = metrics.get('buy_volume', 0.0)
        sell_volume = metrics.get('sell_volume', 0.0)
        buy_volume_text = f"{buy_volume:,.0f}"
        sell_volume_text = f"{sell_volume:,.0f}"
        if self._metrics_cache["buy_volume"] != buy_volume_text:
            self.buy_volume_label.setText(buy_volume_text)
            self._metrics_cache["buy_volume"] = buy_volume_text
        if self._metrics_cache["sell_volume"] != sell_volume_text:
            self.sell_volume_label.setText(sell_volume_text)
            self._metrics_cache["sell_volume"] = sell_volume_text

        depth_imbalance = metrics.get('depth_imbalance', 0.0)
        depth_imb_pct = int(depth_imbalance * 100)
        if self._metrics_cache["depth_bar_value"] != depth_imb_pct:
            self.depth_imbalance_bar.setValue(depth_imb_pct)
            self._metrics_cache["depth_bar_value"] = depth_imb_pct

        if depth_imbalance > 0.3:
            depth_state = "bid"
        elif depth_imbalance < -0.3:
            depth_state = "ask"
        else:
            depth_state = "neutral"
        if self._metrics_cache["depth_state"] != depth_state:
            if depth_state == "bid":
                self.depth_imb_label.setText("Bid Heavy")
                self.depth_imb_label.setStyleSheet("color: green; font-weight: bold;")
                self.depth_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            elif depth_state == "ask":
                self.depth_imb_label.setText("Ask Heavy")
                self.depth_imb_label.setStyleSheet("color: red; font-weight: bold;")
                self.depth_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            else:
                self.depth_imb_label.setText("Neutral")
                self.depth_imb_label.setStyleSheet("color: gray;")
                self.depth_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: gray; }")
            self._metrics_cache["depth_state"] = depth_state

        volume_imbalance = metrics.get('volume_imbalance', 0.0)
        vol_imb_pct = int(volume_imbalance * 100)
        if self._metrics_cache["volume_bar_value"] != vol_imb_pct:
            self.volume_imbalance_bar.setValue(vol_imb_pct)
            self._metrics_cache["volume_bar_value"] = vol_imb_pct

        if volume_imbalance > 0.3:
            volume_state = "buy"
        elif volume_imbalance < -0.3:
            volume_state = "sell"
        else:
            volume_state = "neutral"
        if self._metrics_cache["volume_state"] != volume_state:
            if volume_state == "buy":
                self.vol_imb_label.setText("Buy Pressure")
                self.vol_imb_label.setStyleSheet("color: green; font-weight: bold;")
                self.volume_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            elif volume_state == "sell":
                self.vol_imb_label.setText("Sell Pressure")
                self.vol_imb_label.setStyleSheet("color: red; font-weight: bold;")
                self.volume_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            else:
                self.vol_imb_label.setText("Neutral")
                self.vol_imb_label.setStyleSheet("color: gray;")
                self.volume_imbalance_bar.setStyleSheet("QProgressBar::chunk { background-color: gray; }")
            self._metrics_cache["volume_state"] = volume_state

        self._update_alerts(metrics)

    def _update_alerts(self, metrics: Dict[str, Any]):
        """Update alert labels"""
        # Large Order Detection
        large_order_detected = metrics.get('large_order_detected', False)
        if self._alerts_cache['large_order'] != large_order_detected:
            if large_order_detected:
                direction = metrics.get('large_order_direction', 'unknown')
                self.large_order_alert.setText(f"⚠️ Large {direction.upper()} order detected!")
                self.large_order_alert.show()
            else:
                self.large_order_alert.hide()
            self._alerts_cache['large_order'] = large_order_detected

        # Absorption
        absorption_detected = metrics.get('absorption_detected', False)
        if self._alerts_cache['absorption'] != absorption_detected:
            if absorption_detected:
                self.absorption_alert.setText("🔵 Price absorption detected - Support/Resistance forming")
                self.absorption_alert.show()
            else:
                self.absorption_alert.hide()
            self._alerts_cache['absorption'] = absorption_detected

        # Exhaustion
        exhaustion_detected = metrics.get('exhaustion_detected', False)
        if self._alerts_cache['exhaustion'] != exhaustion_detected:
            if exhaustion_detected:
                self.exhaustion_alert.setText("🔴 Exhaustion detected - Potential reversal")
                self.exhaustion_alert.setStyleSheet(self._exhaustion_active_style)
            else:
                self.exhaustion_alert.setText("")
                self.exhaustion_alert.setStyleSheet(self._exhaustion_inactive_style)
            self._alerts_cache['exhaustion'] = exhaustion_detected

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
        """Refresh display (called by timer) - fetches real-time DOM data"""
        if not self.dom_service:
            logger.warning("No DOM service available")
            return

        try:
            # Get latest DOM snapshot from database
            snapshot = self.dom_service.get_latest_dom_snapshot(self.current_symbol)
            if not snapshot:
                return

            # Compute order flow metrics using analyzer
            if self.order_flow_analyzer:
                # Estimate volume (would normally come from tick data)
                # For now, use depth as proxy
                buy_volume = snapshot['bid_depth'] * 0.5 if self.last_snapshot is None else \
                    max(0, snapshot['bid_depth'] - self.last_snapshot.get('bid_depth', 0))
                sell_volume = snapshot['ask_depth'] * 0.5 if self.last_snapshot is None else \
                    max(0, snapshot['ask_depth'] - self.last_snapshot.get('ask_depth', 0))

                # Compute detailed metrics
                # snapshot['timestamp'] is already in milliseconds from provider
                timestamp_ms = snapshot['timestamp'] if isinstance(snapshot['timestamp'], int) else int(snapshot['timestamp'].timestamp() * 1000)
                metrics = self.order_flow_analyzer.compute_metrics(
                    timestamp=timestamp_ms,
                    symbol=self.current_symbol,
                    timeframe='1m',
                    bid_price=snapshot['best_bid'],
                    ask_price=snapshot['best_ask'],
                    bid_size=snapshot['bid_depth'],
                    ask_size=snapshot['ask_depth'],
                    buy_volume=buy_volume,
                    sell_volume=sell_volume,
                    current_price=snapshot['mid_price']
                )

                # Convert to dict format for update_metrics
                metrics_dict = {
                    'spread': metrics.bid_ask_spread,
                    'spread_zscore': metrics.spread_zscore,
                    'bid_depth': metrics.bid_depth,
                    'ask_depth': metrics.ask_depth,
                    'buy_volume': metrics.buy_volume,
                    'sell_volume': metrics.sell_volume,
                    'depth_imbalance': metrics.depth_imbalance,
                    'volume_imbalance': metrics.volume_imbalance,
                    'large_order_detected': metrics.large_order_count > 0,
                    'large_order_direction': 'BUY' if metrics.volume_imbalance > 0 else 'SELL',
                    'absorption_detected': metrics.absorption_detected,
                    'exhaustion_detected': metrics.exhaustion_detected,
                }

                self.update_metrics(metrics_dict)
                
                # Update order books display
                self._update_order_books(snapshot.get('bids', []), snapshot.get('asks', []))

                # Generate and display signals
                if snapshot['mid_price'] > 0:
                    # Estimate ATR as 0.5% of price for forex
                    atr = snapshot['mid_price'] * 0.005
                    signals = self.order_flow_analyzer.generate_signals(
                        metrics, snapshot['mid_price'], atr
                    )

                    # Convert signals to dict format
                    signals_data = []
                    for sig in signals[-5:]:  # Show last 5 signals
                        signals_data.append({
                            'timestamp': sig.timestamp,
                            'signal_type': sig.signal_type.value,
                            'direction': sig.direction,
                            'strength': sig.strength,
                            'confidence': sig.confidence,
                            'status': 'active'
                        })
                    if signals_data:
                        self.update_signals(signals_data)

            else:
                # Fallback: use basic metrics from snapshot
                metrics_dict = {
                    'spread': snapshot['spread'],
                    'spread_zscore': 0.0,
                    'bid_depth': snapshot['bid_depth'],
                    'ask_depth': snapshot['ask_depth'],
                    'buy_volume': 0.0,
                    'sell_volume': 0.0,
                    'depth_imbalance': snapshot['depth_imbalance'],
                    'volume_imbalance': 0.0,
                    'large_order_detected': False,
                    'absorption_detected': False,
                    'exhaustion_detected': False,
                }
                self.update_metrics(metrics_dict)

            # Store for next iteration
            self.last_snapshot = snapshot

        except Exception as e:
            logger.error(f"Error refreshing order flow display: {e}", exc_info=True)

    def on_symbol_changed(self, symbol: str):
        """Handle symbol change"""
        self.current_symbol = symbol
        self.current_metrics = {}
        self.signals_data = []
        self.last_snapshot = None

        # Reset analyzer state if available
        if self.order_flow_analyzer:
            self.order_flow_analyzer.reset()

        # Immediately fetch new data
        self.refresh_display()

        logger.info(f"Order Flow Panel switched to {symbol}")
    
    def _update_order_books(self, bids: list, asks: list):
        """Update order book tables with bids and asks.
        
        Args:
            bids: List of bid orders [{'price': float, 'size': float}, ...]
            asks: List of ask orders [{'price': float, 'size': float}, ...]
        """
        try:
            # Ensure tables keep the fixed row count
            if self.bids_table.rowCount() != self._order_book_rows:
                self.bids_table.setRowCount(self._order_book_rows)
            if self.asks_table.rowCount() != self._order_book_rows:
                self.asks_table.setRowCount(self._order_book_rows)

            # Update bids table (buy orders)
            self.bids_table.blockSignals(True)
            try:
                for row in range(self._order_book_rows):
                    if row < len(bids):
                        bid = bids[row]
                        price_text = f"{bid['price']:.5f}"
                        size_text = f"{bid['size']:.0f}"
                    else:
                        price_text = ""
                        size_text = ""
                    self._set_table_value(self.bids_table, row, 0, price_text, QColor('green'))
                    self._set_table_value(self.bids_table, row, 1, size_text, QColor('green'))
            finally:
                self.bids_table.blockSignals(False)

            # Update asks table (sell orders)
            self.asks_table.blockSignals(True)
            try:
                for row in range(self._order_book_rows):
                    if row < len(asks):
                        ask = asks[row]
                        price_text = f"{ask['price']:.5f}"
                        size_text = f"{ask['size']:.0f}"
                    else:
                        price_text = ""
                        size_text = ""
                    self._set_table_value(self.asks_table, row, 0, price_text, QColor('red'))
                    self._set_table_value(self.asks_table, row, 1, size_text, QColor('red'))
            finally:
                self.asks_table.blockSignals(False)

        except Exception as e:
            logger.error(f"Error updating order books: {e}", exc_info=True)

    def clear_data(self):
        """Clear all data"""
        self.current_metrics = {}
        self.signals_data = []
        self._update_signals_table()

    def _set_table_value(self, table: QTableWidget, row: int, col: int, text: str, color: QColor):
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, col, item)
        if item.text() != text:
            item.setText(text)
        item.setForeground(color)
