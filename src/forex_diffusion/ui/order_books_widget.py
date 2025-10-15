"""
Order Books (DOM - Depth of Market) Widget

Displays real-time order book data with:
- Bid levels (descending)
- Ask levels (ascending)
- Spread, mid price, imbalance
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont
from loguru import logger


class OrderBooksWidget(QWidget):
    """
    Widget for displaying order book (DOM) data.

    Features:
    - Live bid/ask levels with price and volume
    - Spread and mid price display
    - Order book imbalance visualization
    - Color-coded buy/sell pressure
    """

    # Signal emitted when user clicks on a price level (for quick order entry)
    priceLevelClicked = Signal(float, str)  # (price, side: 'bid' or 'ask')

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._current_symbol: Optional[str] = None
        self._bids: List[List[float]] = []  # [[price, volume], ...]
        self._asks: List[List[float]] = []  # [[price, volume], ...]
        self._mid_price: Optional[float] = None
        self._spread: Optional[float] = None
        self._imbalance: float = 0.0

        self._max_levels = 10  # Display top 10 levels per side

        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)  # Ridotto da 5 a 2
        layout.setSpacing(2)  # Ridotto da 5 a 2

        # Header
        header_layout = QHBoxLayout()
        self.symbol_label = QLabel("--")
        self.symbol_label.setStyleSheet("font-weight: bold; font-size: 9px; color: #e0e0e0;")  # Ridotto da 11px a 9px
        header_layout.addWidget(self.symbol_label)

        header_layout.addStretch()

        self.status_label = QLabel("●")
        self.status_label.setStyleSheet("color: #888888; font-size: 14px;")
        self.status_label.setToolTip("Disconnected")
        header_layout.addWidget(self.status_label)

        layout.addLayout(header_layout)

        # Asks table (ascending order - lowest ask at bottom)
        self.asks_table = QTableWidget()
        self.asks_table.setColumnCount(3)
        self.asks_table.setHorizontalHeaderLabels(["Price", "Volume", "Total"])
        self.asks_table.horizontalHeader().setStretchLastSection(True)
        self.asks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.asks_table.verticalHeader().setVisible(False)
        self.asks_table.verticalHeader().setDefaultSectionSize(16)  # Ridotto altezza righe
        self.asks_table.setMaximumHeight(150)
        self.asks_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                gridline-color: #3a3a3a;
                border: 1px solid #3a3a3a;
                font-size: 8px;
            }
            QTableWidget::item {
                padding: 1px;
                color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #3a3a3a;
                color: #c0c0c0;
                font-size: 8px;
                padding: 1px;
                border: 1px solid #4a4a4a;
            }
        """)
        layout.addWidget(self.asks_table)

        # Spread and mid price display
        spread_layout = QVBoxLayout()
        spread_layout.setSpacing(2)

        self.spread_label = QLabel("Spread: --")
        self.spread_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spread_label.setStyleSheet("font-size: 10px; color: #ffa500; font-weight: bold;")
        spread_layout.addWidget(self.spread_label)

        self.mid_price_label = QLabel("Mid: --")
        self.mid_price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mid_price_label.setStyleSheet("font-size: 11px; color: #00bfff; font-weight: bold;")
        spread_layout.addWidget(self.mid_price_label)

        layout.addLayout(spread_layout)

        # Bids table (descending order - highest bid at top)
        self.bids_table = QTableWidget()
        self.bids_table.setColumnCount(3)
        self.bids_table.setHorizontalHeaderLabels(["Price", "Volume", "Total"])
        self.bids_table.horizontalHeader().setStretchLastSection(True)
        self.bids_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bids_table.verticalHeader().setVisible(False)
        self.bids_table.verticalHeader().setDefaultSectionSize(16)  # Ridotto altezza righe
        self.bids_table.setMaximumHeight(150)
        self.bids_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                gridline-color: #3a3a3a;
                border: 1px solid #3a3a3a;
                font-size: 8px;
            }
            QTableWidget::item {
                padding: 1px;
                color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #3a3a3a;
                color: #c0c0c0;
                padding: 3px;
                border: 1px solid #4a4a4a;
                font-size: 9px;
            }
        """)
        layout.addWidget(self.bids_table)

        # Imbalance bar
        imbalance_layout = QVBoxLayout()
        imbalance_layout.setSpacing(2)

        imbalance_label = QLabel("Order Flow Imbalance")
        imbalance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        imbalance_label.setStyleSheet("font-size: 9px; color: #a0a0a0;")
        imbalance_layout.addWidget(imbalance_label)

        # Progress bar for imbalance (-1 to +1)
        self.imbalance_bar = QProgressBar()
        self.imbalance_bar.setRange(-100, 100)
        self.imbalance_bar.setValue(0)
        self.imbalance_bar.setTextVisible(True)
        self.imbalance_bar.setFormat("%p%")
        self.imbalance_bar.setMaximumHeight(15)
        self.imbalance_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dc143c, stop:0.5 #888888, stop:1 #00ff00
                );
            }
        """)
        imbalance_layout.addWidget(self.imbalance_bar)

        self.imbalance_label = QLabel("Bid: -- | Ask: --")
        self.imbalance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imbalance_label.setStyleSheet("font-size: 8px; color: #808080;")
        imbalance_layout.addWidget(self.imbalance_label)

        layout.addLayout(imbalance_layout)

        layout.addStretch()

        # Connect click handlers
        self.bids_table.cellClicked.connect(self._on_bid_clicked)
        self.asks_table.cellClicked.connect(self._on_ask_clicked)

    @Slot(str, dict)
    def update_order_book(self, symbol: str, book_data: Dict[str, Any]):
        """
        Update the order book display with new data.

        Args:
            symbol: Symbol name (e.g., "EURUSD")
            book_data: Dict with keys:
                - bids: [[price, volume], ...]
                - asks: [[price, volume], ...]
                - mid_price: float
                - spread: float
                - imbalance: float (-1 to +1)
        """
        try:
            self._current_symbol = symbol
            self._bids = book_data.get('bids', [])[:self._max_levels]
            self._asks = book_data.get('asks', [])[:self._max_levels]
            self._mid_price = book_data.get('mid_price')
            self._spread = book_data.get('spread')
            self._imbalance = book_data.get('imbalance', 0.0)

            self._update_display()

            # Update status indicator
            self.status_label.setText("●")
            self.status_label.setStyleSheet("color: #00ff00; font-size: 14px;")
            self.status_label.setToolTip("Connected")

        except Exception as e:
            logger.error(f"Error updating order book display: {e}")

    def _update_display(self):
        """Update all display components."""
        # Update symbol
        self.symbol_label.setText(self._current_symbol or "--")

        # Update asks table (reversed to show lowest ask at bottom)
        self.asks_table.setRowCount(len(self._asks))
        cumulative_ask = 0.0
        for i, (price, volume) in enumerate(reversed(self._asks)):
            cumulative_ask += volume

            # Price
            price_item = QTableWidgetItem(f"{price:.5f}")
            price_item.setForeground(QColor(220, 20, 60))  # Crimson for asks
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.asks_table.setItem(i, 0, price_item)

            # Volume
            volume_item = QTableWidgetItem(f"{volume:.2f}")
            volume_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.asks_table.setItem(i, 1, volume_item)

            # Cumulative
            cumulative_item = QTableWidgetItem(f"{cumulative_ask:.2f}")
            cumulative_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            cumulative_item.setForeground(QColor(150, 150, 150))
            self.asks_table.setItem(i, 2, cumulative_item)

        # Update bids table
        self.bids_table.setRowCount(len(self._bids))
        cumulative_bid = 0.0
        for i, (price, volume) in enumerate(self._bids):
            cumulative_bid += volume

            # Price
            price_item = QTableWidgetItem(f"{price:.5f}")
            price_item.setForeground(QColor(0, 255, 0))  # Green for bids
            price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.bids_table.setItem(i, 0, price_item)

            # Volume
            volume_item = QTableWidgetItem(f"{volume:.2f}")
            volume_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.bids_table.setItem(i, 1, volume_item)

            # Cumulative
            cumulative_item = QTableWidgetItem(f"{cumulative_bid:.2f}")
            cumulative_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            cumulative_item.setForeground(QColor(150, 150, 150))
            self.bids_table.setItem(i, 2, cumulative_item)

        # Update spread and mid price
        if self._spread is not None:
            self.spread_label.setText(f"Spread: {self._spread:.5f}")
        else:
            self.spread_label.setText("Spread: --")

        if self._mid_price is not None:
            self.mid_price_label.setText(f"Mid: {self._mid_price:.5f}")
        else:
            self.mid_price_label.setText("Mid: --")

        # Update imbalance
        imbalance_pct = int(self._imbalance * 100)
        self.imbalance_bar.setValue(imbalance_pct)

        # Calculate total volumes
        total_bid_volume = sum(vol for _, vol in self._bids)
        total_ask_volume = sum(vol for _, vol in self._asks)
        self.imbalance_label.setText(
            f"Bid: {total_bid_volume:.2f} | Ask: {total_ask_volume:.2f}"
        )

    def _on_bid_clicked(self, row: int, column: int):
        """Handle click on bid price level."""
        if column == 0 and row < len(self._bids):  # Clicked on price column
            price = self._bids[row][0]
            self.priceLevelClicked.emit(price, 'bid')
            logger.debug(f"Bid level clicked: {price}")

    def _on_ask_clicked(self, row: int, column: int):
        """Handle click on ask price level."""
        if column == 0:  # Clicked on price column
            # Asks are reversed in display
            actual_index = len(self._asks) - 1 - row
            if 0 <= actual_index < len(self._asks):
                price = self._asks[actual_index][0]
                self.priceLevelClicked.emit(price, 'ask')
                logger.debug(f"Ask level clicked: {price}")

    @Slot()
    def clear(self):
        """Clear all data and reset display."""
        self._current_symbol = None
        self._bids = []
        self._asks = []
        self._mid_price = None
        self._spread = None
        self._imbalance = 0.0

        self.symbol_label.setText("--")
        self.spread_label.setText("Spread: --")
        self.mid_price_label.setText("Mid: --")
        self.imbalance_bar.setValue(0)
        self.imbalance_label.setText("Bid: -- | Ask: --")

        self.bids_table.setRowCount(0)
        self.asks_table.setRowCount(0)

        # Update status indicator
        self.status_label.setText("●")
        self.status_label.setStyleSheet("color: #888888; font-size: 14px;")
        self.status_label.setToolTip("Disconnected")

    @Slot()
    def set_connected(self):
        """Mark as connected."""
        self.status_label.setText("●")
        self.status_label.setStyleSheet("color: #00ff00; font-size: 14px;")
        self.status_label.setToolTip("Connected")

    @Slot()
    def set_disconnected(self):
        """Mark as disconnected."""
        self.status_label.setText("●")
        self.status_label.setStyleSheet("color: #dc143c; font-size: 14px;")
        self.status_label.setToolTip("Disconnected")
    
    def update_dom(self, bids: List[Dict], asks: List[Dict]):
        """Update DOM with bids/asks from provider.
        
        Args:
            bids: List of {'price': float, 'size': float}
            asks: List of {'price': float, 'size': float}
        """
        try:
            # Convert to format expected by update_book
            bids_formatted = [[b['price'], b['size']] for b in bids if 'price' in b and 'size' in b]
            asks_formatted = [[a['price'], a['size']] for a in asks if 'price' in a and 'size' in a]
            
            # Calculate mid, spread, imbalance
            mid = None
            spread = None
            imbalance = 0.0
            
            if bids_formatted and asks_formatted:
                best_bid = bids_formatted[0][0]
                best_ask = asks_formatted[0][0]
                mid = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                
                total_bid_vol = sum(b[1] for b in bids_formatted)
                total_ask_vol = sum(a[1] for a in asks_formatted)
                if total_bid_vol + total_ask_vol > 0:
                    imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
            
            book_data = {
                'bids': bids_formatted,
                'asks': asks_formatted,
                'mid_price': mid,
                'spread': spread,
                'imbalance': imbalance
            }
            
            self.update_order_book(self._current_symbol or "EUR/USD", book_data)
            
        except Exception as e:
            logger.error(f"Error in update_dom: {e}", exc_info=True)
