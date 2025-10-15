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
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QSizePolicy
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
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Header unico: Order Books | Spread | Mid
        header_layout = QHBoxLayout()
        
        title_label = QLabel("📖 Order Books")
        title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #e0e0e0;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.spread_label = QLabel("Spread: --")
        self.spread_label.setStyleSheet("font-size: 9px; color: #ffa500; font-weight: bold;")
        header_layout.addWidget(self.spread_label)
        
        self.mid_price_label = QLabel("Mid: --")
        self.mid_price_label.setStyleSheet("font-size: 9px; color: #00bfff; font-weight: bold; margin-left: 10px;")
        header_layout.addWidget(self.mid_price_label)

        layout.addLayout(header_layout)

        # Tabella unica per bids + asks (10 righe: 5 bids sopra, 5 asks sotto)
        self.book_table = QTableWidget()
        self.book_table.setColumnCount(3)
        self.book_table.setHorizontalHeaderLabels(["Price", "Volume", "Total"])
        self.book_table.setRowCount(10)  # 5 bids + 5 asks
        self.book_table.horizontalHeader().setStretchLastSection(True)
        self.book_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.book_table.verticalHeader().setVisible(False)
        self.book_table.verticalHeader().setDefaultSectionSize(22)  # Row height with 2px margins
        self.book_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.book_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                gridline-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                font-size: 9px;
            }
            QTableWidget::item {
                padding: 2px 4px 2px 4px;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #ffffff;
                font-size: 9px;
                padding: 3px;
                border: 1px solid #3a3a3a;
            }
        """)
        layout.addWidget(self.book_table)
        
        layout.addStretch()

        # Connect click handler
        self.book_table.cellClicked.connect(self._on_book_clicked)

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

        except Exception as e:
            logger.error(f"Error updating order book display: {e}")

    def _update_display(self):
        """Update all display components."""
        # Update spread and mid in header
        if self._spread is not None:
            self.spread_label.setText(f"Spread: {self._spread:.5f}")
        else:
            self.spread_label.setText("Spread: --")

        if self._mid_price is not None:
            self.mid_price_label.setText(f"Mid: {self._mid_price:.5f}")
        else:
            self.mid_price_label.setText("Mid: --")

        # Populate unified table: 5 bids (rows 0-4) + 5 asks (rows 5-9)
        # Bids: highest to lowest (row 0 = best bid)
        bids_to_show = self._bids[:5]  # Take first 5 (highest)
        cumulative_bid = 0.0
        for i, (price, volume) in enumerate(bids_to_show):
            cumulative_bid += volume
            
            # Green gradient (darker = best level, lighter = deeper)
            intensity = int(95 + (i * 40))  # 95, 135, 175, 215, 255 (inverted)
            bg_color = QColor(0, intensity, 0)
            
            self._set_row_data(i, price, volume, cumulative_bid, bg_color)

        # Asks: lowest to highest (row 5 = best ask)
        asks_to_show = self._asks[:5]  # Take first 5 (lowest)
        cumulative_ask = 0.0
        for i, (price, volume) in enumerate(asks_to_show):
            cumulative_ask += volume
            
            # Red gradient (darker = deeper level, lighter = best) - OPPOSITE of bids
            intensity = int(255 - (i * 40))  # 255, 215, 175, 135, 95 (best=light, deep=dark)
            bg_color = QColor(intensity, 0, 0)
            
            self._set_row_data(5 + i, price, volume, cumulative_ask, bg_color)
    
    def _set_row_data(self, row: int, price: float, volume: float, cumulative: float, bg_color: QColor):
        """Set data for a single row with background color."""
        # Price
        price_item = QTableWidgetItem(f"{price:.5f}")
        price_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        price_item.setBackground(bg_color)
        self.book_table.setItem(row, 0, price_item)

        # Volume
        volume_item = QTableWidgetItem(f"{volume:.3f}")
        volume_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        volume_item.setBackground(bg_color)
        self.book_table.setItem(row, 1, volume_item)

        # Cumulative
        cumulative_item = QTableWidgetItem(f"{cumulative:.3f}")
        cumulative_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cumulative_item.setBackground(bg_color)
        self.book_table.setItem(row, 2, cumulative_item)

    def _on_book_clicked(self, row: int, column: int):
        """Handle click on price level in unified table."""
        if column == 0:  # Clicked on price column
            if row < 5:  # Bids (rows 0-4)
                if row < len(self._bids):
                    price = self._bids[row][0]
                    self.priceLevelClicked.emit(price, 'bid')
                    logger.debug(f"Bid level clicked: {price}")
            else:  # Asks (rows 5-9)
                ask_index = row - 5
                if ask_index < len(self._asks):
                    price = self._asks[ask_index][0]
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

        self.spread_label.setText("Spread: --")
        self.mid_price_label.setText("Mid: --")

        # Clear all rows in unified table
        for row in range(10):
            for col in range(3):
                self.book_table.setItem(row, col, QTableWidgetItem(""))
    
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
