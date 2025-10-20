"""
Positions Table Widget

Real-time display of open trading positions with P&L tracking,
color coding, and position management capabilities.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QMenu, QMessageBox, QAbstractItemView
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QCursor
from loguru import logger


class PositionsTableWidget(QWidget):
    """
    Widget for displaying open positions in real-time.

    Features:
    - Real-time P&L updates with color coding
    - Context menu for position management
    - Double-click to highlight on chart
    - Auto-refresh every 1 second
    """

    # Signals
    position_selected = Signal(dict)  # Emitted when position double-clicked
    close_position_requested = Signal(str)  # position_id
    modify_sl_requested = Signal(str, float)  # position_id, new_sl
    modify_tp_requested = Signal(str, float)  # position_id, new_tp

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._positions: Dict[str, Dict[str, Any]] = {}
        self._trading_engine = None

        self._setup_ui()
        self._setup_refresh_timer()

        logger.info("PositionsTableWidget initialized")

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Symbol", "Direction", "Size", "Entry", "Current",
            "P&L ($)", "P&L (%)", "Stop Loss", "Take Profit",
            "Duration", "R:R", "Trailing"
        ])

        # Table settings
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        # Column resize modes
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Symbol
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Direction
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Size
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Entry
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Current
        header.setSectionResizeMode(5, QHeaderView.Stretch)  # P&L ($)
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # P&L (%)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Stop Loss
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # Take Profit
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)  # Duration
        header.setSectionResizeMode(10, QHeaderView.ResizeToContents)  # R:R
        header.setSectionResizeMode(11, QHeaderView.ResizeToContents)  # Trailing

        # Connect signals
        self.table.doubleClicked.connect(self._on_row_double_clicked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.table)

    def _setup_refresh_timer(self):
        """Setup auto-refresh timer (1 second)."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.update_positions)
        self.refresh_timer.start(1000)  # 1 second

    def set_trading_engine(self, engine):
        """Set the trading engine to fetch positions from."""
        self._trading_engine = engine
        logger.info(f"Trading engine set: {type(engine).__name__}")

    @Slot()
    def update_positions(self):
        """Update positions from trading engine."""
        if not self._trading_engine:
            return

        try:
            # Get positions from trading engine
            positions = self._trading_engine.get_open_positions()

            # Update internal cache
            self._positions = {pos['position_id']: pos for pos in positions}

            # Update table
            self._refresh_table()

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    def _refresh_table(self):
        """Refresh table with current positions."""
        # Block signals during update
        self.table.setUpdatesEnabled(False)

        # Clear and resize
        self.table.setRowCount(len(self._positions))

        # Populate rows
        for row, (pos_id, pos) in enumerate(self._positions.items()):
            self._populate_row(row, pos)

        # Re-enable updates
        self.table.setUpdatesEnabled(True)

    def _populate_row(self, row: int, pos: Dict[str, Any]):
        """Populate a single row with position data."""
        # Symbol
        symbol_item = QTableWidgetItem(pos.get('symbol', ''))
        self.table.setItem(row, 0, symbol_item)

        # Direction (▲ for long, ▼ for short)
        direction = pos.get('direction', '').upper()
        direction_symbol = "▲" if direction == "LONG" else "▼"
        direction_item = QTableWidgetItem(direction_symbol)
        direction_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 1, direction_item)

        # Size
        size = pos.get('size', 0.0)
        size_item = QTableWidgetItem(f"{size:.2f}")
        size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 2, size_item)

        # Entry Price
        entry = pos.get('entry_price', 0.0)
        entry_item = QTableWidgetItem(f"{entry:.5f}")
        entry_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 3, entry_item)

        # Current Price
        current = pos.get('current_price', 0.0)
        current_item = QTableWidgetItem(f"{current:.5f}")
        current_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 4, current_item)

        # P&L ($)
        pnl = pos.get('unrealized_pnl', 0.0)
        pnl_item = QTableWidgetItem(f"${pnl:+.2f}")
        pnl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 5, pnl_item)

        # P&L (%)
        pnl_pct = pos.get('unrealized_pnl_pct', 0.0)
        pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
        pnl_pct_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 6, pnl_pct_item)

        # Color code P&L cells
        color = self._get_pnl_color(pnl_pct)
        pnl_item.setBackground(QBrush(color))
        pnl_pct_item.setBackground(QBrush(color))

        # Stop Loss
        sl = pos.get('stop_loss', 0.0)
        sl_item = QTableWidgetItem(f"{sl:.5f}" if sl > 0 else "-")
        sl_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 7, sl_item)

        # Take Profit
        tp = pos.get('take_profit', 0.0)
        tp_item = QTableWidgetItem(f"{tp:.5f}" if tp > 0 else "-")
        tp_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 8, tp_item)

        # Duration
        entry_time = pos.get('entry_time')
        if entry_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            duration = datetime.now() - entry_time
            duration_str = self._format_duration(duration.total_seconds())
        else:
            duration_str = "-"
        duration_item = QTableWidgetItem(duration_str)
        duration_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 9, duration_item)

        # R:R (Risk/Reward ratio)
        rr = self._calculate_rr(pos)
        rr_item = QTableWidgetItem(f"1:{rr:.1f}" if rr > 0 else "-")
        rr_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 10, rr_item)

        # Trailing Stop
        trailing = pos.get('trailing_stop_active', False)
        trailing_item = QTableWidgetItem("✓" if trailing else "✗")
        trailing_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 11, trailing_item)

    def _get_pnl_color(self, pnl_pct: float) -> QColor:
        """Get color based on P&L percentage."""
        if pnl_pct > 1.0:
            return QColor(144, 238, 144)  # Light green (profit >1%)
        elif pnl_pct > 0:
            return QColor(200, 255, 200)  # Very light green (small profit)
        elif pnl_pct > -1.0:
            return QColor(255, 200, 200)  # Very light red (small loss)
        else:
            return QColor(255, 160, 160)  # Light red (loss <-1%)

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(seconds / 86400)
            hours = int((seconds % 86400) / 3600)
            return f"{days}d {hours}h"

    def _calculate_rr(self, pos: Dict[str, Any]) -> float:
        """Calculate Risk/Reward ratio."""
        entry = pos.get('entry_price', 0.0)
        sl = pos.get('stop_loss', 0.0)
        tp = pos.get('take_profit', 0.0)

        if entry == 0 or sl == 0 or tp == 0:
            return 0.0

        direction = pos.get('direction', '').upper()

        if direction == "LONG":
            risk = entry - sl
            reward = tp - entry
        else:  # SHORT
            risk = sl - entry
            reward = entry - tp

        if risk <= 0:
            return 0.0

        return reward / risk

    @Slot()
    def _on_row_double_clicked(self):
        """Handle row double-click - highlight on chart."""
        current_row = self.table.currentRow()
        if current_row < 0:
            return

        # Get position from row
        pos_id = list(self._positions.keys())[current_row]
        position = self._positions[pos_id]

        # Emit signal to highlight on chart
        self.position_selected.emit(position)

        logger.info(f"Position selected for chart highlight: {pos_id}")

    @Slot(object)
    def _show_context_menu(self, pos):
        """Show context menu on right-click."""
        current_row = self.table.currentRow()
        if current_row < 0:
            return

        # Get position
        pos_id = list(self._positions.keys())[current_row]
        position = self._positions[pos_id]

        # Create menu
        menu = QMenu(self)

        close_action = menu.addAction("Close Position")
        menu.addSeparator()
        modify_sl_action = menu.addAction("Modify Stop Loss")
        modify_tp_action = menu.addAction("Modify Take Profit")
        menu.addSeparator()
        add_action = menu.addAction("Add to Position")
        view_action = menu.addAction("View Details")

        # Execute menu
        action = menu.exec(QCursor.pos())

        if action == close_action:
            self._close_position(pos_id, position)
        elif action == modify_sl_action:
            self._modify_stop_loss(pos_id, position)
        elif action == modify_tp_action:
            self._modify_take_profit(pos_id, position)
        elif action == add_action:
            self._add_to_position(pos_id, position)
        elif action == view_action:
            self._view_details(pos_id, position)

    def _close_position(self, pos_id: str, position: Dict[str, Any]):
        """Close position with confirmation."""
        reply = QMessageBox.question(
            self,
            "Close Position",
            f"Close position {position['symbol']} {position['direction']}?\n"
            f"Current P&L: ${position.get('unrealized_pnl', 0):.2f}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.close_position_requested.emit(pos_id)
            logger.info(f"Close position requested: {pos_id}")

    def _modify_stop_loss(self, pos_id: str, position: Dict[str, Any]):
        """Modify stop loss (placeholder)."""
        from PySide6.QtWidgets import QInputDialog

        current_sl = position.get('stop_loss', 0.0)

        new_sl, ok = QInputDialog.getDouble(
            self,
            "Modify Stop Loss",
            f"Current SL: {current_sl:.5f}\nEnter new Stop Loss:",
            current_sl,
            0.0,
            999999.0,
            5
        )

        if ok:
            self.modify_sl_requested.emit(pos_id, new_sl)
            logger.info(f"Modify SL requested: {pos_id} -> {new_sl}")

    def _modify_take_profit(self, pos_id: str, position: Dict[str, Any]):
        """Modify take profit (placeholder)."""
        from PySide6.QtWidgets import QInputDialog

        current_tp = position.get('take_profit', 0.0)

        new_tp, ok = QInputDialog.getDouble(
            self,
            "Modify Take Profit",
            f"Current TP: {current_tp:.5f}\nEnter new Take Profit:",
            current_tp,
            0.0,
            999999.0,
            5
        )

        if ok:
            self.modify_tp_requested.emit(pos_id, new_tp)
            logger.info(f"Modify TP requested: {pos_id} -> {new_tp}")

    def _add_to_position(self, pos_id: str, position: Dict[str, Any]):
        """Add to existing position (placeholder)."""
        QMessageBox.information(
            self,
            "Add to Position",
            f"Add to position feature coming soon.\n"
            f"Position: {position['symbol']} {position['direction']}"
        )

    def _view_details(self, pos_id: str, position: Dict[str, Any]):
        """View position details (placeholder)."""
        details = "\n".join([f"{k}: {v}" for k, v in position.items()])
        QMessageBox.information(
            self,
            "Position Details",
            details
        )

    def clear_positions(self):
        """Clear all positions from table."""
        self._positions.clear()
        self.table.setRowCount(0)

    def closeEvent(self, event):
        """Stop refresh timer on close."""
        self.refresh_timer.stop()
        super().closeEvent(event)
