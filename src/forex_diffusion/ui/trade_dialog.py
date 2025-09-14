# src/forex_diffusion/ui/trade_dialog.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QDoubleSpinBox, QLineEdit, QPushButton, QDialogButtonBox, QWidget, QGridLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class TradeDialog(QDialog):
    """Order ticket dialog (Market/Limit) similar to MT UI."""
    def __init__(self, parent=None, symbols: Optional[List[str]] = None, current_symbol: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Order")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setStyleSheet("""
            QDialog { background-color: #0f1115; color: #e0e0e0; }
            QLineEdit, QDoubleSpinBox, QComboBox { background: #161a21; color: #dfe7f1; border: 1px solid #2a2f3a; padding: 4px; }
            QPushButton { background: #1c1f26; color: #e0e0e0; border: 1px solid #2a2f3a; padding: 6px 10px; border-radius: 4px; }
            QPushButton:hover { background: #242a35; }
            QLabel { color: #cfd6e1; }
        """)
        lay = QVBoxLayout(self)

        grid = QGridLayout()
        r = 0

        grid.addWidget(QLabel("Symbol:"), r, 0)
        self.symbol_combo = QComboBox()
        for s in (symbols or []):
            self.symbol_combo.addItem(s)
        if current_symbol:
            idx = self.symbol_combo.findText(current_symbol)
            if idx >= 0: self.symbol_combo.setCurrentIndex(idx)
        grid.addWidget(self.symbol_combo, r, 1, 1, 3); r += 1

        grid.addWidget(QLabel("Volume:"), r, 0)
        self.vol_spin = QDoubleSpinBox(); self.vol_spin.setDecimals(2); self.vol_spin.setRange(0.01, 1000.0); self.vol_spin.setValue(1.00)
        grid.addWidget(self.vol_spin, r, 1)

        grid.addWidget(QLabel("Type:"), r, 2)
        self.type_combo = QComboBox(); self.type_combo.addItems(["Market", "Limit"])
        grid.addWidget(self.type_combo, r, 3); r += 1

        grid.addWidget(QLabel("Price:"), r, 0)
        self.price_spin = QDoubleSpinBox(); self.price_spin.setDecimals(5); self.price_spin.setRange(0.0, 1e9); self.price_spin.setValue(0.0)
        grid.addWidget(self.price_spin, r, 1)

        grid.addWidget(QLabel("Stop Loss:"), r, 2)
        self.sl_spin = QDoubleSpinBox(); self.sl_spin.setDecimals(5); self.sl_spin.setRange(0.0, 1e9); self.sl_spin.setValue(0.0)
        grid.addWidget(self.sl_spin, r, 3); r += 1

        grid.addWidget(QLabel("Take Profit:"), r, 0)
        self.tp_spin = QDoubleSpinBox(); self.tp_spin.setDecimals(5); self.tp_spin.setRange(0.0, 1e9); self.tp_spin.setValue(0.0)
        grid.addWidget(self.tp_spin, r, 1)

        grid.addWidget(QLabel("Comment:"), r, 2)
        self.comment_edit = QLineEdit()
        grid.addWidget(self.comment_edit, r, 3); r += 1

        lay.addLayout(grid)
# src/forex_diffusion/ui/trade_dialog.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QDoubleSpinBox, QLineEdit, QPushButton, QDialogButtonBox, QWidget, QGridLayout
)
from PySide6.QtCore import Qt

LOT_SIZE = 100_000  # standard FX lot

class TradeDialog(QDialog):
    """Order ticket dialog: Market/Limit/Stop/StopLimit; Lots/Units; SL/TP; TIF."""
    def __init__(self, parent=None, symbols: Optional[List[str]] = None, current_symbol: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Order")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setStyleSheet("""
            QDialog { background-color: #0f1115; color: #e0e0e0; }
            QLineEdit, QDoubleSpinBox, QComboBox { background: #161a21; color: #dfe7f1; border: 1px solid #2a2f3a; padding: 4px; }
            QPushButton { background: #1c1f26; color: #e0e0e0; border: 1px solid #2a2f3a; padding: 6px 10px; border-radius: 4px; }
            QPushButton:hover { background: #242a35; }
            QLabel { color: #cfd6e1; }
        """)

        lay = QVBoxLayout(self)
        grid = QGridLayout(); r = 0

        # Symbol
        grid.addWidget(QLabel("Symbol:"), r, 0)
        self.symbol_combo = QComboBox()
        for s in (symbols or []): self.symbol_combo.addItem(s)
        if current_symbol:
            idx = self.symbol_combo.findText(current_symbol)
            if idx >= 0: self.symbol_combo.setCurrentIndex(idx)
        grid.addWidget(self.symbol_combo, r, 1, 1, 3); r += 1

        # Quantity mode: Lots / Units
        grid.addWidget(QLabel("Qty Mode:"), r, 0)
        self.qty_mode_combo = QComboBox(); self.qty_mode_combo.addItems(["Units","Lots"])
        grid.addWidget(self.qty_mode_combo, r, 1)

        # Volume and Type
        grid.addWidget(QLabel("Volume:"), r, 2)
        self.vol_spin = QDoubleSpinBox(); self.vol_spin.setDecimals(2); self.vol_spin.setRange(0.01, 1e9); self.vol_spin.setValue(LOT_SIZE)  # default units
        grid.addWidget(self.vol_spin, r, 3); r += 1

        grid.addWidget(QLabel("Type:"), r, 0)
        self.type_combo = QComboBox(); self.type_combo.addItems(["Market","Limit","Stop","StopLimit"])
        grid.addWidget(self.type_combo, r, 1)

        grid.addWidget(QLabel("Price:"), r, 2)
        self.price_spin = QDoubleSpinBox(); self.price_spin.setDecimals(5); self.price_spin.setRange(0.0, 1e9); self.price_spin.setValue(0.0)
        grid.addWidget(self.price_spin, r, 3); r += 1

        # SL/TP
        grid.addWidget(QLabel("Stop Loss:"), r, 0)
        self.sl_spin = QDoubleSpinBox(); self.sl_spin.setDecimals(5); self.sl_spin.setRange(0.0, 1e9); self.sl_spin.setValue(0.0)
        grid.addWidget(self.sl_spin, r, 1)

        grid.addWidget(QLabel("Take Profit:"), r, 2)
        self.tp_spin = QDoubleSpinBox(); self.tp_spin.setDecimals(5); self.tp_spin.setRange(0.0, 1e9); self.tp_spin.setValue(0.0)
        grid.addWidget(self.tp_spin, r, 3); r += 1

        # TIF + Comment
        grid.addWidget(QLabel("TIF:"), r, 0)
        self.tif_combo = QComboBox(); self.tif_combo.addItems(["GTC","Day"])
        grid.addWidget(self.tif_combo, r, 1)

        grid.addWidget(QLabel("Comment:"), r, 2)
        self.comment_edit = QLineEdit()
        grid.addWidget(self.comment_edit, r, 3); r += 1

        lay.addLayout(grid)

        # Action buttons
        btn_row = QHBoxLayout()
        self.sell_btn = QPushButton("Sell")
        self.buy_btn = QPushButton("Buy")
        btn_row.addWidget(self.sell_btn); btn_row.addWidget(self.buy_btn)
        lay.addLayout(btn_row)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        lay.addWidget(self.buttons)

        self.sell_btn.clicked.connect(lambda: self._finish("SELL"))
        self.buy_btn.clicked.connect(lambda: self._finish("BUY"))
        self.buttons.rejected.connect(self.reject)

        self._order: Optional[Dict[str, Any]] = None

        # Sync qty mode defaults
        self.qty_mode_combo.currentTextChanged.connect(self._on_qty_mode)

    def _on_qty_mode(self, mode: str):
        if mode == "Lots":
            # convert current units to lots view
            try:
                units = float(self.vol_spin.value())
            except Exception:
                units = float(LOT_SIZE)
            lots = max(0.01, units / LOT_SIZE)
            self.vol_spin.setValue(lots)
            self.vol_spin.setSingleStep(0.01)
        else:
            # units mode
            try:
                lots = float(self.vol_spin.value())
            except Exception:
                lots = 1.0
            units = max(1.0, lots * LOT_SIZE)
            self.vol_spin.setValue(units)
            self.vol_spin.setSingleStep(1000.0)

    def _finish(self, side: str):
        mode = self.qty_mode_combo.currentText()
        vol = float(self.vol_spin.value())
        volume_units = int(vol * LOT_SIZE) if mode == "Lots" else int(vol)
        self._order = {
            "symbol": self.symbol_combo.currentText(),
            "side": side,
            "type": self.type_combo.currentText().upper(),
            "volume": volume_units,
            "price": float(self.price_spin.value()) if self.type_combo.currentText() in ("Limit","Stop","StopLimit") else None,
            "sl": float(self.sl_spin.value()) if self.sl_spin.value() > 0 else None,
            "tp": float(self.tp_spin.value()) if self.tp_spin.value() > 0 else None,
            "tif": self.tif_combo.currentText(),
            "comment": self.comment_edit.text().strip(),
        }
        self.accept()

    def get_order(self) -> Dict[str, Any]:
        return self._order or {}
        # Action buttons
        btn_row = QHBoxLayout()
        self.sell_btn = QPushButton("Sell by Market")
        self.buy_btn = QPushButton("Buy by Market")
        btn_row.addWidget(self.sell_btn); btn_row.addWidget(self.buy_btn)
        lay.addLayout(btn_row)

        # Ok/Cancel
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        lay.addWidget(self.buttons)

        # events
        self.sell_btn.clicked.connect(lambda: self._finish("SELL"))
        self.buy_btn.clicked.connect(lambda: self._finish("BUY"))
        self.buttons.rejected.connect(self.reject)

        self._order: Optional[Dict[str, Any]] = None

    def _finish(self, side: str):
        self._order = {
            "symbol": self.symbol_combo.currentText(),
            "side": side,
            "type": self.type_combo.currentText().upper(),
            "volume": float(self.vol_spin.value()),
            "price": float(self.price_spin.value()) if self.type_combo.currentText() == "Limit" else None,
            "sl": float(self.sl_spin.value()) if self.sl_spin.value() > 0 else None,
            "tp": float(self.tp_spin.value()) if self.tp_spin.value() > 0 else None,
            "comment": self.comment_edit.text().strip(),
        }
        self.accept()

    def get_order(self) -> Dict[str, Any]:
        return self._order or {}
