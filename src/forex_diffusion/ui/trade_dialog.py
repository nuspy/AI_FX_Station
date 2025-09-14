# src/forex_diffusion/ui/trade_dialog.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QDoubleSpinBox, QLineEdit, QPushButton, QDialogButtonBox
)
from PySide6.QtCore import Qt

LOT_SIZE = 100_000  # standard FX lot size

class TradeDialog(QDialog):
    """Order ticket (Market/Limit/Stop/StopLimit) con Units/Lots, SL/TP e TIF."""
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
        for s in (symbols or []):
            self.symbol_combo.addItem(s)
        if current_symbol:
            idx = self.symbol_combo.findText(current_symbol)
            if idx >= 0:
                self.symbol_combo.setCurrentIndex(idx)
        grid.addWidget(self.symbol_combo, r, 1, 1, 3); r += 1

        # Quantity mode
        grid.addWidget(QLabel("Qty Mode:"), r, 0)
        self.qty_mode = QComboBox(); self.qty_mode.addItems(["Units","Lots"])
        grid.addWidget(self.qty_mode, r, 1)

        # Volume
        grid.addWidget(QLabel("Volume:"), r, 2)
        self.vol = QDoubleSpinBox(); self.vol.setDecimals(2); self.vol.setRange(0.01, 1e12); self.vol.setValue(LOT_SIZE)
        grid.addWidget(self.vol, r, 3); r += 1

        # Type
        grid.addWidget(QLabel("Type:"), r, 0)
        self.type_combo = QComboBox(); self.type_combo.addItems(["Market","Limit","Stop","StopLimit"])
        grid.addWidget(self.type_combo, r, 1)

        # Price (per ordini pending)
        grid.addWidget(QLabel("Price:"), r, 2)
        self.price = QDoubleSpinBox(); self.price.setDecimals(5); self.price.setRange(0.0, 1e9)
        grid.addWidget(self.price, r, 3); r += 1

        # SL / TP
        grid.addWidget(QLabel("Stop Loss:"), r, 0)
        self.sl = QDoubleSpinBox(); self.sl.setDecimals(5); self.sl.setRange(0.0, 1e9)
        grid.addWidget(self.sl, r, 1)

        grid.addWidget(QLabel("Take Profit:"), r, 2)
        self.tp = QDoubleSpinBox(); self.tp.setDecimals(5); self.tp.setRange(0.0, 1e9)
        grid.addWidget(self.tp, r, 3); r += 1

        # TIF + Comment
        grid.addWidget(QLabel("TIF:"), r, 0)
        self.tif = QComboBox(); self.tif.addItems(["GTC","Day"])
        grid.addWidget(self.tif, r, 1)

        grid.addWidget(QLabel("Comment:"), r, 2)
        self.comment = QLineEdit()
        grid.addWidget(self.comment, r, 3); r += 1

        lay.addLayout(grid)

        # Actions
        btnrow = QHBoxLayout()
        self.btn_sell = QPushButton("Sell")
        self.btn_buy = QPushButton("Buy")
        btnrow.addWidget(self.btn_sell); btnrow.addWidget(self.btn_buy)
        lay.addLayout(btnrow)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        lay.addWidget(self.buttons)

        # Events
        self.buttons.rejected.connect(self.reject)
        self.btn_sell.clicked.connect(lambda: self._finish("SELL"))
        self.btn_buy.clicked.connect(lambda: self._finish("BUY"))
        self.qty_mode.currentTextChanged.connect(self._sync_qty_mode)

        self._order: Optional[Dict[str, Any]] = None
        self._sync_qty_mode(self.qty_mode.currentText())

    def _sync_qty_mode(self, mode: str):
        if mode == "Lots":
            try:
                units = float(self.vol.value())
            except Exception:
                units = LOT_SIZE
            lots = max(0.01, units / LOT_SIZE)
            self.vol.setValue(lots)
            self.vol.setSingleStep(0.01)
        else:
            try:
                lots = float(self.vol.value())
            except Exception:
                lots = 1.0
            units = max(1.0, lots * LOT_SIZE)
            self.vol.setValue(units)
            self.vol.setSingleStep(1000.0)

    def _finish(self, side: str):
        mode = self.qty_mode.currentText()
        vol_val = float(self.vol.value())
        volume_units = int(vol_val * LOT_SIZE) if mode == "Lots" else int(vol_val)
        self._order = {
            "symbol": self.symbol_combo.currentText(),
            "side": side,
            "type": self.type_combo.currentText().upper(),
            "volume": volume_units,
            "price": float(self.price.value()) if self.type_combo.currentText() in ("Limit","Stop","StopLimit") else None,
            "sl": float(self.sl.value()) if self.sl.value() > 0 else None,
            "tp": float(self.tp.value()) if self.tp.value() > 0 else None,
            "tif": self.tif.currentText(),
            "comment": self.comment.text().strip(),
        }
        self.accept()

    def get_order(self) -> Dict[str, Any]:
        return self._order or {}
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

