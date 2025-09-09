# src/forex_diffusion/ui/indicators_dialog.py
from __future__ import annotations

from typing import Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QSpinBox, QLineEdit, QPushButton, QWidget, QComboBox
)


class IndicatorsDialog(QDialog):
    """
    Dialog to choose which indicators to overlay and set simple parameters.
    Returns a dict of enabled indicators and their params on accept via .result()
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Indicatori")
        self.layout = QVBoxLayout(self)

        # indicator controls
        self.controls = {}

        # SMA
        h = QHBoxLayout()
        self.sma_cb = QCheckBox("SMA")
        self.sma_span = QSpinBox()
        self.sma_span.setRange(1, 500)
        self.sma_span.setValue(20)
        h.addWidget(self.sma_cb)
        h.addWidget(QLabel("window:"))
        h.addWidget(self.sma_span)
        self.layout.addLayout(h)
        self.controls["sma"] = (self.sma_cb, {"window": self.sma_span})

        # EMA
        h = QHBoxLayout()
        self.ema_cb = QCheckBox("EMA")
        self.ema_span = QSpinBox()
        self.ema_span.setRange(1, 500)
        self.ema_span.setValue(20)
        h.addWidget(self.ema_cb)
        h.addWidget(QLabel("span:"))
        h.addWidget(self.ema_span)
        self.layout.addLayout(h)
        self.controls["ema"] = (self.ema_cb, {"span": self.ema_span})

        # Bollinger
        h = QHBoxLayout()
        self.bb_cb = QCheckBox("Bollinger")
        self.bb_window = QSpinBox()
        self.bb_window.setRange(1, 500)
        self.bb_window.setValue(20)
        self.bb_std = QLineEdit("2.0")
        h.addWidget(self.bb_cb)
        h.addWidget(QLabel("window:"))
        h.addWidget(self.bb_window)
        h.addWidget(QLabel("n_std:"))
        h.addWidget(self.bb_std)
        self.layout.addLayout(h)
        self.controls["bollinger"] = (self.bb_cb, {"window": self.bb_window, "n_std": self.bb_std})

        # RSI
        h = QHBoxLayout()
        self.rsi_cb = QCheckBox("RSI")
        self.rsi_period = QSpinBox()
        self.rsi_period.setRange(1, 500)
        self.rsi_period.setValue(14)
        h.addWidget(self.rsi_cb)
        h.addWidget(QLabel("period:"))
        h.addWidget(self.rsi_period)
        self.layout.addLayout(h)
        self.controls["rsi"] = (self.rsi_cb, {"period": self.rsi_period})

        # MACD
        h = QHBoxLayout()
        self.macd_cb = QCheckBox("MACD")
        self.macd_fast = QSpinBox(); self.macd_fast.setRange(1, 200); self.macd_fast.setValue(12)
        self.macd_slow = QSpinBox(); self.macd_slow.setRange(1, 500); self.macd_slow.setValue(26)
        self.macd_signal = QSpinBox(); self.macd_signal.setRange(1, 200); self.macd_signal.setValue(9)
        h.addWidget(self.macd_cb)
        h.addWidget(QLabel("fast:")); h.addWidget(self.macd_fast)
        h.addWidget(QLabel("slow:")); h.addWidget(self.macd_slow)
        h.addWidget(QLabel("signal:")); h.addWidget(self.macd_signal)
        self.layout.addLayout(h)
        self.controls["macd"] = (self.macd_cb, {"fast": self.macd_fast, "slow": self.macd_slow, "signal": self.macd_signal})

        # Buttons
        btn_h = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Annulla")
        btn_h.addWidget(self.ok_btn)
        btn_h.addWidget(self.cancel_btn)
        self.layout.addLayout(btn_h)

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def result(self) -> Dict[str, Any]:
        """
        Return dict of enabled indicators and their parameter values.
        Example:
          {"sma": {"window":20}, "bollinger":{"window":20,"n_std":2.0}}
        """
        out: Dict[str, Any] = {}
        for name, (cb, params) in self.controls.items():
            try:
                if not cb.isChecked():
                    continue
                pvals = {}
                for k, widget in params.items():
                    try:
                        if isinstance(widget, QSpinBox):
                            pvals[k] = int(widget.value())
                        elif isinstance(widget, QLineEdit):
                            pvals[k] = float(widget.text())
                        else:
                            # generic
                            pvals[k] = widget.value() if hasattr(widget, "value") else widget.text()
                    except Exception:
                        pvals[k] = None
                out[name] = pvals
            except Exception:
                continue
        return out
