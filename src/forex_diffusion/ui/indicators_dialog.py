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
    def __init__(self, parent: QWidget | None = None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Indicatori")
        self.layout = QVBoxLayout(self)

        # indicator controls
        self.controls = {}

        # helper palette
        self._palette = [("blue","#1f77b4"),("red","#d62728"),("green","#2ca02c"),("orange","#ff7f0e"),("purple","#9467bd"),("black","#000000")]

        # SMA
        h = QHBoxLayout()
        self.sma_cb = QCheckBox("SMA")
        self.sma_span = QSpinBox()
        self.sma_span.setRange(1, 500)
        self.sma_span.setValue(20)
        h.addWidget(self.sma_cb)
        h.addWidget(QLabel("window:"))
        h.addWidget(self.sma_span)
        # color selector for SMA
        self.sma_color = QComboBox()
        for name, hexv in self._palette:
            self.sma_color.addItem(f"{name} ({hexv})", hexv)
        h.addWidget(QLabel("color:"))
        h.addWidget(self.sma_color)
        self.layout.addLayout(h)
        self.controls["sma"] = (self.sma_cb, {"window": self.sma_span, "color": self.sma_color})

        # EMA
        h = QHBoxLayout()
        self.ema_cb = QCheckBox("EMA")
        self.ema_span = QSpinBox()
        self.ema_span.setRange(1, 500)
        self.ema_span.setValue(20)
        self.ema_color = QComboBox()
        for name, hexv in self._palette:
            self.ema_color.addItem(f"{name} ({hexv})", hexv)
        h.addWidget(self.ema_cb)
        h.addWidget(QLabel("span:"))
        h.addWidget(self.ema_span)
        h.addWidget(QLabel("color:"))
        h.addWidget(self.ema_color)
        self.layout.addLayout(h)
        self.controls["ema"] = (self.ema_cb, {"span": self.ema_span, "color": self.ema_color})

        # Bollinger
        h = QHBoxLayout()
        self.bb_cb = QCheckBox("Bollinger")
        self.bb_window = QSpinBox()
        self.bb_window.setRange(1, 500)
        self.bb_window.setValue(20)
        self.bb_std = QLineEdit("2.0")
        self.bb_color = QComboBox()
        for name, hexv in self._palette:
            self.bb_color.addItem(f"{name} ({hexv})", hexv)
        h.addWidget(self.bb_cb)
        h.addWidget(QLabel("window:"))
        h.addWidget(self.bb_window)
        h.addWidget(QLabel("n_std:"))
        h.addWidget(self.bb_std)
        h.addWidget(QLabel("color:"))
        h.addWidget(self.bb_color)
        self.layout.addLayout(h)
        self.controls["bollinger"] = (self.bb_cb, {"window": self.bb_window, "n_std": self.bb_std, "color": self.bb_color})

        # RSI
        h = QHBoxLayout()
        self.rsi_cb = QCheckBox("RSI")
        self.rsi_period = QSpinBox()
        self.rsi_period.setRange(1, 500)
        self.rsi_period.setValue(14)
        self.rsi_color = QComboBox()
        for name, hexv in self._palette:
            self.rsi_color.addItem(f"{name} ({hexv})", hexv)
        h.addWidget(self.rsi_cb)
        h.addWidget(QLabel("period:"))
        h.addWidget(self.rsi_period)
        h.addWidget(QLabel("color:"))
        h.addWidget(self.rsi_color)
        self.layout.addLayout(h)
        self.controls["rsi"] = (self.rsi_cb, {"period": self.rsi_period, "color": self.rsi_color})

        # MACD
        h = QHBoxLayout()
        self.macd_cb = QCheckBox("MACD")
        self.macd_fast = QSpinBox(); self.macd_fast.setRange(1, 200); self.macd_fast.setValue(12)
        self.macd_slow = QSpinBox(); self.macd_slow.setRange(1, 500); self.macd_slow.setValue(26)
        self.macd_signal = QSpinBox(); self.macd_signal.setRange(1, 200); self.macd_signal.setValue(9)
        self.macd_color = QComboBox()
        for name, hexv in self._palette:
            self.macd_color.addItem(f"{name} ({hexv})", hexv)
        h.addWidget(self.macd_cb)
        h.addWidget(QLabel("fast:")); h.addWidget(self.macd_fast)
        h.addWidget(QLabel("slow:")); h.addWidget(self.macd_slow)
        h.addWidget(QLabel("signal:")); h.addWidget(self.macd_signal)
        h.addWidget(QLabel("color:")); h.addWidget(self.macd_color)
        self.layout.addLayout(h)
        self.controls["macd"] = (self.macd_cb, {"fast": self.macd_fast, "slow": self.macd_slow, "signal": self.macd_signal, "color": self.macd_color})

        # Buttons
        btn_h = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Annulla")
        btn_h.addWidget(self.ok_btn)
        btn_h.addWidget(self.cancel_btn)
        self.layout.addLayout(btn_h)

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        # if initial config provided, prepopulate controls
        if initial:
            try:
                for name, params in initial.items():
                    if name not in self.controls:
                        continue
                    cb, widget_map = self.controls[name]
                    try:
                        cb.setChecked(True)
                    except Exception:
                        pass
                    for k, w in widget_map.items():
                        if k not in params:
                            continue
                        val = params.get(k)
                        try:
                            if isinstance(w, QSpinBox):
                                w.setValue(int(val))
                            elif isinstance(w, QLineEdit):
                                w.setText(str(val))
                            elif isinstance(w, QComboBox):
                                # try to set by data (hex) first then by text
                                found = False
                                for idx in range(w.count()):
                                    if w.itemData(idx) == val or w.itemText(idx).startswith(str(val)):
                                        w.setCurrentIndex(idx)
                                        found = True
                                        break
                                if not found:
                                    # append custom color text
                                    w.addItem(str(val), val)
                                    w.setCurrentIndex(w.count()-1)
                            else:
                                # fallback ignore
                                pass
                        except Exception:
                            pass
            except Exception:
                pass
</llm-patch>

<llm-patch path="/src/forex_diffusion/ui/chart_tab.py" matcher="BeforeAfter">
Persist indicator config to configs/indicators.json and auto-load/apply on init
<!--Separator-->
Before:

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
                        # SpinBox numeric
                        if isinstance(widget, QSpinBox):
                            pvals[k] = int(widget.value())
                        # LineEdit numeric
                        elif isinstance(widget, QLineEdit):
                            pvals[k] = float(widget.text())
                        # ComboBox for colors / other
                        elif isinstance(widget, QComboBox):
                            # pick user data if present (hex), else text
                            data = widget.currentData()
                            pvals[k] = data if data is not None else widget.currentText()
                        else:
                            # generic (try value/text)
                            pvals[k] = widget.value() if hasattr(widget, "value") else widget.text()
                    except Exception:
                        pvals[k] = None
                out[name] = pvals
            except Exception:
                continue
        return out
