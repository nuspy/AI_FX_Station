"""
Dialog for configuring prediction settings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog,
    QDialogButtonBox, QSpinBox, QCheckBox, QHBoxLayout, QLabel
)
from loguru import logger

CONFIG_FILE = Path(__file__).resolve().parents[3] / "configs" / "prediction_settings.json"

class PredictionSettingsDialog(QDialog):
    """
    A dialog window for setting prediction parameters (basic + advanced),
    which are persisted to a JSON file.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prediction Settings")
        self.setMinimumWidth(480)

        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()

        # Model Path
        self.model_path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_model_path)
        model_h = QHBoxLayout()
        model_h.addWidget(self.model_path_edit)
        model_h.addWidget(self.browse_button)
        self.form_layout.addRow("Model Path:", model_h)

        # Horizons
        self.horizons_edit = QLineEdit("1m, 5m, 15m")
        self.form_layout.addRow("Horizons (comma-separated):", self.horizons_edit)

        # N_samples
        self.n_samples_spinbox = QSpinBox()
        self.n_samples_spinbox.setRange(1, 10000)
        self.n_samples_spinbox.setValue(200)
        self.form_layout.addRow("Number of Samples (N_samples):", self.n_samples_spinbox)

        # Conformal Calibration
        self.conformal_checkbox = QCheckBox("Apply Conformal Calibration")
        self.conformal_checkbox.setChecked(True)
        self.form_layout.addRow(self.conformal_checkbox)

        # --- Basic indicators (exposed to basic dialog) ---
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(1, 500)
        self.warmup_spin.setValue(16)
        self.form_layout.addRow("Warmup Bars:", self.warmup_spin)

        self.atr_n_spin = QSpinBox()
        self.atr_n_spin.setRange(1, 200)
        self.atr_n_spin.setValue(14)
        self.form_layout.addRow("ATR n:", self.atr_n_spin)

        self.rsi_n_spin = QSpinBox()
        self.rsi_n_spin.setRange(2, 200)
        self.rsi_n_spin.setValue(14)
        self.form_layout.addRow("RSI n:", self.rsi_n_spin)

        self.bb_n_spin = QSpinBox()
        self.bb_n_spin.setRange(2, 200)
        self.bb_n_spin.setValue(20)
        self.form_layout.addRow("Bollinger window n:", self.bb_n_spin)

        self.rv_window_spin = QSpinBox()
        self.rv_window_spin.setRange(1, 1000)
        self.rv_window_spin.setValue(60)
        self.form_layout.addRow("RV window:", self.rv_window_spin)

        # --- Advanced indicators ---
        self.ema_fast_spin = QSpinBox()
        self.ema_fast_spin.setRange(1, 200)
        self.ema_fast_spin.setValue(12)
        self.form_layout.addRow("EMA fast span:", self.ema_fast_spin)

        self.ema_slow_spin = QSpinBox()
        self.ema_slow_spin.setRange(1, 400)
        self.ema_slow_spin.setValue(26)
        self.form_layout.addRow("EMA slow span:", self.ema_slow_spin)

        self.don_n_spin = QSpinBox()
        self.don_n_spin.setRange(1, 400)
        self.don_n_spin.setValue(20)
        self.form_layout.addRow("Donchian n:", self.don_n_spin)

        self.hurst_window_spin = QSpinBox()
        self.hurst_window_spin.setRange(1, 1024)
        self.hurst_window_spin.setValue(64)
        self.form_layout.addRow("Hurst window:", self.hurst_window_spin)

        self.keltner_k_spin = QSpinBox()
        self.keltner_k_spin.setRange(1, 10)
        self.keltner_k_spin.setValue(1)
        self.form_layout.addRow("Keltner multiplier (k):", self.keltner_k_spin)

        # max forecasts
        self.max_forecasts_spin = QSpinBox()
        self.max_forecasts_spin.setRange(1, 100)
        self.max_forecasts_spin.setValue(5)
        self.form_layout.addRow("Max forecasts to display:", self.max_forecasts_spin)

        # Test history bars (for TestingPoint forecasts)
        self.test_history_spin = QSpinBox()
        self.test_history_spin.setRange(1, 10000)
        self.test_history_spin.setValue(128)
        self.form_layout.addRow("Test history bars (N):", self.test_history_spin)

        # auto predict options
        auto_h = QHBoxLayout()
        self.auto_checkbox = QCheckBox("Auto predict")
        self.auto_interval_spin = QSpinBox()
        self.auto_interval_spin.setRange(1, 86400)
        self.auto_interval_spin.setValue(60)
        auto_h.addWidget(self.auto_checkbox)
        auto_h.addWidget(QLabel("Interval (s):"))
        auto_h.addWidget(self.auto_interval_spin)
        self.form_layout.addRow(auto_h)

        self.layout.addLayout(self.form_layout)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self.load_settings()

    def _browse_model_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def load_settings(self):
        """Loads settings from the JSON config file."""
        if not CONFIG_FILE.exists():
            logger.debug(f"Prediction settings file not found: {CONFIG_FILE}")
            return
        try:
            with open(CONFIG_FILE, "r") as f:
                settings = json.load(f)
            self.model_path_edit.setText(settings.get("model_path", ""))
            self.horizons_edit.setText(", ".join(settings.get("horizons", ["1m", "5m", "15m"])))
            self.n_samples_spinbox.setValue(settings.get("N_samples", 200))
            self.conformal_checkbox.setChecked(settings.get("apply_conformal", True))

            self.warmup_spin.setValue(int(settings.get("warmup_bars", 16)))
            self.atr_n_spin.setValue(int(settings.get("atr_n", 14)))
            self.rsi_n_spin.setValue(int(settings.get("rsi_n", 14)))
            self.bb_n_spin.setValue(int(settings.get("bb_n", 20)))
            self.rv_window_spin.setValue(int(settings.get("rv_window", 60)))

            self.ema_fast_spin.setValue(int(settings.get("ema_fast", 12)))
            self.ema_slow_spin.setValue(int(settings.get("ema_slow", 26)))
            self.don_n_spin.setValue(int(settings.get("don_n", 20)))
            self.hurst_window_spin.setValue(int(settings.get("hurst_window", 64)))
            self.keltner_k_spin.setValue(int(settings.get("keltner_k", 1)))

            self.max_forecasts_spin.setValue(int(settings.get("max_forecasts", 5)))
            self.test_history_spin.setValue(int(settings.get("test_history_bars", 128)))
            self.auto_checkbox.setChecked(bool(settings.get("auto_predict", False)))
            self.auto_interval_spin.setValue(int(settings.get("auto_interval_seconds", 60)))

            logger.info(f"Loaded prediction settings from {CONFIG_FILE}")
        except Exception as e:
            logger.exception(f"Failed to load prediction settings: {e}")

    def save_settings(self):
        """Saves the current settings to the JSON config file."""
        settings = {
            "model_path": self.model_path_edit.text(),
            "horizons": [h.strip() for h in self.horizons_edit.text().split(",") if h.strip()],
            "N_samples": self.n_samples_spinbox.value(),
            "apply_conformal": self.conformal_checkbox.isChecked(),

            "warmup_bars": int(self.warmup_spin.value()),
            "atr_n": int(self.atr_n_spin.value()),
            "rsi_n": int(self.rsi_n_spin.value()),
            "bb_n": int(self.bb_n_spin.value()),
            "rv_window": int(self.rv_window_spin.value()),

            "ema_fast": int(self.ema_fast_spin.value()),
            "ema_slow": int(self.ema_slow_spin.value()),
            "don_n": int(self.don_n_spin.value()),
            "hurst_window": int(self.hurst_window_spin.value()),
            "keltner_k": int(self.keltner_k_spin.value()),

            "max_forecasts": int(self.max_forecasts_spin.value()),
            "test_history_bars": int(self.test_history_spin.value()),
            "auto_predict": bool(self.auto_checkbox.isChecked()),
            "auto_interval_seconds": int(self.auto_interval_spin.value()),
        }
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(settings, f, indent=4)
            logger.info(f"Saved prediction settings to {CONFIG_FILE}")
        except Exception as e:
            logger.exception(f"Failed to save prediction settings: {e}")

    def accept(self):
        """Saves settings and closes the dialog."""
        self.save_settings()
        super().accept()

    @staticmethod
    def get_settings() -> Dict[str, Any]:
        """Static method to retrieve the latest saved settings."""
        if not CONFIG_FILE.exists():
            return {}
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
