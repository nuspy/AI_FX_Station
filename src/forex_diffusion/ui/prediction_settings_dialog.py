"""
Dialog for configuring prediction settings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog,
    QDialogButtonBox, QSpinBox, QCheckBox
)
from loguru import logger

CONFIG_FILE = Path(__file__).resolve().parents[3] / "configs" / "prediction_settings.json"

class PredictionSettingsDialog(QDialog):
    """
    A dialog window for setting prediction parameters, which are persisted to a JSON file.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prediction Settings")
        self.setMinimumWidth(400)

        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()

        # Model Path
        self.model_path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_model_path)
        model_layout = QVBoxLayout()
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.browse_button)
        self.form_layout.addRow("Model Path:", model_layout)

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
