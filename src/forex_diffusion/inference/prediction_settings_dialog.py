from __future__ import annotations

from typing import Dict, Any

from PySide6.QtWidgets import QDialog


class PredictionSettings:
    """Placeholder apply interface for Basic/Advanced presets."""
    def __init__(self):
        self.basic: Dict[str, Any] = {}
        self.advanced: Dict[str, Any] = {}

    def apply_basic(self, cfg: Dict[str, Any]) -> None:
        self.basic.update(cfg)

    def apply_advanced(self, cfg: Dict[str, Any]) -> None:
        self.advanced.update(cfg)



