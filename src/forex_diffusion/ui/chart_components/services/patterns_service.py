# src/forex_diffusion/ui/chart_components/services/patterns_service.py
from __future__ import annotations

from typing import Optional
from PySide6.QtCore import QObject, Signal, QTimer

class PatternsService(QObject):
    """
    Safe QObject-based service used by the Chart tab.
    Emits `events_ready` with a list of events. For now, we emit an **empty list**
    to avoid schema mismatches with PatternEvent(**e) downstream.
    """
    events_ready = Signal(list)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        if parent is not None and not isinstance(parent, QObject):
            parent = None
        super().__init__(parent)
        self._chart_enabled: bool = True
        self._candle_enabled: bool = True
        self._history_enabled: bool = False

    # ---- toggles ----
    def set_chart_enabled(self, enabled: bool) -> None:
        self._chart_enabled = bool(enabled)

    def set_candle_enabled(self, enabled: bool) -> None:
        self._candle_enabled = bool(enabled)

    def set_history_enabled(self, enabled: bool) -> None:
        self._history_enabled = bool(enabled)

    def chart_enabled(self) -> bool:
        return self._chart_enabled

    def candle_enabled(self) -> bool:
        return self._candle_enabled

    def history_enabled(self) -> bool:
        return self._history_enabled

    # ---- detection ----
    def detect_async(self, df) -> None:
        """Non-blocking no-op detector: always emits an empty list.
        This guarantees compatibility with `PatternEvent(**e)` parsing downstream.
        """
        QTimer.singleShot(0, lambda: self.events_ready.emit([]))
