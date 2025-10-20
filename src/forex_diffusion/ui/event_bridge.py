# src/forex_diffusion/ui/event_bridge.py
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from loguru import logger
from typing import Any

class EventBridge(QObject):
    """
    Bridge between in-process event_bus and Qt main thread.
    """
    tickReceived = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.debug("EventBridge initialized")

    def _on_event(self, payload: Any) -> None:
        """
        Callback registered with event_bus.subscribe('tick', ...).
        This method is called from a background thread. It emits a Qt signal,
        which safely queues the payload for processing on the main UI thread.
        """
        try:
            logger.debug("EventBridge received event, emitting tickReceived signal...")
            self.tickReceived.emit(payload)
        except Exception as e:
            logger.exception(f"EventBridge failed to emit tick: {e}")
