# src/forex_diffusion/ui/event_bridge.py
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from loguru import logger
from typing import Any

class EventBridge(QObject):
    """
    Bridge between in-process event_bus and Qt main thread.
    Subscribe event_bus to call bridge._on_event(payload).
    The bridge emits tickReceived(payload) signal which is delivered to UI thread via Qt queued connections.
    """
    tickReceived = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            logger.debug("EventBridge initialized")
        except Exception:
            pass

    def _on_event(self, payload: Any) -> None:
        """
        Callback to be registered with event_bus.subscribe('tick', bridge._on_event).
        Can be called from any thread; emitting a Qt Signal will deliver to UI thread.
        """
        try:
            # emit payload to UI; Qt will queue the signal to the object's thread (UI thread)
            self.tickReceived.emit(payload)
        except Exception as e:
            try:
                logger.exception("EventBridge failed to emit tick: {}", e)
            except Exception:
                pass
