# src/forex_diffusion/utils/event_bus.py
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List

_lock = threading.Lock()
_subscribers: Dict[str, List[Callable[[Any], None]]] = {}


def subscribe(event: str, callback: Callable[[Any], None]) -> None:
    """Subscribe callback to event."""
    with _lock:
        lst = _subscribers.get(event)
        if lst is None:
            _subscribers[event] = [callback]
        else:
            lst.append(callback)


def unsubscribe(event: str, callback: Callable[[Any], None]) -> None:
    with _lock:
        lst = _subscribers.get(event)
        if not lst:
            return
        try:
            lst.remove(callback)
        except ValueError:
            pass


def publish(event: str, payload: Any) -> None:
    """Publish payload to all subscribers (copied under lock)."""
    with _lock:
        lst = list(_subscribers.get(event, []))
    for cb in lst:
        try:
            cb(payload)
        except Exception:
            # subscribers should handle exceptions; ignore here
            pass
