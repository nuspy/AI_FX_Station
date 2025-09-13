# src/forex_diffusion/utils/event_bus.py
from __future__ import annotations

"""
Thread-safe, in-process event bus using a direct callback model.
"""

from typing import Any, Callable, Dict, List
import threading
from loguru import logger

_registry: Dict[str, List[Callable[[Any], None]]] = {}
_registry_lock = threading.RLock()

def subscribe(topic: str, callback: Callable[[Any], None]) -> None:
    """Register a callback to be invoked when a topic is published."""
    with _registry_lock:
        lst = _registry.setdefault(topic, [])
        if callback not in lst:
            lst.append(callback)
            logger.debug(f"event_bus: Subscribed {getattr(callback, '__name__', str(callback))} to topic '{topic}'")

def unsubscribe(topic: str, callback: Callable[[Any], None]) -> None:
    """Remove a previously registered callback."""
    with _registry_lock:
        if topic in _registry and callback in _registry[topic]:
            _registry[topic].remove(callback)
            if not _registry[topic]:
                del _registry[topic]

def publish(topic: str, payload: Any) -> None:
    """
    Publish an event by directly invoking all subscribed callbacks for the topic.
    """
    logger.critical("+" * 80)
    logger.critical(f"--- EVENT_BUS: PUBLISH CALLED FOR TOPIC: {topic} ---")
    logger.critical("+" * 80)
    
    callbacks_to_run: List[Callable[[Any], None]] = []
    with _registry_lock:
        if topic in _registry:
            callbacks_to_run = list(_registry[topic])
    
    for callback in callbacks_to_run:
        try:
            callback(payload)
        except Exception as e:
            logger.exception(f"event_bus: Error in callback for topic '{topic}': {e}")

# --- Compatibility Aliases ---
pub = publish
sub = subscribe
unsub = unsubscribe

def debug_status() -> dict:
    """Return a diagnostic snapshot of subscriber counts."""
    with _registry_lock:
        return {"subscribers": {k: len(v) for k, v in _registry.items()}}
