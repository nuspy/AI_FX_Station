# src/forex_diffusion/utils/event_bus.py
from __future__ import annotations

"""
Thread-safe in-process event bus (pub/sub).
Provides: subscribe(topic, callback), unsubscribe(topic, callback), publish(topic, payload).
Callbacks are invoked synchronously in the publishing thread; exceptions in callbacks are logged and ignored.
"""

from typing import Any, Callable, Dict, List
import threading
from loguru import logger

# Internal registry and lock
_registry: Dict[str, List[Callable[[Any], None]]] = {}
_registry_lock = threading.RLock()

def subscribe(topic: str, callback: Callable[[Any], None]) -> None:
    """Subscribe a callable to a topic."""
    try:
        with _registry_lock:
            lst = _registry.setdefault(topic, [])
            if callback not in lst:
                lst.append(callback)
                logger.debug("event_bus: subscribed to topic='{}' callback={}", topic, getattr(callback, "__name__", str(callback)))
    except Exception as e:
        logger.exception("event_bus.subscribe failed for topic=%s: %s", topic, e)

def unsubscribe(topic: str, callback: Callable[[Any], None]) -> None:
    """Unsubscribe a callable from a topic."""
    try:
        with _registry_lock:
            lst = _registry.get(topic)
            if not lst:
                return
            try:
                lst.remove(callback)
            except ValueError:
                pass
            if not lst:
                _registry.pop(topic, None)
            logger.debug("event_bus: unsubscribed from topic='{}' callback={}", topic, getattr(callback, "__name__", str(callback)))
    except Exception as e:
        logger.exception("event_bus.unsubscribe failed for topic=%s: %s", topic, e)

def publish(topic: str, payload: Any) -> None:
    """Publish payload to all subscribers for the topic. Exceptions in subscribers are logged and ignored."""
    try:
        with _registry_lock:
            subs = list(_registry.get(topic, []))
        logger.debug("event_bus: publish topic='{}' payload_type='{}' subscribers={}", topic, type(payload).__name__, len(subs))
        for cb in subs:
            try:
                cb(payload)
            except Exception as e:
                logger.exception("event_bus: subscriber for topic='%s' raised: %s", topic, e)
    except Exception as e:
        logger.exception("event_bus.publish failed for topic=%s: %s", topic, e)

# Backwards-compatible aliases
pub = publish
sub = subscribe
unsub = unsubscribe
