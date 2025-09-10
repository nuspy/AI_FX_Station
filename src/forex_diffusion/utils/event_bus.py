# src/forex_diffusion/utils/event_bus.py
from __future__ import annotations

"""
Thread-safe in-process event bus (queue-based).
- publish(topic, payload) appends payload to an internal per-topic queue (safe from any thread).
- take_pending(topic) returns and clears queued payloads (call from UI thread).
- subscribe/unsubscribe remain for in-process direct callbacks if needed, but publish no longer calls subscribers directly.
This avoids executing Qt-related callbacks from non-Qt threads.
"""

from typing import Any, Callable, Dict, List
import threading
from loguru import logger

# Internal registry and lock for subscribers (not invoked by publish to avoid threading issues)
_registry: Dict[str, List[Callable[[Any], None]]] = {}
_registry_lock = threading.RLock()

# Per-topic payload queues for safe cross-thread publication
_queues: Dict[str, List[Any]] = {}
_queues_lock = threading.RLock()

def subscribe(topic: str, callback: Callable[[Any], None]) -> None:
    """Register a callback (not automatically invoked by publish anymore)."""
    try:
        with _registry_lock:
            lst = _registry.setdefault(topic, [])
            if callback not in lst:
                lst.append(callback)
                logger.debug("event_bus: subscribed to topic='{}' callback={}", topic, getattr(callback, "__name__", str(callback)))
    except Exception as e:
        logger.exception("event_bus.subscribe failed for topic=%s: %s", topic, e)

def unsubscribe(topic: str, callback: Callable[[Any], None]) -> None:
    """Remove a previously registered callback."""
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
    """
    Publish: append payload to per-topic queue.
    Does NOT invoke subscribers directly to avoid executing UI code from non-UI threads.
    """
    try:
        with _queues_lock:
            q = _queues.setdefault(topic, [])
            q.append(payload)
            sz = len(q)
        # INFO so it's visible in standard logs and easier to trace cross-thread publishes
        logger.info(f"event_bus: publish -> topic='{topic}' queued payload_type='{type(payload).__name__}' queue_size={sz}")
    except Exception as e:
        logger.exception("event_bus.publish failed for topic=%s: %s", topic, e)

def take_pending(topic: str) -> List[Any]:
    """
    Consume and return all pending payloads for a topic; to be called from the main/UI thread.
    """
    try:
        with _queues_lock:
            lst = _queues.get(topic, [])
            if not lst:
                return []
            out = list(lst)
            _queues[topic] = []
        logger.debug("event_bus: take_pending -> topic='{}' returned_count={}", topic, len(out))
        return out
    except Exception as e:
        logger.exception("event_bus.take_pending failed for topic=%s: %s", topic, e)
        return []

# Backwards-compatible aliases
pub = publish
sub = subscribe
unsub = unsubscribe

def debug_status() -> dict:
    """
    Return a diagnostic snapshot: subscriber counts and queue sizes.
    """
    try:
        with _registry_lock:
            subs_map = {k: len(v) for k, v in _registry.items()}
        with _queues_lock:
            queues_map = {k: len(v) for k, v in _queues.items()}
        logger.debug("event_bus.debug_status: subscribers=%s queues=%s", subs_map, queues_map)
        return {"subscribers": subs_map, "queues": queues_map}
    except Exception as e:
        logger.exception("event_bus.debug_status failed: %s", e)
        return {"subscribers": {}, "queues": {}}

def get_subscribers(topic: str) -> list:
    """
    Return a shallow copy of subscriber callables for a topic (thread-safe).
    Deduplicate by identity (id) to avoid multiple invocations of the same callable.
    """
    try:
        with _registry_lock:
            lst = list(_registry.get(topic, []) )
        # dedupe by id while preserving order
        seen = set()
        out = []
        for cb in lst:
            iid = id(cb)
            if iid in seen:
                continue
            seen.add(iid)
            out.append(cb)
        return out
    except Exception as e:
        logger.exception("event_bus.get_subscribers failed for topic=%s: %s", topic, e)
        return []
