from __future__ import annotations
from typing import Optional
from weakref import WeakKeyDictionary

# Controller -> service (nessun setattr: compatibile con __slots__)
_SERVICES = WeakKeyDictionary()

def get_patterns_service(controller, view, create: bool = True):
    """Restituisce il service dal registry; lo crea lazy se create=True."""
    ps = _SERVICES.get(controller)
    if ps is None and create:
        from .patterns_service import PatternsService
        ps = PatternsService(view, controller)
        _SERVICES[controller] = ps
    return ps

def call_patterns_detection(controller, view, df):
    """Invoca la detection in modo debounced e non bloccante."""
    from loguru import logger
    try:
        ps = get_patterns_service(controller, view, create=True)
        ps.detect_async(df)
    except Exception as e:
        # Non propagare in UI: solo log di debug
        logger.debug(f"patterns hook (safe) error: {e}")

def set_patterns_toggle(controller, view, *, chart: Optional[bool]=None,
                        candle: Optional[bool]=None, history: Optional[bool]=None):
    """API di comodo per attivare/disattivare i tre toggle senza toccare attributi del controller."""
    ps = get_patterns_service(controller, view, create=True)
    if chart is not None:   ps.set_chart_enabled(bool(chart))
    if candle is not None:  ps.set_candle_enabled(bool(candle))
    if history is not None: ps.set_history_enabled(bool(history))
