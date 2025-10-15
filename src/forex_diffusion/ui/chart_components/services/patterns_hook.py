# src/forex_diffusion/ui/chart_components/services/patterns_hook.py
from __future__ import annotations

import logging
from typing import Optional
from weakref import WeakKeyDictionary

from PySide6.QtCore import QObject

from .patterns_service import  PatternsService as PatternsService

log = logging.getLogger(__name__)

_PS_BY_CTRL: "WeakKeyDictionary[object, PatternsService]" = WeakKeyDictionary()

def _as_qobject_parent(view) -> Optional[QObject]:
    candidates = (
        getattr(view, "widget", None),
        getattr(view, "parent", None),
        getattr(view, "parentWidget", None),
    )
    for cand in candidates:
        obj = cand() if callable(cand) else cand
        if isinstance(obj, QObject):
            return obj
    return view if isinstance(view, QObject) else None

def get_patterns_service(controller, view, create: bool = False) -> Optional[PatternsService]:
    ps = _PS_BY_CTRL.get(controller)
    if ps is None and create:
        parent = _as_qobject_parent(view)
        try:
            ps = PatternsService(parent=parent)
        except TypeError:
            ps = PatternsService(parent=None)
        _PS_BY_CTRL[controller] = ps
    return ps

def set_patterns_toggle(controller, view, *, chart=None, candle=None, history=None) -> None:
    ps = get_patterns_service(controller, view, create=True)
    if ps is None:
        log.error("PatternsService not available; toggle ignored")
        return
    if chart is not None:
        ps.set_chart_enabled(bool(chart))
    if candle is not None:
        ps.set_candle_enabled(bool(candle))
    if history is not None:
        ps.set_history_enabled(bool(history))

# Optional integration points: safe no-ops if not present.

try:
    from ....ui.pattern_overlay import PatternOverlayRenderer  # type: ignore
except Exception:
    class PatternOverlayRenderer:
        def __init__(self) -> None:
            self._axes = None
            self._canvas = None
        def set_axes(self, ax, canvas) -> None:
            self._axes, self._canvas = ax, canvas
        def set_events(self, events, x_mode: str = "auto") -> None:
            pass

_REND_BY_CTRL: "WeakKeyDictionary[object, PatternOverlayRenderer]" = WeakKeyDictionary()

def ensure_renderer(controller, view) -> PatternOverlayRenderer:
    r = _REND_BY_CTRL.get(controller)
    if r is None:
        r = PatternOverlayRenderer()
        _REND_BY_CTRL[controller] = r
    return r

def set_patterns_axes(controller, view) -> None:
    r = ensure_renderer(controller, view)
    ax = getattr(view, "price_ax", None) or getattr(view, "ax_price", None) or getattr(view, "axes_price", None) or getattr(view, "plot_ax", None)   or getattr(view, "ax", None)
    canvas = getattr(view, "canvas", None) or getattr(view, "mpl_canvas", None)  or getattr(view, "figure_canvas", None) or getattr(view, "fig_canvas", None)
    if ax is None or canvas is None:
        log.debug("patterns set_axes: axes/canvas not found on view")
        return
    r.set_axes(ax, canvas)

def call_patterns_detection(controller, view, df) -> None:
    ps = get_patterns_service(controller, view, create=True)
    if ps is None:
        log.error("PatternsService not available; detection skipped")
        return
    r = ensure_renderer(controller, view)

    def _on_ready(events):
        r.set_events(events, x_mode="auto")

    set_patterns_axes(controller, view)
    if not hasattr(ps, "_conn_done"):
        ps.events_ready.connect(_on_ready)
        ps._conn_done = True
    ps.detect_async(df)
