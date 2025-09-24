
from __future__ import annotations
from typing import Optional, List, Any

from .patterns_service import PatternsService

# convenience accessors used by ChartTabUI/plot_service

def get_patterns_service(controller, view, create: bool = False) -> Optional[PatternsService]:
    svc = getattr(controller, "patterns_service", None)
    if svc is None and create:
        svc = PatternsService(controller, view)
        controller.patterns_service = svc
    return svc

def set_patterns_toggle(controller, view, *, chart: Optional[bool]=None,
                        candle: Optional[bool]=None, history: Optional[bool]=None):
    ps = get_patterns_service(controller, view, create=True)
    if chart is not None:   ps.set_chart_enabled(bool(chart))
    if candle is not None:  ps.set_candle_enabled(bool(candle))
    if history is not None: ps.set_history_enabled(bool(history))

def set_patterns_axes(controller, view, ax):
    ps = get_patterns_service(controller, view, create=True)
    try:
        ps.set_axes(ax)
    except Exception as e:
        # allow log on caller
        raise

def call_patterns_detection(controller, view, df):
    ps = get_patterns_service(controller, view, create=True)
    # instrada tutto su on_update_plot (che è idempotente)
    if hasattr(ps, "on_update_plot"):
        ps.on_update_plot(df)
    else:
        # fallback per versioni precedenti
        ps.detect_async(df)
