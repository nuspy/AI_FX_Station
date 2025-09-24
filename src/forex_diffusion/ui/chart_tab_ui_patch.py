
from __future__ import annotations
from typing import Any
import logging

from .chart_components.services.patterns_hook import (
    set_patterns_axes, set_patterns_toggle, call_patterns_detection
)

log = logging.getLogger(__name__)


def _wire_pattern_checkboxes(self) -> None:
    """Call this inside ChartTabUI.__init__ AFTER axes/canvas are ready."""
    # Bind renderer to axes
    try:
        set_patterns_axes(self, self)
    except Exception as ex:
        log.debug("patterns: set_axes failed: %s", ex)

    # Find checkboxes on the topbar
    chk_chart = getattr(self, "chart_patterns_chk", None) or getattr(self, "chartPatternsCheckbox", None) or getattr(self, "chart_patterns_checkbox", None)
    chk_candle = getattr(self, "candle_patterns_chk", None) or getattr(self, "candlePatternsCheckbox", None) or getattr(self, "candlestick_patterns_checkbox", None)
    chk_hist = getattr(self, "history_patterns_chk", None) or getattr(self, "historyPatternsCheckbox", None) or getattr(self, "patterns_history_checkbox", None)

    # Sync initial
    init_chart = bool(chk_chart.isChecked()) if chk_chart else False
    init_candle = bool(chk_candle.isChecked()) if chk_candle else False
    init_hist = bool(chk_hist.isChecked()) if chk_hist else False
    set_patterns_toggle(self, self, chart=init_chart, candle=init_candle, history=init_hist)

    # Connect toggles
    if chk_chart:
        chk_chart.toggled.connect(lambda on: set_patterns_toggle(self, self, chart=on))
    if chk_candle:
        chk_candle.toggled.connect(lambda on: set_patterns_toggle(self, self, candle=on))
    if chk_hist:
        chk_hist.toggled.connect(lambda on: set_patterns_toggle(self, self, history=on))

    # Expose a method to be called by your plot update with the latest df
    def scan_patterns_now(df):
        try:
            call_patterns_detection(self, self, df)
        except Exception as ex:
            log.error("Scan patterns failed: %s", ex)

    setattr(self, "scan_patterns_now", scan_patterns_now)
