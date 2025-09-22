from __future__ import annotations
from typing import List, Optional
import pandas as pd
from loguru import logger
from .base import ChartServiceBase
from patterns.registry import PatternRegistry
from ...patterns.engine import PatternEvent
from ...patterns.info_provider import PatternInfoProvider
from ...ui.pattern_overlay import PatternOverlayRenderer

class PatternsService(ChartServiceBase):
    def __init__(self, view, controller) -> None:
        super().__init__(view, controller)
        self._enabled_chart = False
        self._enabled_candle = False
        self._enabled_history = False
        self._events: List[PatternEvent] = []
        self.registry = PatternRegistry()
        self.info = PatternInfoProvider(self._default_info_path())
        self.renderer = PatternOverlayRenderer(controller, self.info)

    def _default_info_path(self):
        from pathlib import Path
        return Path(self.view._app_root or ".") / "configs" / "pattern_info.json"

    # toggles
    def set_chart_enabled(self, on: bool):
        self._enabled_chart = bool(on)
        self._repaint()

    def set_candle_enabled(self, on: bool):
        self._enabled_candle = bool(on)
        self._repaint()

    def set_history_enabled(self, on: bool):
        self._enabled_history = bool(on)
        self._repaint()

    def on_update_plot(self, df: pd.DataFrame):
        try:
            kinds = []
            if self._enabled_chart: kinds.append("chart")
            if self._enabled_candle: kinds.append("candle")
            if not kinds:
                self._events.clear()
                self.renderer.clear()
                return
            # precompute dt for overlay
            try:
                df = df.copy()
                df["ts_dt"] = pd.to_datetime(df["ts_utc"], unit="ms", utc=True).tz_convert(None)
            except Exception:
                pass
            evs: List[PatternEvent] = []
            for det in self.registry.detectors(kinds=kinds):
                try:
                    evs.extend(det.detect(df))
                except Exception as e:
                    logger.debug(f"Detector {getattr(det,'key','?')} failed: {e}")
            self._events = evs
            self.renderer.draw(self._events)
        except Exception as e:
            logger.exception("Patterns on_update_plot failed: {}", e)

    def _repaint(self):
        self.renderer.draw(self._events or [])
