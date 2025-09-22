from __future__ import annotations
from typing import List, Dict, Any, Optional
import json, os
from pathlib import Path
from .broadening import make_broadening_detectors
from .candles import make_candle_detectors
from .engine import DetectorBase

class PatternRegistry:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path

    def detectors(self, kinds: Optional[List[str]] = None) -> List[DetectorBase]:
        dets = []
        if kinds is None or "chart" in kinds:
            dets.extend(make_broadening_detectors())
        if kinds is None or "candle" in kinds:
            dets.extend(make_candle_detectors())
        return dets
