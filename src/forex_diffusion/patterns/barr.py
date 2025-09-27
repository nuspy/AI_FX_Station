from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

from .engine import PatternEvent, DetectorBase
from .primitives import atr, time_array

class BARRDetector(DetectorBase):
    kind = "chart"
    def __init__(self, key: str, top: bool, window: int = 260, slope_min: float = 0.002, max_events: int = 20) -> None:
        self.key = key; self.top = top; self.window = window; self.slope_min = slope_min; self.max_events = max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        ts = time_array(df); c = df["close"].astype(float).to_numpy(); a = atr(df, 14).to_numpy()
        n = len(df); out: List[PatternEvent] = []
        for end in range(self.window, n):
            start = end - self.window; seg = c[start:end+1]; x = np.arange(len(seg), dtype=float)
            m = np.polyfit(x, seg, 1)[0]
            if abs(m) < self.slope_min: continue
            early_mean = float(np.mean(seg[: max(3, len(seg) // 3)])); last_val = float(seg[-1])
            if self.top and m > 0 and last_val < early_mean:
                out.append(PatternEvent(self.key, "chart", "bear", pd.to_datetime(ts[start]), pd.to_datetime(ts[end]), last_val, 0.50, float(a[end]), 3, self.window, None, 40.0, {}))
                if len(out) >= self.max_events: break
            if (not self.top) and m < 0 and last_val > early_mean:
                out.append(PatternEvent(self.key, "chart", "bull", pd.to_datetime(ts[start]), pd.to_datetime(ts[end]), last_val, 0.50, float(a[end]), 3, self.window, None, 40.0, {}))
                if len(out) >= self.max_events: break
        return out

def make_barr_detectors() -> List[BARRDetector]:
    return [BARRDetector("barr_top", True), BARRDetector("barr_bottom", False)]
