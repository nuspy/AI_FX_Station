from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

from .engine import PatternEvent, DetectorBase
from .primitives import atr, time_array

class CupHandleDetector(DetectorBase):
    kind = "chart"
    def __init__(self, key: str, window: int = 220, handle_max: int = 40, tol_symmetry: float = 0.015, min_depth_atr: float = 2.0, max_events: int = 20) -> None:
        self.key = key; self.window = window; self.handle_max = handle_max; self.tol_symm = tol_symmetry; self.min_depth_atr = min_depth_atr; self.max_events = max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        ts = time_array(df); c = df["close"].astype(float).to_numpy(); a = atr(df, 14).to_numpy()
        n = len(df); out: List[PatternEvent] = []
        for end in range(self.window, n):
            start = end - self.window; seg = c[start:end+1]; low_idx = int(np.argmin(seg))
            left = float(seg[0]); right = float(seg[-1]); bottom = float(seg[low_idx])
            if not (left * (1 - self.tol_symm) <= right <= left * (1 + self.tol_symm)): continue
            depth = left - bottom
            if depth <= 0 or depth < self.min_depth_atr * float(a[end]): continue
            hstart = end - min(self.handle_max, self.window // 4); handle = c[hstart:end+1]
            handle_range = float(handle.max() - handle.min())
            if handle.max() <= handle[-1] and handle_range <= 1.2 * float(a[end]):
                confirm_idx = end; direction = "bull"; price = float(c[confirm_idx]); target = float(right + depth)
                out.append(PatternEvent(self.key, "chart", direction, pd.to_datetime(ts[start+low_idx]), pd.to_datetime(ts[confirm_idx]), price, 0.58, float(a[end]), 5, self.window, target, 40.0, {"cup_low": start + low_idx}))
                if len(out) >= self.max_events: break
        return out

def make_cup_detectors() -> List[CupHandleDetector]:
    return [CupHandleDetector("cup_and_handle")]
