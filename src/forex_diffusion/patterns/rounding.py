from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

from .engine import PatternEvent, DetectorBase
from .primitives import atr, time_array

class RoundingDetector(DetectorBase):
    kind = "chart"
    def __init__(self, key: str, top: bool, window: int = 200, max_events: int = 20) -> None:
        self.key = key; self.top = top; self.window = window; self.max_events = max_events

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        ts = time_array(df); c = df["close"].astype(float).to_numpy(); a = atr(df, 14).to_numpy()
        n = len(df); out: List[PatternEvent] = []
        for end in range(self.window, n):
            start = end - self.window; seg = c[start:end+1]
            s1 = np.gradient(seg); s2 = np.gradient(s1); curv = float(np.mean(s2))

            # Calculate target and failure prices based on pattern direction
            current_price = float(seg[-1])
            atr_value = float(a[end])

            if self.top and curv < 0:
                # Rounding top (bearish): target below, failure above
                target_price = current_price - (2.0 * atr_value)  # Target 2 ATR below
                failure_price = current_price + (1.0 * atr_value)  # Stop 1 ATR above

                out.append(PatternEvent(
                    pattern_key=self.key,
                    kind="chart",
                    direction="bear",
                    start_ts=pd.to_datetime(ts[start]),
                    confirm_ts=pd.to_datetime(ts[end]),
                    state="confirmed",
                    score=0.7,
                    scale_atr=atr_value,
                    touches=3,
                    bars_span=self.window,
                    target_price=target_price,
                    failure_price=failure_price,
                    horizon_bars=40,
                    overlay={}
                ))
            elif (not self.top) and curv > 0:
                # Rounding bottom (bullish): target above, failure below
                target_price = current_price + (2.0 * atr_value)  # Target 2 ATR above
                failure_price = current_price - (1.0 * atr_value)  # Stop 1 ATR below

                out.append(PatternEvent(
                    pattern_key=self.key,
                    kind="chart",
                    direction="bull",
                    start_ts=pd.to_datetime(ts[start]),
                    confirm_ts=pd.to_datetime(ts[end]),
                    state="confirmed",
                    score=0.7,
                    scale_atr=atr_value,
                    touches=3,
                    bars_span=self.window,
                    target_price=target_price,
                    failure_price=failure_price,
                    horizon_bars=40,
                    overlay={}
                ))
            if len(out) >= self.max_events: break
        return out

def make_rounding_detectors() -> List[RoundingDetector]:
    return [RoundingDetector("rounding_top", True), RoundingDetector("rounding_bottom", False)]
