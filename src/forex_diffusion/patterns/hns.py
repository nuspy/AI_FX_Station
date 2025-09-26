from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from .engine import PatternEvent, DetectorBase
from .primitives import atr

class HeadAndShouldersDetector(DetectorBase):
    key="head_and_shoulders"
    kind="chart"
    def __init__(self, inverse: bool=False, window:int=120, tol:float=0.02, max_events:int=30) -> None:
        self.inverse = inverse; self.window=window; self.tol=tol; self.max_events=max_events
        self.key = "inverse_head_and_shoulders" if inverse else "head_and_shoulders"

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        c = df["close"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        ts = pd.to_datetime(df["time"] if "time" in df.columns else df.index).to_numpy()
        a = atr(df, 14).to_numpy()
        events: List[PatternEvent] = []
        n=len(df)
        for end in range(self.window, n):
            start = end - self.window
            # naive three-peak detection
            seg = h[start:end+1] if not self.inverse else -l[start:end+1]
            idx = np.arange(len(seg))
            # find local maxima (or minima if inverse)
            locs = [i for i in range(1,len(seg)-1) if seg[i]>seg[i-1] and seg[i]>seg[i+1]]
            if len(locs) < 3: 
                continue
            # try sliding triplets
            for i in range(len(locs)-2):
                L, H, R = locs[i], locs[i+1], locs[i+2]
                # Head higher than shoulders by tol
                if seg[H] > seg[L]*(1+self.tol) and seg[H] > seg[R]*(1+self.tol) and abs(seg[L]-seg[R]) <= seg[H]*self.tol*2:
                    # neckline between troughs
                    confirm_idx = end
                    direction = "bear" if not self.inverse else "bull"
                    events.append(PatternEvent(self.key,"chart",direction, ts[start+L], ts[confirm_idx], "confirmed", 0.55, float(a[end]), 3, self.window, None, self.window//3, {"peaks":(start+L,start+H,start+R)}))
                    if len(events)>=self.max_events: return events
        return events

def make_hns_detectors() -> List[HeadAndShouldersDetector]:
    return [HeadAndShouldersDetector(False), HeadAndShouldersDetector(True)]
