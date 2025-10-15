from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

from .engine import PatternEvent, DetectorBase
from .primitives import atr, time_array

def _zigzag(price: np.ndarray, pct: float = 0.35) -> List[int]:
    """Minizigzag: pivot quando il movimento supera pct%% del prezzo."""
    if price is None or len(price) < 3:
        return []
    pivots = [0]
    last = float(price[0])
    direction = 0
    for i in range(1, len(price)):
        cur = float(price[i])
        if last == 0:
            continue
        change = (cur - last) / last
        if direction >= 0 and change >= (pct / 100.0):
            pivots.append(i); direction = -1; last = cur
        elif direction <= 0 and change <= -(pct / 100.0):
            pivots.append(i); direction = +1; last = cur
    if pivots[-1] != len(price) - 1:
        pivots.append(len(price) - 1)
    return pivots

def _ratio(a: float, b: float) -> float:
    try: return abs(float(a) / float(b))
    except Exception: return float("inf")

class HarmonicDetector(DetectorBase):
    kind = "chart"
    def __init__(self, key: str, ratios: Tuple[Tuple[float, float], ...], pct_tol: float = 0.12, window: int = 240, max_events: int = 30, direction: str = "neutral") -> None:
        self.key = key; self.ratios = ratios; self.pct_tol = pct_tol; self.window = window; self.max_events = max_events; self.direction = direction

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        ts = time_array(df); c = df["close"].astype(float).to_numpy(); a = atr(df, 14).to_numpy()
        n = len(df); out: List[PatternEvent] = []

        # PERFORMANCE FIX: Compute ZigZag once for entire dataset
        full_pivots = _zigzag(c, pct=0.30)
        if len(full_pivots) < 5:
            return out

        # Use step size to reduce iterations significantly
        step_size = max(20, self.window // 10)  # Much larger steps

        for end in range(self.window, n, step_size):
            # Find pivots within current window using pre-computed pivots
            start = end - self.window
            window_pivots = [p for p in full_pivots if start <= p <= end]

            if len(window_pivots) < 5:
                continue

            # Take last 5 pivots for pattern matching
            X, A, B, C, D = window_pivots[-5:]
            XA = c[A] - c[X]; AB = c[B] - c[A]; BC = c[C] - c[B]; CD = c[D] - c[C]
            rs = (_ratio(AB, -XA), _ratio(BC, -AB), _ratio(CD, -BC))
            ok = True
            for (lo, hi), r in zip(self.ratios, rs):
                if not (lo * (1 - self.pct_tol) <= r <= hi * (1 + self.pct_tol)): ok = False; break
            if not ok: continue
            direction = self.direction if self.direction != "neutral" else ("bull" if CD < 0 else "bear")
            confirm = D; price = float(c[D]); target = float(price + (price - c[C]))
            out.append(PatternEvent(self.key, "chart", direction, pd.to_datetime(ts[A]), pd.to_datetime(ts[confirm]), price, 0.60, float(a[end]), 5, end - start, target, 40.0, {"pivots": [X, A, B, C, D]}))
            if len(out) >= self.max_events: break
        return out

def make_harmonic_detectors():
    specs = {
        "harmonic_gartley_bull": ((0.60, 0.72), (0.38, 0.88), (1.20, 1.80)),
        "harmonic_gartley_bear": ((0.60, 0.72), (0.38, 0.88), (1.20, 1.80)),
        "harmonic_bat_bull": ((0.38, 0.52), (0.38, 0.88), (1.60, 2.60)),
        "harmonic_bat_bear": ((0.38, 0.52), (0.38, 0.88), (1.60, 2.60)),
        "harmonic_butterfly_bull": ((0.72, 0.90), (0.38, 0.88), (1.60, 2.60)),
        "harmonic_butterfly_bear": ((0.72, 0.90), (0.38, 0.88), (1.60, 2.60)),
        "harmonic_crab_bull": ((0.38, 0.62), (0.38, 0.88), (2.20, 3.60)),
        "harmonic_crab_bear": ((0.38, 0.62), (0.38, 0.88), (2.20, 3.60)),
        "harmonic_cypher_bull": ((0.28, 0.42), (1.10, 1.40), (0.62, 1.00)),
        "harmonic_cypher_bear": ((0.28, 0.42), (1.10, 1.40), (0.62, 1.00)),
        "harmonic_shark_bull": ((0.86, 1.13), (1.13, 1.61), (0.62, 1.00)),
        "harmonic_shark_bear": ((0.86, 1.13), (1.13, 1.61), (0.62, 1.00)),
        "harmonic_abcd_bull": ((0.62, 0.90), (0.62, 0.90), (1.00, 1.60)),
        "harmonic_abcd_bear": ((0.62, 0.90), (0.62, 0.90), (1.00, 1.60)),
    }
    dets: List[HarmonicDetector] = []
    for key, ratios in specs.items():
        direction = "bull" if key.endswith("bull") else "bear"
        dets.append(HarmonicDetector(key, ratios, direction=direction))
    return dets
