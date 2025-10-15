from __future__ import annotations
from typing import List

from .wedges import WedgeDetector
from .triangles import TriangleDetector
from .channels import ChannelDetector
from .flags import FlagDetector
from .rectangle import RectangleDetector

def make_param_variants() -> List[object]:
    dets: List[object] = []

    # Wedges
    for key, asc in [("wedge_ascending_tight", True), ("wedge_ascending_loose", True), ("wedge_descending_tight", False), ("wedge_descending_loose", False)]:
        dets.append(WedgeDetector(key=key, ascending=asc, min_span=20 if "tight" in key else 40, max_span=140 if "tight" in key else 220, min_touches=5 if "tight" in key else 4, max_events=25))

    # Triangles
    for mode in ["ascending", "descending", "symmetrical"]:
        for variant in ["narrow", "wide"]:
            key = f"{mode}_triangle_{variant}"
            dets.append(TriangleDetector(key=key, mode=mode, min_span=18 if variant=="narrow" else 36, max_span=160 if variant=="narrow" else 240, min_touches=5 if variant=="narrow" else 4, max_events=25))

    # Channels
    for rising in [True, False]:
        for variant in ["tight", "wide"]:
            key = ("rising" if rising else "falling") + f"_channel_{variant}"
            dets.append(ChannelDetector(key=key, rising=rising, min_span=24 if variant=="tight" else 40, max_span=180 if variant=="tight" else 260, max_events=25))

    # Flags & Pennants
    for direction in ["bull", "bear"]:
        for variant in ["tight", "wide"]:
            dets.append(FlagDetector(key=f"{direction}_flag_{variant}", direction=direction, impulse_mult=2.2 if variant=="tight" else 1.6, window=36 if variant=="tight" else 52, pennant=False, max_events=30))
            dets.append(FlagDetector(key=f"{direction}_pennant_{variant}", direction=direction, impulse_mult=2.2 if variant=="tight" else 1.6, window=36 if variant=="tight" else 52, pennant=True, max_events=30))

    # Rectangle / Range
    dets.append(RectangleDetector(key="rectangle_tight", window=60, tightness=0.7, max_events=40))
    dets.append(RectangleDetector(key="rectangle_loose", window=120, tightness=1.1, max_events=40))

    return dets
