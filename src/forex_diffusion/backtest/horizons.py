from __future__ import annotations

import math
import re
from typing import List, Tuple


# Units mapping in seconds
UNIT2SEC = {"s": 1, "m": 60, "h": 3600}


_SINGLE_RE = re.compile(r"^\s*(\d+)\s*([smh])\s*$", re.IGNORECASE)
_RANGE_RE = re.compile(r"^\s*\((\d+)\s*-\s*(\d+)\)\s*([smh])\s*$", re.IGNORECASE)


def _expand_token(token: str) -> List[Tuple[str, int]]:
    """Expand a single token into (label, seconds) items.
    Supports formats: 'Xs', 'Ym', 'Zh', '(A-B)u'.
    """
    token = token.strip()
    if not token:
        return []
    m = _SINGLE_RE.match(token)
    if m:
        val = int(m.group(1))
        unit = m.group(2).lower()
        sec = val * UNIT2SEC[unit]
        return [(f"{val}{unit}", sec)]
    m = _RANGE_RE.match(token)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        unit = m.group(3).lower()
        if a > b:
            a, b = b, a
        return [(f"{v}{unit}", v * UNIT2SEC[unit]) for v in range(a, b + 1)]
    raise ValueError(f"Invalid horizon token: '{token}'")


def parse_horizons(hstr: str, max_expanded: int = 100, max_horizon_sec: int = 48 * 3600) -> Tuple[List[str], List[int]]:
    """Parse horizons string into normalized labels and sorted seconds list.

    Rules:
    - tokens separated by commas
    - ignores spaces
    - supports single and ranged tokens
    - deduplicates and sorts by seconds
    - clamps to max_expanded and max_horizon_sec
    Returns (labels, seconds)
    """
    if hstr is None:
        return [], []
    raw_tokens = [t for t in (hstr.split(",") if isinstance(hstr, str) else [])]
    items: List[Tuple[str, int]] = []
    for t in raw_tokens:
        try:
            items.extend(_expand_token(t))
        except ValueError:
            # skip invalid tokens silently to be robust in UI; they can be validated upstream
            continue
    # filter by max horizon seconds
    items = [(lab, sec) for (lab, sec) in items if sec <= max_horizon_sec and sec >= 1]
    # dedup by seconds, keep smallest label occurrence
    by_sec = {}
    for lab, sec in items:
        if sec not in by_sec:
            by_sec[sec] = lab
    secs_sorted = sorted(by_sec.keys())
    labels_sorted = [by_sec[s] for s in secs_sorted]
    if len(secs_sorted) > max_expanded:
        secs_sorted = secs_sorted[:max_expanded]
        labels_sorted = labels_sorted[:max_expanded]
    return labels_sorted, secs_sorted


def bars_ahead_for_timeframe(horizon_sec: int, timeframe: str) -> int:
    """Compute bars_ahead = ceil(H_sec / tf_sec).
    timeframe is like '1m','5m','15m','30m','1h','4h','1d'.
    """
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        tf_sec = int(tf[:-1]) * 60
    elif tf.endswith("h"):
        tf_sec = int(tf[:-1]) * 3600
    elif tf.endswith("d"):
        tf_sec = int(tf[:-1]) * 86400
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    return max(1, math.ceil(horizon_sec / tf_sec))


__all__ = ["UNIT2SEC", "parse_horizons", "bars_ahead_for_timeframe"]


