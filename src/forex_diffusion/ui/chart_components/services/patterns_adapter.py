
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def _to_ms(ts: Any) -> Optional[int]:
    if ts is None:
        return None
    if hasattr(ts, "value"):  # pandas Timestamp (ns)
        return int(int(ts.value) // 1_000_000)
    if isinstance(ts, (np.integer, int)):
        return int(ts)
    if isinstance(ts, (float, np.floating)):
        # already float seconds or mdates; let the overlay convert
        return int(ts)
    try:
        return int(ts)
    except Exception:
        return None


def normalize_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map various detector payloads into the renderer schema."""
    norm: List[Dict[str, Any]] = []
    for e in raw_events or []:
        norm.append({
            "name": e.get("name") or e.get("key") or "Pattern",
            "kind": e.get("kind") or e.get("type") or "chart",
            "direction": e.get("direction"),
            "ts_start": _to_ms(e.get("ts_start") or e.get("start_ts") or e.get("start")),
            "ts_end": _to_ms(e.get("ts_end") or e.get("end_ts") or e.get("end")),
            "confirm_ts": _to_ms(e.get("confirm_ts") or e.get("confirm") or e.get("pivot_ts")),
            "target_price": e.get("target_price"),
            "info_html": e.get("info_html"),
        })
    return norm
