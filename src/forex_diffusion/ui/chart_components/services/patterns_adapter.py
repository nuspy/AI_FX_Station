
from __future__ import annotations
"""
Optional adapter – currently unused, but kept for forward-compat.
You can map raw detector outputs to PatternEvent dicts here if needed.
"""
from typing import Dict, Any

def adapt_detector_event(d: Dict[str, Any]) -> Dict[str, Any]:
    # Map common aliases to expected keys
    out = dict(d)
    if "t_start" in out and "start_ts" not in out:
        out["start_ts"] = out["t_start"]
    if "t_confirm" in out and "confirm_ts" not in out:
        out["confirm_ts"] = out["t_confirm"]
    if "pattern" in out and "name" not in out:
        out["name"] = out["pattern"]
    return out
