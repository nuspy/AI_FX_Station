from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
import pandas as pd

Kind = Literal["chart","candle"]
State = Literal["forming","confirmed"]
Direction = Literal["bull","bear","neutral"]

@dataclass
class PatternEvent:
    pattern_key: str
    kind: Kind
    direction: Direction
    start_ts: pd.Timestamp
    confirm_ts: pd.Timestamp
    state: State = "confirmed"
    score: float = 0.0
    scale_atr: float = 0.0
    touches: int = 0
    bars_span: int = 0
    target_price: Optional[float] = None
    failure_price: Optional[float] = None
    horizon_bars: Optional[int] = None
    overlay: Dict[str, Any] = field(default_factory=dict)

class DetectorBase:
    """Base class for all detectors. Must be causal: only use data <= confirm_ts."""
    key: str = "base"
    kind: Kind = "chart"

    def detect(self, df: pd.DataFrame) -> List[PatternEvent]:
        raise NotImplementedError("Detector must implement detect()")

