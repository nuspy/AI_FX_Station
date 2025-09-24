from __future__ import annotations
from typing import Iterable, List, Optional

# Import espliciti dei nostri detector
from .candles import make_candle_detectors
from .broadening import make_broadening_detectors

class PatternRegistry:
    """
    Registry minimale e deterministico. Restituisce i detector richiesti
    per tipo: "chart" e/o "candle". Niente magia, niente discovery dinamica.
    """
    def __init__(self) -> None:
        pass

    def detectors(self, kinds: Optional[Iterable[str]] = None) -> List[object]:
        kinds_set = set(kinds or [])
        want_chart  = not kinds_set or ("chart"  in kinds_set)
        want_candle = not kinds_set or ("candle" in kinds_set)

        out: List[object] = []
        if want_chart:
            out.extend(make_broadening_detectors())
        if want_candle:
            out.extend(make_candle_detectors())
        return out
