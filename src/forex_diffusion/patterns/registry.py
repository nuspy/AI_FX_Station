# src/forex_diffusion/patterns/registry.py
from __future__ import annotations
from typing import Iterable, List, Optional

# Import espliciti dei nostri detector (già esistenti nel tuo progetto)
from .candles import make_candle_detectors
from .broadening import make_broadening_detectors


class PatternRegistry:
    """
    Registry minimale e deterministico.
    - "chart"  -> detector di pattern di struttura/price-action (es. broadening, wedges, ecc.)
    - "candle" -> detector di candele/patterns a candele (es. engulfing, harami, ecc.)
    Se kinds è None o vuoto, ritorna entrambi i gruppi.
    """
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


# --------- Funzioni "compat" usate da patterns_service.py ---------

def get_chart_detectors() -> List[object]:
    """Ritorna solo i detector 'chart' (compat con vecchie import)."""
    return make_broadening_detectors()


def get_candle_detectors() -> List[object]:
    """Ritorna solo i detector 'candle' (compat con vecchie import)."""
    return make_candle_detectors()


def get_detectors(kinds: Optional[Iterable[str]] = None) -> List[object]:
    """Convenience: come PatternRegistry().detectors(kinds)."""
    return PatternRegistry().detectors(kinds)


__all__ = [
    "PatternRegistry",
    "get_chart_detectors",
    "get_candle_detectors",
    "get_detectors",
]
