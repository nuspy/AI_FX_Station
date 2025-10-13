from __future__ import annotations
from typing import Iterable, List, Optional
from loguru import logger

# Import espliciti dei nostri detector
from .candles import make_candle_detectors
from .candles_advanced import make_advanced_candle_detectors
from .broadening import make_broadening_detectors
from .wedges import make_wedge_detectors
from .triangles import make_triangle_detectors
from .rectangle import make_rectangle_detectors
from .diamond import make_diamond_detectors
from .double_triple import make_double_triple_detectors
from .channels import make_channel_detectors
from .flags import make_flag_detectors
from .hns import make_hns_detectors
from .elliott_wave import make_elliott_wave_detectors
from .harmonic_patterns import make_harmonic_pattern_detectors
from .advanced_chart_patterns import make_advanced_chart_pattern_detectors

# MED-003: Import vectorized detectors
try:
    from .vectorized_detectors import (
        make_vectorized_candle_detectors,
        make_vectorized_three_candle_detectors
    )
    _HAS_VECTORIZED = True
except ImportError as e:
    _HAS_VECTORIZED = False
    logger.warning(f"Vectorized detectors not available: {e}")


class PatternRegistry:
    """
    Registry minimale e deterministico. Restituisce i detector richiesti
    per tipo: "chart" e/o "candle". Niente magia, niente discovery dinamica.
    
    MED-003: Supports vectorized detectors for 10-100x performance boost.
    """
    
    def __init__(self, use_vectorized: bool = True) -> None:
        """
        Initialize pattern registry.
        
        Args:
            use_vectorized: Use vectorized detectors when available (default: True)
        """
        self.use_vectorized = use_vectorized and _HAS_VECTORIZED
        if self.use_vectorized:
            logger.info("✅ Vectorized pattern detection enabled (10-100x speedup)")

    def detectors(self, kinds: Optional[Iterable[str]] = None) -> List[object]:
        """
        Get pattern detectors of specified kinds.
        
        Args:
            kinds: Pattern kinds to include ("chart", "candle"), or None for all
            
        Returns:
            List of pattern detector instances
        """
        kinds_set = set(kinds or [])
        want_chart  = not kinds_set or ("chart"  in kinds_set)
        want_candle = not kinds_set or ("candle" in kinds_set)

        out: List[object] = []
        
        if want_chart:
            # Basic chart patterns
            out.extend(make_broadening_detectors())
            out.extend(make_wedge_detectors())
            out.extend(make_triangle_detectors())
            out.extend(make_rectangle_detectors())
            out.extend(make_diamond_detectors())
            out.extend(make_double_triple_detectors())
            out.extend(make_channel_detectors())
            out.extend(make_flag_detectors())
            out.extend(make_hns_detectors())

            # Advanced chart patterns
            out.extend(make_elliott_wave_detectors())
            out.extend(make_harmonic_pattern_detectors())
            out.extend(make_advanced_chart_pattern_detectors())

        if want_candle:
            # MED-003: Use vectorized detectors if available
            if self.use_vectorized:
                logger.debug("Using vectorized candlestick detectors")
                out.extend(make_vectorized_candle_detectors())
                out.extend(make_vectorized_three_candle_detectors())
            else:
                logger.debug("Using loop-based candlestick detectors")
                out.extend(make_candle_detectors())
            
            # Advanced candlestick patterns (not yet vectorized)
            out.extend(make_advanced_candle_detectors())
            
        return out
