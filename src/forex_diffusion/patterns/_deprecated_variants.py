"""
DEPRECATED: Parameter variants system

This module is deprecated and no longer used in the pattern detection system.
Parameter optimization is now handled by:
- DatabaseParameterSelector (patterns/parameter_selector.py)
- OptimizationEngine (training/optimization/engine.py)
- NSGA-II multi-objective genetic algorithm

Historical note: This file was used to create parameter variations for testing.
Kept for reference purposes only.

Date deprecated: 2025-10-13
Replaced by: Database-driven parameter selection with historical performance
"""
from __future__ import annotations
from typing import List

# Original implementation (disabled)
# from .wedges import WedgeDetector
# from .triangles import TriangleDetector
# from .channels import ChannelDetector
# from .flags import FlagDetector
# from .rectangle import RectangleDetector

def make_param_variants() -> List:
    """
    DEPRECATED: Creates parameter variants for testing.
    
    This function is no longer called and returns empty list.
    Use DatabaseParameterSelector.get_optimal_parameters() instead.
    """
    return []
