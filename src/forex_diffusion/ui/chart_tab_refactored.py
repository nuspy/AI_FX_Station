"""
Refactored ChartTab implementation - backward compatible entry point.

This file provides the refactored ChartTabUI while maintaining compatibility
with existing imports. Use this during the transition period.

Usage:
    # Old way (still works)
    from forex_diffusion.ui.chart_tab_ui import ChartTabUI

    # New way (recommended)
    from forex_diffusion.ui.chart_tab_refactored import ChartTabUI

    # Or directly from the package
    from forex_diffusion.ui.chart_tab import ChartTabUI
"""
from __future__ import annotations

# Import the refactored implementation
from .chart_tab import ChartTabUI as RefactoredChartTabUI

# Export with the original name for compatibility
ChartTabUI = RefactoredChartTabUI

# Also export the individual components for advanced usage
from .chart_tab.chart_tab_base import DraggableOverlay
from .chart_tab.ui_builder import UIBuilderMixin
from .chart_tab.event_handlers import EventHandlersMixin
from .chart_tab.controller_proxy import ControllerProxyMixin
from .chart_tab.patterns_mixin import PatternsMixin
from .chart_tab.overlay_manager import OverlayManagerMixin

__all__ = [
    'ChartTabUI',
    'DraggableOverlay',
    'UIBuilderMixin',
    'EventHandlersMixin',
    'ControllerProxyMixin',
    'PatternsMixin',
    'OverlayManagerMixin'
]