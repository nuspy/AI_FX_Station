"""
Chart Tab UI Components - Refactored Structure

This package contains the refactored chart tab components:
- chart_tab_base.py: Main ChartTab class and DraggableOverlay
- ui_builder.py: UI construction methods
- event_handlers.py: Event handling methods
- controller_proxy.py: Controller passthrough methods
- patterns_mixin.py: Pattern detection integration
- overlay_manager.py: Overlay and drawing management
"""

from .chart_tab_base import ChartTabUI, DraggableOverlay

__all__ = ['ChartTabUI', 'DraggableOverlay']