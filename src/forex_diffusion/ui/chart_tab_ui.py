"""Chart Tab UI - Modular Implementation Bridge"""
from __future__ import annotations

from .chart_tab import ChartTabUI as RefactoredChartTabUI
from .chart_tab import DraggableOverlay

ChartTabUI = RefactoredChartTabUI

__all__ = ['ChartTabUI', 'DraggableOverlay']