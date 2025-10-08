#!/usr/bin/env python
"""Test imports to find the error"""

try:
    print("Testing chart_controller import...")
    from src.forex_diffusion.ui.chart_components.controllers.chart_controller import ChartTabController
    print("SUCCESS: chart_controller imported")
except Exception as e:
    print(f"ERROR in chart_controller: {e}")
    import traceback
    traceback.print_exc()
