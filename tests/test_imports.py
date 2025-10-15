#!/usr/bin/env python3
"""
Test individual imports from app.py to find the blocking one
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Disable WebSocket to avoid blocking
os.environ['FOREX_ENABLE_WS'] = '0'

print("Testing app.py imports individually...")

imports_to_test = [
    ("Basic imports", ["os", "typing"]),
    ("Loguru", ["loguru"]),
    ("PySide6", ["PySide6.QtWidgets"]),
    ("DBService", ["forex_diffusion.services.db_service"]),
    ("DBWriter", ["forex_diffusion.services.db_writer"]),
    ("MarketDataService", ["forex_diffusion.services.marketdata"]),
    ("TiingoWSConnector", ["forex_diffusion.services.tiingo_ws_connector"]),
    ("AggregatorService", ["forex_diffusion.services.aggregator"]),
    ("UIController", ["forex_diffusion.ui.controllers"]),
    ("TrainingTab", ["forex_diffusion.ui.training_tab"]),
    ("SignalsTab", ["forex_diffusion.ui.signals_tab"]),
    ("ChartTabUI", ["forex_diffusion.ui.chart_tab_ui"]),
    ("BacktestingTab", ["forex_diffusion.ui.backtesting_tab"]),
]

for description, modules in imports_to_test:
    print(f"\nTesting {description}...")
    try:
        for module in modules:
            print(f"  Importing {module}...")
            __import__(module)
            print(f"  SUCCESS: {module}")
        print(f"SUCCESS: {description} imports completed")
    except Exception as e:
        print(f"ERROR: {description} failed - {e}")
        print(f"Blocking module: {module}")
        import traceback
        traceback.print_exc()
        break  # Stop at first failing import

print("\nImport testing completed")