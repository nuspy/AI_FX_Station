#!/usr/bin/env python3
"""
Test importing the full app module to find the blocking point
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Disable WebSocket to avoid blocking
os.environ['FOREX_ENABLE_WS'] = '0'

print("Testing full app.py module import...")

try:
    print("Step 1: About to import forex_diffusion.ui.app...")
    import forex_diffusion.ui.app
    print("Step 2: SUCCESS - forex_diffusion.ui.app imported")

    print("Step 3: About to access setup_ui function...")
    setup_ui = forex_diffusion.ui.app.setup_ui
    print("Step 4: SUCCESS - setup_ui function accessed")

    print("All imports successful!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")