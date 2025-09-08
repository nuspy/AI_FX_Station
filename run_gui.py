#!/usr/bin/env python3
"""
Run GUI helper: executes the UI module using package context.

Usage:
  python scripts/run_gui.py

This ensures relative imports inside src.forex_diffusion.ui.* work correctly
regardless of the current working directory.
"""
import runpy
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path (parent of scripts/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Run the UI module as a script within package context
runpy.run_module("src.forex_diffusion.ui.app", run_name="__main__")
