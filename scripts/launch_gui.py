#!/usr/bin/env python3
"""
launch_gui.py - cross-platform Python launcher for the GUI.

Usage:
  # from project root, with .venv activated (PowerShell)
  python .\\scripts\launch_gui.py

This script calls the existing scripts/run_gui.py using the current Python interpreter
to avoid invoking a .ps1 from Python incorrectly.
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    run_gui_path = repo_root / "scripts" / "run_gui.py"
    if not run_gui_path.exists():
        print("Error: scripts/run_gui.py not found in repo root:", run_gui_path)
        sys.exit(2)

    py = sys.executable or "python"
    print("Using Python:", py)
    print("Launching GUI helper:", run_gui_path)
    # forward environment and working directory
    try:
        ret = subprocess.run([py, str(run_gui_path)], cwd=str(repo_root), shell=False)
        if ret.returncode != 0:
            print("GUI helper exited with code", ret.returncode)
            sys.exit(ret.returncode)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)
    except Exception as e:
        print("Failed to launch GUI helper:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
