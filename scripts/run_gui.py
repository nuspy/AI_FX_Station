#!/usr/bin/env python3
"""
scripts/run_gui.py

Standalone runner for the MagicForex GUI.
"""
from __future__ import annotations

import os
import sys
import argparse
import subprocess
import atexit
from pathlib import Path

# Configure transformers to use PyTorch instead of TensorFlow
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

# Disable sageattention (not used, avoid import overhead)
os.environ['DISABLE_SAGE_ATTENTION'] = '1'
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QStatusBar, QSizePolicy
from loguru import logger

# --- Global variable for the simulator process ---
simulator_process = None

# Add project root to path to allow relative imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.forex_diffusion.ui.menus import MainMenuBar
from src.forex_diffusion.ui.viewer import ViewerWidget
from src.forex_diffusion.ui.app import setup_ui

def cleanup():
    """Ensures the simulator process is terminated when the main app exits."""
    global simulator_process
    if simulator_process:
        logger.info("Terminating Tiingo WebSocket simulator...")
        simulator_process.terminate()
        simulator_process.wait()
        logger.info("Simulator terminated.")

def main():
    """Initializes the QApplication and the main window."""
    global simulator_process

    parser = argparse.ArgumentParser(description="Forex-Diffusion GUI Runner")
    parser.add_argument(
        "--testserver",
        action="store_true",
        help="Run the integrated Tiingo WebSocket simulator and connect to it."
    )
    args = parser.parse_args()

    if args.testserver:
        logger.warning("--- USING TEST SERVER MODE ---")
        try:
            simulator_script_path = project_root / "tests" / "manual_tests" / "tiingo_ws_simulator.py"
            if not simulator_script_path.exists():
                raise FileNotFoundError(f"Simulator script not found at {simulator_script_path}")
            
            simulator_process = subprocess.Popen([sys.executable, str(simulator_script_path)])
            logger.info(f"Launched Tiingo simulator process (PID: {simulator_process.pid})")
            atexit.register(cleanup)

        except Exception as e:
            logger.exception(f"Failed to start the Tiingo simulator: {e}")
            sys.exit(1)

    logger.info("Starting GUI (run_gui.py)...")

    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_win.setWindowTitle("Forex-Diffusion")
    # main_win.setGeometry(100, 100, 1600, 900)

    central_widget = QWidget()
    main_win.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)

    menu_bar = MainMenuBar(main_win)
    main_win.setMenuBar(menu_bar)

    status_bar = QStatusBar()
    status_label = QLabel("Status: Initializing...")
    status_bar.addWidget(status_label)
    main_win.setStatusBar(status_bar)

    viewer = ViewerWidget()
    
    setup_ui(
        main_window=main_win, 
        layout=layout, 
        menu_bar=menu_bar, 
        viewer=viewer, 
        status_label=status_label,
        use_test_server=args.testserver
    )
    main_win.show()
    main_win.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
