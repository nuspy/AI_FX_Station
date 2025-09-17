#!/usr/bin/env python3
"""
Quick smoke test for the PySide6 GUI.
Shows the main window and quits automatically after a short delay.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QStatusBar


def main():
    # add project root to sys.path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.forex_diffusion.ui.menus import MainMenuBar
    from src.forex_diffusion.ui.viewer import ViewerWidget
    from src.forex_diffusion.ui.app import setup_ui

    import os
    # run headless-friendly
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # skip network/backfill during smoke
    os.environ.setdefault("UI_SKIP_WS", "1")
    os.environ.setdefault("UI_SKIP_BACKFILL", "1")
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Forex-Diffusion (Smoke)")
    win.resize(1280, 800)

    central = QWidget()
    win.setCentralWidget(central)
    layout = QVBoxLayout(central)

    menu_bar = MainMenuBar(win)
    win.setMenuBar(menu_bar)

    status_bar = QStatusBar()
    status_label = QLabel("Status: Smoke startup...")
    status_bar.addWidget(status_label)
    win.setStatusBar(status_bar)

    viewer = ViewerWidget()
    setup_ui(
        main_window=win,
        layout=layout,
        menu_bar=menu_bar,
        viewer=viewer,
        status_label=status_label,
        use_test_server=False,
    )

    win.show()

    # auto-quit after 5 seconds
    QTimer.singleShot(4500, lambda: win.close())
    QTimer.singleShot(5000, app.quit)
    # hard failsafe: terminate process after 7 seconds
    QTimer.singleShot(7000, lambda: os._exit(0))  # noqa: E701
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


