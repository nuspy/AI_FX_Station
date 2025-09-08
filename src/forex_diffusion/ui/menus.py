from __future__ import annotations

from PySide6.QtWidgets import QMenuBar
from PySide6.QtCore import QObject, Signal


class MenuSignals(QObject):
    """
    Simple container of menu-related signals. Controllers can connect to these.
    Add more signals as needed by the application.
    """
    openRequested = Signal()
    saveRequested = Signal()
    settingsRequested = Signal()
    exitRequested = Signal()

    def __init__(self):
        super().__init__()


class MainMenuBar(QMenuBar):
    """
    Minimal MainMenuBar exposing a `signals` attribute to allow controllers to bind menu actions.
    Real application may populate menus and emit the signals accordingly.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = MenuSignals()
