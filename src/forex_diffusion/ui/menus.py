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
    ingestRequested = Signal()
    trainRequested = Signal()
    forecastRequested = Signal()
    calibrationRequested = Signal()
    backtestRequested = Signal()
    realtimeToggled = Signal(bool)
    configRequested = Signal()
    predictionSettingsRequested = Signal()

    def __init__(self):
        super().__init__()


class MainMenuBar(QMenuBar):
    """
    MainMenuBar exposing a `signals` attribute to allow controllers to bind menu actions.
    Populates menus and emits signals accordingly.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = MenuSignals()

        # File Menu
        file_menu = self.addMenu("&File")
        file_menu.addAction("&Open", self.signals.openRequested.emit)
        file_menu.addAction("&Save", self.signals.saveRequested.emit)
        file_menu.addSeparator()
        file_menu.addAction("&Settings", self.signals.settingsRequested.emit)
        file_menu.addSeparator()
        file_menu.addAction("&Exit", self.signals.exitRequested.emit)

        # Data Menu
        data_menu = self.addMenu("&Data")
        data_menu.addAction("&Ingest/Backfill", self.signals.ingestRequested.emit)

        # Model Menu
        model_menu = self.addMenu("&Model")
        model_menu.addAction("&Train", self.signals.trainRequested.emit)
        model_menu.addAction("&Calibrate", self.signals.calibrationRequested.emit)
        model_menu.addAction("&Backtest", self.signals.backtestRequested.emit)

        # Prediction Menu
        prediction_menu = self.addMenu("&Prediction")
        prediction_menu.addAction("Prediction &Settings...", self.signals.predictionSettingsRequested.emit)
        prediction_menu.addAction("&Make Prediction", self.signals.forecastRequested.emit)

        # Realtime Menu
        realtime_menu = self.addMenu("&Realtime")
        realtime_action = realtime_menu.addAction("&Enable Realtime")
        realtime_action.setCheckable(True)
        realtime_action.toggled.connect(self.signals.realtimeToggled.emit)
