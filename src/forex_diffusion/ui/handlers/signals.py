# ui/handlers/signals.py
# Signal definitions for the ForexGPT UI controllers
from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class UIControllerSignals(QObject):
    forecastReady = Signal(object, object)  # (pd.DataFrame, quantiles_dict)
    error = Signal(str)
    status = Signal(str)