# ui/controllers.py
# Compatibility module - imports from refactored modules for backward compatibility
from __future__ import annotations

# Import all components from their new locations
from .controllers.ui_controller import UIController, _preimport_sklearn_for_unpickle
from .controllers.training_controller import TrainingController, TrainingControllerSignals
from .handlers.signals import UIControllerSignals
from .workers.forecast_worker import ForecastWorker

# Maintain backward compatibility
__all__ = [
    "UIController",
    "UIControllerSignals",
    "ForecastWorker",
    "TrainingController",
    "TrainingControllerSignals",
    "_preimport_sklearn_for_unpickle"
]