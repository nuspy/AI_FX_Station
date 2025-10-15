"""
Two-Phase Training Pipeline

Implements external loop (model training) and internal loop (inference backtest)
architecture for efficient AI model optimization with regime-based selection.
"""

from .training_orchestrator import TrainingOrchestrator
from .inference_backtester import InferenceBacktester
from .regime_manager import RegimeManager
from .checkpoint_manager import CheckpointManager
from .model_file_manager import ModelFileManager
from .config_grid import compute_config_hash, generate_config_grid, validate_config
from .database import (
    TrainingRun, InferenceBacktest, RegimeDefinition,
    RegimeBestModel, TrainingQueue, get_session
)

__all__ = [
    'TrainingOrchestrator',
    'InferenceBacktester',
    'RegimeManager',
    'CheckpointManager',
    'ModelFileManager',
    'compute_config_hash',
    'generate_config_grid',
    'validate_config',
    'TrainingRun',
    'InferenceBacktest',
    'RegimeDefinition',
    'RegimeBestModel',
    'TrainingQueue',
    'get_session'
]
