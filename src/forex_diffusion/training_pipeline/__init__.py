# src/forex_diffusion/training_pipeline/__init__.py
"""
New AI Training Pipeline - Two-Phase Training System

This module implements an efficient two-phase training and backtesting architecture:
- Phase 1 (External Loop): Train models with different configurations
- Phase 2 (Internal Loop): Backtest inference strategies on trained models

Key Components:
- TrainingOrchestrator: Manages the training grid and coordinates phases
- InferenceBacktester: Runs fast backtests on trained models
- RegimeManager: Tracks and evaluates performance by market regime
- CheckpointManager: Handles interruption and resume logic
"""

__version__ = '1.0.0'

from .training_orchestrator import TrainingOrchestrator
from .inference_backtester import InferenceBacktester
from .regime_manager import RegimeManager
from .checkpoint_manager import CheckpointManager
from .model_file_manager import ModelFileManager
from .database_models import (
    TrainingRun,
    InferenceBacktest,
    RegimeDefinition,
    RegimeBestModel,
    TrainingQueue
)

__all__ = [
    'TrainingOrchestrator',
    'InferenceBacktester',
    'RegimeManager',
    'CheckpointManager',
    'ModelFileManager',
    'TrainingRun',
    'InferenceBacktest',
    'RegimeDefinition',
    'RegimeBestModel',
    'TrainingQueue',
]
