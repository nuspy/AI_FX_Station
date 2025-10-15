"""
Agentic Pattern Optimization and Backtesting System

This package provides comprehensive optimization infrastructure for pattern recognition parameters
with multi-objective optimization, regime-aware adaptation, and scalable backtesting.
"""

from .engine import OptimizationEngine
from .multi_objective import ParetoOptimizer, MultiObjectiveEvaluator
from .parameter_space import ParameterSpace, ParameterDefinition
from .backtest_runner import BacktestRunner, BacktestResult
from .regime_classifier import RegimeClassifier
from .task_manager import TaskManager, TaskStatus
from .logging_reporter import LoggingReporter
from .regime_aware_optimizer import (
    RegimeAwareOptimizer,
    RegimeAwareOptimizationResult,
    RegimeParameters
)

__all__ = [
    'OptimizationEngine',
    'ParetoOptimizer',
    'MultiObjectiveEvaluator',
    'ParameterSpace',
    'ParameterDefinition',
    'BacktestRunner',
    'BacktestResult',
    'RegimeClassifier',
    'TaskManager',
    'TaskStatus',
    'LoggingReporter',
    'RegimeAwareOptimizer',
    'RegimeAwareOptimizationResult',
    'RegimeParameters',
]