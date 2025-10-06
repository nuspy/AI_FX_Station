# backtest package initializer
from .engine import BacktestEngine

# kernc/backtesting.py integration
from .kernc_integration import (
    ForexDiffusionStrategy,
    run_backtest,
    optimize_strategy,
    generate_predictions_from_model,
    prepare_ohlcv_dataframe
)

# Optimization engines
from .genetic_optimizer import GeneticOptimizer, ParameterSpace, create_parameter_space_from_ranges
from .hybrid_optimizer import HybridOptimizer
from .optimization_db import OptimizationDB

# Integrated backtest system
from .integrated_backtest import (
    IntegratedBacktester,
    BacktestConfig,
    BacktestResult,
    Trade
)

__all__ = [
    # Original backtest engine
    "BacktestEngine",
    # kernc integration
    "ForexDiffusionStrategy",
    "run_backtest",
    "optimize_strategy",
    "generate_predictions_from_model",
    "prepare_ohlcv_dataframe",
    # Optimizers
    "GeneticOptimizer",
    "HybridOptimizer",
    "ParameterSpace",
    "create_parameter_space_from_ranges",
    # Database
    "OptimizationDB",
    # Integrated backtest
    "IntegratedBacktester",
    "BacktestConfig",
    "BacktestResult",
    "Trade"
]
