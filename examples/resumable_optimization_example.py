"""
Example: Resumable Hyperparameter Optimization

Demonstrates how the checkpoint/resume system works:

Scenario:
1. Start optimization with 100 models
2. System crashes after training model #50 and during backtest #50
3. Restart script → automatically resumes from backtest #50
4. Continues until completion

Usage:
    python examples/resumable_optimization_example.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from loguru import logger

from forex_diffusion.backtest.resumable_optimizer import (
    ResumableOptimizer,
    OptimizationConfig,
    run_resumable_optimization
)
from forex_diffusion.backtest.genetic_optimizer import ParameterSpace


def generate_sample_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1H')

    # Random walk price
    returns = np.random.randn(n_samples) * 0.001
    price = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.randn(n_samples) * 0.0002),
        'high': price * (1 + np.abs(np.random.randn(n_samples)) * 0.0005),
        'low': price * (1 - np.abs(np.random.randn(n_samples)) * 0.0005),
        'close': price,
        'volume': np.random.randint(1000, 10000, n_samples),
    })

    return data


# Define a simple strategy class (mock for example)
class SimpleStrategy:
    """Simple mock strategy for demonstration"""
    def __init__(self, **params):
        self.params = params


def progress_callback(task_id: str, progress: dict):
    """Callback to display progress"""
    logger.info(
        f"Task {task_id} completed | "
        f"Progress: {progress['completed']}/{progress['total']} "
        f"({progress['progress']*100:.1f}%)"
    )


def main():
    """Run resumable optimization example"""

    # Configuration
    WORKFLOW_ID = "example_optimization_eurusd_1h"
    SYMBOL = "EUR/USD"
    TIMEFRAME = "1h"
    HORIZON = 24  # 24 hours ahead

    logger.info("=" * 80)
    logger.info("Resumable Hyperparameter Optimization Example")
    logger.info("=" * 80)
    logger.info(f"Workflow ID: {WORKFLOW_ID}")
    logger.info(f"Symbol: {SYMBOL}, Timeframe: {TIMEFRAME}, Horizon: {HORIZON}")
    logger.info("")

    # Generate sample data
    logger.info("Generating sample data...")
    data = generate_sample_data(n_samples=5000)
    logger.info(f"Data shape: {data.shape}")

    # Define parameter spaces to optimize
    param_spaces = [
        ParameterSpace(name="learning_rate", min_value=0.0001, max_value=0.01, log_scale=True),
        ParameterSpace(name="batch_size", min_value=16, max_value=128, step=16),
        ParameterSpace(name="n_layers", min_value=2, max_value=6, step=1),
        ParameterSpace(name="dropout", min_value=0.1, max_value=0.5, step=0.1),
    ]

    logger.info(f"Parameter spaces: {[p.name for p in param_spaces]}")
    logger.info("")

    # Create optimization configuration
    config = OptimizationConfig(
        workflow_id=WORKFLOW_ID,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        horizon=HORIZON,
        data_df=data,
        param_spaces=param_spaces,
        strategy_class=SimpleStrategy,
        population_size=10,  # Small for demonstration
        n_generations=5,     # Small for demonstration
        enable_validation=True,
        validation_horizons=[12, 24, 48],
        auto_save_interval=30,  # Save every 30 seconds
    )

    # Create optimizer
    optimizer = ResumableOptimizer(config)

    # Check if resuming
    if optimizer.is_resumed:
        logger.warning("🔄 RESUMING from previous checkpoint!")
        progress = optimizer.get_progress()
        logger.info(f"   Previous progress: {progress['completed']}/{progress['total']} tasks completed")
        logger.info(f"   Current stage: {progress['stage']}")
        logger.info("")
    else:
        logger.info("✨ Starting NEW optimization workflow")
        logger.info("")

    # Run optimization
    logger.info("Starting optimization...")
    logger.info("=" * 80)

    try:
        result = optimizer.run(callback=progress_callback)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ Optimization COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Total tasks: {result['progress']['total']}")
        logger.info(f"Completed: {result['progress']['completed']}")
        logger.info(f"Failed: {result['progress']['failed']}")
        logger.info(f"Elapsed time: {result['elapsed_time']:.1f}s")

        # Get Pareto front
        pareto_front = optimizer.get_pareto_front()
        logger.info(f"Pareto front: {len(pareto_front)} optimal solutions")

        if pareto_front:
            logger.info("")
            logger.info("Top 3 Pareto solutions:")
            for i, solution in enumerate(pareto_front[:3], 1):
                logger.info(f"  {i}. Params: {solution['params']}")
                logger.info(f"     Return: {solution['result']['return']:.2f}%")
                logger.info(f"     Sharpe: {solution['result']['sharpe_ratio']:.2f}")
                logger.info(f"     Max DD: {solution['result']['max_drawdown']:.2f}%")

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning("⚠️ Optimization INTERRUPTED by user")
        logger.warning("=" * 80)

        progress = optimizer.get_progress()
        logger.info(f"Progress at interruption: {progress['completed']}/{progress['total']} tasks")
        logger.info(f"Current stage: {progress['stage']}")
        logger.info("")
        logger.info("💡 To resume, run this script again!")
        logger.info(f"   Workflow ID: {WORKFLOW_ID}")

        return 1

    except Exception as e:
        logger.exception("❌ Optimization FAILED!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
