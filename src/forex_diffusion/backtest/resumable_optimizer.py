"""
Resumable Hyperparameter Optimizer

Wraps genetic optimizer with checkpoint/resume capability.
Handles interruptions during: training → backtest → validation sequences.

Example:
    - Train model #100 ✓
    - Start backtest #100... ✗ SYSTEM CRASH
    - Resume: continues from backtest #100
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from .genetic_optimizer import GeneticOptimizer, ParameterSpace
from .kernc_integration import run_backtest
from ..training.checkpoint_manager import resume_or_create_workflow
from ..validation.multi_horizon import MultiHorizonValidator


@dataclass
class OptimizationConfig:
    """Configuration for resumable optimization"""
    workflow_id: str
    symbol: str
    timeframe: str
    horizon: int

    # Data
    data_path: Optional[Path] = None
    data_df: Optional[pd.DataFrame] = None

    # Parameter space
    param_spaces: List[ParameterSpace] = None

    # Genetic algorithm settings
    population_size: int = 50
    n_generations: int = 20
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1

    # Training settings
    training_script: str = "src.forex_diffusion.training.train"
    training_args_template: Dict[str, Any] = None

    # Backtest settings
    strategy_class: type = None
    initial_capital: float = 100000.0

    # Validation settings
    enable_validation: bool = True
    validation_horizons: List[int] = None

    # Checkpoint settings
    checkpoint_dir: Optional[Path] = None
    auto_save_interval: int = 300  # 5 minutes


class ResumableOptimizer:
    """
    Hyperparameter optimizer with automatic checkpoint/resume.

    Workflow:
    1. Generate parameter combinations (genetic algorithm)
    2. For each combination:
       a. Train model → checkpoint
       b. Run backtest → checkpoint
       c. Run validation → checkpoint
    3. Return Pareto front of solutions

    Resume behavior:
    - If interrupted during training #100: resume training #100
    - If interrupted after training #100: skip training, run backtest #100
    - If interrupted after backtest #100: skip backtest, run validation #100
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize resumable optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config

        # Initialize checkpoint manager
        self.checkpoint_mgr, self.is_resumed = resume_or_create_workflow(
            workflow_id=config.workflow_id,
            config=self._config_to_dict(),
            task_generator=self._generate_task_plan,
            checkpoint_dir=config.checkpoint_dir
        )

        # Load data
        if config.data_df is not None:
            self.data = config.data_df
        elif config.data_path:
            self.data = pd.read_csv(config.data_path)
        else:
            raise ValueError("Must provide either data_df or data_path")

        # Genetic optimizer (for initial parameter generation)
        if not self.is_resumed:
            self.ga_optimizer = GeneticOptimizer(
                strategy_class=config.strategy_class,
                data=self.data,
                param_spaces=config.param_spaces,
                population_size=config.population_size,
                n_generations=config.n_generations,
                crossover_prob=config.crossover_prob,
                mutation_prob=config.mutation_prob
            )

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'workflow_id': self.config.workflow_id,
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'horizon': self.config.horizon,
            'population_size': self.config.population_size,
            'n_generations': self.config.n_generations,
        }

    def _generate_task_plan(self, config: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Generate task plan for workflow.

        Returns:
            List of (task_id, task_type, params) tuples
        """
        logger.info("Generating optimization task plan...")

        # Run genetic algorithm to get parameter combinations
        # (This is fast - just generates combinations, doesn't evaluate)
        ga_result = self.ga_optimizer.optimize(verbose=False)

        tasks = []
        for i, solution in enumerate(ga_result['pareto_front']):
            task_prefix = f"task_{i:03d}"

            # Task 1: Training
            tasks.append((
                f"{task_prefix}_train",
                "training",
                {
                    'model_params': solution['params'],
                    'symbol': self.config.symbol,
                    'timeframe': self.config.timeframe,
                    'horizon': self.config.horizon,
                }
            ))

            # Task 2: Backtest
            tasks.append((
                f"{task_prefix}_backtest",
                "backtest",
                {
                    'model_params': solution['params'],
                    'strategy_class': self.config.strategy_class.__name__,
                    'initial_capital': self.config.initial_capital,
                }
            ))

            # Task 3: Validation (if enabled)
            if self.config.enable_validation:
                tasks.append((
                    f"{task_prefix}_validation",
                    "validation",
                    {
                        'model_params': solution['params'],
                        'validation_horizons': self.config.validation_horizons or [self.config.horizon],
                    }
                ))

        logger.info(f"Generated {len(tasks)} tasks for {len(ga_result['pareto_front'])} parameter combinations")
        return tasks

    def run(self, callback: Optional[Callable[[str, Dict], None]] = None) -> Dict[str, Any]:
        """
        Run optimization with automatic checkpointing.

        Args:
            callback: Optional callback(task_id, progress_info) for progress updates

        Returns:
            Optimization results
        """
        if self.is_resumed:
            logger.info(f"Resuming optimization '{self.config.workflow_id}'")
            progress = self.checkpoint_mgr.get_progress()
            logger.info(f"Progress: {progress['completed']}/{progress['total']} tasks completed")

        start_time = time.time()

        while True:
            # Get next task
            task = self.checkpoint_mgr.get_next_task()

            if task is None:
                # All tasks complete
                break

            logger.info(f"Executing task: {task.task_id} ({task.task_type})")

            # Start task
            self.checkpoint_mgr.start_task(task.task_id)

            try:
                # Execute task based on type
                if task.task_type == "training":
                    result, artifacts = self._run_training(task)
                elif task.task_type == "backtest":
                    result, artifacts = self._run_backtest(task)
                elif task.task_type == "validation":
                    result, artifacts = self._run_validation(task)
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")

                # Mark task complete
                self.checkpoint_mgr.complete_task(task.task_id, result, artifacts)

                # Progress callback
                if callback:
                    callback(task.task_id, self.checkpoint_mgr.get_progress())

            except Exception as e:
                logger.exception(f"Task {task.task_id} failed: {e}")
                self.checkpoint_mgr.fail_task(task.task_id, str(e))

                # Decide: continue with next task or fail entire workflow?
                # For now, continue with next task
                continue

            # Auto-save checkpoint
            self.checkpoint_mgr.auto_save()

        # Collect results
        elapsed = time.time() - start_time
        results = self.checkpoint_mgr.get_results()

        logger.info(f"Optimization complete in {elapsed:.1f}s: {len(results)} results")

        return {
            'workflow_id': self.config.workflow_id,
            'results': results,
            'progress': self.checkpoint_mgr.get_progress(),
            'elapsed_time': elapsed,
        }

    def _run_training(self, task: 'TaskCheckpoint') -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Run model training.

        Returns:
            Tuple of (result_dict, artifacts_dict)
        """
        import subprocess
        import sys

        params = task.params
        model_params = params['model_params']

        # Build training command
        args = [
            sys.executable, '-m', self.config.training_script,
            '--symbol', params['symbol'],
            '--timeframe', params['timeframe'],
            '--horizon', str(params['horizon']),
        ]

        # Add model parameters
        for key, value in model_params.items():
            args.extend([f'--{key}', str(value)])

        # Add template args
        if self.config.training_args_template:
            for key, value in self.config.training_args_template.items():
                args.extend([f'--{key}', str(value)])

        # Run training
        logger.info(f"Running training: {' '.join(args)}")
        result = subprocess.run(args, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")

        # Parse output for model path
        model_path = self._extract_model_path(result.stdout)

        return {
            'success': True,
            'model_path': model_path,
            'stdout': result.stdout[-1000:],  # Last 1000 chars
        }, {
            'model': model_path
        }

    def _run_backtest(self, task: 'TaskCheckpoint') -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Run backtest on trained model.

        Returns:
            Tuple of (result_dict, artifacts_dict)
        """
        # Get model from previous training task
        train_task_id = task.task_id.replace('_backtest', '_train')
        train_task = self.checkpoint_mgr._find_task(train_task_id)

        if not train_task or not train_task.artifacts.get('model'):
            raise RuntimeError(f"Training task {train_task_id} not completed or missing model")

        model_path = train_task.artifacts['model']

        # Run backtest
        backtest_result = run_backtest(
            strategy_class=self.config.strategy_class,
            data=self.data,
            model_path=model_path,
            initial_capital=task.params['initial_capital']
        )

        return {
            'success': True,
            'return': backtest_result.get('return', 0.0),
            'sharpe_ratio': backtest_result.get('sharpe_ratio', 0.0),
            'max_drawdown': backtest_result.get('max_drawdown', 0.0),
            'n_trades': backtest_result.get('n_trades', 0),
        }, {}

    def _run_validation(self, task: 'TaskCheckpoint') -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Run multi-horizon validation.

        Returns:
            Tuple of (result_dict, artifacts_dict)
        """
        # Get model from training task
        train_task_id = task.task_id.replace('_validation', '_train')
        train_task = self.checkpoint_mgr._find_task(train_task_id)

        if not train_task or not train_task.artifacts.get('model'):
            raise RuntimeError(f"Training task {train_task_id} not completed or missing model")

        model_path = train_task.artifacts['model']

        # Run validation
        validator = MultiHorizonValidator(
            model_path=model_path,
            data=self.data,
            horizons=task.params['validation_horizons']
        )

        validation_result = validator.validate()

        return {
            'success': True,
            'optimal_horizon': validation_result.get('optimal_horizon'),
            'horizon_metrics': validation_result.get('metrics_by_horizon', {}),
        }, {}

    def _extract_model_path(self, stdout: str) -> str:
        """Extract model path from training stdout"""
        # Look for line like: "[OK] saved checkpoint to /path/to/model.ckpt"
        for line in stdout.split('\n'):
            if 'saved checkpoint to' in line.lower() or '[ok]' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.endswith('.ckpt') or part.endswith('.pkl'):
                        return part

        raise RuntimeError("Could not extract model path from training output")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress"""
        return self.checkpoint_mgr.get_progress()

    def get_results(self) -> List[Dict[str, Any]]:
        """Get results from completed tasks"""
        return self.checkpoint_mgr.get_results()

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        Extract Pareto front from completed backtests.

        Returns:
            List of Pareto-optimal solutions
        """
        results = self.get_results()

        # Extract backtest results
        backtest_results = [
            r for r in results
            if r['task_type'] == 'backtest' and r['result']['success']
        ]

        if not backtest_results:
            return []

        # Compute Pareto front (minimize drawdown, maximize return and Sharpe)
        pareto_solutions = []

        for result in backtest_results:
            is_dominated = False

            for other in backtest_results:
                if other == result:
                    continue

                # Check if 'other' dominates 'result'
                better_return = other['result']['return'] >= result['result']['return']
                better_sharpe = other['result']['sharpe_ratio'] >= result['result']['sharpe_ratio']
                better_dd = other['result']['max_drawdown'] <= result['result']['max_drawdown']

                # At least one strictly better
                strictly_better = (
                    other['result']['return'] > result['result']['return'] or
                    other['result']['sharpe_ratio'] > result['result']['sharpe_ratio'] or
                    other['result']['max_drawdown'] < result['result']['max_drawdown']
                )

                if better_return and better_sharpe and better_dd and strictly_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_solutions.append(result)

        logger.info(f"Pareto front: {len(pareto_solutions)} solutions out of {len(backtest_results)}")

        return pareto_solutions


# Convenience function
def run_resumable_optimization(
    workflow_id: str,
    symbol: str,
    timeframe: str,
    horizon: int,
    data: pd.DataFrame,
    param_spaces: List[ParameterSpace],
    strategy_class: type,
    **kwargs
) -> Dict[str, Any]:
    """
    Run resumable hyperparameter optimization.

    Args:
        workflow_id: Unique workflow identifier
        symbol: Trading symbol
        timeframe: Timeframe
        horizon: Forecast horizon
        data: OHLCV data
        param_spaces: Parameter spaces to optimize
        strategy_class: Backtest strategy class
        **kwargs: Additional configuration options

    Returns:
        Optimization results
    """
    config = OptimizationConfig(
        workflow_id=workflow_id,
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        data_df=data,
        param_spaces=param_spaces,
        strategy_class=strategy_class,
        **kwargs
    )

    optimizer = ResumableOptimizer(config)
    return optimizer.run()
