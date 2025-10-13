"""
Core optimization engine for agentic pattern parameter optimization.

This module provides the main orchestration for multi-objective pattern optimization
with support for resume/idempotency, parallel execution, and regime-aware optimization.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .multi_objective import ParetoOptimizer, MultiObjectiveEvaluator
from .parameter_space import ParameterSpace
from .backtest_runner import BacktestRunner, BacktestResult
from .regime_classifier import RegimeClassifier
from .task_manager import TaskManager, TaskStatus
from .invalidation_rules import InvalidationRuleEngine
from .early_stopping import EarlyStoppingManager
from .logging_reporter import LoggingReporter
from .genetic_algorithm import GeneticAlgorithm, GAConfig

# Database models
from ...services.db_service import DBService

# Parameter validator
from ...patterns.parameter_validator import validate_parameters

@dataclass
class OptimizationConfig:
    """Configuration for optimization studies"""

    # Matrix dimensions
    pattern_key: str
    direction: str  # bull, bear
    asset: str
    timeframe: str
    regime_tag: Optional[str] = None

    # Dataset configuration
    dataset_1_config: Dict[str, Any] = field(default_factory=dict)
    dataset_2_config: Optional[Dict[str, Any]] = None

    # Multi-objective settings
    is_multi_objective: bool = True
    objective_weights: Optional[Dict[str, float]] = None

    # Parameter space
    parameter_ranges: Dict[str, Any] = field(default_factory=dict)

    # Optimization limits
    max_trials: int = 1000
    max_duration_hours: float = 24.0
    max_parallel_workers: int = 32

    # Early stopping
    early_stopping_alpha: float = 0.8
    min_trades_for_pruning: int = 10
    min_duration_months: int = 3

    # Invalidation rules
    k_time_multiplier: float = 4.0
    k_loss_multiplier: float = 4.0
    quantile_threshold: float = 0.75

    # Walk-forward validation
    walk_forward_months: int = 6
    purge_days: int = 1
    embargo_days: int = 2

    # Constraints
    min_signals_required: int = 20
    min_temporal_coverage_months: int = 6
    max_drawdown_threshold: float = 0.20

    # Recency weighting
    recency_decay_months: float = 12.0
    max_history_years: int = 10

@dataclass
class OptimizationResult:
    """Results from an optimization study"""

    study_id: int
    total_trials: int
    completed_trials: int
    pruned_trials: int
    failed_trials: int

    best_parameters: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)

    execution_time_seconds: float = 0.0
    convergence_metrics: Dict[str, Any] = field(default_factory=dict)

    # Per-dataset results
    d1_metrics: Optional[Dict[str, Any]] = None
    d2_metrics: Optional[Dict[str, Any]] = None

    # Regime breakdown
    regime_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class OptimizationEngine:
    """
    Main orchestration engine for agentic pattern optimization.

    Handles:
    - N-dimensional matrix management (pattern x direction x asset x timeframe x regime)
    - Multi-objective optimization with Pareto frontiers
    - Resume/idempotency via TaskID hashing
    - Parallel execution across 32 threads
    - Dynamic early stopping and pruning
    - Parameter promotion and rollback
    """

    def __init__(self, db_service: DBService, data_root: Path):
        self.db_service = db_service
        self.data_root = Path(data_root)

        # Core components
        self.parameter_space = ParameterSpace()
        self.backtest_runner = BacktestRunner()
        self.regime_classifier = RegimeClassifier()
        self.task_manager = TaskManager(db_service)
        self.pareto_optimizer = ParetoOptimizer()

        # Resource management
        from .resource_manager import ResourceManager, Priority
        self.resource_manager = ResourceManager()
        self.is_running = False
        self.is_paused = False
        self.pause_reason = None
        self.current_allocation = None
        self.evaluator = MultiObjectiveEvaluator()
        self.invalidation_engine = InvalidationRuleEngine()
        self.early_stopping = EarlyStoppingManager()

        # State tracking
        self._active_studies: Dict[int, OptimizationConfig] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._shutdown_requested = False

        logger.info("OptimizationEngine initialized")

    def create_study(self, config: OptimizationConfig) -> int:
        """
        Create a new optimization study for the given matrix cell.

        Args:
            config: Study configuration

        Returns:
            Study ID
        """
        # Create study in database
        study_id = self.task_manager.create_study(
            pattern_key=config.pattern_key,
            direction=config.direction,
            asset=config.asset,
            timeframe=config.timeframe,
            regime_tag=config.regime_tag,
            config=config
        )

        # Initialize parameter space for this study
        self.parameter_space.initialize_ranges(
            pattern_key=config.pattern_key,
            timeframe=config.timeframe,
            custom_ranges=config.parameter_ranges
        )

        logger.info(
            f"Created optimization study {study_id} for "
            f"{config.pattern_key}_{config.direction}_{config.asset}_{config.timeframe}"
            f"{'_' + config.regime_tag if config.regime_tag else ''}"
        )

        return study_id

    async def run_study(self, study_id: int) -> OptimizationResult:
        """
        Execute an optimization study with parallel trial execution.

        Args:
            study_id: Study identifier

        Returns:
            Optimization results
        """
        start_time = time.time()

        # Load study configuration
        config = self.task_manager.get_study_config(study_id)
        if not config:
            raise ValueError(f"Study {study_id} not found")

        self._active_studies[study_id] = config

        try:
            # Update study status
            self.task_manager.update_study_status(study_id, "running")

            # Initialize optimization components
            await self._initialize_study(study_id, config)

            # Generate and execute trials
            result = await self._execute_trials(study_id, config)

            # Post-process results
            await self._finalize_study(study_id, result)

            result.execution_time_seconds = time.time() - start_time

            logger.info(
                f"Study {study_id} completed: {result.completed_trials} trials, "
                f"best score: {result.best_score:.4f}"
            )

            return result

        except Exception as e:
            logger.exception(f"Study {study_id} failed: {e}")
            self.task_manager.update_study_status(study_id, "failed")
            raise

        finally:
            self._active_studies.pop(study_id, None)

    async def _initialize_study(self, study_id: int, config: OptimizationConfig) -> None:
        """Initialize study components and validate datasets."""

        # Validate dataset configurations
        for dataset_id, dataset_config in [("D1", config.dataset_1_config),
                                          ("D2", config.dataset_2_config)]:
            if dataset_config:
                await self._validate_dataset(dataset_id, dataset_config)

        # Initialize regime classifications if needed
        if config.regime_tag:
            await self.regime_classifier.ensure_classifications(
                start_date=datetime.now() - timedelta(days=config.max_history_years * 365),
                end_date=datetime.now(),
                regime_tag=config.regime_tag
            )

        # Initialize invalidation rule thresholds
        await self.invalidation_engine.initialize_thresholds(
            pattern_key=config.pattern_key,
            direction=config.direction,
            asset=config.asset,
            timeframe=config.timeframe,
            quantile=config.quantile_threshold
        )

    async def _execute_trials(self, study_id: int, config: OptimizationConfig) -> OptimizationResult:
        """Execute optimization trials with parallel workers."""

        result = OptimizationResult(study_id=study_id, total_trials=0,
                                   completed_trials=0, pruned_trials=0, failed_trials=0)

        # Setup parallel execution
        max_workers = min(config.max_parallel_workers, 32)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            # Generate trial candidates using quasi-random sampling
            trial_queue = await self._generate_trial_queue(study_id, config)
            result.total_trials = len(trial_queue)

            # Execute trials in parallel batches
            active_futures = []
            trial_results = []

            while trial_queue or active_futures:
                # Submit new trials up to worker limit
                while len(active_futures) < max_workers and trial_queue:
                    trial_params = trial_queue.pop(0)
                    future = self._executor.submit(
                        self._execute_single_trial, study_id, config, trial_params
                    )
                    active_futures.append((future, trial_params))

                # Process completed trials
                if active_futures:
                    completed_futures = []
                    for future, trial_params in active_futures:
                        if future.done():
                            completed_futures.append((future, trial_params))

                    for future, trial_params in completed_futures:
                        active_futures.remove((future, trial_params))

                        try:
                            trial_result = future.result()
                            trial_results.append(trial_result)

                            if trial_result.status == TaskStatus.COMPLETED:
                                result.completed_trials += 1
                            elif trial_result.status == TaskStatus.PRUNED:
                                result.pruned_trials += 1
                            else:
                                result.failed_trials += 1

                            # Check early stopping conditions
                            if await self._should_stop_early(study_id, trial_results):
                                logger.info(f"Early stopping triggered for study {study_id}")
                                trial_queue.clear()  # Stop submitting new trials

                        except Exception as e:
                            logger.exception(f"Trial execution failed: {e}")
                            result.failed_trials += 1

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(0.1)

                # Check time limit
                if time.time() - self._study_start_time > config.max_duration_hours * 3600:
                    logger.warning(f"Study {study_id} hit time limit")
                    break

            # Process final results
            await self._process_final_results(study_id, config, trial_results, result)

        finally:
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

        return result

    def _execute_single_trial(self, study_id: int, config: OptimizationConfig,
                             trial_params: Dict[str, Any]) -> Any:
        """Execute a single optimization trial (runs in thread pool)."""

        trial_id = None
        try:
            # Create trial record with deterministic TaskID
            trial_id = self.task_manager.create_trial(
                study_id=study_id,
                form_params=trial_params["form_parameters"],
                action_params=trial_params["action_parameters"],
                trial_number=trial_params["trial_number"]
            )

            # Update trial status
            self.task_manager.update_trial_status(trial_id, TaskStatus.RUNNING)

            # Execute backtest on both datasets
            metrics = {}

            for dataset_id, dataset_config in [("D1", config.dataset_1_config),
                                              ("D2", config.dataset_2_config)]:
                if dataset_config:
                    dataset_metrics = self._run_backtest_for_dataset(
                        trial_id, config, trial_params, dataset_id, dataset_config
                    )
                    metrics[dataset_id] = dataset_metrics

            # Apply invalidation rules
            metrics = self.invalidation_engine.apply_rules(
                metrics, config.k_time_multiplier, config.k_loss_multiplier
            )

            # Calculate multi-objective scores
            if config.is_multi_objective and len(metrics) > 1:
                scores = self.evaluator.evaluate_multi_objective(metrics)
            else:
                scores = self.evaluator.evaluate_single_objective(
                    metrics, config.objective_weights
                )

            # Store trial metrics
            for dataset_id, dataset_metrics in metrics.items():
                self.task_manager.store_trial_metrics(
                    trial_id, dataset_id, dataset_metrics, scores.get(dataset_id, {})
                )

            # Update trial completion
            self.task_manager.update_trial_status(trial_id, TaskStatus.COMPLETED)

            return {
                "trial_id": trial_id,
                "status": TaskStatus.COMPLETED,
                "metrics": metrics,
                "scores": scores
            }

        except Exception as e:
            logger.exception(f"Trial {trial_id} failed: {e}")
            if trial_id:
                self.task_manager.update_trial_status(trial_id, TaskStatus.FAILED, str(e))
            return {
                "trial_id": trial_id,
                "status": TaskStatus.FAILED,
                "error": str(e)
            }

    def _run_backtest_for_dataset(self, trial_id: int, config: OptimizationConfig,
                                 trial_params: Dict[str, Any], dataset_id: str,
                                 dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute backtest for a specific dataset."""

        # Load historical data
        data = self._load_dataset(config.asset, config.timeframe, dataset_config)

        # Create pattern detector with trial parameters
        detector = self._create_detector(
            config.pattern_key, trial_params["form_parameters"]
        )

        # Run backtest with walk-forward validation
        backtest_result = self.backtest_runner.run_walk_forward(
            data=data,
            detector=detector,
            action_params=trial_params["action_parameters"],
            walk_forward_months=config.walk_forward_months,
            purge_days=config.purge_days,
            embargo_days=config.embargo_days
        )

        # Calculate recency-weighted metrics
        recency_metrics = self._calculate_recency_weighted_metrics(
            backtest_result, config.recency_decay_months
        )

        # Merge all metrics
        metrics = {
            **backtest_result.metrics,
            **recency_metrics,
            "temporal_coverage_months": backtest_result.temporal_coverage_months,
            "total_signals": backtest_result.total_signals
        }

        return metrics

    async def _generate_trial_queue(self, study_id: int,
                                   config: OptimizationConfig) -> List[Dict[str, Any]]:
        """Generate queue of trial parameter combinations."""

        # Check for existing incomplete trials (resume functionality)
        existing_trials = self.task_manager.get_incomplete_trials(study_id)
        trial_queue = []

        # Resume incomplete trials first
        for trial in existing_trials:
            trial_queue.append({
                "trial_number": trial.trial_number,
                "form_parameters": trial.form_parameters,
                "action_parameters": trial.action_parameters,
                "is_resume": True
            })

        # Generate new trials using quasi-random sampling
        remaining_trials = config.max_trials - len(trial_queue)
        if remaining_trials > 0:
            new_trials = self.parameter_space.generate_sobol_samples(
                config.pattern_key, remaining_trials
            )

            for i, (form_params, action_params) in enumerate(new_trials):
                trial_queue.append({
                    "trial_number": len(trial_queue) + 1,
                    "form_parameters": form_params,
                    "action_parameters": action_params,
                    "is_resume": False
                })

        logger.info(f"Generated {len(trial_queue)} trials for study {study_id}")
        return trial_queue

    async def _should_stop_early(self, study_id: int,
                                trial_results: List[Dict[str, Any]]) -> bool:
        """Check if early stopping conditions are met."""

        if len(trial_results) < 10:  # Need minimum trials
            return False

        config = self._active_studies.get(study_id)
        if not config:
            return False

        return self.early_stopping.should_stop(
            trial_results, config.early_stopping_alpha,
            config.min_trades_for_pruning, config.min_duration_months
        )

    async def _process_final_results(self, study_id: int, config: OptimizationConfig,
                                    trial_results: List[Dict[str, Any]],
                                    result: OptimizationResult) -> None:
        """Process and store final optimization results."""

        completed_trials = [t for t in trial_results if t["status"] == TaskStatus.COMPLETED]

        if not completed_trials:
            logger.warning(f"Study {study_id} has no completed trials")
            return

        # Calculate Pareto frontier for multi-objective
        if config.is_multi_objective and config.dataset_2_config:
            pareto_front = self.pareto_optimizer.calculate_pareto_front(completed_trials)
            result.pareto_front = pareto_front

            # Select best compromise solution
            best_trial = self.pareto_optimizer.select_best_compromise(pareto_front)
        else:
            # Single objective - select best by combined score
            best_trial = max(completed_trials,
                           key=lambda t: t["scores"].get("combined_score", 0))

        if best_trial:
            result.best_parameters = {
                "form_parameters": best_trial.get("form_parameters", {}),
                "action_parameters": best_trial.get("action_parameters", {})
            }
            result.best_score = best_trial["scores"].get("combined_score", 0)

            # Store best parameters
            await self.task_manager.store_best_parameters(
                study_id, best_trial, config.regime_tag
            )

        # Calculate convergence metrics
        result.convergence_metrics = self._calculate_convergence_metrics(trial_results)

        # Update study status
        self.task_manager.update_study_status(study_id, "completed")

    async def _validate_dataset(self, dataset_id: str, dataset_config: Dict[str, Any]) -> None:
        """Validate dataset configuration and availability."""
        # Implementation would check data availability, format, etc.
        pass

    def _load_dataset(self, asset: str, timeframe: str,
                     dataset_config: Dict[str, Any]) -> pd.DataFrame:
        """Load historical data for backtesting."""
        # Implementation would load from database or files
        # This is a placeholder
        return pd.DataFrame()

    def _create_detector(self, pattern_key: str,
                        form_params: Dict[str, Any]) -> Any:
        """Create pattern detector with specified parameters."""
        # Implementation would create appropriate detector instance
        # This is a placeholder
        pass

    def _calculate_recency_weighted_metrics(self, backtest_result: Any,
                                          decay_months: float) -> Dict[str, Any]:
        """Calculate recency-weighted performance metrics."""
        # Implementation would apply exponential decay weighting
        return {}

    def _calculate_convergence_metrics(self, trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate optimization convergence metrics."""
        if not trial_results:
            return {}

        scores = [t["scores"].get("combined_score", 0) for t in trial_results
                 if t["status"] == TaskStatus.COMPLETED]

        if not scores:
            return {}

        return {
            "best_score_progression": scores,
            "final_best_score": max(scores),
            "score_variance": np.var(scores),
            "convergence_rate": len(scores) / len(trial_results)
        }

    def get_study_status(self, study_id: int) -> Dict[str, Any]:
        """Get current status and progress of optimization study."""
        return self.task_manager.get_study_status(study_id)

    def pause_study(self, study_id: int) -> None:
        """Pause an ongoing optimization study."""
        self.task_manager.update_study_status(study_id, "paused")
        logger.info(f"Study {study_id} paused")

    def resume_study(self, study_id: int) -> None:
        """Resume a paused optimization study."""
        self.task_manager.update_study_status(study_id, "running")
        logger.info(f"Study {study_id} resumed")

    def stop_study(self, study_id: int) -> None:
        """Stop an optimization study."""
        self._shutdown_requested = True
        self.task_manager.update_study_status(study_id, "completed")
        logger.info(f"Study {study_id} stopped")

    def promote_parameters(self, study_id: int, regime_tag: Optional[str] = None,
                          reason: str = "Manual promotion") -> None:
        """Promote best parameters to production configuration with validation."""
        # Get parameters before promotion
        study = self.task_manager.get_study(study_id)
        if not study:
            logger.error(f"Study {study_id} not found")
            return
        
        best_params = study.get('best_parameters', {})
        pattern_key = study.get('pattern_key', 'unknown')
        
        # Validate parameters before promotion
        is_valid, error = validate_parameters(pattern_key, best_params)
        if not is_valid:
            logger.error(f"Parameter validation failed for study {study_id}: {error}")
            raise ValueError(f"Invalid parameters: {error}")
        
        logger.info(f"Parameters validated successfully for study {study_id}")
        
        # Proceed with promotion
        self.task_manager.promote_best_parameters(study_id, regime_tag, reason)
        logger.info(f"Parameters promoted for study {study_id}")

    def rollback_parameters(self, study_id: int, version: int,
                           reason: str = "Manual rollback") -> None:
        """Rollback parameters to previous version."""
        self.task_manager.rollback_parameters(study_id, version, reason)
        logger.info(f"Parameters rolled back for study {study_id} to version {version}")

    def cleanup(self) -> None:
        """Cleanup resources and shutdown workers."""
        self._shutdown_requested = True
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.info("OptimizationEngine cleaned up")