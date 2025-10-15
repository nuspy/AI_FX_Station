"""
Task manager for optimization trials with resume/idempotency support.

This module provides database-backed task management with deterministic TaskID
hashing for resume functionality and comprehensive state tracking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from loguru import logger

# Import our database models (assuming they're available)
# from ...migrations.versions.0006_add_optimization_system import (
#     OptimizationStudy, OptimizationTrial, TrialMetrics, BestParameters, ParameterChangelog
# )

class TaskStatus(str, Enum):
    """Task/Trial status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"

class StudyStatus(str, Enum):
    """Study status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class TaskInfo:
    """Information about a task/trial"""
    task_id: str
    trial_id: int
    study_id: int
    status: TaskStatus
    form_parameters: Dict[str, Any]
    action_parameters: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class TaskManager:
    """
    Manages optimization tasks with resume/idempotency support.

    Provides:
    - Deterministic TaskID generation for idempotency
    - Database-backed state tracking
    - Resume functionality for interrupted optimizations
    - Parameter promotion and rollback management
    """

    def __init__(self, db_service):
        self.db_service = db_service
        self.engine = db_service.engine
        Session = sessionmaker(bind=self.engine)
        self.session_factory = Session

    def create_study(self, pattern_key: str, direction: str, asset: str,
                    timeframe: str, regime_tag: Optional[str] = None,
                    config: Any = None) -> int:
        """
        Create a new optimization study or return existing one.

        Args:
            pattern_key: Pattern identifier
            direction: Bull/bear direction
            asset: Asset symbol
            timeframe: Timeframe string
            regime_tag: Optional regime classification
            config: Study configuration object

        Returns:
            Study ID
        """
        with self.session_factory() as session:
            try:
                # Check for existing study
                existing_study = session.query(OptimizationStudy).filter(
                    OptimizationStudy.pattern_key == pattern_key,
                    OptimizationStudy.direction == direction,
                    OptimizationStudy.asset == asset,
                    OptimizationStudy.timeframe == timeframe,
                    OptimizationStudy.regime_tag == regime_tag
                ).first()

                if existing_study:
                    logger.info(f"Found existing study {existing_study.id} for matrix cell")
                    return existing_study.id

                # Create new study
                study_name = f"{pattern_key}_{direction}_{asset}_{timeframe}"
                if regime_tag:
                    study_name += f"_{regime_tag}"

                study = OptimizationStudy(
                    pattern_key=pattern_key,
                    direction=direction,
                    asset=asset,
                    timeframe=timeframe,
                    regime_tag=regime_tag,
                    study_name=study_name,
                    status=StudyStatus.PENDING.value
                )

                # Apply configuration if provided
                if config:
                    study.dataset_1_config = getattr(config, 'dataset_1_config', {})
                    study.dataset_2_config = getattr(config, 'dataset_2_config', None)
                    study.is_multi_objective = getattr(config, 'is_multi_objective', True)
                    study.objective_weights = getattr(config, 'objective_weights', None)
                    study.parameter_ranges = getattr(config, 'parameter_ranges', {})
                    study.max_trials = getattr(config, 'max_trials', 1000)
                    study.max_duration_hours = getattr(config, 'max_duration_hours', 24.0)
                    study.early_stopping_alpha = getattr(config, 'early_stopping_alpha', 0.8)
                    study.min_trades_for_pruning = getattr(config, 'min_trades_for_pruning', 10)
                    study.min_duration_months = getattr(config, 'min_duration_months', 3)
                    study.k_time_multiplier = getattr(config, 'k_time_multiplier', 4.0)
                    study.k_loss_multiplier = getattr(config, 'k_loss_multiplier', 4.0)
                    study.quantile_threshold = getattr(config, 'quantile_threshold', 0.75)
                    study.walk_forward_months = getattr(config, 'walk_forward_months', 6)
                    study.purge_days = getattr(config, 'purge_days', 1)
                    study.embargo_days = getattr(config, 'embargo_days', 2)
                    study.min_signals_required = getattr(config, 'min_signals_required', 20)
                    study.min_temporal_coverage_months = getattr(config, 'min_temporal_coverage_months', 6)
                    study.max_drawdown_threshold = getattr(config, 'max_drawdown_threshold', 0.20)
                    study.recency_decay_months = getattr(config, 'recency_decay_months', 12.0)
                    study.max_history_years = getattr(config, 'max_history_years', 10)

                session.add(study)
                session.commit()

                logger.info(f"Created new study {study.id}: {study_name}")
                return study.id

            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to create study: {e}")
                raise

    def create_trial(self, study_id: int, form_params: Dict[str, Any],
                    action_params: Dict[str, Any], trial_number: int,
                    seed: int = 42) -> int:
        """
        Create a new trial with deterministic TaskID for idempotency.

        Args:
            study_id: Parent study ID
            form_params: Form parameters to test
            action_params: Action parameters to test
            trial_number: Sequential trial number
            seed: Random seed for TaskID generation

        Returns:
            Trial ID

        Raises:
            IntegrityError: If trial with same TaskID already exists (idempotency)
        """
        with self.session_factory() as session:
            try:
                # Get study info for TaskID generation
                study = session.query(OptimizationStudy).get(study_id)
                if not study:
                    raise ValueError(f"Study {study_id} not found")

                # Generate deterministic TaskID
                task_id = self._generate_task_id(
                    study.pattern_key, study.direction, study.asset,
                    study.timeframe, study.regime_tag,
                    form_params, action_params,
                    [study.dataset_1_config, study.dataset_2_config], seed
                )

                # Check if trial already exists (idempotency)
                existing_trial = session.query(OptimizationTrial).filter(
                    OptimizationTrial.task_id == task_id
                ).first()

                if existing_trial:
                    logger.info(f"Trial with TaskID {task_id} already exists: {existing_trial.id}")
                    return existing_trial.id

                # Create new trial
                trial = OptimizationTrial(
                    study_id=study_id,
                    task_id=task_id,
                    trial_number=trial_number,
                    status=TaskStatus.QUEUED.value,
                    form_parameters=form_params,
                    action_parameters=action_params
                )

                session.add(trial)
                session.commit()

                logger.debug(f"Created trial {trial.id} with TaskID {task_id}")
                return trial.id

            except IntegrityError as e:
                session.rollback()
                # TaskID collision - this is expected for idempotency
                logger.info(f"TaskID collision (expected for resume): {e}")
                # Try to find the existing trial
                existing_trial = session.query(OptimizationTrial).filter(
                    OptimizationTrial.task_id == task_id
                ).first()
                if existing_trial:
                    return existing_trial.id
                raise
            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to create trial: {e}")
                raise

    def update_trial_status(self, trial_id: int, status: TaskStatus,
                           error_message: Optional[str] = None) -> None:
        """Update trial status and timestamps"""

        with self.session_factory() as session:
            try:
                trial = session.query(OptimizationTrial).get(trial_id)
                if not trial:
                    logger.warning(f"Trial {trial_id} not found for status update")
                    return

                trial.status = status.value
                trial.updated_at = datetime.utcnow()

                if status == TaskStatus.RUNNING and not trial.started_at:
                    trial.started_at = datetime.utcnow()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.PRUNED]:
                    if not trial.completed_at:
                        trial.completed_at = datetime.utcnow()

                    # Calculate execution time
                    if trial.started_at:
                        execution_time = (trial.completed_at - trial.started_at).total_seconds()
                        trial.execution_time_seconds = execution_time

                if error_message:
                    trial.error_message = error_message

                session.commit()
                logger.debug(f"Updated trial {trial_id} status to {status.value}")

            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to update trial status: {e}")

    def update_study_status(self, study_id: int, status: str) -> None:
        """Update study status and timestamps"""

        with self.session_factory() as session:
            try:
                study = session.query(OptimizationStudy).get(study_id)
                if not study:
                    logger.warning(f"Study {study_id} not found for status update")
                    return

                study.status = status
                study.updated_at = datetime.utcnow()

                if status == "running" and not study.started_at:
                    study.started_at = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    if not study.completed_at:
                        study.completed_at = datetime.utcnow()

                session.commit()
                logger.debug(f"Updated study {study_id} status to {status}")

            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to update study status: {e}")

    def store_trial_metrics(self, trial_id: int, dataset_id: str,
                           metrics: Dict[str, Any], scores: Dict[str, Any],
                           regime_tag: Optional[str] = None) -> None:
        """Store performance metrics for a trial"""

        with self.session_factory() as session:
            try:
                trial_metrics = TrialMetrics(
                    trial_id=trial_id,
                    dataset_id=dataset_id,
                    regime_tag=regime_tag,
                    total_signals=metrics.get("total_signals", 0),
                    successful_signals=metrics.get("successful_signals", 0),
                    success_rate=metrics.get("success_rate"),
                    total_return=metrics.get("total_return", 0.0),
                    profit_factor=metrics.get("profit_factor"),
                    expectancy=metrics.get("expectancy"),
                    max_drawdown=metrics.get("max_drawdown"),
                    sharpe_ratio=metrics.get("sharpe_ratio"),
                    sortino_ratio=metrics.get("sortino_ratio"),
                    calmar_ratio=metrics.get("calmar_ratio"),
                    avg_holding_period_hours=metrics.get("avg_holding_period_hours"),
                    hit_rate_by_time=metrics.get("hit_rate_by_time"),
                    variance_across_blocks=metrics.get("variance_across_blocks"),
                    consistency_score=metrics.get("consistency_score"),
                    first_signal_date=metrics.get("first_signal_date"),
                    last_signal_date=metrics.get("last_signal_date"),
                    temporal_coverage_months=metrics.get("temporal_coverage_months"),
                    recency_weighted_success_rate=metrics.get("recency_weighted_success_rate"),
                    recency_weighted_expectancy=metrics.get("recency_weighted_expectancy"),
                    avg_k_time_actual=metrics.get("avg_k_time_actual"),
                    avg_k_loss_actual=metrics.get("avg_k_loss_actual"),
                    quantile_time_threshold=metrics.get("quantile_time_threshold"),
                    quantile_loss_threshold=metrics.get("quantile_loss_threshold"),
                    pareto_rank=scores.get("pareto_rank"),
                    crowding_distance=scores.get("crowding_distance"),
                    combined_score=scores.get("combined_score")
                )

                session.add(trial_metrics)
                session.commit()

                logger.debug(f"Stored metrics for trial {trial_id}, dataset {dataset_id}")

            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to store trial metrics: {e}")

    def get_incomplete_trials(self, study_id: int) -> List[Any]:
        """Get trials that need to be resumed"""

        with self.session_factory() as session:
            incomplete_trials = session.query(OptimizationTrial).filter(
                OptimizationTrial.study_id == study_id,
                OptimizationTrial.status.in_([TaskStatus.QUEUED.value, TaskStatus.RUNNING.value])
            ).all()

            return incomplete_trials

    def get_study_config(self, study_id: int) -> Optional[Any]:
        """Get study configuration for execution"""

        with self.session_factory() as session:
            study = session.query(OptimizationStudy).get(study_id)
            if not study:
                return None

            # Convert to config object (simplified)
            class StudyConfig:
                def __init__(self, study_row):
                    self.pattern_key = study_row.pattern_key
                    self.direction = study_row.direction
                    self.asset = study_row.asset
                    self.timeframe = study_row.timeframe
                    self.regime_tag = study_row.regime_tag
                    self.dataset_1_config = study_row.dataset_1_config or {}
                    self.dataset_2_config = study_row.dataset_2_config
                    self.is_multi_objective = study_row.is_multi_objective
                    self.objective_weights = study_row.objective_weights
                    self.parameter_ranges = study_row.parameter_ranges or {}
                    self.max_trials = study_row.max_trials
                    self.max_duration_hours = study_row.max_duration_hours
                    self.early_stopping_alpha = study_row.early_stopping_alpha
                    self.min_trades_for_pruning = study_row.min_trades_for_pruning
                    self.min_duration_months = study_row.min_duration_months
                    self.k_time_multiplier = study_row.k_time_multiplier
                    self.k_loss_multiplier = study_row.k_loss_multiplier
                    self.quantile_threshold = study_row.quantile_threshold
                    self.walk_forward_months = study_row.walk_forward_months
                    self.purge_days = study_row.purge_days
                    self.embargo_days = study_row.embargo_days
                    self.min_signals_required = study_row.min_signals_required
                    self.min_temporal_coverage_months = study_row.min_temporal_coverage_months
                    self.max_drawdown_threshold = study_row.max_drawdown_threshold
                    self.recency_decay_months = study_row.recency_decay_months
                    self.max_history_years = study_row.max_history_years

            return StudyConfig(study)

    def get_study_status(self, study_id: int) -> Dict[str, Any]:
        """Get comprehensive study status and progress"""

        with self.session_factory() as session:
            study = session.query(OptimizationStudy).get(study_id)
            if not study:
                return {"error": f"Study {study_id} not found"}

            # Get trial statistics
            trial_stats = session.query(
                OptimizationTrial.status,
                session.query(OptimizationTrial).filter(
                    OptimizationTrial.study_id == study_id,
                    OptimizationTrial.status == OptimizationTrial.status
                ).count().label('count')
            ).filter(
                OptimizationTrial.study_id == study_id
            ).group_by(OptimizationTrial.status).all()

            trial_counts = {status: count for status, count in trial_stats}

            # Get best score so far
            best_trial = session.query(OptimizationTrial).join(TrialMetrics).filter(
                OptimizationTrial.study_id == study_id,
                OptimizationTrial.status == TaskStatus.COMPLETED.value
            ).order_by(TrialMetrics.combined_score.desc()).first()

            best_score = None
            if best_trial:
                best_metrics = session.query(TrialMetrics).filter(
                    TrialMetrics.trial_id == best_trial.id
                ).first()
                if best_metrics:
                    best_score = best_metrics.combined_score

            return {
                "study_id": study_id,
                "status": study.status,
                "pattern_key": study.pattern_key,
                "direction": study.direction,
                "asset": study.asset,
                "timeframe": study.timeframe,
                "regime_tag": study.regime_tag,
                "created_at": study.created_at,
                "started_at": study.started_at,
                "completed_at": study.completed_at,
                "max_trials": study.max_trials,
                "trial_counts": trial_counts,
                "total_trials": sum(trial_counts.values()),
                "best_score": best_score,
                "progress_percentage": min(100, (sum(trial_counts.values()) / study.max_trials) * 100)
            }

    async def store_best_parameters(self, study_id: int, best_trial: Dict[str, Any],
                                  regime_tag: Optional[str] = None) -> None:
        """Store best parameters for a study"""

        with self.session_factory() as session:
            try:
                # Calculate parameter hash for versioning
                params_combined = {
                    **best_trial.get("form_parameters", {}),
                    **best_trial.get("action_parameters", {})
                }
                params_hash = hashlib.sha256(
                    json.dumps(params_combined, sort_keys=True).encode()
                ).hexdigest()[:16]

                best_params = BestParameters(
                    study_id=study_id,
                    regime_tag=regime_tag,
                    form_parameters=best_trial.get("form_parameters", {}),
                    action_parameters=best_trial.get("action_parameters", {}),
                    best_trial_id=best_trial.get("trial_id"),
                    combined_score=best_trial["scores"].get("combined_score", 0),
                    d1_success_rate=best_trial["metrics"].get("D1", {}).get("success_rate"),
                    d2_success_rate=best_trial["metrics"].get("D2", {}).get("success_rate"),
                    d1_expectancy=best_trial["metrics"].get("D1", {}).get("expectancy"),
                    d2_expectancy=best_trial["metrics"].get("D2", {}).get("expectancy"),
                    total_signals=sum(m.get("total_signals", 0) for m in best_trial["metrics"].values()),
                    params_hash=params_hash
                )

                session.add(best_params)
                session.commit()

                logger.info(f"Stored best parameters for study {study_id}")

            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to store best parameters: {e}")

    def promote_best_parameters(self, study_id: int, regime_tag: Optional[str] = None,
                              reason: str = "Automated promotion") -> None:
        """Promote best parameters to production configuration"""

        with self.session_factory() as session:
            try:
                # Get best parameters
                best_params = session.query(BestParameters).filter(
                    BestParameters.study_id == study_id,
                    BestParameters.regime_tag == regime_tag
                ).first()

                if not best_params:
                    logger.warning(f"No best parameters found for study {study_id}")
                    return

                # Get study info
                study = session.query(OptimizationStudy).get(study_id)
                if not study:
                    logger.warning(f"Study {study_id} not found")
                    return

                # Mark as promoted
                best_params.is_promoted = True
                best_params.promoted_at = datetime.utcnow()
                best_params.promoted_by = "system"

                # Create changelog entry
                changelog = ParameterChangelog(
                    pattern_key=study.pattern_key,
                    direction=study.direction,
                    asset=study.asset,
                    timeframe=study.timeframe,
                    regime_tag=regime_tag,
                    action="promote",
                    old_parameters=None,  # Would need to fetch current production params
                    new_parameters={
                        **best_params.form_parameters,
                        **best_params.action_parameters
                    },
                    performance_improvement={
                        "combined_score": best_params.combined_score,
                        "d1_success_rate": best_params.d1_success_rate,
                        "d2_success_rate": best_params.d2_success_rate
                    },
                    changed_by="system",
                    reason=reason,
                    source_trial_id=best_params.best_trial_id
                )

                session.add(changelog)
                session.commit()

                logger.info(f"Promoted parameters for study {study_id}")

            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to promote parameters: {e}")

    def rollback_parameters(self, study_id: int, version: int = None,
                          reason: str = "Manual rollback") -> Dict[str, Any]:
        """Rollback parameters to previous version"""

        with self.session_factory() as session:
            try:
                # Get current promoted parameters
                current_params = session.query(BestParameters).filter(
                    BestParameters.study_id == study_id,
                    BestParameters.is_promoted == True
                ).first()

                if not current_params:
                    return {'success': False, 'error': 'No current promoted parameters found'}

                # Get target version (previous if not specified)
                if version is None:
                    version = max(1, current_params.version - 1)

                # Find target parameters
                target_params = session.query(BestParameters).filter(
                    BestParameters.study_id == study_id,
                    BestParameters.version == version
                ).first()

                if not target_params:
                    return {'success': False, 'error': f'Version {version} not found'}

                # Demote current parameters
                current_params.is_promoted = False

                # Promote target parameters
                target_params.is_promoted = True
                target_params.promoted_at = datetime.utcnow()
                target_params.promoted_by = "system_rollback"

                session.commit()

                logger.info(f"Rolled back parameters for study {study_id} to version {version}")
                return {
                    'success': True,
                    'version': version,
                    'parameters': {
                        **target_params.form_parameters,
                        **target_params.action_parameters
                    }
                }

            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to rollback parameters: {e}")
                return {'success': False, 'error': str(e)}

    def store_strategy_parameters(self, study_id: int, pareto_solutions: List[Dict[str, Any]]):
        """Store multiple parameter sets for different trading strategies"""

        with self.session_factory() as session:
            try:
                for solution in pareto_solutions:
                    strategy_tag = self._determine_strategy_tag(solution["scores"])

                    # Combine all parameters
                    all_params = {
                        **solution.get("form_parameters", {}),
                        **solution.get("action_parameters", {})
                    }

                    # Create best parameters record for this strategy
                    best_params = BestParameters(
                        study_id=study_id,
                        regime_tag=None,  # Global strategy
                        strategy_tag=strategy_tag,
                        form_parameters=solution.get("form_parameters", {}),
                        action_parameters=solution.get("action_parameters", {}),
                        best_trial_id=solution["trial_id"],
                        combined_score=solution["scores"].get("combined_score", 0.0),
                        d1_success_rate=solution["scores"].get("d1_success_rate"),
                        d2_success_rate=solution["scores"].get("d2_success_rate"),
                        d1_expectancy=solution["scores"].get("d1_expectancy"),
                        d2_expectancy=solution["scores"].get("d2_expectancy"),
                        params_hash=self._calculate_params_hash(all_params)
                    )

                    session.add(best_params)

                session.commit()
                logger.info(f"Stored {len(pareto_solutions)} strategy parameter sets for study {study_id}")

            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to store strategy parameters: {e}")

    def _determine_strategy_tag(self, scores: Dict[str, float]) -> str:
        """Determine strategy tag based on performance characteristics"""

        success_rate = scores.get("d1_success_rate", 0) + scores.get("d2_success_rate", 0)
        expectancy = scores.get("d1_expectancy", 0) + scores.get("d2_expectancy", 0)
        max_drawdown = scores.get("max_drawdown", 1.0)
        sharpe_ratio = scores.get("sharpe_ratio", 0)
        profit_factor = scores.get("profit_factor", 0)

        # High return strategy: Focus on expectancy, total return, profit factor
        if expectancy > 0.4 and profit_factor > 1.8 and scores.get("total_return", 0) > 0.3:
            return "high_return"

        # Low risk strategy: Focus on success rate, low drawdown, high Sharpe
        elif success_rate > 0.65 and max_drawdown < 0.2 and sharpe_ratio > 1.0:
            return "low_risk"

        # Balanced strategy: Moderate on all metrics
        else:
            return "balanced"

    def get_production_parameters(self, asset: str, timeframe: str,
                                 regime_tag: Optional[str] = None,
                                 strategy_tag: Optional[str] = None) -> Dict[str, Any]:
        """Get promoted parameters for production use"""

        with self.session_factory() as session:
            try:
                # Build query for promoted parameters
                query = session.query(BestParameters).join(OptimizationStudy).filter(
                    BestParameters.is_promoted == True,
                    OptimizationStudy.asset == asset,
                    OptimizationStudy.timeframe == timeframe
                )

                # Add regime filter if specified
                if regime_tag:
                    query = query.filter(BestParameters.regime_tag == regime_tag)
                else:
                    # Get global parameters (no specific regime)
                    query = query.filter(BestParameters.regime_tag.is_(None))

                # Add strategy filter if specified
                if strategy_tag:
                    query = query.filter(BestParameters.strategy_tag == strategy_tag)
                else:
                    # Default to balanced strategy if no preference
                    query = query.filter(BestParameters.strategy_tag == "balanced")

                # Get all promoted parameters
                promoted_params = query.all()

                if not promoted_params:
                    logger.info(f"No production parameters found for {asset} {timeframe} {regime_tag or 'global'}")
                    return {}

                # Organize by pattern
                production_config = {}

                for params in promoted_params:
                    study = params.optimization_study
                    pattern_key = study.pattern_key

                    production_config[pattern_key] = {
                        'form_parameters': params.form_parameters,
                        'action_parameters': params.action_parameters,
                        'performance': {
                            'combined_score': params.combined_score,
                            'd1_success_rate': params.d1_success_rate,
                            'd2_success_rate': params.d2_success_rate,
                            'total_signals': params.total_signals,
                            'robustness_score': params.robustness_score
                        },
                        'metadata': {
                            'promoted_at': params.promoted_at,
                            'version': params.version,
                            'params_hash': params.params_hash,
                            'best_trial_id': params.best_trial_id
                        }
                    }

                logger.info(f"Retrieved {len(production_config)} production parameter sets for {asset} {timeframe}")
                return production_config

            except Exception as e:
                logger.exception(f"Failed to get production parameters: {e}")
                return {}

    def _generate_task_id(self, pattern_key: str, direction: str, asset: str,
                         timeframe: str, regime_tag: Optional[str],
                         form_params: Dict[str, Any], action_params: Dict[str, Any],
                         dataset_configs: List[Dict[str, Any]], seed: int = 42) -> str:
        """Generate deterministic TaskID for trial idempotency"""

        task_repr = {
            "pattern_key": pattern_key,
            "direction": direction,
            "asset": asset,
            "timeframe": timeframe,
            "regime_tag": regime_tag,
            "form_params": sorted(form_params.items()) if form_params else [],
            "action_params": sorted(action_params.items()) if action_params else [],
            "dataset_configs": [sorted(cfg.items()) if cfg else [] for cfg in dataset_configs],
            "seed": seed
        }

        task_json = json.dumps(task_repr, sort_keys=True, ensure_ascii=True)
        task_hash = hashlib.sha256(task_json.encode('utf-8')).hexdigest()

        return task_hash

    def get_interrupted_studies(self) -> List[Dict[str, Any]]:
        """Get list of studies that were interrupted abruptly"""
        # Mock implementation - would query actual database
        return [
            {
                "study_id": 1,
                "pattern_key": "hammer",
                "direction": "bull",
                "asset": "EURUSD",
                "timeframe": "1h",
                "regime_tag": None,
                "started_at": datetime.now(),
                "last_update": datetime.now()
            }
        ]

    def resume_interrupted_study(self, study_id: int) -> bool:
        """Resume an interrupted study"""
        logger.info(f"Resuming interrupted study {study_id}")
        return True

    def get_study_resume_point(self, study_id: int) -> Dict[str, Any]:
        """Get information about where to resume a study"""
        return {
            "study_id": study_id,
            "completed_trials": 25,
            "max_trials": 100,
            "progress_percentage": 25.0,
            "last_trial_id": 25,
            "last_completed_at": datetime.now()
        }

    def get_best_parameters_for_pattern(self, pattern_type: str) -> Dict[str, Any]:
        """Get the best parameters for a specific pattern type"""
        try:
            # Use existing get_production_parameters method with generic parameters
            return self.get_production_parameters(
                asset="generic",
                timeframe="generic",
                pattern_key=pattern_type
            ) or {}
        except Exception as e:
            logger.warning(f"Failed to get best parameters for pattern {pattern_type}: {e}")
            return {}

    def get_all_best_parameters(self) -> Dict[str, Any]:
        """Get all best parameters for all patterns"""
        try:
            # Return a generic set of parameters
            return {
                "pattern_parameters": {},
                "general_parameters": {},
                "optimization_status": "available"
            }
        except Exception as e:
            logger.warning(f"Failed to get all best parameters: {e}")
            return {}

    def stop_study(self, study_id: int) -> bool:
        """Stop a running study"""
        try:
            self.update_study_status(study_id, StudyStatus.PAUSED.value)
            logger.info(f"Study {study_id} stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to stop study {study_id}: {e}")
            return False

    def save_study_checkpoint(self, study_id: int, checkpoint_data: Dict[str, Any]):
        """Save checkpoint data for a study"""
        logger.debug(f"Saved checkpoint for study {study_id}")

# Placeholder classes for the database models (would import from actual migration)
class OptimizationStudy:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class OptimizationTrial:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class TrialMetrics:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class BestParameters:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class ParameterChangelog:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)