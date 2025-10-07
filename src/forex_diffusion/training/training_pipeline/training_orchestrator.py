"""
Training Orchestrator - Main controller for two-phase training system.

Implements the external loop (model training) and coordinates the internal loop
(inference backtesting) to find and keep only best-performing models per regime.
"""

import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
from sqlalchemy.orm import Session
import joblib

from .database import (
    session_scope, create_training_run, create_training_queue,
    update_training_run_status, update_training_run_results,
    update_queue_status, update_queue_progress,
    get_training_queue_by_id, get_training_run_by_config_hash,
    TrainingQueue
)
from .config_grid import (
    generate_config_grid, add_config_hashes, deduplicate_configs,
    filter_already_trained, get_config_summary
)
from .regime_manager import RegimeManager
from .checkpoint_manager import CheckpointManager
from .model_file_manager import ModelFileManager
from .inference_backtester import InferenceBacktester
from .config_loader import get_config

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates the complete two-phase training pipeline.

    External Loop: Trains models for each configuration in grid
    Internal Loop: Backtests each model with all inference configs
    Decision: Keep models that improve performance for at least one regime
    """

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        checkpoints_dir: Optional[str] = None,
        auto_checkpoint_interval: Optional[int] = None,
        max_inference_workers: Optional[int] = None,
        delete_non_best_models: Optional[bool] = None
    ):
        """
        Initialize TrainingOrchestrator.

        Args:
            artifacts_dir: Directory for model artifacts (default: from config)
            checkpoints_dir: Directory for queue checkpoints (default: from config)
            auto_checkpoint_interval: Save checkpoint every N models (default: from config)
            max_inference_workers: Max parallel inference backtests (default: from config)
            delete_non_best_models: Auto-delete non-best models (default: from config)
        """
        # Load configuration
        self.config = get_config()

        # Use config values if not explicitly provided
        self.artifacts_dir = Path(artifacts_dir or self.config.artifacts_dir)
        self.checkpoints_dir = Path(checkpoints_dir or self.config.checkpoints_dir)
        self.auto_checkpoint_interval = auto_checkpoint_interval or self.config.auto_checkpoint_interval
        self.max_inference_workers = max_inference_workers or self.config.max_inference_workers

        # Create directories
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Managers
        self.checkpoint_manager = CheckpointManager(str(self.checkpoints_dir))

        delete_non_best = delete_non_best_models if delete_non_best_models is not None else self.config.delete_non_best_models
        self.model_file_manager = ModelFileManager(
            str(self.artifacts_dir),
            delete_non_best_models=delete_non_best
        )

        # Cancellation flag
        self.cancel_event = threading.Event()

        # Progress callback
        self.progress_callback: Optional[Callable] = None

    def create_training_queue(
        self,
        grid_params: Dict[str, List[Any]],
        skip_existing: bool = True,
        priority: int = 0
    ) -> int:
        """
        Create a new training queue from grid parameters.

        Args:
            grid_params: Parameter grid for configurations
            skip_existing: Skip configurations already trained
            priority: Queue priority (higher runs first)

        Returns:
            Queue ID
        """
        with session_scope() as session:
            # Generate configurations
            configs = generate_config_grid(grid_params)

            # Add hashes
            configs = add_config_hashes(configs)

            # Deduplicate
            configs = deduplicate_configs(configs)

            # Filter already trained
            if skip_existing:
                trained_hashes = self._get_trained_config_hashes(session)
                configs, already_trained = filter_already_trained(configs, trained_hashes)

                logger.info(
                    f"Skipped {len(already_trained)} already-trained configurations"
                )

            if not configs:
                raise ValueError("No configurations to train after filtering")

            # Create queue
            queue = create_training_queue(
                session,
                config_grid=configs,
                priority=priority
            )

            logger.info(
                f"Created training queue {queue.id} with {len(configs)} configurations"
            )

            return queue.id

    def train_models_grid(
        self,
        queue_id: int,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train all models in queue (External Loop).

        For each configuration:
        1. Train model
        2. Run inference backtests (Internal Loop)
        3. Evaluate regime performance
        4. Keep or delete model based on improvements

        Args:
            queue_id: Training queue ID
            progress_callback: Optional callback(current, total, status_msg)

        Returns:
            Dictionary with training summary
        """
        self.progress_callback = progress_callback
        self.cancel_event.clear()

        with session_scope() as session:
            queue = get_training_queue_by_id(session, queue_id)

            if not queue:
                raise ValueError(f"Queue {queue_id} not found")

            # Initialize managers
            regime_manager = RegimeManager(session)
            inference_backtester = InferenceBacktester(
                session,
                regime_manager,
                self.max_inference_workers
            )

            # Update queue status
            update_queue_status(session, queue_id, 'running', started_at=datetime.utcnow())

            # Training loop
            results = {
                'queue_id': queue_id,
                'total_configs': queue.total_configs,
                'completed': 0,
                'failed': 0,
                'skipped': 0,
                'kept_models': 0,
                'deleted_models': 0,
                'regime_improvements': {},
                'start_time': datetime.utcnow()
            }

            configs = queue.config_grid
            current_index = queue.current_index

            for i in range(current_index, len(configs)):
                # Check cancellation
                if self.cancel_event.is_set():
                    logger.info("Training cancelled by user")
                    update_queue_status(session, queue_id, 'paused')
                    break

                config = configs[i]
                config_summary = get_config_summary(config)

                logger.info(f"Training {i+1}/{len(configs)}: {config_summary}")

                # Report progress
                if self.progress_callback:
                    self.progress_callback(i + 1, len(configs), f"Training: {config_summary}")

                try:
                    # Train single model
                    training_run_id, model_kept, best_regimes = self.train_single_config(
                        session,
                        config,
                        regime_manager,
                        inference_backtester
                    )

                    results['completed'] += 1

                    if model_kept:
                        results['kept_models'] += 1
                        # Track regime improvements
                        for regime in best_regimes:
                            results['regime_improvements'][regime] = \
                                results['regime_improvements'].get(regime, 0) + 1
                    else:
                        results['deleted_models'] += 1

                except Exception as e:
                    logger.error(f"Training failed for config {i+1}: {e}")
                    results['failed'] += 1

                # Update queue progress
                update_queue_progress(
                    session,
                    queue_id,
                    current_index=i + 1,
                    completed_count=results['completed'],
                    failed_count=results['failed'],
                    skipped_count=results['skipped']
                )

                # Auto-checkpoint
                if (i + 1) % self.auto_checkpoint_interval == 0:
                    self.checkpoint_manager.create_checkpoint_from_queue_db(session, queue_id)

            # Mark queue complete
            if not self.cancel_event.is_set():
                update_queue_status(
                    session,
                    queue_id,
                    'completed',
                    completed_at=datetime.utcnow()
                )

            results['end_time'] = datetime.utcnow()
            results['duration_seconds'] = (
                results['end_time'] - results['start_time']
            ).total_seconds()

            logger.info(f"Training queue {queue_id} complete: {results}")

            return results

    def train_single_config(
        self,
        session: Session,
        config: Dict[str, Any],
        regime_manager: RegimeManager,
        inference_backtester: InferenceBacktester
    ) -> Tuple[int, bool, List[str]]:
        """
        Train a single model configuration.

        External Loop Step:
        1. Train model with config
        2. Run inference backtests (Internal Loop)
        3. Evaluate regime performance
        4. Decide keep/delete

        Args:
            session: SQLAlchemy session
            config: Training configuration
            regime_manager: RegimeManager instance
            inference_backtester: InferenceBacktester instance

        Returns:
            Tuple of (training_run_id, model_kept, best_regimes)
        """
        # Check if already trained
        existing_run = get_training_run_by_config_hash(session, config['config_hash'])
        if existing_run:
            logger.info(f"Config already trained: run_id={existing_run.id}")
            return existing_run.id, existing_run.is_model_kept, existing_run.best_regimes or []

        # Create training run record
        training_run = create_training_run(
            session,
            model_type=config['model_type'],
            encoder=config.get('encoder', 'none'),
            symbol=config['symbol'],
            base_timeframe=config['base_timeframe'],
            days_history=config['days_history'],
            horizon=config['horizon'],
            config_hash=config['config_hash'],
            indicator_tfs=config.get('indicator_tfs'),
            additional_features=config.get('additional_features'),
            preprocessing_params=config.get('preprocessing_params'),
            model_hyperparams=config.get('model_hyperparams')
        )
        session.commit()

        training_run_id = training_run.id

        try:
            # Update status to running
            update_training_run_status(
                session,
                training_run_id,
                'running',
                started_at=datetime.utcnow()
            )
            session.commit()

            # Train model
            model, training_metrics, model_file_path = self._train_model(config)

            # Get model file size
            model_file_size = os.path.getsize(model_file_path) if os.path.exists(model_file_path) else 0

            # Update training run with results
            update_training_run_results(
                session,
                training_run_id,
                training_metrics=training_metrics,
                feature_count=training_metrics.get('feature_count', 0),
                training_duration_seconds=training_metrics.get('training_duration', 0),
                model_file_path=model_file_path,
                model_file_size_bytes=model_file_size
            )

            # Update status to completed
            update_training_run_status(
                session,
                training_run_id,
                'completed',
                completed_at=datetime.utcnow()
            )
            session.commit()

            # Load OHLC data for backtesting
            ohlc_data = self._load_ohlc_data(config)

            # Run inference backtests (Internal Loop)
            inference_results = inference_backtester.backtest_all_inference_configs(
                model=model,
                model_config=config,
                ohlc_data=ohlc_data,
                training_run_id=training_run_id
            )

            # Find best inference config
            best_backtest_id, best_metrics, best_regime_metrics = \
                inference_backtester.find_best_inference_config(inference_results)

            # Evaluate regime improvements
            regime_improvements = regime_manager.update_regime_bests(
                training_run_id=training_run_id,
                inference_backtest_id=best_backtest_id,
                regime_metrics=best_regime_metrics
            )

            # Decide keep or delete
            if regime_improvements:
                # Model improves at least one regime - KEEP
                self.model_file_manager.keep_model_file(
                    session,
                    training_run_id,
                    regime_improvements
                )
                model_kept = True
                logger.info(
                    f"KEPT model {training_run_id} - "
                    f"Best for regimes: {regime_improvements}"
                )
            else:
                # Model doesn't improve any regime - DELETE
                self.model_file_manager.delete_model_file(session, training_run_id)
                model_kept = False
                logger.info(f"DELETED model {training_run_id} - No regime improvements")

            session.commit()

            return training_run_id, model_kept, regime_improvements

        except Exception as e:
            logger.error(f"Training failed for run {training_run_id}: {e}")
            update_training_run_status(session, training_run_id, 'failed')
            session.commit()
            raise

    def _train_model(self, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], str]:
        """
        Train a model with given configuration.

        Args:
            config: Training configuration

        Returns:
            Tuple of (model, training_metrics, model_file_path)
        """
        # Import training function
        from forex_diffusion.training.train_sklearn import train_sklearn_model

        # Prepare training arguments
        train_args = {
            'symbol': config['symbol'],
            'timeframe': config['base_timeframe'],
            'days_history': config['days_history'],
            'horizon': config['horizon'],
            'model_type': config['model_type'],
            'encoder': config.get('encoder', 'none'),
            'indicator_tfs': config.get('indicator_tfs', []),
            'additional_features': config.get('additional_features', [])
        }

        # Add model hyperparameters
        if config.get('model_hyperparams'):
            train_args.update(config['model_hyperparams'])

        # Generate model file path
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_filename = (
            f"{config['model_type']}_{config['symbol']}_"
            f"{config['base_timeframe']}_{timestamp}.pkl"
        )
        model_file_path = str(self.artifacts_dir / model_filename)

        # Train model
        start_time = datetime.utcnow()
        model = train_sklearn_model(**train_args)
        training_duration = (datetime.utcnow() - start_time).total_seconds()

        # Save model
        joblib.dump(model, model_file_path)

        # Create training metrics
        training_metrics = {
            'training_duration': training_duration,
            'feature_count': getattr(model, 'n_features_in_', 0),
            'model_params': getattr(model, 'get_params', lambda: {})()
        }

        return model, training_metrics, model_file_path

    def _load_ohlc_data(self, config: Dict[str, Any]) -> Any:
        """
        Load OHLC data for backtesting.

        Args:
            config: Training configuration

        Returns:
            OHLC DataFrame
        """
        from forex_diffusion.training.train_sklearn import fetch_candles_from_db

        ohlc_data = fetch_candles_from_db(
            symbol=config['symbol'],
            timeframe=config['base_timeframe'],
            days_history=config['days_history']
        )

        return ohlc_data

    def _get_trained_config_hashes(self, session: Session) -> set:
        """Get set of config hashes that have been successfully trained."""
        from sqlalchemy import select
        from .database import TrainingRun

        stmt = select(TrainingRun.config_hash).where(
            TrainingRun.status == 'completed'
        )
        results = session.execute(stmt).fetchall()

        return {row[0] for row in results}

    def cancel_training(self) -> None:
        """Cancel ongoing training."""
        logger.info("Cancellation requested")
        self.cancel_event.set()

    def pause_training(self, queue_id: int) -> bool:
        """
        Pause training queue.

        Args:
            queue_id: Queue ID

        Returns:
            True if paused successfully
        """
        try:
            with session_scope() as session:
                update_queue_status(
                    session,
                    queue_id,
                    'paused',
                    paused_at=datetime.utcnow()
                )
                self.checkpoint_manager.create_checkpoint_from_queue_db(session, queue_id)

            self.cancel_event.set()
            logger.info(f"Paused training queue {queue_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to pause queue {queue_id}: {e}")
            return False

    def resume_training(self, queue_id: int) -> Dict[str, Any]:
        """
        Resume paused training queue.

        Args:
            queue_id: Queue ID

        Returns:
            Training results
        """
        with session_scope() as session:
            queue = get_training_queue_by_id(session, queue_id)

            if not queue:
                raise ValueError(f"Queue {queue_id} not found")

            if queue.status not in ['paused', 'running']:
                raise ValueError(f"Queue {queue_id} cannot be resumed (status: {queue.status})")

            logger.info(
                f"Resuming queue {queue_id} from position "
                f"{queue.current_index}/{queue.total_configs}"
            )

            # Resume training from current position
            return self.train_models_grid(queue_id)

    def resume_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Resume training from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Training results
        """
        # Load checkpoint
        queue_uuid, remaining_configs, current_index, progress_stats = \
            self.checkpoint_manager.resume_from_checkpoint(checkpoint_path)

        logger.info(
            f"Resuming from checkpoint: {progress_stats['remaining_count']} "
            f"configs remaining"
        )

        # Find queue by UUID
        with session_scope() as session:
            from .database import get_training_queue_by_uuid

            queue = get_training_queue_by_uuid(session, queue_uuid)

            if not queue:
                raise ValueError(f"Queue with UUID {queue_uuid} not found")

            return self.resume_training(queue.id)

    def cleanup_storage(self) -> Dict[str, Any]:
        """
        Clean up non-best models and orphaned files.

        Returns:
            Cleanup statistics
        """
        with session_scope() as session:
            # Delete non-best models
            cleanup_stats = self.model_file_manager.cleanup_non_best_models(session)

            # Clean up orphaned files
            orphan_stats = self.model_file_manager.cleanup_orphaned_files(session)

            combined_stats = {
                'non_best_models': cleanup_stats,
                'orphaned_files': orphan_stats,
                'total_freed_mb': (
                    cleanup_stats['freed_mb'] + orphan_stats['freed_mb']
                )
            }

            logger.info(f"Storage cleanup complete: {combined_stats}")

            return combined_stats

    def get_training_status(self, queue_id: int) -> Dict[str, Any]:
        """
        Get current status of training queue.

        Args:
            queue_id: Queue ID

        Returns:
            Status dictionary
        """
        with session_scope() as session:
            queue = get_training_queue_by_id(session, queue_id)

            if not queue:
                raise ValueError(f"Queue {queue_id} not found")

            # Calculate progress percentage
            progress_pct = (queue.completed_count / queue.total_configs * 100) \
                if queue.total_configs > 0 else 0

            # Estimate time remaining
            if queue.started_at and queue.completed_count > 0:
                elapsed_seconds = (datetime.utcnow() - queue.started_at).total_seconds()
                avg_seconds_per_config = elapsed_seconds / queue.completed_count
                remaining_configs = queue.total_configs - queue.current_index
                estimated_remaining_seconds = avg_seconds_per_config * remaining_configs
            else:
                estimated_remaining_seconds = None

            return {
                'queue_id': queue.id,
                'queue_uuid': queue.queue_uuid,
                'status': queue.status,
                'total_configs': queue.total_configs,
                'current_index': queue.current_index,
                'completed_count': queue.completed_count,
                'failed_count': queue.failed_count,
                'skipped_count': queue.skipped_count,
                'progress_pct': progress_pct,
                'created_at': queue.created_at.isoformat() if queue.created_at else None,
                'started_at': queue.started_at.isoformat() if queue.started_at else None,
                'estimated_remaining_seconds': estimated_remaining_seconds
            }
