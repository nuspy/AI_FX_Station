# src/forex_diffusion/training_pipeline/training_orchestrator.py
"""
Training Orchestrator - Manages the two-phase training pipeline.

This is the main entry point for the new training system. It coordinates:
1. Configuration grid generation
2. External loop (model training)
3. Internal loop (inference backtesting)
4. Regime performance evaluation
5. Model file management
6. Checkpoint save/resume
"""

from __future__ import annotations
import hashlib
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import threading
from loguru import logger

from .database_models import TrainingRun, TrainingQueue
from .inference_backtester import InferenceBacktester
from .regime_manager import RegimeManager
from .checkpoint_manager import CheckpointManager
from .model_file_manager import ModelFileManager
from ..utils.config import get_config


class TrainingOrchestrator:
    """
    Orchestrates the two-phase training pipeline.
    
    Workflow:
    1. User creates configuration grid (mutually exclusive parameters)
    2. create_training_queue() → Creates queue in database
    3. start_training_queue() → Begins external loop
    4. For each config:
       a. train_single_config() → Trains model
       b. backtest_inference_configs() → Internal loop
       c. evaluate_regime_performance() → Compare to bests
       d. keep or delete model file based on results
    5. Handle interruption/resume via CheckpointManager
    """
    
    def __init__(self, db_session, artifacts_dir: Path = None):
        """
        Initialize orchestrator.
        
        Args:
            db_session: SQLAlchemy database session
            artifacts_dir: Directory for model artifacts (default from config)
        """
        self.db = db_session
        self.cfg = get_config()
        self.artifacts_dir = artifacts_dir or Path(self.cfg.model.artifacts_dir)
        
        # Sub-managers
        self.inference_backtester = InferenceBacktester(db_session)
        self.regime_manager = RegimeManager(db_session)
        self.checkpoint_mgr = CheckpointManager(self.artifacts_dir / "checkpoints")
        self.model_file_mgr = ModelFileManager(self.artifacts_dir)
        
        # Cancellation flag for graceful shutdown
        self._cancellation_event = threading.Event()
        
        logger.info("TrainingOrchestrator initialized")
    
    def generate_config_grid(
        self,
        model_types: List[str],
        encoders: List[str],
        symbols: List[str],
        timeframes: List[str],
        days_history_list: List[int],
        horizon_list: List[int],
        indicator_tfs: Dict[str, List[str]],
        additional_features: Dict[str, bool],
        preprocessing_params: Dict[str, Any],
        model_hyperparams: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate configuration grid from parameter lists.
        
        This creates the Cartesian product of mutually exclusive parameters,
        combined with the shared combinable parameters.
        
        Args:
            model_types: List of model types to try (e.g., ['ridge', 'rf'])
            encoders: List of encoders (e.g., ['none', 'pca'])
            symbols: List of symbols (e.g., ['EUR/USD'])
            timeframes: List of timeframes (e.g., ['1h', '4h'])
            days_history_list: List of day ranges (e.g., [7, 30, 90])
            horizon_list: List of horizons (e.g., [5, 10, 20])
            indicator_tfs: Indicator timeframe selections (combinable)
            additional_features: Feature flags (combinable)
            preprocessing_params: Preprocessing parameters (combinable)
            model_hyperparams: Model-specific hyperparameters (optional)
        
        Returns:
            List of configuration dictionaries
        """
        grid = []
        
        # Cartesian product of mutually exclusive parameters
        for model_type in model_types:
            for encoder in encoders:
                for symbol in symbols:
                    for timeframe in timeframes:
                        for days_history in days_history_list:
                            for horizon in horizon_list:
                                config = {
                                    # Mutually exclusive
                                    'model_type': model_type,
                                    'encoder': encoder,
                                    'symbol': symbol,
                                    'base_timeframe': timeframe,
                                    'days_history': days_history,
                                    'horizon': horizon,
                                    
                                    # Combinable (shared across all configs)
                                    'indicator_tfs': indicator_tfs.copy(),
                                    'additional_features': additional_features.copy(),
                                    'preprocessing_params': preprocessing_params.copy(),
                                    'model_hyperparams': model_hyperparams.copy() if model_hyperparams else {}
                                }
                                
                                grid.append(config)
        
        logger.info(f"Generated configuration grid: {len(grid)} configurations")
        return grid
    
    def compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute SHA256 hash of configuration for deduplication.
        
        Ensures deterministic hashing by sorting all keys recursively.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            64-character hex hash
        """
        def sort_dict_recursively(d):
            """Sort dictionary keys recursively."""
            if isinstance(d, dict):
                return {k: sort_dict_recursively(v) for k, v in sorted(d.items())}
            elif isinstance(d, list):
                return [sort_dict_recursively(item) for item in d]
            else:
                return d
        
        canonical = sort_dict_recursively(config)
        json_str = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
        hash_value = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        
        return hash_value
    
    def config_exists_in_db(self, config: Dict[str, Any]) -> Optional[TrainingRun]:
        """
        Check if configuration already exists in database.
        
        Args:
            config: Configuration to check
        
        Returns:
            Existing TrainingRun if found, None otherwise
        """
        config_hash = self.compute_config_hash(config)
        existing = self.db.query(TrainingRun).filter(
            TrainingRun.config_hash == config_hash
        ).first()
        
        return existing
    
    def create_training_queue(
        self,
        config_grid: List[Dict[str, Any]],
        priority: int = 0,
        max_parallel: int = 1
    ) -> TrainingQueue:
        """
        Create a new training queue in the database.
        
        Args:
            config_grid: List of configurations to train
            priority: Queue priority (higher = more important)
            max_parallel: Max parallel training jobs
        
        Returns:
            Created TrainingQueue object
        """
        queue = TrainingQueue(
            queue_uuid=str(uuid.uuid4()),
            config_grid=config_grid,
            current_index=0,
            total_configs=len(config_grid),
            status='pending',
            completed_count=0,
            failed_count=0,
            skipped_count=0,
            priority=priority,
            max_parallel=max_parallel
        )
        
        self.db.add(queue)
        self.db.commit()
        
        logger.info(f"Created training queue: {queue.queue_uuid} ({len(config_grid)} configs)")
        return queue
    
    def start_training_queue(
        self,
        queue_uuid: str,
        resume_from_checkpoint: bool = False
    ) -> str:
        """
        Start or resume training a queue.
        
        Args:
            queue_uuid: UUID of queue to train
            resume_from_checkpoint: If True, resume from last checkpoint
        
        Returns:
            Status: 'COMPLETED', 'PAUSED', or 'CANCELLED'
        """
        # Load queue
        queue = self.db.query(TrainingQueue).filter(
            TrainingQueue.queue_uuid == queue_uuid
        ).first()
        
        if not queue:
            raise ValueError(f"Queue not found: {queue_uuid}")
        
        # Resume from checkpoint if requested
        start_index = 0
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_mgr.load_latest_checkpoint(queue_uuid)
            if checkpoint:
                start_index = checkpoint['current_index']
                logger.info(f"Resuming from checkpoint: {start_index}/{queue.total_configs}")
        
        # Clear cancellation flag
        self._cancellation_event.clear()
        
        # Update queue status
        queue.status = 'running'
        queue.started_at = datetime.utcnow()
        self.db.commit()
        
        # Load current regime bests
        regime_bests = self.regime_manager.load_regime_bests()
        
        # External loop: iterate configurations
        for index in range(start_index, queue.total_configs):
            config = queue.config_grid[index]
            
            # Check for cancellation
            if self._cancellation_event.is_set():
                logger.info("Cancellation requested - saving checkpoint")
                self.checkpoint_mgr.save_checkpoint(queue, index, regime_bests)
                queue.status = 'paused'
                queue.paused_at = datetime.utcnow()
                self.db.commit()
                return 'PAUSED'
            
            # Skip if already trained (deduplication)
            existing = self.config_exists_in_db(config)
            if existing:
                logger.info(f"Config {index+1}/{queue.total_configs} already trained, skipping")
                queue.skipped_count += 1
                queue.current_index = index + 1
                self.db.commit()
                continue
            
            # Train model
            try:
                logger.info(f"Training config {index+1}/{queue.total_configs}: {config['symbol']} {config['base_timeframe']} {config['model_type']}")
                training_run = self.train_single_config(config, queue_uuid)
                
                # Internal loop: inference backtest
                logger.info(f"Running inference backtests for training_run {training_run.id}")
                inference_results = self.inference_backtester.backtest_all_inference_configs(training_run)
                
                # Evaluate against regimes
                improvements = self.regime_manager.evaluate_regime_improvements(
                    inference_results,
                    regime_bests
                )
                
                # Decide: keep or delete model
                if improvements:
                    self.model_file_mgr.keep_model_file(training_run)
                    self.regime_manager.update_regime_bests(improvements)
                    regime_names = list(improvements.keys())
                    logger.info(f"✓ Model kept - improved regimes: {regime_names}")
                    training_run.best_regimes = regime_names
                else:
                    self.model_file_mgr.delete_model_file(training_run)
                    logger.info(f"✗ Model deleted - no improvements")
                
                # Update database
                training_run.status = 'completed'
                training_run.completed_at = datetime.utcnow()
                self.db.commit()
                
                # Update queue progress
                queue.completed_count += 1
                queue.current_index = index + 1
                self.db.commit()
                
                # Auto-checkpoint every 10 models
                if (index + 1) % 10 == 0:
                    self.checkpoint_mgr.save_checkpoint(queue, index + 1, regime_bests)
                
            except Exception as e:
                logger.exception(f"Training failed for config {index+1}: {e}")
                
                # Mark as failed in database
                training_run = TrainingRun(
                    run_uuid=str(uuid.uuid4()),
                    status='failed',
                    config_hash=self.compute_config_hash(config),
                    **config
                )
                self.db.add(training_run)
                
                queue.failed_count += 1
                queue.current_index = index + 1
                self.db.commit()
                
                # Continue to next config
                continue
        
        # Mark queue as completed
        queue.status = 'completed'
        queue.completed_at = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"Queue completed: {queue.completed_count} completed, {queue.failed_count} failed, {queue.skipped_count} skipped")
        return 'COMPLETED'
    
    def train_single_config(self, config: Dict[str, Any], queue_uuid: str = None) -> TrainingRun:
        """
        Train a single model configuration.
        
        This method calls the actual training code (train_sklearn.py or train.py)
        and creates a TrainingRun entry in the database.
        
        Args:
            config: Configuration dictionary
            queue_uuid: Optional queue UUID for tracking
        
        Returns:
            Created TrainingRun object
        """
        # Create training run entry
        training_run = TrainingRun(
            run_uuid=str(uuid.uuid4()),
            status='running',
            created_by='training_queue' if queue_uuid else 'manual',
            config_hash=self.compute_config_hash(config),
            **config  # Unpack all config fields
        )
        training_run.started_at = datetime.utcnow()
        
        self.db.add(training_run)
        self.db.commit()
        
        # TODO: Call actual training code here
        # This would invoke train_sklearn.py or train.py with the config
        # For now, we'll simulate training
        
        # Placeholder for actual training logic
        # In reality, this would:
        # 1. Call train_sklearn.py or train.py with subprocess
        # 2. Wait for completion
        # 3. Load trained model file
        # 4. Extract training metrics
        # 5. Update training_run with results
        
        logger.warning("TODO: Implement actual training call in train_single_config()")
        
        # Simulate training completion
        training_run.status = 'completed'
        training_run.completed_at = datetime.utcnow()
        training_run.training_duration_seconds = 120.0  # Placeholder
        training_run.training_metrics = {'mae': 0.05, 'rmse': 0.08}  # Placeholder
        training_run.model_file_path = str(self.artifacts_dir / f"model_{training_run.run_uuid}.pkl")
        training_run.model_file_size_bytes = 1024 * 100  # Placeholder
        
        self.db.commit()
        
        return training_run
    
    def request_cancellation(self):
        """Request graceful cancellation of current training queue."""
        logger.info("Cancellation requested")
        self._cancellation_event.set()
    
    def get_queue_status(self, queue_uuid: str) -> Dict[str, Any]:
        """
        Get status of a training queue.
        
        Args:
            queue_uuid: Queue UUID
        
        Returns:
            Status dictionary
        """
        queue = self.db.query(TrainingQueue).filter(
            TrainingQueue.queue_uuid == queue_uuid
        ).first()
        
        if not queue:
            raise ValueError(f"Queue not found: {queue_uuid}")
        
        return queue.to_dict()
