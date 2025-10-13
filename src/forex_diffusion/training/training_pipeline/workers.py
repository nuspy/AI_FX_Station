"""
QThread worker classes for async training operations in GUI.

Provides worker threads that run training operations without blocking the GUI,
with progress signals and proper exception handling.
"""

from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any, List, Optional
import logging

from .training_orchestrator import TrainingOrchestrator
from .database import session_scope

logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    """
    Worker thread for running training queue.

    Signals:
        progress: (current, total, status_message)
        finished: (results_dict)
        error: (error_message)
        started: ()
        cancelled: ()
    """

    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, status
    finished = pyqtSignal(dict)  # results
    error = pyqtSignal(str)  # error message
    started = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(
        self,
        queue_id: int,
        orchestrator: Optional[TrainingOrchestrator] = None,
        parent=None
    ):
        """
        Initialize TrainingWorker.

        Args:
            queue_id: Training queue ID
            orchestrator: TrainingOrchestrator instance (or creates new one)
            parent: Parent QObject
        """
        super().__init__(parent)
        self.queue_id = queue_id
        self.orchestrator = orchestrator or TrainingOrchestrator()
        self._is_cancelled = False

    def run(self):
        """Run training in background thread."""
        try:
            self.started.emit()

            logger.info(f"Starting training worker for queue {self.queue_id}")

            # Set progress callback
            def progress_callback(current: int, total: int, status: str):
                if not self._is_cancelled:
                    self.progress.emit(current, total, status)

            self.orchestrator.progress_callback = progress_callback

            # Run training
            results = self.orchestrator.train_models_grid(self.queue_id, progress_callback)

            # Check if cancelled
            if self._is_cancelled:
                self.cancelled.emit()
                logger.info(f"Training worker {self.queue_id} cancelled")
            else:
                self.finished.emit(results)
                logger.info(f"Training worker {self.queue_id} completed: {results}")

        except Exception as e:
            logger.error(f"Training worker {self.queue_id} error: {e}", exc_info=True)
            self.error.emit(str(e))

    def cancel(self):
        """Cancel training."""
        self._is_cancelled = True
        self.orchestrator.cancel_training()
        logger.info(f"Cancellation requested for training worker {self.queue_id}")


class QueueCreationWorker(QThread):
    """
    Worker thread for creating training queue from grid parameters.

    Signals:
        finished: (queue_id)
        error: (error_message)
        progress: (status_message)
    """

    finished = pyqtSignal(int)  # queue_id
    error = pyqtSignal(str)
    progress = pyqtSignal(str)  # status message

    def __init__(
        self,
        grid_params: Dict[str, List[Any]],
        skip_existing: bool = True,
        priority: int = 0,
        orchestrator: Optional[TrainingOrchestrator] = None,
        parent=None
    ):
        """
        Initialize QueueCreationWorker.

        Args:
            grid_params: Parameter grid
            skip_existing: Skip already-trained configs
            priority: Queue priority
            orchestrator: TrainingOrchestrator instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.grid_params = grid_params
        self.skip_existing = skip_existing
        self.priority = priority
        self.orchestrator = orchestrator or TrainingOrchestrator()

    def run(self):
        """Create queue in background."""
        try:
            self.progress.emit("Generating configuration grid...")

            queue_id = self.orchestrator.create_training_queue(
                grid_params=self.grid_params,
                skip_existing=self.skip_existing,
                priority=self.priority
            )

            self.progress.emit(f"Queue {queue_id} created successfully")
            self.finished.emit(queue_id)

            logger.info(f"Queue creation worker completed: queue_id={queue_id}")

        except Exception as e:
            logger.error(f"Queue creation worker error: {e}", exc_info=True)
            self.error.emit(str(e))


class StorageCleanupWorker(QThread):
    """
    Worker thread for storage cleanup operations.

    Signals:
        finished: (cleanup_stats)
        error: (error_message)
        progress: (status_message)
    """

    finished = pyqtSignal(dict)  # cleanup stats
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        orchestrator: Optional[TrainingOrchestrator] = None,
        parent=None
    ):
        """
        Initialize StorageCleanupWorker.

        Args:
            orchestrator: TrainingOrchestrator instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.orchestrator = orchestrator or TrainingOrchestrator()

    def run(self):
        """Run cleanup in background."""
        try:
            self.progress.emit("Cleaning up non-best models...")

            cleanup_stats = self.orchestrator.cleanup_storage()

            self.progress.emit(
                f"Cleanup complete: freed {cleanup_stats['total_freed_mb']:.2f} MB"
            )
            self.finished.emit(cleanup_stats)

            logger.info(f"Storage cleanup worker completed: {cleanup_stats}")

        except Exception as e:
            logger.error(f"Storage cleanup worker error: {e}", exc_info=True)
            self.error.emit(str(e))


class RegimeSummaryWorker(QThread):
    """
    Worker thread for loading regime summary data.

    Signals:
        finished: (regime_summary)
        error: (error_message)
    """

    finished = pyqtSignal(dict)  # regime summary
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        """
        Initialize RegimeSummaryWorker.

        Args:
            parent: Parent QObject
        """
        super().__init__(parent)

    def run(self):
        """Load regime summary in background."""
        try:
            from .regime_manager import RegimeManager

            with session_scope() as session:
                regime_manager = RegimeManager(session)
                summary = regime_manager.get_regime_summary()

            self.finished.emit(summary)

            logger.info("Regime summary worker completed")

        except Exception as e:
            logger.error(f"Regime summary worker error: {e}", exc_info=True)
            self.error.emit(str(e))


class TrainingHistoryWorker(QThread):
    """
    Worker thread for loading training history.

    Signals:
        finished: (training_runs)
        error: (error_message)
    """

    finished = pyqtSignal(list)  # list of training runs
    error = pyqtSignal(str)

    def __init__(
        self,
        symbol: Optional[str] = None,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        parent=None
    ):
        """
        Initialize TrainingHistoryWorker.

        Args:
            symbol: Filter by symbol
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum results
            parent: Parent QObject
        """
        super().__init__(parent)
        self.symbol = symbol
        self.model_type = model_type
        self.status = status
        self.limit = limit

    def run(self):
        """Load training history in background."""
        try:
            from .database import get_training_runs_summary

            with session_scope() as session:
                runs = get_training_runs_summary(
                    session,
                    symbol=self.symbol,
                    model_type=self.model_type,
                    status=self.status,
                    limit=self.limit
                )

                # Convert to dicts for signal
                runs_data = [
                    {
                        'id': run.id,
                        'run_uuid': run.run_uuid,
                        'status': run.status,
                        'model_type': run.model_type,
                        'symbol': run.symbol,
                        'base_timeframe': run.base_timeframe,
                        'created_at': run.created_at.isoformat() if run.created_at else None,
                        'is_model_kept': run.is_model_kept,
                        'best_regimes': run.best_regimes
                    }
                    for run in runs
                ]

            self.finished.emit(runs_data)

            logger.info(f"Training history worker completed: {len(runs_data)} runs")

        except Exception as e:
            logger.error(f"Training history worker error: {e}", exc_info=True)
            self.error.emit(str(e))


class StorageStatsWorker(QThread):
    """
    Worker thread for calculating storage statistics.

    Signals:
        finished: (storage_stats)
        error: (error_message)
    """

    finished = pyqtSignal(dict)  # storage stats
    error = pyqtSignal(str)

    def __init__(
        self,
        orchestrator: Optional[TrainingOrchestrator] = None,
        parent=None
    ):
        """
        Initialize StorageStatsWorker.

        Args:
            orchestrator: TrainingOrchestrator instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.orchestrator = orchestrator or TrainingOrchestrator()

    def run(self):
        """Calculate storage stats in background."""
        try:
            with session_scope() as session:
                stats = self.orchestrator.model_file_manager.get_storage_stats(session)

            self.finished.emit(stats)

            logger.info(f"Storage stats worker completed: {stats}")

        except Exception as e:
            logger.error(f"Storage stats worker error: {e}", exc_info=True)
            self.error.emit(str(e))


class QueueStatusWorker(QThread):
    """
    Worker thread for monitoring queue status.

    Signals:
        status_updated: (status_dict)
        error: (error_message)
    """

    status_updated = pyqtSignal(dict)  # queue status
    error = pyqtSignal(str)

    def __init__(
        self,
        queue_id: int,
        orchestrator: Optional[TrainingOrchestrator] = None,
        parent=None
    ):
        """
        Initialize QueueStatusWorker.

        Args:
            queue_id: Queue ID to monitor
            orchestrator: TrainingOrchestrator instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.queue_id = queue_id
        self.orchestrator = orchestrator or TrainingOrchestrator()
        self._is_stopped = False

    def run(self):
        """Monitor queue status."""
        try:
            import time

            while not self._is_stopped:
                status = self.orchestrator.get_training_status(self.queue_id)
                self.status_updated.emit(status)

                # Stop if queue is completed or cancelled
                if status['status'] in ['completed', 'cancelled']:
                    break

                # Sleep before next check
                time.sleep(2)  # Update every 2 seconds

        except Exception as e:
            logger.error(f"Queue status worker error: {e}", exc_info=True)
            self.error.emit(str(e))

    def stop(self):
        """Stop monitoring."""
        self._is_stopped = True
