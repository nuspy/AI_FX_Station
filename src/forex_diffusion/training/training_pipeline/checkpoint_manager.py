"""
Checkpoint management for training queue interruption and resume.

Provides functionality to save/load training queue state to JSON checkpoints,
enabling interruption and resumption of long-running training jobs.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Checkpoint version for compatibility checking
CHECKPOINT_VERSION = "1.0"


class CheckpointManager:
    """
    Manages checkpoint creation and restoration for training queues.

    Checkpoints store the complete state of a training queue, enabling
    interruption and resumption without loss of progress.
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints/training_pipeline"):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        queue_uuid: str,
        config_grid: List[Dict[str, Any]],
        current_index: int,
        completed_count: int,
        failed_count: int,
        skipped_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save training queue state to checkpoint file.

        Args:
            queue_uuid: Unique identifier for the queue
            config_grid: Complete list of configurations
            current_index: Current position in grid
            completed_count: Number of completed trainings
            failed_count: Number of failed trainings
            skipped_count: Number of skipped trainings
            metadata: Optional additional metadata

        Returns:
            Path to saved checkpoint file

        Raises:
            IOError: If checkpoint cannot be written
        """
        checkpoint_data = {
            'version': CHECKPOINT_VERSION,
            'queue_uuid': queue_uuid,
            'timestamp': datetime.utcnow().isoformat(),
            'state': {
                'config_grid': config_grid,
                'current_index': current_index,
                'total_configs': len(config_grid),
                'completed_count': completed_count,
                'failed_count': failed_count,
                'skipped_count': skipped_count
            },
            'metadata': metadata or {}
        }

        # Generate checkpoint filename with timestamp
        timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.checkpoint_dir / f"queue_{queue_uuid}_{timestamp_str}.json"

        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            logger.info(f"Saved checkpoint: {checkpoint_file}")
            logger.info(
                f"Progress: {completed_count}/{len(config_grid)} completed, "
                f"{failed_count} failed, {skipped_count} skipped"
            )

            return str(checkpoint_file)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise IOError(f"Could not save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training queue state from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint data

        Raises:
            IOError: If checkpoint cannot be read
            ValueError: If checkpoint format is invalid
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            # Validate checkpoint
            is_valid, error_msg = self.validate_checkpoint(checkpoint_data)
            if not is_valid:
                raise ValueError(f"Invalid checkpoint: {error_msg}")

            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            logger.info(
                f"Queue {checkpoint_data['queue_uuid']}: "
                f"{checkpoint_data['state']['current_index']}/{checkpoint_data['state']['total_configs']} "
                f"(saved at {checkpoint_data['timestamp']})"
            )

            return checkpoint_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse checkpoint JSON: {e}")
            raise IOError(f"Corrupt checkpoint file: {e}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise IOError(f"Could not load checkpoint: {e}")

    def validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate checkpoint data structure and version.

        Args:
            checkpoint_data: Checkpoint dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check version
        if 'version' not in checkpoint_data:
            return False, "Missing version field"

        if checkpoint_data['version'] != CHECKPOINT_VERSION:
            return False, f"Incompatible version: {checkpoint_data['version']} (expected {CHECKPOINT_VERSION})"

        # Check required fields
        required_fields = ['queue_uuid', 'timestamp', 'state']
        for field in required_fields:
            if field not in checkpoint_data:
                return False, f"Missing required field: {field}"

        # Check state structure
        state = checkpoint_data['state']
        required_state_fields = [
            'config_grid', 'current_index', 'total_configs',
            'completed_count', 'failed_count', 'skipped_count'
        ]
        for field in required_state_fields:
            if field not in state:
                return False, f"Missing state field: {field}"

        # Validate data types
        if not isinstance(state['config_grid'], list):
            return False, "config_grid must be a list"

        if not isinstance(state['current_index'], int):
            return False, "current_index must be an integer"

        if state['current_index'] < 0 or state['current_index'] > state['total_configs']:
            return False, f"Invalid current_index: {state['current_index']}"

        # Validate counts
        total_processed = (
            state['completed_count'] +
            state['failed_count'] +
            state['skipped_count']
        )
        if total_processed > state['total_configs']:
            return False, f"Processed count ({total_processed}) exceeds total configs ({state['total_configs']})"

        return True, None

    def list_checkpoints(self, queue_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoint files.

        Args:
            queue_uuid: Optional filter by queue UUID

        Returns:
            List of checkpoint info dictionaries sorted by timestamp (newest first)
        """
        checkpoints = []

        # Find all checkpoint files
        pattern = f"queue_{queue_uuid}_*.json" if queue_uuid else "queue_*.json"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))

        for checkpoint_file in checkpoint_files:
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)

                checkpoints.append({
                    'path': str(checkpoint_file),
                    'filename': checkpoint_file.name,
                    'queue_uuid': data.get('queue_uuid'),
                    'timestamp': data.get('timestamp'),
                    'progress': f"{data['state']['current_index']}/{data['state']['total_configs']}",
                    'completed': data['state']['completed_count'],
                    'failed': data['state']['failed_count'],
                    'skipped': data['state']['skipped_count'],
                    'size_bytes': checkpoint_file.stat().st_size
                })

            except Exception as e:
                logger.warning(f"Could not read checkpoint {checkpoint_file}: {e}")
                continue

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

        return checkpoints

    def get_latest_checkpoint(self, queue_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint for a queue.

        Args:
            queue_uuid: Queue UUID

        Returns:
            Checkpoint data dictionary, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints(queue_uuid)

        if not checkpoints:
            logger.info(f"No checkpoints found for queue {queue_uuid}")
            return None

        latest = checkpoints[0]
        logger.info(f"Found latest checkpoint for queue {queue_uuid}: {latest['filename']}")

        return self.load_checkpoint(latest['path'])

    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            os.remove(checkpoint_path)
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_path}: {e}")
            return False

    def cleanup_old_checkpoints(
        self,
        queue_uuid: str,
        keep_latest: int = 3,
        max_age_days: Optional[int] = None
    ) -> int:
        """
        Clean up old checkpoint files for a queue.

        Args:
            queue_uuid: Queue UUID
            keep_latest: Number of most recent checkpoints to keep
            max_age_days: Delete checkpoints older than this many days

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints(queue_uuid)

        if not checkpoints:
            return 0

        deleted_count = 0

        # Keep the N most recent
        checkpoints_to_delete = checkpoints[keep_latest:]

        for checkpoint in checkpoints_to_delete:
            # Check age if specified
            if max_age_days is not None:
                checkpoint_time = datetime.fromisoformat(checkpoint['timestamp'])
                age_days = (datetime.utcnow() - checkpoint_time).days

                if age_days < max_age_days:
                    continue  # Skip, not old enough

            # Delete this checkpoint
            if self.delete_checkpoint(checkpoint['path']):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old checkpoints for queue {queue_uuid}")

        return deleted_count

    def create_checkpoint_from_queue_db(
        self,
        session,
        queue_id: int
    ) -> Optional[str]:
        """
        Create a checkpoint from a training queue database record.

        Args:
            session: SQLAlchemy session
            queue_id: Training queue ID

        Returns:
            Path to checkpoint file, or None if queue not found
        """
        from .database import get_training_queue_by_id

        queue = get_training_queue_by_id(session, queue_id)

        if not queue:
            logger.error(f"Queue {queue_id} not found in database")
            return None

        metadata = {
            'queue_id': queue.id,
            'status': queue.status,
            'priority': queue.priority,
            'started_at': queue.started_at.isoformat() if queue.started_at else None,
            'created_at': queue.created_at.isoformat() if queue.created_at else None
        }

        return self.save_checkpoint(
            queue_uuid=queue.queue_uuid,
            config_grid=queue.config_grid,
            current_index=queue.current_index,
            completed_count=queue.completed_count,
            failed_count=queue.failed_count,
            skipped_count=queue.skipped_count,
            metadata=metadata
        )

    def resume_from_checkpoint(
        self,
        checkpoint_path: str
    ) -> tuple[str, List[Dict[str, Any]], int, Dict[str, int]]:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (queue_uuid, remaining_configs, current_index, progress_stats)
        """
        checkpoint_data = self.load_checkpoint(checkpoint_path)

        state = checkpoint_data['state']
        queue_uuid = checkpoint_data['queue_uuid']

        # Get remaining configurations
        config_grid = state['config_grid']
        current_index = state['current_index']
        remaining_configs = config_grid[current_index:]

        progress_stats = {
            'total_configs': state['total_configs'],
            'current_index': current_index,
            'completed_count': state['completed_count'],
            'failed_count': state['failed_count'],
            'skipped_count': state['skipped_count'],
            'remaining_count': len(remaining_configs)
        }

        logger.info(
            f"Resuming queue {queue_uuid} from checkpoint: "
            f"{progress_stats['remaining_count']} configs remaining"
        )

        return queue_uuid, remaining_configs, current_index, progress_stats

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all checkpoints.

        Returns:
            Dictionary with checkpoint statistics
        """
        all_checkpoints = self.list_checkpoints()

        total_size = sum(cp['size_bytes'] for cp in all_checkpoints)
        unique_queues = len(set(cp['queue_uuid'] for cp in all_checkpoints))

        return {
            'total_checkpoints': len(all_checkpoints),
            'unique_queues': unique_queues,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'checkpoint_dir': str(self.checkpoint_dir),
            'checkpoints': all_checkpoints
        }
