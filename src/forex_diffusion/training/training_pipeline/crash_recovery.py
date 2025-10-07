"""
Automatic crash recovery system for training pipeline.

Detects interrupted training queues on startup and prompts user to resume.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from sqlalchemy.orm import Session

from .database import (
    session_scope, get_all_training_queues, TrainingQueue,
    update_queue_status
)
from .config_loader import get_config

logger = logging.getLogger(__name__)


class CrashRecoveryManager:
    """
    Manages automatic detection and recovery of crashed training queues.

    Checks for queues that were interrupted (status='running' with no recent
    activity) and provides options to resume them.
    """

    def __init__(self):
        """Initialize crash recovery manager."""
        self.config = get_config()
        self.detect_crash_after_minutes = self.config.detect_crash_after_minutes

    def detect_crashed_queues(self) -> List[Dict[str, Any]]:
        """
        Detect training queues that appear to have crashed.

        A queue is considered crashed if:
        1. Status is 'running' or 'paused'
        2. No activity (updated_at) for more than detect_crash_after_minutes
        3. Not completed or cancelled

        Returns:
            List of crashed queue information dictionaries
        """
        crashed_queues = []

        with session_scope() as session:
            # Get all non-completed queues
            all_queues = get_all_training_queues(session, status_filter=None)

            crash_threshold = datetime.now() - timedelta(
                minutes=self.detect_crash_after_minutes
            )

            for queue in all_queues:
                # Check if queue appears crashed
                if self._is_queue_crashed(queue, crash_threshold):
                    crashed_info = {
                        'id': queue.id,
                        'name': queue.queue_name,
                        'status': queue.status,
                        'created_at': queue.created_at,
                        'updated_at': queue.updated_at,
                        'current_index': queue.current_index,
                        'total_configs': queue.total_configs,
                        'progress_pct': (queue.current_index / queue.total_configs * 100)
                                       if queue.total_configs > 0 else 0,
                        'kept_models': queue.models_kept,
                        'deleted_models': queue.models_deleted
                    }
                    crashed_queues.append(crashed_info)

                    logger.warning(
                        f"Detected crashed queue: {queue.queue_name} "
                        f"(ID: {queue.id}, {crashed_info['progress_pct']:.1f}% complete)"
                    )

        return crashed_queues

    def _is_queue_crashed(
        self,
        queue: TrainingQueue,
        crash_threshold: datetime
    ) -> bool:
        """
        Check if a queue appears to have crashed.

        Args:
            queue: TrainingQueue instance
            crash_threshold: Datetime threshold for considering crashed

        Returns:
            True if queue appears crashed
        """
        # Queue must be in 'running' status
        if queue.status != 'running':
            return False

        # Must not be completed
        if queue.current_index >= queue.total_configs:
            return False

        # Check if updated_at is older than threshold
        # If updated_at is None, use created_at
        last_activity = queue.updated_at or queue.created_at

        if last_activity < crash_threshold:
            return True

        return False

    def mark_queue_as_crashed(self, queue_id: int) -> bool:
        """
        Mark a queue as crashed (paused with crash flag).

        Args:
            queue_id: Queue ID to mark as crashed

        Returns:
            True if successful
        """
        try:
            with session_scope() as session:
                update_queue_status(session, queue_id, 'paused')
                logger.info(f"Marked queue {queue_id} as crashed (paused)")
                return True

        except Exception as e:
            logger.error(f"Failed to mark queue {queue_id} as crashed: {e}")
            return False

    def get_recovery_recommendations(
        self,
        crashed_queues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recovery recommendations for crashed queues.

        Args:
            crashed_queues: List of crashed queue info

        Returns:
            List of recommendations with actions
        """
        recommendations = []

        for queue_info in crashed_queues:
            progress_pct = queue_info['progress_pct']
            current_index = queue_info['current_index']
            total_configs = queue_info['total_configs']
            remaining = total_configs - current_index

            # Determine recommendation based on progress
            if progress_pct < 10:
                recommendation = "Consider restarting from scratch"
                priority = "low"
            elif progress_pct < 50:
                recommendation = "Resume to save partial progress"
                priority = "medium"
            else:
                recommendation = "Resume to complete remaining work"
                priority = "high"

            rec = {
                'queue_id': queue_info['id'],
                'queue_name': queue_info['name'],
                'progress_pct': progress_pct,
                'remaining_configs': remaining,
                'recommendation': recommendation,
                'priority': priority,
                'actions': ['resume', 'restart', 'cancel']
            }

            recommendations.append(rec)

        # Sort by priority (high first)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])

        return recommendations

    def auto_resume_if_enabled(self) -> Optional[List[Dict[str, Any]]]:
        """
        Automatically resume crashed queues if config option is enabled.

        Returns:
            List of resumed queue info, or None if auto-resume disabled
        """
        if not self.config.auto_resume_on_crash:
            return None

        crashed_queues = self.detect_crashed_queues()

        if not crashed_queues:
            logger.info("No crashed queues detected")
            return None

        resumed = []

        for queue_info in crashed_queues:
            queue_id = queue_info['id']

            try:
                with session_scope() as session:
                    # Update status to paused (ready for resume)
                    update_queue_status(session, queue_id, 'paused')

                    logger.info(
                        f"Auto-prepared queue {queue_id} for resume "
                        f"({queue_info['progress_pct']:.1f}% complete)"
                    )

                    resumed.append(queue_info)

            except Exception as e:
                logger.error(f"Failed to auto-prepare queue {queue_id}: {e}")

        return resumed

    def cleanup_old_crashed_queues(self, days_old: int = 30) -> int:
        """
        Clean up very old crashed queues.

        Args:
            days_old: Clean up queues older than this many days

        Returns:
            Number of queues cleaned up
        """
        cleanup_threshold = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0

        with session_scope() as session:
            all_queues = get_all_training_queues(session, status_filter=None)

            for queue in all_queues:
                if queue.status == 'running' and queue.created_at < cleanup_threshold:
                    # Mark as cancelled
                    update_queue_status(session, queue.id, 'cancelled')
                    cleaned_count += 1

                    logger.info(
                        f"Cleaned up old crashed queue: {queue.queue_name} "
                        f"(created {queue.created_at})"
                    )

        return cleaned_count


def check_and_report_crashed_queues() -> Optional[List[Dict[str, Any]]]:
    """
    Convenience function to check for crashed queues and return recommendations.

    Returns:
        List of recommendations for crashed queues, or None if none found
    """
    manager = CrashRecoveryManager()

    # Detect crashed queues
    crashed = manager.detect_crashed_queues()

    if not crashed:
        return None

    # Get recommendations
    recommendations = manager.get_recovery_recommendations(crashed)

    return recommendations


def auto_recover_on_startup() -> Optional[Dict[str, Any]]:
    """
    Run automatic recovery check on application startup.

    Checks for crashed queues and either:
    1. Auto-resumes them if config.auto_resume_on_crash is True
    2. Returns info about crashed queues for manual intervention

    Returns:
        Dictionary with:
        - 'auto_resumed': List of auto-resumed queues (if enabled)
        - 'requires_manual': List of recommendations (if not auto-enabled)
        - None if no crashed queues found
    """
    manager = CrashRecoveryManager()

    # Try auto-resume first
    if manager.config.auto_resume_on_crash:
        resumed = manager.auto_resume_if_enabled()

        if resumed:
            return {
                'auto_resumed': resumed,
                'requires_manual': []
            }

    # If not auto-resume or nothing to resume, check for manual intervention
    crashed = manager.detect_crashed_queues()

    if not crashed:
        return None

    recommendations = manager.get_recovery_recommendations(crashed)

    return {
        'auto_resumed': [],
        'requires_manual': recommendations
    }
