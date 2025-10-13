"""
Checkpoint Manager - Robust checkpoint and resume system for hyperparameter optimization workflows.

Handles:
- Training checkpoints
- Backtest checkpoints
- Validation checkpoints
- Hyperparameter search state
- Automatic resume from last completed step
"""

from __future__ import annotations

import json
import time
import shutil
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict, field

from loguru import logger

from ..utils.user_settings import SETTINGS_DIR


class WorkflowStage(Enum):
    """Stages in optimization workflow"""

    SETUP = "setup"
    TRAINING = "training"
    BACKTEST = "backtest"
    VALIDATION = "validation"
    COMPLETE = "complete"
    FAILED = "failed"


class TaskStatus(Enum):
    """Status of individual task"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskCheckpoint:
    """Checkpoint for a single task (e.g., train model #100)"""

    task_id: str
    task_type: str  # 'training', 'backtest', 'validation'
    status: TaskStatus
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskCheckpoint":
        """Create from dictionary"""
        data = data.copy()
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for entire optimization workflow"""

    workflow_id: str
    stage: WorkflowStage
    config: Dict[str, Any]
    tasks: List[TaskCheckpoint]
    started_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "stage": self.stage.value,
            "config": self.config,
            "tasks": [t.to_dict() for t in self.tasks],
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        """Create from dictionary"""
        data = data.copy()
        data["stage"] = WorkflowStage(data["stage"])
        data["tasks"] = [TaskCheckpoint.from_dict(t) for t in data["tasks"]]
        return cls(**data)


class CheckpointManager:
    """
    Manages checkpoints for optimization workflows with automatic resume capability.

    Features:
    - Atomic checkpoint writes (write to temp, then rename)
    - Automatic backup of previous checkpoints
    - Resume from last completed step
    - Progress tracking
    - Cleanup of old checkpoints
    """

    def __init__(
        self,
        workflow_id: str,
        checkpoint_dir: Optional[Path] = None,
        max_backups: int = 5,
        auto_save_interval: int = 60,  # seconds
    ):
        """
        Initialize checkpoint manager.

        Args:
            workflow_id: Unique workflow identifier
            checkpoint_dir: Directory to store checkpoints
            max_backups: Maximum number of checkpoint backups to keep
            auto_save_interval: Auto-save interval in seconds
        """
        self.workflow_id = workflow_id
        self.checkpoint_dir = checkpoint_dir or (SETTINGS_DIR / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_backups = max_backups
        self.auto_save_interval = auto_save_interval

        self.checkpoint_file = self.checkpoint_dir / f"{workflow_id}.json"
        self.backup_dir = self.checkpoint_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Current workflow state
        self.workflow: Optional[WorkflowCheckpoint] = None
        self.last_save_time = 0.0

        # Load existing checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self) -> bool:
        """Load existing checkpoint if available"""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    self.workflow = WorkflowCheckpoint.from_dict(data)

                logger.info(
                    f"Loaded checkpoint for workflow '{self.workflow_id}' "
                    f"(stage: {self.workflow.stage.value}, tasks: {len(self.workflow.tasks)})"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

        return False

    def initialize_workflow(
        self,
        config: Dict[str, Any],
        task_plan: List[
            Tuple[str, str, Dict[str, Any]]
        ],  # (task_id, task_type, params)
    ) -> WorkflowCheckpoint:
        """
        Initialize a new workflow or resume existing one.

        Args:
            config: Workflow configuration
            task_plan: List of tasks to execute

        Returns:
            WorkflowCheckpoint instance
        """
        if self.workflow is not None:
            logger.info(f"Resuming existing workflow '{self.workflow_id}'")
            return self.workflow

        # Create new workflow
        tasks = [
            TaskCheckpoint(
                task_id=task_id,
                task_type=task_type,
                status=TaskStatus.PENDING,
                params=params,
            )
            for task_id, task_type, params in task_plan
        ]

        self.workflow = WorkflowCheckpoint(
            workflow_id=self.workflow_id,
            stage=WorkflowStage.SETUP,
            config=config,
            tasks=tasks,
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        self._save_checkpoint()

        logger.info(
            f"Initialized new workflow '{self.workflow_id}' " f"with {len(tasks)} tasks"
        )

        return self.workflow

    def get_next_task(self) -> Optional[TaskCheckpoint]:
        """
        Get next pending task to execute.

        Returns:
            TaskCheckpoint or None if all tasks complete
        """
        if self.workflow is None:
            return None

        for task in self.workflow.tasks:
            if task.status == TaskStatus.PENDING:
                return task

        return None

    def start_task(self, task_id: str) -> bool:
        """
        Mark task as in progress.

        Args:
            task_id: Task identifier

        Returns:
            bool: True if successful
        """
        task = self._find_task(task_id)
        if not task:
            logger.error(f"Task '{task_id}' not found")
            return False

        if task.status != TaskStatus.PENDING:
            logger.warning(
                f"Task '{task_id}' already started (status: {task.status.value})"
            )
            return False

        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now().isoformat()

        self._update_workflow_stage()
        self._save_checkpoint()

        logger.info(f"Started task '{task_id}' ({task.task_type})")
        return True

    def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Mark task as completed with results.

        Args:
            task_id: Task identifier
            result: Task result data
            artifacts: Paths to artifacts (model files, etc.)

        Returns:
            bool: True if successful
        """
        task = self._find_task(task_id)
        if not task:
            logger.error(f"Task '{task_id}' not found")
            return False

        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = datetime.now().isoformat()

        if artifacts:
            task.artifacts.update(artifacts)

        self._update_workflow_stage()
        self._save_checkpoint()

        logger.info(f"Completed task '{task_id}' ({task.task_type})")
        return True

    def fail_task(self, task_id: str, error: str) -> bool:
        """
        Mark task as failed.

        Args:
            task_id: Task identifier
            error: Error message

        Returns:
            bool: True if successful
        """
        task = self._find_task(task_id)
        if not task:
            logger.error(f"Task '{task_id}' not found")
            return False

        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = datetime.now().isoformat()

        self.workflow.stage = WorkflowStage.FAILED

        self._save_checkpoint()

        logger.error(f"Failed task '{task_id}': {error}")
        return True

    def skip_task(self, task_id: str, reason: str) -> bool:
        """
        Skip a task (e.g., if conditions not met).

        Args:
            task_id: Task identifier
            reason: Reason for skipping

        Returns:
            bool: True if successful
        """
        task = self._find_task(task_id)
        if not task:
            return False

        task.status = TaskStatus.SKIPPED
        task.error = f"Skipped: {reason}"
        task.completed_at = datetime.now().isoformat()

        self._save_checkpoint()

        logger.info(f"Skipped task '{task_id}': {reason}")
        return True

    def get_progress(self) -> Dict[str, Any]:
        """
        Get workflow progress statistics.

        Returns:
            Dictionary with progress info
        """
        if not self.workflow:
            return {"total": 0, "completed": 0, "progress": 0.0}

        total = len(self.workflow.tasks)
        completed = sum(
            1 for t in self.workflow.tasks if t.status == TaskStatus.COMPLETED
        )
        in_progress = sum(
            1 for t in self.workflow.tasks if t.status == TaskStatus.IN_PROGRESS
        )
        failed = sum(1 for t in self.workflow.tasks if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.workflow.tasks if t.status == TaskStatus.SKIPPED)

        return {
            "workflow_id": self.workflow_id,
            "stage": self.workflow.stage.value,
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "skipped": skipped,
            "pending": total - completed - in_progress - failed - skipped,
            "progress": completed / total if total > 0 else 0.0,
            "started_at": self.workflow.started_at,
            "updated_at": self.workflow.updated_at,
        }

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get results from all completed tasks.

        Returns:
            List of task results
        """
        if not self.workflow:
            return []

        results = []
        for task in self.workflow.tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                results.append(
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "params": task.params,
                        "result": task.result,
                        "artifacts": task.artifacts,
                    }
                )

        return results

    def is_resumable(self) -> bool:
        """Check if workflow can be resumed"""
        if not self.workflow:
            return False

        # Can resume if not complete and not failed
        if self.workflow.stage in (WorkflowStage.COMPLETE, WorkflowStage.FAILED):
            return False

        # Has pending tasks
        return any(t.status == TaskStatus.PENDING for t in self.workflow.tasks)

    def auto_save(self):
        """Auto-save checkpoint if interval elapsed"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.auto_save_interval:
            self._save_checkpoint()

    def _find_task(self, task_id: str) -> Optional[TaskCheckpoint]:
        """Find task by ID"""
        if not self.workflow:
            return None

        for task in self.workflow.tasks:
            if task.task_id == task_id:
                return task

        return None

    def _update_workflow_stage(self):
        """Update workflow stage based on task statuses"""
        if not self.workflow:
            return

        all_tasks = self.workflow.tasks
        completed = [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in all_tasks if t.status == TaskStatus.FAILED]
        in_progress = [t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS]

        if failed:
            self.workflow.stage = WorkflowStage.FAILED
        elif len(completed) == len(all_tasks):
            self.workflow.stage = WorkflowStage.COMPLETE
        elif in_progress:
            # Determine stage based on current task type
            current_task = in_progress[0]
            if current_task.task_type == "training":
                self.workflow.stage = WorkflowStage.TRAINING
            elif current_task.task_type == "backtest":
                self.workflow.stage = WorkflowStage.BACKTEST
            elif current_task.task_type == "validation":
                self.workflow.stage = WorkflowStage.VALIDATION

    def _save_checkpoint(self):
        """Save checkpoint to file (atomic write)"""
        if not self.workflow:
            return

        self.workflow.updated_at = datetime.now().isoformat()

        # Create backup of existing checkpoint
        if self.checkpoint_file.exists():
            self._backup_checkpoint()

        # Atomic write: write to temp file, then rename
        temp_file = self.checkpoint_file.with_suffix(".tmp")

        try:
            with open(temp_file, "w") as f:
                json.dump(self.workflow.to_dict(), f, indent=2)

            # Atomic rename
            temp_file.replace(self.checkpoint_file)

            self.last_save_time = time.time()

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _backup_checkpoint(self):
        """Create backup of current checkpoint"""
        if not self.checkpoint_file.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{self.workflow_id}_{timestamp}.json"

        try:
            shutil.copy2(self.checkpoint_file, backup_file)

            # Cleanup old backups
            self._cleanup_old_backups()

        except Exception as e:
            logger.warning(f"Failed to create checkpoint backup: {e}")

    def _cleanup_old_backups(self):
        """Remove old backup files, keeping only max_backups most recent"""
        backups = sorted(
            self.backup_dir.glob(f"{self.workflow_id}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove old backups
        for backup in backups[self.max_backups :]:
            try:
                backup.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete old backup {backup}: {e}")

    def clear_checkpoint(self):
        """Clear checkpoint (use with caution!)"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()

            self.workflow = None
            logger.info(f"Cleared checkpoint for workflow '{self.workflow_id}'")

        except Exception as e:
            logger.error(f"Failed to clear checkpoint: {e}")


def resume_or_create_workflow(
    workflow_id: str,
    config: Dict[str, Any],
    task_generator: callable,  # Function that returns List[Tuple[task_id, task_type, params]]
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[CheckpointManager, bool]:
    """
    Convenience function to resume existing workflow or create new one.

    Args:
        workflow_id: Workflow identifier
        config: Workflow configuration
        task_generator: Function that generates task plan
        checkpoint_dir: Optional checkpoint directory

    Returns:
        Tuple of (CheckpointManager, is_resumed)
    """
    manager = CheckpointManager(workflow_id, checkpoint_dir)

    if manager.workflow is not None and manager.is_resumable():
        logger.info(f"Resuming workflow '{workflow_id}' from checkpoint")
        is_resumed = True
    else:
        # Generate task plan
        task_plan = task_generator(config)
        manager.initialize_workflow(config, task_plan)
        is_resumed = False

    return manager, is_resumed
