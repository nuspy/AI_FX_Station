"""
Model file lifecycle management for training pipeline.

Handles keeping, deleting, and cleanup of model files based on regime
performance. Only models that are best for at least one regime are kept.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from sqlalchemy.orm import Session

from .database import (
    TrainingRun, get_training_run_by_id, get_models_to_delete,
    get_storage_stats, mark_model_as_kept
)

logger = logging.getLogger(__name__)


class ModelFileManager:
    """
    Manages model file storage and cleanup.

    Responsible for marking models as kept/deletable and removing model
    files that don't improve performance for any regime.
    """

    def __init__(
        self,
        artifacts_dir: str = "./artifacts",
        delete_non_best_models: bool = True
    ):
        """
        Initialize ModelFileManager.

        Args:
            artifacts_dir: Directory where model files are stored
            delete_non_best_models: If True, automatically delete non-best models
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.delete_non_best_models = delete_non_best_models

        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def keep_model_file(
        self,
        session: Session,
        training_run_id: int,
        best_regimes: List[str]
    ) -> bool:
        """
        Mark a model as kept with the regimes it's best for.

        Args:
            session: SQLAlchemy session
            training_run_id: Training run ID
            best_regimes: List of regime names where this model is best

        Returns:
            True if marked successfully, False otherwise
        """
        try:
            mark_model_as_kept(session, training_run_id, best_regimes)
            session.commit()

            run = get_training_run_by_id(session, training_run_id)
            if run and run.model_file_path:
                logger.info(
                    f"Marked model as kept: run_id={training_run_id}, "
                    f"regimes={best_regimes}, path={run.model_file_path}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to mark model as kept: {e}")
            session.rollback()
            return False

    def delete_model_file(
        self,
        session: Session,
        training_run_id: int
    ) -> bool:
        """
        Delete a model file from filesystem and update database.

        Args:
            session: SQLAlchemy session
            training_run_id: Training run ID

        Returns:
            True if deleted successfully, False otherwise
        """
        run = get_training_run_by_id(session, training_run_id)

        if not run:
            logger.error(f"Training run {training_run_id} not found")
            return False

        if not run.model_file_path:
            logger.warning(f"Training run {training_run_id} has no model file path")
            return False

        model_path = Path(run.model_file_path)

        try:
            # Delete file if it exists
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Deleted model file: {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")

            # Update database
            run.model_file_path = None
            run.model_file_size_bytes = None
            run.is_model_kept = False
            session.commit()

            return True

        except Exception as e:
            logger.error(f"Failed to delete model file: {e}")
            session.rollback()
            return False

    def cleanup_non_best_models(self, session: Session) -> Dict[str, Any]:
        """
        Delete all model files that are not marked as kept.

        Args:
            session: SQLAlchemy session

        Returns:
            Dictionary with cleanup statistics
        """
        if not self.delete_non_best_models:
            logger.info("Auto-delete disabled, skipping cleanup")
            return {
                'deleted_count': 0,
                'freed_bytes': 0,
                'failed_count': 0,
                'message': 'Auto-delete disabled'
            }

        # Get models to delete
        models_to_delete = get_models_to_delete(session)

        deleted_count = 0
        failed_count = 0
        freed_bytes = 0

        logger.info(f"Found {len(models_to_delete)} models to delete")

        for run in models_to_delete:
            file_size = run.model_file_size_bytes or 0

            if self.delete_model_file(session, run.id):
                deleted_count += 1
                freed_bytes += file_size
            else:
                failed_count += 1

        freed_mb = freed_bytes / (1024 * 1024)

        logger.info(
            f"Cleanup complete: deleted {deleted_count} models, "
            f"freed {freed_mb:.2f} MB, {failed_count} failures"
        )

        return {
            'deleted_count': deleted_count,
            'freed_bytes': freed_bytes,
            'freed_mb': freed_mb,
            'failed_count': failed_count
        }

    def cleanup_orphaned_files(self, session: Session) -> Dict[str, Any]:
        """
        Find and remove model files that have no database record.

        Scans artifacts directory for model files not referenced in database.

        Args:
            session: SQLAlchemy session

        Returns:
            Dictionary with orphan cleanup statistics
        """
        # Get all model file paths from database
        all_runs = session.query(TrainingRun).filter(
            TrainingRun.model_file_path.isnot(None)
        ).all()

        db_model_paths = {Path(run.model_file_path) for run in all_runs}

        # Scan artifacts directory for model files
        model_extensions = ['.pkl', '.joblib', '.h5', '.pt', '.pth', '.onnx', '.bin']
        found_files = []

        for ext in model_extensions:
            found_files.extend(self.artifacts_dir.rglob(f'*{ext}'))

        # Find orphans (files not in database)
        orphaned_files = [f for f in found_files if f not in db_model_paths]

        deleted_count = 0
        failed_count = 0
        freed_bytes = 0

        logger.info(f"Found {len(orphaned_files)} orphaned model files")

        for orphan_file in orphaned_files:
            try:
                file_size = orphan_file.stat().st_size
                orphan_file.unlink()
                deleted_count += 1
                freed_bytes += file_size
                logger.info(f"Deleted orphaned file: {orphan_file}")

            except Exception as e:
                logger.error(f"Failed to delete orphaned file {orphan_file}: {e}")
                failed_count += 1

        freed_mb = freed_bytes / (1024 * 1024)

        logger.info(
            f"Orphan cleanup complete: deleted {deleted_count} files, "
            f"freed {freed_mb:.2f} MB, {failed_count} failures"
        )

        return {
            'orphaned_count': len(orphaned_files),
            'deleted_count': deleted_count,
            'freed_bytes': freed_bytes,
            'freed_mb': freed_mb,
            'failed_count': failed_count
        }

    def get_storage_stats(self, session: Session) -> Dict[str, Any]:
        """
        Get storage statistics for model files.

        Args:
            session: SQLAlchemy session

        Returns:
            Dictionary with storage statistics
        """
        db_stats = get_storage_stats(session)

        # Calculate directory size
        total_dir_size = sum(
            f.stat().st_size
            for f in self.artifacts_dir.rglob('*')
            if f.is_file()
        )

        return {
            **db_stats,
            'artifacts_dir': str(self.artifacts_dir),
            'total_dir_size_bytes': total_dir_size,
            'total_dir_size_mb': total_dir_size / (1024 * 1024),
            'potential_savings_mb': (
                db_stats['total_size_bytes'] - db_stats['kept_size_bytes']
            ) / (1024 * 1024)
        }

    def verify_model_files(self, session: Session) -> Dict[str, Any]:
        """
        Verify that model files referenced in database actually exist.

        Args:
            session: SQLAlchemy session

        Returns:
            Dictionary with verification results
        """
        all_runs = session.query(TrainingRun).filter(
            TrainingRun.model_file_path.isnot(None)
        ).all()

        missing_files = []
        existing_files = []
        total_size_existing = 0

        for run in all_runs:
            model_path = Path(run.model_file_path)

            if model_path.exists():
                existing_files.append(run.id)
                total_size_existing += run.model_file_size_bytes or 0
            else:
                missing_files.append({
                    'run_id': run.id,
                    'expected_path': str(model_path),
                    'is_kept': run.is_model_kept
                })

        logger.info(
            f"Verification: {len(existing_files)} files exist, "
            f"{len(missing_files)} missing"
        )

        return {
            'total_records': len(all_runs),
            'existing_files': len(existing_files),
            'missing_files': len(missing_files),
            'missing_details': missing_files,
            'total_size_existing_mb': total_size_existing / (1024 * 1024)
        }

    def get_model_file_info(
        self,
        session: Session,
        training_run_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model file.

        Args:
            session: SQLAlchemy session
            training_run_id: Training run ID

        Returns:
            Dictionary with model file info, or None if not found
        """
        run = get_training_run_by_id(session, training_run_id)

        if not run or not run.model_file_path:
            return None

        model_path = Path(run.model_file_path)

        file_exists = model_path.exists()
        actual_size = model_path.stat().st_size if file_exists else 0

        return {
            'training_run_id': run.id,
            'run_uuid': run.run_uuid,
            'model_file_path': str(model_path),
            'file_exists': file_exists,
            'is_kept': run.is_model_kept,
            'best_regimes': run.best_regimes,
            'db_size_bytes': run.model_file_size_bytes,
            'actual_size_bytes': actual_size,
            'size_mismatch': (run.model_file_size_bytes or 0) != actual_size,
            'model_type': run.model_type,
            'created_at': run.created_at.isoformat() if run.created_at else None
        }

    def compress_old_models(
        self,
        session: Session,
        older_than_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compress model files older than specified days.

        Note: This is a placeholder. Actual compression implementation
        would require specific compression libraries and format handling.

        Args:
            session: SQLAlchemy session
            older_than_days: Compress models older than this

        Returns:
            Dictionary with compression statistics
        """
        logger.warning("Model compression not yet implemented")

        return {
            'compressed_count': 0,
            'saved_bytes': 0,
            'message': 'Compression not implemented'
        }

    def export_storage_report(
        self,
        session: Session,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a detailed storage report.

        Args:
            session: SQLAlchemy session
            output_path: Optional path for report file

        Returns:
            Report text
        """
        stats = self.get_storage_stats(session)
        verification = self.verify_model_files(session)

        report_lines = [
            "=" * 60,
            "MODEL STORAGE REPORT",
            "=" * 60,
            "",
            "STORAGE SUMMARY:",
            f"  Artifacts Directory: {stats['artifacts_dir']}",
            f"  Total Models: {stats['total_models_count']}",
            f"  Kept Models: {stats['kept_models_count']}",
            f"  Deletable Models: {stats['deletable_count']}",
            "",
            "SIZE BREAKDOWN:",
            f"  Total Size: {stats['total_size_mb']:.2f} MB",
            f"  Kept Size: {stats['kept_size_mb']:.2f} MB",
            f"  Deletable Size: {stats['deletable_size_mb']:.2f} MB",
            f"  Potential Savings: {stats['potential_savings_mb']:.2f} MB",
            "",
            "FILE VERIFICATION:",
            f"  Database Records: {verification['total_records']}",
            f"  Existing Files: {verification['existing_files']}",
            f"  Missing Files: {verification['missing_files']}",
            "",
            "=" * 60
        ]

        report_text = "\n".join(report_lines)

        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Storage report saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")

        return report_text
