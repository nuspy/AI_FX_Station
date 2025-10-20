"""
Model artifact management system for versioned model storage and loading.

Handles checkpoints, metadata, preprocessing stats, and configuration.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import shutil

import numpy as np
from loguru import logger


class ModelArtifact:
    """
    Container for all model artifacts.

    Includes:
    - Model checkpoint (.ckpt)
    - Metadata (.meta.json)
    - Configuration (.config.json)
    - Preprocessing statistics (.stats.npz)
    - Training history (.history.json)
    """

    def __init__(
        self,
        checkpoint_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, np.ndarray]] = None,
        history: Optional[Dict[str, List[float]]] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata = metadata or {}
        self.config = config or {}
        self.stats = stats or {}
        self.history = history or {}

        # Derived paths
        self.base_path = self.checkpoint_path.parent
        self.base_name = self.checkpoint_path.stem

    @property
    def meta_path(self) -> Path:
        """Path to metadata file."""
        return self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".meta.json")

    @property
    def config_path(self) -> Path:
        """Path to config file."""
        return self.base_path / f"{self.base_name}.config.json"

    @property
    def stats_path(self) -> Path:
        """Path to stats file."""
        return self.base_path / f"{self.base_name}.stats.npz"

    @property
    def history_path(self) -> Path:
        """Path to history file."""
        return self.base_path / f"{self.base_name}.history.json"

    def save(self):
        """Save all artifacts to disk."""
        # Metadata
        if self.metadata:
            self.meta_path.write_text(json.dumps(self.metadata, indent=2))
            logger.info(f"Saved metadata: {self.meta_path}")

        # Config
        if self.config:
            self.config_path.write_text(json.dumps(self.config, indent=2))
            logger.info(f"Saved config: {self.config_path}")

        # Stats
        if self.stats:
            np.savez(self.stats_path, **self.stats)
            logger.info(f"Saved stats: {self.stats_path}")

        # History
        if self.history:
            self.history_path.write_text(json.dumps(self.history, indent=2))
            logger.info(f"Saved history: {self.history_path}")

    @classmethod
    def load(cls, checkpoint_path: Path) -> 'ModelArtifact':
        """
        Load all artifacts from checkpoint path.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            ModelArtifact instance
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        artifact = cls(checkpoint_path=checkpoint_path)

        # Load metadata
        if artifact.meta_path.exists():
            artifact.metadata = json.loads(artifact.meta_path.read_text())

        # Load config
        if artifact.config_path.exists():
            artifact.config = json.loads(artifact.config_path.read_text())

        # Load stats
        if artifact.stats_path.exists():
            stats_data = np.load(artifact.stats_path)
            artifact.stats = {k: stats_data[k] for k in stats_data.files}

        # Load history
        if artifact.history_path.exists():
            artifact.history = json.loads(artifact.history_path.read_text())

        logger.info(f"Loaded artifact: {checkpoint_path.name}")

        return artifact

    def get_preprocessing_transform(self):
        """
        Get preprocessing transform function from stats.

        Returns:
            Function that transforms input data
        """
        if 'mu' not in self.stats or 'sigma' not in self.stats:
            raise ValueError("Preprocessing stats (mu, sigma) not found")

        mu = self.stats['mu']
        sigma = self.stats['sigma']

        def transform(x: np.ndarray) -> np.ndarray:
            """Standardize input."""
            return (x - mu) / (sigma + 1e-8)

        def inverse_transform(x_norm: np.ndarray) -> np.ndarray:
            """Denormalize output."""
            return x_norm * sigma + mu

        return transform, inverse_transform

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'checkpoint_path': str(self.checkpoint_path),
            'metadata': self.metadata,
            'config': self.config,
            'stats': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.stats.items()},
            'history': self.history
        }


class ArtifactManager:
    """
    Manages model artifacts across multiple versions.

    Provides versioning, cataloging, and retrieval of model checkpoints.
    """

    def __init__(self, artifacts_dir: Path):
        """
        Initialize artifact manager.

        Args:
            artifacts_dir: Root directory for artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.catalog_path = self.artifacts_dir / "catalog.json"
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> Dict[str, Any]:
        """Load catalog from disk."""
        if self.catalog_path.exists():
            return json.loads(self.catalog_path.read_text())
        return {'artifacts': [], 'metadata': {}}

    def _save_catalog(self):
        """Save catalog to disk."""
        self.catalog_path.write_text(json.dumps(self.catalog, indent=2))

    def register_artifact(
        self,
        artifact: ModelArtifact,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register artifact in catalog.

        Args:
            artifact: Model artifact
            version: Version string (auto-generated if None)
            tags: Optional tags for filtering
            description: Optional description

        Returns:
            Artifact ID
        """
        if version is None:
            # Auto-generate version from timestamp
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        artifact_id = f"{artifact.base_name}_{version}"

        # Save artifact files
        artifact.save()

        # Add to catalog
        entry = {
            'id': artifact_id,
            'version': version,
            'checkpoint_path': str(artifact.checkpoint_path),
            'created_at': datetime.now().isoformat(),
            'tags': tags or [],
            'description': description or "",
            'metadata': artifact.metadata,
            'config': artifact.config
        }

        self.catalog['artifacts'].append(entry)
        self._save_catalog()

        logger.info(f"Registered artifact: {artifact_id}")

        return artifact_id

    def get_artifact(
        self,
        artifact_id: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        latest: bool = False
    ) -> Optional[ModelArtifact]:
        """
        Retrieve artifact from catalog.

        Args:
            artifact_id: Specific artifact ID
            version: Version to retrieve
            tags: Filter by tags
            latest: Get latest artifact

        Returns:
            ModelArtifact or None
        """
        candidates = self.catalog['artifacts']

        # Filter by ID
        if artifact_id:
            candidates = [a for a in candidates if a['id'] == artifact_id]

        # Filter by version
        if version:
            candidates = [a for a in candidates if a['version'] == version]

        # Filter by tags
        if tags:
            candidates = [a for a in candidates if any(t in a['tags'] for t in tags)]

        if not candidates:
            return None

        # Get latest if requested
        if latest:
            candidates.sort(key=lambda x: x['created_at'], reverse=True)

        selected = candidates[0]

        # Load artifact
        checkpoint_path = Path(selected['checkpoint_path'])
        return ModelArtifact.load(checkpoint_path)

    def list_artifacts(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List artifacts in catalog.

        Args:
            tags: Filter by tags
            limit: Maximum number to return

        Returns:
            List of artifact metadata dicts
        """
        artifacts = self.catalog['artifacts']

        # Filter by tags
        if tags:
            artifacts = [a for a in artifacts if any(t in a['tags'] for t in tags)]

        # Sort by creation time (newest first)
        artifacts.sort(key=lambda x: x['created_at'], reverse=True)

        # Limit
        return artifacts[:limit]

    def delete_artifact(self, artifact_id: str, remove_files: bool = True):
        """
        Delete artifact from catalog.

        Args:
            artifact_id: Artifact ID to delete
            remove_files: Whether to delete files from disk
        """
        # Find artifact
        artifact_entry = next((a for a in self.catalog['artifacts'] if a['id'] == artifact_id), None)

        if not artifact_entry:
            logger.warning(f"Artifact not found: {artifact_id}")
            return

        # Remove files if requested
        if remove_files:
            checkpoint_path = Path(artifact_entry['checkpoint_path'])
            artifact = ModelArtifact.load(checkpoint_path)

            for path in [checkpoint_path, artifact.meta_path, artifact.config_path,
                         artifact.stats_path, artifact.history_path]:
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted: {path}")

        # Remove from catalog
        self.catalog['artifacts'] = [a for a in self.catalog['artifacts'] if a['id'] != artifact_id]
        self._save_catalog()

        logger.info(f"Deleted artifact: {artifact_id}")

    def export_artifact(self, artifact_id: str, export_dir: Path) -> Path:
        """
        Export artifact to directory.

        Args:
            artifact_id: Artifact ID
            export_dir: Export directory

        Returns:
            Path to exported artifact directory
        """
        # Get artifact
        artifact_entry = next((a for a in self.catalog['artifacts'] if a['id'] == artifact_id), None)

        if not artifact_entry:
            raise ValueError(f"Artifact not found: {artifact_id}")

        # Load artifact
        checkpoint_path = Path(artifact_entry['checkpoint_path'])
        artifact = ModelArtifact.load(checkpoint_path)

        # Create export directory
        export_path = Path(export_dir) / artifact_id
        export_path.mkdir(parents=True, exist_ok=True)

        # Copy all files
        for src_path in [checkpoint_path, artifact.meta_path, artifact.config_path,
                         artifact.stats_path, artifact.history_path]:
            if src_path.exists():
                dst_path = export_path / src_path.name
                shutil.copy2(src_path, dst_path)

        logger.info(f"Exported artifact to: {export_path}")

        return export_path

    def cleanup_old_artifacts(self, max_saved: int = 10, keep_best: bool = True):
        """
        Enforce artifact cleanup policy by removing oldest artifacts.

        PROC-003: Implements automatic cleanup to prevent disk space issues.

        Args:
            max_saved: Maximum number of artifacts to keep (default: 10)
            keep_best: If True, keep artifacts tagged with 'best' or 'production'

        Returns:
            Number of artifacts deleted
        """
        artifacts = self.catalog['artifacts']

        if len(artifacts) <= max_saved:
            logger.debug(f"Artifact count ({len(artifacts)}) within limit ({max_saved})")
            return 0

        # Separate protected artifacts (tagged as 'best' or 'production')
        protected = []
        deletable = []

        for artifact in artifacts:
            tags = artifact.get('tags', [])
            if keep_best and any(tag in ['best', 'production', 'protected'] for tag in tags):
                protected.append(artifact)
            else:
                deletable.append(artifact)

        # Sort deletable by creation time (oldest first)
        deletable.sort(key=lambda x: x['created_at'])

        # Calculate how many to delete
        total_artifacts = len(protected) + len(deletable)
        num_to_delete = total_artifacts - max_saved

        if num_to_delete <= 0:
            logger.info(f"No cleanup needed: {total_artifacts} artifacts ({len(protected)} protected)")
            return 0

        # Ensure we don't delete more than available deletable artifacts
        num_to_delete = min(num_to_delete, len(deletable))

        # Delete oldest artifacts
        deleted_count = 0
        for artifact in deletable[:num_to_delete]:
            try:
                self.delete_artifact(artifact['id'], remove_files=True)
                deleted_count += 1
                logger.info(f"Cleaned up old artifact: {artifact['id']}")
            except Exception as e:
                logger.error(f"Failed to delete artifact {artifact['id']}: {e}")

        logger.info(
            f"Artifact cleanup complete: deleted {deleted_count}/{num_to_delete} artifacts "
            f"(kept {len(protected)} protected, {len(deletable) - deleted_count} recent)"
        )

        return deleted_count

    def get_artifacts_disk_usage(self) -> Dict[str, Any]:
        """
        Calculate total disk usage of artifacts directory.

        PROC-003: Provides disk usage information for GUI warnings.

        Returns:
            Dictionary with disk usage statistics
        """
        total_size = 0
        file_count = 0
        artifact_count = len(self.catalog['artifacts'])

        # Calculate size of all artifacts
        for artifact_entry in self.catalog['artifacts']:
            checkpoint_path = Path(artifact_entry['checkpoint_path'])

            # Check all associated files
            try:
                artifact = ModelArtifact.load(checkpoint_path)
                for path in [checkpoint_path, artifact.meta_path, artifact.config_path,
                             artifact.stats_path, artifact.history_path]:
                    if path.exists():
                        total_size += path.stat().st_size
                        file_count += 1
            except Exception as e:
                logger.debug(f"Could not calculate size for {artifact_entry['id']}: {e}")

        # Convert to human-readable format
        size_mb = total_size / (1024 * 1024)
        size_gb = total_size / (1024 * 1024 * 1024)

        return {
            'total_bytes': total_size,
            'total_mb': round(size_mb, 2),
            'total_gb': round(size_gb, 2),
            'file_count': file_count,
            'artifact_count': artifact_count,
            'artifacts_dir': str(self.artifacts_dir)
        }


def create_artifact_from_checkpoint(
    checkpoint_path: Path,
    symbol: str,
    timeframe: str,
    horizon: int,
    channel_order: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    **kwargs
) -> ModelArtifact:
    """
    Create model artifact from training checkpoint.

    Args:
        checkpoint_path: Path to Lightning checkpoint
        symbol: Trading pair
        timeframe: Timeframe
        horizon: Prediction horizon
        channel_order: Feature channel names
        mu: Normalization means
        sigma: Normalization std devs
        **kwargs: Additional metadata

    Returns:
        ModelArtifact instance
    """
    metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'horizon_bars': horizon,
        'channel_order': channel_order,
        'created_at': datetime.now().isoformat(),
        **kwargs
    }

    stats = {
        'mu': mu,
        'sigma': sigma
    }

    artifact = ModelArtifact(
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        stats=stats
    )

    return artifact
