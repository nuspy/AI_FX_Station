"""
SSSD Dataset for Multi-Timeframe Data Loading

Loads and prepares data for SSSD training with multiple timeframes.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from ..config.sssd_config import SSSDConfig


class SSSDTimeSeriesDataset(Dataset):
    """
    Dataset for SSSD multi-timeframe time series data.

    Each sample contains:
    - Features from multiple timeframes (5m, 15m, 1h, 4h)
    - Target price changes at multiple horizons
    - Horizon indices
    """

    def __init__(
        self,
        features_dict: Dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        config: SSSDConfig,
        split: str = "train"
    ):
        """
        Initialize dataset.

        Args:
            features_dict: Dict mapping timeframe to feature DataFrame
                Format: {"5m": df_5m, "15m": df_15m, ...}
                Each df has columns: timestamp, feature_0, feature_1, ...
            targets: DataFrame with target price changes
                Columns: timestamp, target_5m, target_15m, target_1h, target_4h
            config: SSSDConfig object
            split: Dataset split ("train", "val", "test")
        """
        super().__init__()

        self.config = config
        self.split = split
        self.timeframes = config.model.encoder.timeframes
        self.horizons = config.model.horizons.minutes
        self.lookback_bars = config.data.lookback_bars

        # Filter by split
        if split == "train":
            start_date = config.data.train_start
            end_date = config.data.train_end
        elif split == "val":
            start_date = config.data.val_start
            end_date = config.data.val_end
        elif split == "test":
            start_date = config.data.test_start
            end_date = config.data.test_end
        else:
            raise ValueError(f"Unknown split: {split}")

        # Filter features and targets by date
        self.features_dict = {}
        for tf, df in features_dict.items():
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            self.features_dict[tf] = df[mask].reset_index(drop=True)

        mask = (targets['timestamp'] >= start_date) & (targets['timestamp'] <= end_date)
        self.targets = targets[mask].reset_index(drop=True)

        # Get feature columns (exclude timestamp)
        self.feature_cols = [
            col for col in self.features_dict[self.timeframes[0]].columns
            if col != 'timestamp'
        ]

        # Compute valid sample indices
        # Need enough lookback bars for each timeframe
        self._compute_valid_indices()

        logger.info(
            f"Initialized SSSDTimeSeriesDataset ({split}): "
            f"{len(self)} samples, "
            f"timeframes={self.timeframes}, "
            f"horizons={self.horizons}"
        )

    def _compute_valid_indices(self):
        """Compute valid sample indices (with sufficient lookback)."""
        # For each timeframe, we need at least lookback_bars
        min_idx = max(self.lookback_bars.values())

        # Maximum index is constrained by smallest timeframe
        max_idx = min(len(df) for df in self.features_dict.values())

        self.valid_indices = list(range(min_idx, max_idx))

        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No valid samples for split {self.split}. "
                f"Check data availability and lookback settings."
            )

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            (features_dict, targets, horizons)
                features_dict: {"5m": tensor(seq_len, feature_dim), ...}
                targets: tensor(num_horizons, 1)
                horizons: List of horizon indices
        """
        # Get actual index (accounting for lookback)
        actual_idx = self.valid_indices[idx]

        # Extract features for each timeframe
        features = {}

        for tf in self.timeframes:
            lookback = self.lookback_bars[tf]

            # Get lookback window: [actual_idx - lookback : actual_idx]
            start_idx = actual_idx - lookback
            end_idx = actual_idx

            # Extract features
            feature_window = self.features_dict[tf].iloc[start_idx:end_idx][
                self.feature_cols
            ].values

            # Convert to tensor
            features[tf] = torch.tensor(feature_window, dtype=torch.float32)

        # Extract targets for all horizons
        target_values = []
        horizon_indices = []

        for i, h in enumerate(self.horizons):
            target_col = f"target_{h}m"

            if target_col in self.targets.columns:
                target_val = self.targets.iloc[actual_idx][target_col]
                target_values.append(target_val)
                horizon_indices.append(i)
            else:
                logger.warning(f"Missing target column: {target_col}")

        # Convert targets to tensor
        targets = torch.tensor(target_values, dtype=torch.float32).unsqueeze(-1)  # (num_horizons, 1)

        return features, targets, horizon_indices


class SSSDDataModule:
    """
    Data module for SSSD training (PyTorch Lightning style).

    Handles data loading, splitting, and DataLoader creation.
    """

    def __init__(
        self,
        data_path: str | Path,
        config: SSSDConfig,
        feature_pipeline: Optional[object] = None
    ):
        """
        Initialize data module.

        Args:
            data_path: Path to data directory or file
            config: SSSDConfig object
            feature_pipeline: Optional feature engineering pipeline
        """
        self.data_path = Path(data_path)
        self.config = config
        self.feature_pipeline = feature_pipeline

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Load and prepare data."""
        logger.info("Setting up SSSD data module...")

        # Load raw data
        # This is a placeholder - actual implementation depends on data format
        features_dict = self._load_features()
        targets = self._load_targets()

        # Apply feature engineering if provided
        if self.feature_pipeline is not None:
            features_dict = self._apply_feature_pipeline(features_dict)

        # Create datasets
        self.train_dataset = SSSDTimeSeriesDataset(
            features_dict, targets, self.config, split="train"
        )
        self.val_dataset = SSSDTimeSeriesDataset(
            features_dict, targets, self.config, split="val"
        )
        self.test_dataset = SSSDTimeSeriesDataset(
            features_dict, targets, self.config, split="test"
        )

        logger.info(
            f"Data setup complete: "
            f"train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, "
            f"test={len(self.test_dataset)}"
        )

    def _load_features(self) -> Dict[str, pd.DataFrame]:
        """
        Load features from disk.

        Returns:
            Dict mapping timeframe to DataFrame
        """
        # Placeholder - implement actual data loading
        # Could load from Parquet, CSV, DuckDB, etc.

        features_dict = {}

        for tf in self.config.model.encoder.timeframes:
            # Example: load from parquet
            file_path = self.data_path / f"features_{tf}.parquet"

            if file_path.exists():
                df = pd.read_parquet(file_path)
                features_dict[tf] = df
            else:
                raise FileNotFoundError(f"Features file not found: {file_path}")

        return features_dict

    def _load_targets(self) -> pd.DataFrame:
        """
        Load targets from disk.

        Returns:
            DataFrame with target columns
        """
        # Placeholder - implement actual target loading
        file_path = self.data_path / "targets.parquet"

        if file_path.exists():
            return pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"Targets file not found: {file_path}")

    def _apply_feature_pipeline(
        self,
        features_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Apply feature engineering pipeline."""
        # Placeholder - call feature pipeline if needed
        return features_dict

    def train_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Create training DataLoader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            **kwargs
        )

    def val_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Create validation DataLoader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            **kwargs
        )

    def test_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Create test DataLoader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            **kwargs
        )


def collate_fn(batch: List[Tuple]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]:
    """
    Custom collate function for SSSD dataset.

    Args:
        batch: List of (features_dict, targets, horizons) tuples

    Returns:
        Batched (features_dict, targets, horizons)
    """
    # Unpack batch
    features_dicts, targets_list, horizons_list = zip(*batch)

    # Stack features for each timeframe
    batched_features = {}

    for tf in features_dicts[0].keys():
        # Stack all samples for this timeframe
        tf_features = [f[tf] for f in features_dicts]
        batched_features[tf] = torch.stack(tf_features, dim=0)

    # Stack targets
    batched_targets = torch.stack(targets_list, dim=0)

    # Horizons should be same for all samples
    horizons = horizons_list[0]

    return batched_features, batched_targets, horizons
