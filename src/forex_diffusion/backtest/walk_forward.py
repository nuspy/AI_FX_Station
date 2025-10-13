"""
Walk-Forward Validation with Purge and Embargo

Implements proper walk-forward validation to prevent data leakage.
Includes purge period (remove data immediately after training) and
embargo period (remove data immediately after validation).

Reference: "Advances in Financial Machine Learning" by Marcos López de Prado
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
from loguru import logger


@dataclass
class WalkForwardSplit:
    """Single walk-forward split with purge and embargo"""
    split_id: int
    train_start: datetime
    train_end: datetime
    purge_start: datetime
    purge_end: datetime
    val_start: datetime
    val_end: datetime
    embargo_start: datetime
    embargo_end: datetime
    test_start: datetime
    test_end: datetime


class WalkForwardValidator:
    """
    Walk-forward validation with proper purge and embargo to prevent data leakage.
    
    Timeline:
    |--- Train (730d) ---|P|-- Val (90d) --|E|--- Test (90d) ---|
    
    P = Purge (1 day) - removed data immediately after training
    E = Embargo (2 days) - removed data immediately after validation
    
    This ensures:
    - Training data doesn't leak into validation/test
    - Validation parameter selection doesn't bias test results
    - Results are more realistic (no look-ahead bias)
    """
    
    def __init__(
        self,
        train_days: int = 730,
        val_days: int = 90,
        test_days: int = 90,
        purge_days: int = 1,
        embargo_days: int = 2,
        step_days: int = 90
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_days: Training window size in days
            val_days: Validation window size in days
            test_days: Test window size in days
            purge_days: Purge period after training (prevent leakage)
            embargo_days: Embargo period after validation (prevent bias)
            step_days: Step size for advancing window
        """
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.step_days = step_days
        
        logger.info(
            f"WalkForwardValidator initialized: "
            f"train={train_days}d, val={val_days}d, test={test_days}d, "
            f"purge={purge_days}d, embargo={embargo_days}d, step={step_days}d"
        )
    
    def create_splits(
        self,
        data: pd.DataFrame,
        date_column: str = 'timestamp'
    ) -> List[WalkForwardSplit]:
        """
        Create walk-forward splits with purge and embargo.
        
        Args:
            data: DataFrame with datetime index or date column
            date_column: Name of datetime column (if not using index)
            
        Returns:
            List of WalkForwardSplit objects
        """
        # Get datetime series
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        elif date_column in data.columns:
            dates = pd.to_datetime(data[date_column])
        else:
            raise ValueError(f"No datetime index or column '{date_column}' found")
        
        start_date = dates.min()
        end_date = dates.max()
        
        # Calculate minimum required data
        min_required_days = (
            self.train_days + 
            self.purge_days + 
            self.val_days + 
            self.embargo_days + 
            self.test_days
        )
        
        total_days = (end_date - start_date).days
        if total_days < min_required_days:
            raise ValueError(
                f"Insufficient data: {total_days} days available, "
                f"{min_required_days} days required"
            )
        
        splits = []
        split_id = 0
        current_start = start_date
        
        while True:
            # Training period
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_days)
            
            # Purge period
            purge_start = train_end
            purge_end = purge_start + timedelta(days=self.purge_days)
            
            # Validation period
            val_start = purge_end
            val_end = val_start + timedelta(days=self.val_days)
            
            # Embargo period
            embargo_start = val_end
            embargo_end = embargo_start + timedelta(days=self.embargo_days)
            
            # Test period
            test_start = embargo_end
            test_end = test_start + timedelta(days=self.test_days)
            
            # Check if we have enough data for this split
            if test_end > end_date:
                break
            
            split = WalkForwardSplit(
                split_id=split_id,
                train_start=train_start,
                train_end=train_end,
                purge_start=purge_start,
                purge_end=purge_end,
                val_start=val_start,
                val_end=val_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end
            )
            
            splits.append(split)
            
            logger.debug(
                f"Split {split_id}: "
                f"train=[{train_start.date()} to {train_end.date()}], "
                f"purge=[{purge_start.date()} to {purge_end.date()}], "
                f"val=[{val_start.date()} to {val_end.date()}], "
                f"embargo=[{embargo_start.date()} to {embargo_end.date()}], "
                f"test=[{test_start.date()} to {test_end.date()}]"
            )
            
            # Advance to next split
            current_start += timedelta(days=self.step_days)
            split_id += 1
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def get_split_data(
        self,
        data: pd.DataFrame,
        split: WalkForwardSplit,
        date_column: str = 'timestamp'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract train, val, test data for a split (excluding purge/embargo).
        
        Args:
            data: Full dataset
            split: WalkForwardSplit specification
            date_column: Name of datetime column
            
        Returns:
            (train_data, val_data, test_data) tuple
        """
        # Get datetime series
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        elif date_column in data.columns:
            dates = pd.to_datetime(data[date_column])
        else:
            raise ValueError(f"No datetime index or column '{date_column}' found")
        
        # Extract splits (excluding purge and embargo periods)
        train_mask = (dates >= split.train_start) & (dates < split.train_end)
        val_mask = (dates >= split.val_start) & (dates < split.val_end)
        test_mask = (dates >= split.test_start) & (dates < split.test_end)
        
        train_data = data[train_mask].copy()
        val_data = data[val_mask].copy()
        test_data = data[test_mask].copy()
        
        logger.debug(
            f"Split {split.split_id} data: "
            f"train={len(train_data)} rows, "
            f"val={len(val_data)} rows, "
            f"test={len(test_data)} rows"
        )
        
        return train_data, val_data, test_data
    
    def validate_no_leakage(
        self,
        data: pd.DataFrame,
        split: WalkForwardSplit,
        date_column: str = 'timestamp'
    ) -> bool:
        """
        Validate that there's no data leakage in the split.
        
        Checks:
        - No overlap between train/val/test
        - Purge period is actually empty
        - Embargo period is actually empty
        - Chronological order is correct
        
        Returns:
            True if validation passes, False otherwise
        """
        # Get datetime series
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        elif date_column in data.columns:
            dates = pd.to_datetime(data[date_column])
        else:
            return False
        
        # Check purge period is not used
        purge_mask = (dates >= split.purge_start) & (dates < split.purge_end)
        if purge_mask.any():
            purged_count = purge_mask.sum()
            logger.warning(
                f"Split {split.split_id}: Purge period contains {purged_count} rows "
                f"(these will be excluded)"
            )
        
        # Check embargo period is not used
        embargo_mask = (dates >= split.embargo_start) & (dates < split.embargo_end)
        if embargo_mask.any():
            embargoed_count = embargo_mask.sum()
            logger.warning(
                f"Split {split.split_id}: Embargo period contains {embargoed_count} rows "
                f"(these will be excluded)"
            )
        
        # Check chronological order
        if not (split.train_start < split.train_end < split.purge_end < 
                split.val_start < split.val_end < split.embargo_end < 
                split.test_start < split.test_end):
            logger.error(f"Split {split.split_id}: Chronological order violation")
            return False
        
        # Check no overlap
        train_data, val_data, test_data = self.get_split_data(data, split, date_column)
        
        if isinstance(train_data.index, pd.DatetimeIndex):
            train_dates = train_data.index
            val_dates = val_data.index
            test_dates = test_data.index
        else:
            train_dates = pd.to_datetime(train_data[date_column])
            val_dates = pd.to_datetime(val_data[date_column])
            test_dates = pd.to_datetime(test_data[date_column])
        
        # Check train max < val min
        if len(train_dates) > 0 and len(val_dates) > 0:
            if train_dates.max() >= val_dates.min():
                logger.error(
                    f"Split {split.split_id}: Train/val overlap detected "
                    f"(train_max={train_dates.max()}, val_min={val_dates.min()})"
                )
                return False
        
        # Check val max < test min
        if len(val_dates) > 0 and len(test_dates) > 0:
            if val_dates.max() >= test_dates.min():
                logger.error(
                    f"Split {split.split_id}: Val/test overlap detected "
                    f"(val_max={val_dates.max()}, test_min={test_dates.min()})"
                )
                return False
        
        logger.debug(f"Split {split.split_id}: Validation passed (no leakage)")
        return True
