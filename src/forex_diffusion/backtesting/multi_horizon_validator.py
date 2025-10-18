"""
Multi-Horizon Forecast Validator

Validates multi-horizon forecasts from trained models.
Computes MAE, RMSE, CRPS for each horizon separately.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HorizonMetrics:
    """Metrics for a single horizon."""
    horizon_bars: int
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    directional_accuracy: float = 0.0
    num_samples: int = 0


@dataclass
class MultiHorizonValidationResult:
    """Results from multi-horizon validation."""
    model_name: str
    symbol: str
    timeframe: str
    horizons: List[int]
    
    # Per-horizon metrics
    horizon_metrics: Dict[int, HorizonMetrics] = field(default_factory=dict)
    
    # Aggregate metrics
    avg_mae: float = 0.0
    avg_rmse: float = 0.0
    avg_mape: float = 0.0
    avg_directional_accuracy: float = 0.0
    
    # Metadata
    total_samples: int = 0
    validation_start: Optional[datetime] = None
    validation_end: Optional[datetime] = None


class MultiHorizonValidator:
    """
    Validator for multi-horizon forecasts.
    
    Features:
    - Validates predictions for each horizon independently
    - Computes standard metrics (MAE, RMSE, MAPE, Directional Accuracy)
    - Aggregates metrics across horizons
    - Supports both sklearn and Lightning models
    """
    
    def __init__(self, model_path: str):
        """
        Initialize validator.
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        
        # Load model and extract horizons
        self._load_model_info()
        
        logger.info(f"MultiHorizonValidator initialized for {self.model_name} with horizons: {self.horizons}")
    
    def _load_model_info(self):
        """Load model information and extract horizons."""
        try:
            from ..inference.model_metadata_loader import ModelMetadataLoader
            
            loader = ModelMetadataLoader(self.model_path)
            metadata = loader.load_metadata()
            
            self.horizons = metadata.get('horizons', [])
            self.model_name = self.model_path.stem
            self.symbol = metadata.get('symbol', 'EUR/USD')
            self.timeframe = metadata.get('timeframe', '1m')
            
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")
            self.horizons = []
            self.model_name = self.model_path.stem
            self.symbol = 'EUR/USD'
            self.timeframe = '1m'
    
    def validate(
        self,
        test_data: pd.DataFrame,
        predictions_dict: Dict[int, np.ndarray]
    ) -> MultiHorizonValidationResult:
        """
        Validate predictions against test data.
        
        Args:
            test_data: DataFrame with actual prices
                Must have columns: ['close', 'timestamp']
            predictions_dict: Dict mapping horizon -> predictions array
                {15: array([...]), 60: array([...]), 240: array([...])}
        
        Returns:
            MultiHorizonValidationResult with metrics
        """
        result = MultiHorizonValidationResult(
            model_name=self.model_name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            horizons=list(predictions_dict.keys())
        )
        
        # Validate each horizon
        for horizon, predictions in predictions_dict.items():
            metrics = self._validate_single_horizon(
                test_data,
                predictions,
                horizon
            )
            result.horizon_metrics[horizon] = metrics
        
        # Compute aggregate metrics
        if result.horizon_metrics:
            result.avg_mae = np.mean([m.mae for m in result.horizon_metrics.values()])
            result.avg_rmse = np.mean([m.rmse for m in result.horizon_metrics.values()])
            result.avg_mape = np.mean([m.mape for m in result.horizon_metrics.values()])
            result.avg_directional_accuracy = np.mean([
                m.directional_accuracy for m in result.horizon_metrics.values()
            ])
            result.total_samples = sum([m.num_samples for m in result.horizon_metrics.values()])
        
        # Set time range
        if 'timestamp' in test_data.columns:
            result.validation_start = test_data['timestamp'].min()
            result.validation_end = test_data['timestamp'].max()
        
        logger.info(f"Validation completed: Avg MAE={result.avg_mae:.6f}, Avg RMSE={result.avg_rmse:.6f}")
        
        return result
    
    def _validate_single_horizon(
        self,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        horizon: int
    ) -> HorizonMetrics:
        """
        Validate predictions for a single horizon.
        
        Args:
            test_data: Test data DataFrame
            predictions: Predictions array
            horizon: Horizon in bars
        
        Returns:
            HorizonMetrics
        """
        # Extract actual values at horizon
        actuals = []
        valid_predictions = []
        
        for i in range(len(predictions)):
            # Check if we have actual value at horizon
            future_idx = i + horizon
            if future_idx < len(test_data):
                actual = test_data['close'].iloc[future_idx]
                pred = predictions[i]
                
                actuals.append(actual)
                valid_predictions.append(pred)
        
        if len(actuals) == 0:
            logger.warning(f"No valid samples for horizon {horizon}")
            return HorizonMetrics(horizon_bars=horizon)
        
        actuals = np.array(actuals)
        valid_predictions = np.array(valid_predictions)
        
        # Compute metrics
        mae = np.mean(np.abs(actuals - valid_predictions))
        rmse = np.sqrt(np.mean((actuals - valid_predictions) ** 2))
        
        # MAPE (avoid division by zero)
        mape_values = np.abs((actuals - valid_predictions) / np.clip(actuals, 1e-8, None))
        mape = np.mean(mape_values) * 100
        
        # Directional accuracy
        if len(actuals) > 1:
            actual_directions = np.diff(actuals)
            pred_directions = np.diff(valid_predictions)
            directional_accuracy = np.mean(
                np.sign(actual_directions) == np.sign(pred_directions)
            )
        else:
            directional_accuracy = 0.0
        
        return HorizonMetrics(
            horizon_bars=horizon,
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape),
            directional_accuracy=float(directional_accuracy),
            num_samples=len(actuals)
        )
    
    def generate_report(self, result: MultiHorizonValidationResult) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            result: Validation result
        
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MULTI-HORIZON VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Model: {result.model_name}")
        lines.append(f"Symbol: {result.symbol}")
        lines.append(f"Timeframe: {result.timeframe}")
        lines.append(f"Horizons: {result.horizons}")
        lines.append(f"Total Samples: {result.total_samples}")
        
        if result.validation_start and result.validation_end:
            lines.append(f"Period: {result.validation_start} to {result.validation_end}")
        
        lines.append("")
        lines.append("AGGREGATE METRICS:")
        lines.append(f"  Average MAE:   {result.avg_mae:.6f}")
        lines.append(f"  Average RMSE:  {result.avg_rmse:.6f}")
        lines.append(f"  Average MAPE:  {result.avg_mape:.2f}%")
        lines.append(f"  Avg Dir Acc:   {result.avg_directional_accuracy:.4f}")
        lines.append("")
        
        lines.append("PER-HORIZON METRICS:")
        lines.append("-" * 80)
        lines.append(f"{'Horizon':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<10} {'Dir Acc':<10} {'Samples':<10}")
        lines.append("-" * 80)
        
        for horizon in sorted(result.horizon_metrics.keys()):
            metrics = result.horizon_metrics[horizon]
            lines.append(
                f"{horizon:<10} "
                f"{metrics.mae:<12.6f} "
                f"{metrics.rmse:<12.6f} "
                f"{metrics.mape:<10.2f} "
                f"{metrics.directional_accuracy:<10.4f} "
                f"{metrics.num_samples:<10}"
            )
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_report(self, result: MultiHorizonValidationResult, output_path: str):
        """Save validation report to file."""
        report = self.generate_report(result)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to {output_path}")
