"""
Unified Multi-Horizon Predictor

Single interface for both sklearn and Lightning models.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger

from .sklearn_predictor import SklearnMultiHorizonPredictor
from .lightning_predictor import LightningMultiHorizonPredictor


class UnifiedMultiHorizonPredictor:
    """
    Unified predictor that works with both sklearn and Lightning models.
    
    Features:
    - Auto-detects model type from file extension
    - Provides consistent API for both model types
    - Handles interpolation transparently
    - Returns predictions in consistent format
    """
    
    def __init__(self, model_path: str | Path, device: str = 'cpu'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model file (.pkl for sklearn, .ckpt for Lightning)
            device: Device for Lightning models ('cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Detect model type
        self.model_type = self._detect_model_type()
        
        # Load appropriate predictor
        if self.model_type == 'sklearn':
            self.predictor = SklearnMultiHorizonPredictor(model_path)
        elif self.model_type == 'lightning':
            self.predictor = LightningMultiHorizonPredictor(model_path, device=device)
        else:
            raise ValueError(f"Unknown model type for {model_path}")
        
        # Store metadata
        self.horizons = self.predictor.horizons
        self.is_multi_horizon = self.predictor.is_multi_horizon
        self.num_horizons = self.predictor.num_horizons
        
        logger.info(f"Loaded {self.model_type} model with horizons: {self.horizons}")
    
    def _detect_model_type(self) -> str:
        """Detect model type from file extension."""
        suffix = self.model_path.suffix.lower()
        
        if suffix in ['.pkl', '.pickle']:
            return 'sklearn'
        elif suffix in ['.ckpt', '.pt', '.pth']:
            return 'lightning'
        else:
            raise ValueError(f"Unknown model file extension: {suffix}")
    
    def predict(
        self,
        features: np.ndarray,
        horizons: Optional[List[int]] = None,
        num_samples: int = 50,
        return_distribution: bool = False
    ) -> Dict[int, Any]:
        """
        Make predictions.
        
        Args:
            features: Feature array (for sklearn) or patch (for Lightning)
            horizons: Specific horizons to predict (None = all trained horizons)
            num_samples: Number of samples (Lightning only)
            return_distribution: Return full distribution stats
            
        Returns:
            Dict mapping horizon -> prediction
            {15: 0.002, 60: 0.005, 240: 0.012}
            
            If return_distribution=True:
            {15: {'mean': 0.002, 'std': 0.001, ...}, ...}
        """
        # Determine which horizons to predict
        requested_horizons = horizons if horizons is not None else self.horizons
        
        # Check if interpolation needed
        needs_interpolation = not set(requested_horizons).issubset(set(self.horizons))
        
        if needs_interpolation:
            logger.warning(
                f"Interpolation needed: requested {requested_horizons}, "
                f"trained {self.horizons}"
            )
        
        # Make predictions based on model type
        if self.model_type == 'sklearn':
            if needs_interpolation:
                # Sklearn with interpolation
                predictions = self._interpolate_sklearn(features, requested_horizons)
            else:
                # Sklearn without interpolation
                all_preds = self.predictor.predict(features, return_dict=True)
                predictions = {h: all_preds[h] for h in requested_horizons}
        
        elif self.model_type == 'lightning':
            # Lightning handles interpolation internally
            import torch
            if isinstance(features, np.ndarray):
                features_tensor = torch.from_numpy(features).float()
            else:
                features_tensor = features
            
            predictions = self.predictor.predict_with_interpolation(
                features_tensor,
                requested_horizons,
                num_samples=num_samples,
                return_distribution=return_distribution
            )
        
        return predictions
    
    def _interpolate_sklearn(
        self,
        features: np.ndarray,
        requested_horizons: List[int]
    ) -> Dict[int, float]:
        """
        Interpolate sklearn predictions for missing horizons.
        
        Args:
            features: Feature array
            requested_horizons: All horizons to predict
            
        Returns:
            Predictions for all requested horizons
        """
        # Get predictions for trained horizons
        trained_preds = self.predictor.predict(features, return_dict=True)
        
        result = {}
        sorted_trained = sorted(self.horizons)
        
        for h in requested_horizons:
            if h in trained_preds:
                # Direct prediction available
                result[h] = trained_preds[h]
            else:
                # Need interpolation
                lower = None
                upper = None
                
                for th in sorted_trained:
                    if th < h:
                        lower = th
                    elif th > h and upper is None:
                        upper = th
                        break
                
                if lower is not None and upper is not None:
                    # Linear interpolation
                    weight = (h - lower) / (upper - lower)
                    result[h] = (
                        trained_preds[lower] * (1 - weight) +
                        trained_preds[upper] * weight
                    )
                    logger.debug(
                        f"Interpolated horizon {h}: "
                        f"{lower}*{1-weight:.3f} + {upper}*{weight:.3f}"
                    )
                else:
                    # Cannot interpolate
                    logger.error(
                        f"Cannot interpolate horizon {h} - "
                        f"outside trained range {sorted_trained}"
                    )
                    result[h] = 0.0
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = self.predictor.get_info()
        info['unified_predictor'] = True
        return info
