"""
Sklearn Multi-Horizon Predictor

Handles inference for sklearn models trained with multi-horizon support.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from joblib import load
from loguru import logger

from ..utils.horizon_parser import get_model_horizons_from_metadata, validate_inference_horizons


class SklearnMultiHorizonPredictor:
    """
    Predictor for sklearn models with multi-horizon support.
    
    Features:
    - Loads sklearn model from .pkl file
    - Extracts horizon metadata
    - Validates inference horizons match training
    - Returns predictions for all horizons
    """
    
    def __init__(self, model_path: str | Path):
        """
        Initialize predictor from saved model.
        
        Args:
            model_path: Path to .pkl model file
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model payload
        logger.info(f"Loading sklearn model from {self.model_path}")
        self.payload = load(self.model_path)
        
        # Extract components
        self.model = self.payload['model']
        self.scaler_mu = self.payload.get('scaler_mu')
        self.scaler_sigma = self.payload.get('scaler_sigma')
        self.encoder = self.payload.get('encoder')
        self.features = self.payload.get('features', [])
        
        # Extract horizons from metadata
        try:
            self.horizons = get_model_horizons_from_metadata(self.payload)
        except ValueError as e:
            logger.warning(f"Could not extract horizons from metadata: {e}")
            # Fallback: assume single horizon
            self.horizons = [self.payload.get('horizon_bars', 60)]
        
        self.is_multi_horizon = len(self.horizons) > 1
        
        logger.info(f"Model horizons: {self.horizons} ({'multi' if self.is_multi_horizon else 'single'})")
        logger.info(f"Model type: {self.payload.get('model_type', 'unknown')}")
        logger.info(f"Features: {len(self.features)} features")
        
    def predict(
        self,
        features: np.ndarray | pd.DataFrame,
        return_dict: bool = True
    ) -> Dict[int, float] | np.ndarray:
        """
        Make predictions for all trained horizons.
        
        Args:
            features: Feature matrix (N, F) or DataFrame
            return_dict: If True, return dict {horizon: prediction}, else array
            
        Returns:
            If return_dict=True: {15: 0.002, 60: 0.005, 240: 0.012}
            If return_dict=False: np.array([0.002, 0.005, 0.012])
            
        Notes:
            - Automatically applies encoder if present
            - Automatically scales features
            - Returns predictions for ALL training horizons
        """
        # Convert to numpy if DataFrame
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = np.asarray(features)
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Apply encoder if present
        if self.encoder is not None:
            logger.debug(f"Applying encoder: {type(self.encoder).__name__}")
            X = self.encoder.transform(X)
        
        # Apply scaling
        if self.scaler_mu is not None and self.scaler_sigma is not None:
            logger.debug("Applying feature scaling")
            X = (X - self.scaler_mu) / self.scaler_sigma
        
        # Make prediction
        predictions = self.model.predict(X)
        
        # Handle output shape
        if predictions.ndim == 1:
            # Single sample, single horizon
            predictions = predictions.reshape(1, -1)
        elif predictions.ndim == 2 and predictions.shape[0] == X.shape[0]:
            # (N, H) - multi-horizon
            pass
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
        
        # Return format
        if return_dict:
            # Convert to dict {horizon: prediction}
            if predictions.shape[1] != len(self.horizons):
                logger.warning(
                    f"Prediction shape mismatch: got {predictions.shape[1]} values "
                    f"but expected {len(self.horizons)} horizons"
                )
            
            result = {}
            for i, horizon in enumerate(self.horizons):
                if i < predictions.shape[1]:
                    # Take first sample's prediction for this horizon
                    result[horizon] = float(predictions[0, i])
                else:
                    result[horizon] = 0.0
            
            return result
        else:
            # Return raw array (first sample)
            return predictions[0]
    
    def predict_single_horizon(
        self,
        features: np.ndarray | pd.DataFrame,
        horizon: int
    ) -> float:
        """
        Predict for a specific horizon.
        
        Args:
            features: Feature matrix
            horizon: Which horizon to predict (must be in training horizons)
            
        Returns:
            Prediction value for specified horizon
            
        Raises:
            ValueError: If horizon not in training horizons
        """
        if horizon not in self.horizons:
            raise ValueError(
                f"Horizon {horizon} not in training horizons {self.horizons}. "
                f"Model can only predict horizons it was trained on."
            )
        
        predictions_dict = self.predict(features, return_dict=True)
        return predictions_dict[horizon]
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_path': str(self.model_path),
            'model_type': self.payload.get('model_type', 'unknown'),
            'horizons': self.horizons,
            'is_multi_horizon': self.is_multi_horizon,
            'num_horizons': len(self.horizons),
            'num_features': len(self.features),
            'features': self.features,
            'has_encoder': self.encoder is not None,
            'encoder_type': self.payload.get('encoder_type', 'none'),
            'val_mae': self.payload.get('val_mae', None),
        }
