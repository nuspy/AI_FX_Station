"""
Unified Model Loader - Load and manage multiple models (sklearn + lightning).

Supports:
- Mixed model types (sklearn + lightning)
- Ensemble predictions (mean, median, weighted)
- Individual predictions
- Metadata extraction
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger


class UnifiedModelLoader:
    """
    Load and manage multiple models for ensemble or separate inference.
    
    Features:
    - Load sklearn (.pkl) and lightning (.ckpt, .pt, .pth) models
    - Ensemble prediction with multiple aggregation methods
    - Individual predictions for each model
    - Metadata extraction and validation
    """
    
    def __init__(self, model_paths: List[str]):
        """
        Initialize loader.
        
        Args:
            model_paths: List of paths to model files
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.models = []
        self.metadata = []
        self.model_types = []
        
    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all models and extract metadata.
        
        Returns:
            List of model info dictionaries
        """
        from .model_metadata_loader import ModelMetadataLoader
        
        self.models = []
        self.metadata = []
        self.model_types = []
        
        for path in self.model_paths:
            try:
                # Load metadata
                loader = ModelMetadataLoader(path)
                meta = loader.load_metadata()
                
                # Store model type
                model_type = meta['model_type']
                self.model_types.append(model_type)
                self.metadata.append(meta)
                
                # Load predictor (lazy loading - actual model loaded on first predict)
                if model_type == 'sklearn':
                    from .sklearn_predictor import SklearnMultiHorizonPredictor
                    predictor = SklearnMultiHorizonPredictor(str(path))
                elif model_type == 'lightning':
                    from .lightning_predictor import LightningMultiHorizonPredictor
                    predictor = LightningMultiHorizonPredictor(str(path))
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                self.models.append(predictor)
                logger.info(f"Loaded {model_type} model: {path.name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {path}: {e}")
                raise
        
        logger.info(f"Successfully loaded {len(self.models)} models")
        return self.get_models_info()
    
    def get_models_info(self) -> List[Dict[str, Any]]:
        """Get summary info for all loaded models."""
        info = []
        for i, (path, meta, model_type) in enumerate(zip(self.model_paths, self.metadata, self.model_types)):
            info.append({
                'index': i,
                'name': path.name,
                'path': str(path),
                'type': model_type,
                'horizons': meta.get('horizons', []),
                'num_horizons': meta.get('num_horizons', 1),
                'symbol': meta.get('symbol', 'N/A'),
                'timeframe': meta.get('timeframe', 'N/A'),
                'algorithm': meta.get('algorithm', model_type),
            })
        return info
    
    def predict_ensemble(
        self,
        features: np.ndarray,
        requested_horizons: Optional[List[int]] = None,
        method: str = 'mean',
        return_distribution: bool = False,
        num_samples: int = 100,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run ensemble prediction across all models.
        
        Args:
            features: Input features (N, F) or (N, C, L) for diffusion
            requested_horizons: Horizons to predict (None = use model defaults)
            method: Aggregation method ('mean', 'median', 'weighted', 'best')
            return_distribution: Whether to return full distribution
            num_samples: Number of samples for diffusion models
        
        Returns:
            Dictionary mapping horizon -> prediction stats
        """
        logger.info(f"Running ensemble prediction with {len(self.models)} models, method={method}")
        
        # Collect predictions from all models
        all_predictions = []
        
        for i, (model, meta) in enumerate(zip(self.models, self.metadata)):
            try:
                logger.debug(f"Predicting with model {i+1}/{len(self.models)}: {self.model_paths[i].name}")
                
                # Run prediction
                preds = model.predict(
                    features,
                    requested_horizons=requested_horizons,
                    return_distribution=return_distribution,
                    num_samples=num_samples,
                    return_dict=True
                )
                
                all_predictions.append({
                    'predictions': preds,
                    'model_index': i,
                    'model_name': self.model_paths[i].name,
                    'model_type': meta['model_type']
                })
                
            except Exception as e:
                logger.error(f"Prediction failed for model {i}: {e}")
                # Continue with other models
                continue
        
        if not all_predictions:
            raise RuntimeError("All model predictions failed")
        
        # Aggregate predictions
        if method == 'mean':
            result = self._aggregate_mean(all_predictions, return_distribution)
        elif method == 'median':
            result = self._aggregate_median(all_predictions, return_distribution)
        elif method == 'weighted':
            result = self._aggregate_weighted(all_predictions, return_distribution)
        elif method == 'best':
            result = self._aggregate_best(all_predictions, return_distribution)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Add ensemble metadata
        for horizon_data in result.values():
            horizon_data['ensemble_size'] = len(all_predictions)
            horizon_data['ensemble_method'] = method
            horizon_data['models_used'] = [p['model_name'] for p in all_predictions]
        
        logger.info(f"Ensemble prediction complete: {len(result)} horizons")
        return result
    
    def predict_individual(
        self,
        features: np.ndarray,
        requested_horizons: Optional[List[int]] = None,
        return_distribution: bool = False,
        num_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Run predictions for each model separately.
        
        Args:
            features: Input features
            requested_horizons: Horizons to predict
            return_distribution: Whether to return full distribution
            num_samples: Number of samples for diffusion models
        
        Returns:
            List of prediction results, one per model
        """
        logger.info(f"Running individual predictions for {len(self.models)} models")
        
        results = []
        
        for i, (model, meta, path) in enumerate(zip(self.models, self.metadata, self.model_paths)):
            try:
                logger.debug(f"Predicting with model {i+1}/{len(self.models)}: {path.name}")
                
                preds = model.predict(
                    features,
                    requested_horizons=requested_horizons,
                    return_distribution=return_distribution,
                    num_samples=num_samples,
                    return_dict=True
                )
                
                results.append({
                    'model_index': i,
                    'model_name': path.name,
                    'model_type': meta['model_type'],
                    'predictions': preds,
                    'metadata': meta
                })
                
            except Exception as e:
                logger.error(f"Prediction failed for model {i}: {e}")
                results.append({
                    'model_index': i,
                    'model_name': path.name,
                    'model_type': meta['model_type'],
                    'error': str(e)
                })
        
        logger.info(f"Individual predictions complete: {len(results)} models")
        return results
    
    def _aggregate_mean(
        self,
        predictions: List[Dict],
        return_distribution: bool
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate using simple mean."""
        # Get all horizons
        all_horizons = set()
        for pred in predictions:
            all_horizons.update(pred['predictions'].keys())
        
        result = {}
        
        for horizon in sorted(all_horizons):
            # Collect values for this horizon
            means = []
            stds = []
            q05s = []
            q50s = []
            q95s = []
            
            for pred in predictions:
                if horizon in pred['predictions']:
                    p = pred['predictions'][horizon]
                    means.append(p.get('mean', p.get('q50', 0)))
                    if 'std' in p:
                        stds.append(p['std'])
                    if 'q05' in p:
                        q05s.append(p['q05'])
                    if 'q50' in p:
                        q50s.append(p['q50'])
                    if 'q95' in p:
                        q95s.append(p['q95'])
            
            if not means:
                continue
            
            # Compute ensemble statistics
            result[horizon] = {
                'mean': float(np.mean(means)),
                'std': float(np.mean(stds)) if stds else float(np.std(means)),
                'q05': float(np.mean(q05s)) if q05s else float(np.percentile(means, 5)),
                'q50': float(np.mean(q50s)) if q50s else float(np.median(means)),
                'q95': float(np.mean(q95s)) if q95s else float(np.percentile(means, 95)),
            }
        
        return result
    
    def _aggregate_median(
        self,
        predictions: List[Dict],
        return_distribution: bool
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate using median (more robust to outliers)."""
        all_horizons = set()
        for pred in predictions:
            all_horizons.update(pred['predictions'].keys())
        
        result = {}
        
        for horizon in sorted(all_horizons):
            means = []
            stds = []
            q05s = []
            q50s = []
            q95s = []
            
            for pred in predictions:
                if horizon in pred['predictions']:
                    p = pred['predictions'][horizon]
                    means.append(p.get('mean', p.get('q50', 0)))
                    if 'std' in p:
                        stds.append(p['std'])
                    if 'q05' in p:
                        q05s.append(p['q05'])
                    if 'q50' in p:
                        q50s.append(p['q50'])
                    if 'q95' in p:
                        q95s.append(p['q95'])
            
            if not means:
                continue
            
            result[horizon] = {
                'mean': float(np.median(means)),
                'std': float(np.median(stds)) if stds else float(np.std(means)),
                'q05': float(np.median(q05s)) if q05s else float(np.percentile(means, 5)),
                'q50': float(np.median(q50s)) if q50s else float(np.median(means)),
                'q95': float(np.median(q95s)) if q95s else float(np.percentile(means, 95)),
            }
        
        return result
    
    def _aggregate_weighted(
        self,
        predictions: List[Dict],
        return_distribution: bool
    ) -> Dict[int, Dict[str, Any]]:
        """
        Aggregate using weighted mean based on model accuracy.
        
        Note: Weights based on validation MAE if available in metadata,
        otherwise equal weights (same as mean).
        """
        # Extract weights from metadata (inverse MAE)
        weights = []
        for i, pred in enumerate(predictions):
            meta = self.metadata[i]
            mae = meta.get('val_mae')
            if mae and mae > 0:
                # Weight = 1/MAE (better models get higher weight)
                weights.append(1.0 / mae)
            else:
                weights.append(1.0)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(predictions)] * len(predictions)
        
        logger.debug(f"Weighted aggregation weights: {weights}")
        
        # Aggregate
        all_horizons = set()
        for pred in predictions:
            all_horizons.update(pred['predictions'].keys())
        
        result = {}
        
        for horizon in sorted(all_horizons):
            weighted_means = []
            stds = []
            q05s = []
            q50s = []
            q95s = []
            
            for pred, weight in zip(predictions, weights):
                if horizon in pred['predictions']:
                    p = pred['predictions'][horizon]
                    mean_val = p.get('mean', p.get('q50', 0))
                    weighted_means.append(mean_val * weight)
                    if 'std' in p:
                        stds.append(p['std'])
                    if 'q05' in p:
                        q05s.append(p['q05'])
                    if 'q50' in p:
                        q50s.append(p['q50'])
                    if 'q95' in p:
                        q95s.append(p['q95'])
            
            if not weighted_means:
                continue
            
            result[horizon] = {
                'mean': float(sum(weighted_means)),  # Already weighted
                'std': float(np.mean(stds)) if stds else 0.0,
                'q05': float(np.mean(q05s)) if q05s else 0.0,
                'q50': float(np.mean(q50s)) if q50s else 0.0,
                'q95': float(np.mean(q95s)) if q95s else 0.0,
                'weights_used': weights,
            }
        
        return result
    
    def _aggregate_best(
        self,
        predictions: List[Dict],
        return_distribution: bool
    ) -> Dict[int, Dict[str, Any]]:
        """
        Use predictions from best model only.
        
        Best model determined by validation MAE in metadata.
        Falls back to first model if no validation scores available.
        """
        # Find best model (lowest MAE)
        best_idx = 0
        best_mae = float('inf')
        
        for i, meta in enumerate(self.metadata):
            mae = meta.get('val_mae')
            if mae and mae < best_mae:
                best_mae = mae
                best_idx = i
        
        logger.info(f"Using best model (index {best_idx}): {self.model_paths[best_idx].name} (MAE={best_mae})")
        
        # Return predictions from best model
        best_preds = predictions[best_idx]['predictions']
        
        # Add metadata about which model was used
        for horizon_data in best_preds.values():
            horizon_data['best_model_index'] = best_idx
            horizon_data['best_model_name'] = self.model_paths[best_idx].name
            horizon_data['best_model_mae'] = best_mae
        
        return best_preds
