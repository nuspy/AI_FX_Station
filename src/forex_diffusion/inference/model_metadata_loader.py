"""
Model Metadata Loader

Loads model metadata (sidecar) and validates inference compatibility.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
from loguru import logger

from ..utils.horizon_parser import get_model_horizons_from_metadata


class ModelMetadataLoader:
    """
    Loads and validates model metadata for inference.
    
    Features:
    - Loads metadata from .pkl (sklearn) or .ckpt.meta.json (lightning)
    - Extracts training parameters (horizons, symbol, timeframe, etc.)
    - Validates compatibility with inference settings
    - Suggests auto-configuration
    """
    
    def __init__(self, model_path: str | Path):
        """
        Initialize loader.
        
        Args:
            model_path: Path to model file (.pkl or .ckpt)
        """
        self.model_path = Path(model_path)
        self.metadata = {}
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """Detect if model is sklearn or lightning."""
        if self.model_path.suffix == '.pkl':
            return 'sklearn'
        elif self.model_path.suffix == '.ckpt':
            return 'lightning'
        elif self.model_path.suffix == '.pt':
            return 'lightning'
        else:
            return 'unknown'
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from model sidecar.
        
        Returns:
            Metadata dictionary
        """
        if self.model_type == 'sklearn':
            return self._load_sklearn_metadata()
        elif self.model_type == 'lightning':
            return self._load_lightning_metadata()
        else:
            raise ValueError(f"Unknown model type for {self.model_path}")
    
    def _load_sklearn_metadata(self) -> Dict[str, Any]:
        """Load metadata from sklearn .pkl file."""
        try:
            from joblib import load
            payload = load(self.model_path)
            
            # Extract metadata
            horizons = get_model_horizons_from_metadata(payload)
            
            metadata = {
                'model_type': 'sklearn',
                'algorithm': payload.get('model_type', 'unknown'),
                'horizons': horizons,
                'num_horizons': len(horizons),
                'is_multi_horizon': len(horizons) > 1,
                'symbol': payload.get('params_used', {}).get('symbol', 'EUR/USD'),
                'timeframe': payload.get('params_used', {}).get('timeframe', '1m'),
                'features': payload.get('features', []),
                'num_features': len(payload.get('features', [])),
                'encoder_type': payload.get('encoder_type', 'none'),
                'val_mae': payload.get('val_mae'),
                'optimization_strategy': payload.get('optimization_strategy', 'none'),
            }
            
            self.metadata = metadata
            logger.info(f"Loaded sklearn metadata: {metadata['algorithm']}, horizons={metadata['horizons']}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load sklearn metadata: {e}")
            raise
    
    def _load_lightning_metadata(self) -> Dict[str, Any]:
        """Load metadata from lightning .ckpt.meta.json file."""
        try:
            # Look for sidecar .meta.json file
            meta_path = self.model_path.with_suffix(self.model_path.suffix + '.meta.json')
            
            if not meta_path.exists():
                logger.warning(f"No metadata file found: {meta_path}")
                # Try loading from checkpoint hparams
                import torch
                ckpt = torch.load(self.model_path, map_location='cpu')
                hparams = ckpt.get('hyper_parameters', {})
                
                # Extract horizons
                if 'horizons' in hparams:
                    horizons = hparams['horizons']
                elif 'horizon' in hparams:
                    horizon = hparams['horizon']
                    horizons = horizon if isinstance(horizon, list) else [horizon]
                else:
                    horizons = [60]  # Default fallback
                
                metadata = {
                    'model_type': 'lightning',
                    'horizons': horizons,
                    'num_horizons': len(horizons),
                    'is_multi_horizon': len(horizons) > 1,
                    'symbol': hparams.get('symbol', 'EUR/USD'),
                    'timeframe': hparams.get('timeframe', '1m'),
                    'patch_len': hparams.get('patch_len', 64),
                }
            else:
                # Load from meta.json
                with open(meta_path, 'r') as f:
                    payload = json.load(f)
                
                horizons = get_model_horizons_from_metadata(payload)
                
                metadata = {
                    'model_type': 'lightning',
                    'horizons': horizons,
                    'num_horizons': len(horizons),
                    'is_multi_horizon': len(horizons) > 1,
                    'symbol': payload.get('symbol', 'EUR/USD'),
                    'timeframe': payload.get('timeframe', '1m'),
                    'patch_len': payload.get('patch_len', 64),
                    'channel_order': payload.get('channel_order', []),
                }
            
            self.metadata = metadata
            logger.info(f"Loaded lightning metadata: horizons={metadata['horizons']}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load lightning metadata: {e}")
            raise
    
    def validate_inference_settings(
        self,
        inference_horizons: List[int],
        inference_symbol: str,
        inference_timeframe: str
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate inference settings against model metadata.
        
        Args:
            inference_horizons: Horizons requested for inference
            inference_symbol: Symbol for inference
            inference_timeframe: Timeframe for inference
            
        Returns:
            Tuple of (is_compatible, warnings, errors)
            - is_compatible: True if can proceed (with or without warnings)
            - warnings: List of warning messages (non-critical)
            - errors: List of error messages (critical incompatibilities)
        """
        if not self.metadata:
            self.load_metadata()
        
        warnings = []
        errors = []
        
        # Check horizons
        model_horizons = self.metadata['horizons']
        
        if sorted(inference_horizons) == sorted(model_horizons):
            # Perfect match
            pass
        elif set(inference_horizons).issubset(set(model_horizons)):
            # Inference is subset of training - OK but warn
            warnings.append(
                f"Using subset of horizons: {inference_horizons} "
                f"(model trained on {model_horizons})"
            )
        elif set(model_horizons).issubset(set(inference_horizons)):
            # Requesting more horizons than trained - requires interpolation
            missing = set(inference_horizons) - set(model_horizons)
            warnings.append(
                f"INTERPOLATION REQUIRED for horizons: {sorted(missing)}\n"
                f"Model trained on: {model_horizons}\n"
                f"You requested: {inference_horizons}\n"
                f"Recommended: Use trained horizons {model_horizons} for best accuracy"
            )
        else:
            # Completely different horizons - error
            errors.append(
                f"INCOMPATIBLE HORIZONS!\n"
                f"Model trained on: {model_horizons}\n"
                f"Inference requested: {inference_horizons}\n"
                f"Cannot predict unseen horizons.\n"
                f"Options:\n"
                f"  1. Use model horizons: {model_horizons}\n"
                f"  2. Re-train model with: {inference_horizons}"
            )
        
        # Check symbol
        model_symbol = self.metadata.get('symbol', 'EUR/USD')
        if inference_symbol != model_symbol:
            warnings.append(
                f"Symbol mismatch: model trained on {model_symbol}, "
                f"inference on {inference_symbol}"
            )
        
        # Check timeframe
        model_timeframe = self.metadata.get('timeframe', '1m')
        if inference_timeframe != model_timeframe:
            warnings.append(
                f"Timeframe mismatch: model trained on {model_timeframe}, "
                f"inference on {inference_timeframe}"
            )
        
        is_compatible = len(errors) == 0
        
        return is_compatible, warnings, errors
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """
        Get recommended inference settings based on model metadata.
        
        Returns:
            Dictionary of recommended settings
        """
        if not self.metadata:
            self.load_metadata()
        
        from ..utils.horizon_parser import format_horizon_spec
        
        return {
            'symbol': self.metadata.get('symbol', 'EUR/USD'),
            'timeframe': self.metadata.get('timeframe', '1m'),
            'horizons': self.metadata['horizons'],
            'horizon_str': format_horizon_spec(self.metadata['horizons']),
            'num_samples': 50 if self.metadata['model_type'] == 'lightning' else 1,
        }
    
    def get_interpolation_plan(
        self,
        inference_horizons: List[int]
    ) -> Optional[Dict[int, Tuple[int, int, float]]]:
        """
        Get interpolation plan for missing horizons.
        
        Args:
            inference_horizons: Requested horizons
            
        Returns:
            Dict mapping horizon -> (lower_bound, upper_bound, weight)
            or None if no interpolation needed
            
        Example:
            Model trained on [15, 60, 240]
            Inference requests [15, 30, 60, 120, 240]
            
            Returns:
            {
                30: (15, 60, 0.333),    # 30 is 1/3 between 15 and 60
                120: (60, 240, 0.333),  # 120 is 1/3 between 60 and 240
            }
        """
        if not self.metadata:
            self.load_metadata()
        
        model_horizons = sorted(self.metadata['horizons'])
        inference_horizons = sorted(inference_horizons)
        
        missing = set(inference_horizons) - set(model_horizons)
        if not missing:
            return None
        
        interpolation_plan = {}
        
        for h in sorted(missing):
            # Find surrounding horizons
            lower = None
            upper = None
            
            for mh in model_horizons:
                if mh < h:
                    lower = mh
                elif mh > h and upper is None:
                    upper = mh
                    break
            
            if lower is not None and upper is not None:
                # Calculate interpolation weight
                weight = (h - lower) / (upper - lower)
                interpolation_plan[h] = (lower, upper, weight)
            else:
                # Cannot interpolate (outside range)
                logger.warning(f"Cannot interpolate horizon {h} - outside trained range")
        
        return interpolation_plan if interpolation_plan else None
