"""
Enhanced Metadata Manager for complete Training-Inference consistency.

This module manages comprehensive metadata persistence to ensure that all parameters
used during training are properly saved and restored during inference.
"""
from __future__ import annotations

import json
import pickle
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import numpy as np
from loguru import logger

from ..features.unified_pipeline import FeatureConfig


class ModelMetadata:
    """
    Comprehensive metadata container for model persistence.
    Stores all information needed to recreate training conditions during inference.
    """

    def __init__(self):
        self.version = "2.0"  # Metadata schema version
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = None

        # Model file information
        self.model_path = None
        self.file_size = 0

        # Model information
        self.model_type = None  # 'sklearn', 'pytorch', 'lightning'
        self.model_class = None
        self.model_hash = None

        # Training configuration
        self.symbol = None
        self.base_timeframe = None
        self.horizon_bars = None
        self.horizon_minutes = None
        self.horizons = None  # For multi-horizon models (list of horizon values)
        self.is_multi_horizon = False  # Flag for multi-horizon support
        self.days_history = None

        # Feature engineering configuration
        self.feature_config = None  # FeatureConfig object serialized
        self.feature_names = []
        self.num_features = 0

        # Preprocessing parameters
        self.preprocessing_config = {}
        self.standardization_params = {}
        self.normalization_method = None

        # Multi-timeframe configuration
        self.multi_timeframe_enabled = False
        self.multi_timeframe_config = {}
        self.hierarchical_mode = False
        self.query_timeframe = None

        # Standardizer state
        self.standardizer_mu = {}
        self.standardizer_sigma = {}
        self.standardizer_columns = []

        # Training hyperparameters
        self.training_params = {}
        self.optimization_config = {}

        # Performance metrics
        self.validation_metrics = {}
        self.training_metrics = {}

        # Data statistics
        self.data_stats = {}

        # Advanced parameters for inference replication
        self.inference_config = {}

        # Compatibility and validation
        self.required_packages = {}
        self.validation_hash = None

    def set_model_info(self, model: Any, model_type: str):
        """Set model information and generate hash."""
        self.model_type = model_type
        self.model_class = type(model).__name__

        # Generate model hash for validation
        try:
            if hasattr(model, 'state_dict'):  # PyTorch
                model_bytes = str(model.state_dict()).encode()
            elif hasattr(model, 'get_params'):  # sklearn
                model_bytes = str(model.get_params()).encode()
            else:
                model_bytes = str(model).encode()

            self.model_hash = hashlib.sha256(model_bytes).hexdigest()[:16]
        except Exception:
            self.model_hash = None

    def set_training_config(self, **kwargs):
        """Set training configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.training_params[key] = value

    def set_feature_config(self, config: FeatureConfig, feature_names: List[str]):
        """Set feature engineering configuration."""
        self.feature_config = config.to_dict()
        self.feature_names = feature_names.copy()
        self.num_features = len(feature_names)

        # Extract multi-timeframe info
        if config.is_multi_timeframe_enabled():
            self.multi_timeframe_enabled = True
            self.multi_timeframe_config = config.get_multi_timeframe_config()
            self.hierarchical_mode = self.multi_timeframe_config.get("hierarchical_mode", False)
            self.query_timeframe = self.multi_timeframe_config.get("query_timeframe")

    def set_standardizer(self, standardizer):
        """Set standardizer parameters."""
        if standardizer is not None:
            self.standardizer_mu = getattr(standardizer, 'mu', {})
            self.standardizer_sigma = getattr(standardizer, 'sigma', {})
            self.standardizer_columns = getattr(standardizer, 'cols', [])

            # Convert numpy arrays to lists for JSON serialization
            for key in self.standardizer_mu:
                if isinstance(self.standardizer_mu[key], np.ndarray):
                    self.standardizer_mu[key] = self.standardizer_mu[key].tolist()
                elif isinstance(self.standardizer_mu[key], np.number):
                    self.standardizer_mu[key] = float(self.standardizer_mu[key])

            for key in self.standardizer_sigma:
                if isinstance(self.standardizer_sigma[key], np.ndarray):
                    self.standardizer_sigma[key] = self.standardizer_sigma[key].tolist()
                elif isinstance(self.standardizer_sigma[key], np.number):
                    self.standardizer_sigma[key] = float(self.standardizer_sigma[key])

    def set_preprocessing_config(self, **config):
        """Set preprocessing configuration."""
        self.preprocessing_config.update(config)

    def set_training_metrics(self, **metrics):
        """Set training performance metrics."""
        self.training_metrics.update(metrics)

    def set_validation_metrics(self, **metrics):
        """Set validation performance metrics."""
        self.validation_metrics.update(metrics)

    def set_data_stats(self, **stats):
        """Set data statistics."""
        self.data_stats.update(stats)

    def set_inference_config(self, **config):
        """Set inference-specific configuration."""
        self.inference_config.update(config)

    def generate_validation_hash(self):
        """Generate validation hash for integrity checking."""
        validation_data = {
            'feature_names': self.feature_names,
            'feature_config': self.feature_config,
            'standardizer_mu': self.standardizer_mu,
            'standardizer_sigma': self.standardizer_sigma,
            'model_type': self.model_type,
            'horizon_bars': self.horizon_bars,
            'base_timeframe': self.base_timeframe
        }
        validation_str = json.dumps(validation_data, sort_keys=True)
        self.validation_hash = hashlib.sha256(validation_str.encode()).hexdigest()[:16]

    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        self.update_timestamp()
        self.generate_validation_hash()

        return {
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,

            # Model information
            'model_type': self.model_type,
            'model_class': self.model_class,
            'model_hash': self.model_hash,

            # Training configuration
            'symbol': self.symbol,
            'base_timeframe': self.base_timeframe,
            'horizon_bars': self.horizon_bars,
            'horizon_minutes': self.horizon_minutes,
            'horizons': self.horizons,
            'is_multi_horizon': self.is_multi_horizon,
            'days_history': self.days_history,

            # Feature engineering
            'feature_config': self.feature_config,
            'feature_names': self.feature_names,
            'num_features': self.num_features,

            # Preprocessing
            'preprocessing_config': self.preprocessing_config,
            'standardization_params': self.standardization_params,
            'normalization_method': self.normalization_method,

            # Multi-timeframe
            'multi_timeframe_enabled': self.multi_timeframe_enabled,
            'multi_timeframe_config': self.multi_timeframe_config,
            'hierarchical_mode': self.hierarchical_mode,
            'query_timeframe': self.query_timeframe,

            # Standardizer
            'standardizer_mu': self.standardizer_mu,
            'standardizer_sigma': self.standardizer_sigma,
            'standardizer_columns': self.standardizer_columns,

            # Training
            'training_params': self.training_params,
            'optimization_config': self.optimization_config,

            # Metrics
            'validation_metrics': self.validation_metrics,
            'training_metrics': self.training_metrics,

            # Data
            'data_stats': self.data_stats,

            # Inference
            'inference_config': self.inference_config,

            # Validation
            'required_packages': self.required_packages,
            'validation_hash': self.validation_hash
        }
    
    def get(self, key: str, default=None):
        """
        Dict-like get method for backward compatibility.
        
        Allows metadata to be accessed like a dictionary:
        metadata.get('horizons') instead of metadata.horizons
        """
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_path: Optional[str] = None) -> 'ModelMetadata':
        """Create ModelMetadata from dictionary.

        Args:
            data: Dictionary with metadata
            model_path: Optional model path to set if not in data
        """
        metadata = cls()

        for key, value in data.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        # Ensure model_path is set (use parameter if not in data)
        if not hasattr(metadata, 'model_path') or not metadata.model_path:
            if model_path:
                metadata.model_path = model_path

        return metadata

    def validate_compatibility(self, inference_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate compatibility between training metadata and inference configuration.

        Returns:
            Dict with validation results and warnings
        """
        results = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        # Check feature compatibility
        if 'feature_names' in inference_config:
            inference_features = set(inference_config['feature_names'])
            training_features = set(self.feature_names)

            missing_features = training_features - inference_features
            extra_features = inference_features - training_features

            if missing_features:
                results['errors'].append(f"Missing features: {list(missing_features)}")
                results['compatible'] = False

            if extra_features:
                results['warnings'].append(f"Extra features (will be ignored): {list(extra_features)}")

        # Check timeframe compatibility
        if 'timeframe' in inference_config:
            if inference_config['timeframe'] != self.base_timeframe:
                results['warnings'].append(
                    f"Timeframe mismatch: training={self.base_timeframe}, inference={inference_config['timeframe']}"
                )

        # Check multi-timeframe compatibility
        if inference_config.get('multi_timeframe_enabled', False) != self.multi_timeframe_enabled:
            results['errors'].append("Multi-timeframe mode mismatch between training and inference")
            results['compatible'] = False

        return results

    def get_inference_config(self) -> Dict[str, Any]:
        """Get configuration needed for inference."""
        return {
            'feature_config': self.feature_config,
            'feature_names': self.feature_names,
            'standardizer_mu': self.standardizer_mu,
            'standardizer_sigma': self.standardizer_sigma,
            'standardizer_columns': self.standardizer_columns,
            'base_timeframe': self.base_timeframe,
            'horizon_bars': self.horizon_bars,
            'multi_timeframe_enabled': self.multi_timeframe_enabled,
            'multi_timeframe_config': self.multi_timeframe_config,
            'query_timeframe': self.query_timeframe,
            'preprocessing_config': self.preprocessing_config,
            'inference_config': self.inference_config,
            'model_type': self.model_type
        }


class MetadataManager:
    """Manager for model metadata persistence and retrieval."""

    @staticmethod
    def save_metadata(metadata: ModelMetadata, model_path: Union[str, Path], format: str = 'sidecar') -> Path:
        """
        Save metadata to file.

        Args:
            metadata: ModelMetadata object to save
            model_path: Path to the model file
            format: 'sidecar' (separate .meta.json) or 'embedded' (in model file)

        Returns:
            Path to the saved metadata file
        """
        model_path = Path(model_path)

        if format == 'sidecar':
            # Save as separate .meta.json file
            meta_path = Path(str(model_path) + ".meta.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Saved metadata to {meta_path}")
            return meta_path

        elif format == 'embedded':
            # Embed in model file (for pickle files)
            if model_path.suffix.lower() in ('.pkl', '.pickle'):
                # Read existing model
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Add metadata
                if isinstance(model_data, dict):
                    model_data['metadata'] = metadata.to_dict()
                else:
                    model_data = {
                        'model': model_data,
                        'metadata': metadata.to_dict()
                    }

                # Save back
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)

                logger.info(f"Embedded metadata in {model_path}")
                return model_path

            else:
                raise ValueError(f"Embedded format not supported for {model_path.suffix}")

        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def load_metadata(model_path: Union[str, Path]) -> Optional[ModelMetadata]:
        """
        Load metadata from file.

        Args:
            model_path: Path to the model file

        Returns:
            ModelMetadata object or None if not found
        """
        model_path = Path(model_path)

        # Try sidecar file first
        sidecar_path = Path(str(model_path) + ".meta.json")
        if sidecar_path.exists():
            try:
                with open(sidecar_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded metadata from sidecar: {sidecar_path}")
                return ModelMetadata.from_dict(data, model_path=str(model_path))
            except Exception as e:
                logger.warning(f"Failed to load sidecar metadata: {e}")

        # Try embedded metadata
        if model_path.suffix.lower() in ('.pkl', '.pickle'):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                if isinstance(model_data, dict) and 'metadata' in model_data:
                    logger.info(f"Loaded embedded metadata from: {model_path}")
                    return ModelMetadata.from_dict(model_data['metadata'], model_path=str(model_path))

            except Exception as e:
                logger.warning(f"Failed to load embedded metadata: {e}")

        # Try legacy format (for backward compatibility)
        try:
            legacy_metadata = MetadataManager._load_legacy_metadata(model_path)
            if legacy_metadata:
                logger.info(f"Loaded legacy metadata from: {model_path}")
                return legacy_metadata
        except Exception as e:
            logger.debug(f"No legacy metadata found: {e}")

        logger.warning(f"No metadata found for model: {model_path}")
        return None

    @staticmethod
    def _load_legacy_metadata(model_path: Path) -> Optional[ModelMetadata]:
        """Load metadata from legacy format for backward compatibility."""
        # Try loading from pickle file with old format
        if model_path.suffix.lower() in ('.pkl', '.pickle'):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                if isinstance(model_data, dict):
                    # Convert old format to new metadata
                    metadata = ModelMetadata()

                    # Map old keys to new metadata structure
                    if 'features' in model_data:
                        metadata.feature_names = model_data['features']
                        metadata.num_features = len(model_data['features'])

                    if 'std_mu' in model_data:
                        metadata.standardizer_mu = model_data['std_mu']

                    if 'std_sigma' in model_data:
                        metadata.standardizer_sigma = model_data['std_sigma']
                    elif 'std' in model_data:  # Very old format
                        metadata.standardizer_mu = model_data['std']
                        metadata.standardizer_sigma = {k: 1.0 for k in model_data['std'].keys()}

                    # Try to infer model type
                    model = model_data.get('model')
                    if model:
                        if hasattr(model, 'predict'):
                            metadata.model_type = 'sklearn'
                        else:
                            metadata.model_type = 'unknown'

                    return metadata

            except Exception:
                pass

        return None

    @staticmethod
    def create_from_training(
        model: Any,
        model_type: str,
        feature_config: FeatureConfig,
        feature_names: List[str],
        standardizer: Any,
        **kwargs
    ) -> ModelMetadata:
        """
        Create comprehensive metadata from training components.

        Args:
            model: Trained model object
            model_type: Type of model ('sklearn', 'pytorch', 'lightning')
            feature_config: FeatureConfig used during training
            feature_names: List of feature names
            standardizer: Fitted standardizer object
            **kwargs: Additional training parameters

        Returns:
            Complete ModelMetadata object
        """
        metadata = ModelMetadata()

        # Set model information
        metadata.set_model_info(model, model_type)

        # Set feature configuration
        metadata.set_feature_config(feature_config, feature_names)

        # Set standardizer
        metadata.set_standardizer(standardizer)

        # Set training configuration
        metadata.set_training_config(**kwargs)

        # Set preprocessing config from feature config
        preprocessing_config = {
            'relative_ohlc': feature_config.config['base_features']['relative_ohlc'],
            'log_returns': feature_config.config['base_features']['log_returns'],
            'warmup_bars': feature_config.config['warmup_bars'],
            'rv_window': feature_config.config['rv_window']
        }
        metadata.set_preprocessing_config(**preprocessing_config)

        # Set inference config
        inference_config = {
            'apply_conformal': kwargs.get('apply_conformal', True),
            'conformal_alpha': kwargs.get('conformal_alpha', 0.1),
            'model_weight_pct': kwargs.get('model_weight_pct', 100)
        }
        metadata.set_inference_config(**inference_config)

        return metadata