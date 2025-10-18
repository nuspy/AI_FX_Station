"""
Standardized Model Loader for unified model loading across the application.

Consolidates all model loading logic and provides consistent interface for:
- PyTorch models (.pt, .pth)
- Scikit-learn models (.pkl, .pickle, .joblib)
- Custom model artifacts with metadata
- Model validation and compatibility checking
"""
from __future__ import annotations

import os
import pickle
import joblib
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import torch
import numpy as np
from loguru import logger

from .metadata_manager import MetadataManager, ModelMetadata
from .model_path_resolver import ModelPathResolver


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class StandardizedModelLoader:
    """
    Unified model loader with consistent interface and error handling.

    Features:
    - Automatic format detection (.pt/.pth/.pkl/.pickle/.joblib)
    - Metadata extraction and validation
    - Model compatibility checking
    - Consistent error handling and logging
    - Integration with ModelPathResolver and MetadataManager
    """

    def __init__(self, metadata_manager: Optional[MetadataManager] = None):
        self.metadata_manager = metadata_manager or MetadataManager()
        self.path_resolver = ModelPathResolver()

        # Supported loaders
        self._loaders = {
            '.pt': self._load_pytorch,
            '.pth': self._load_pytorch,
            '.ckpt': self._load_pytorch,  # Lightning checkpoints
            '.pkl': self._load_pickle,
            '.pickle': self._load_pickle,
            '.joblib': self._load_joblib
        }

    def load_models(self, settings: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Load multiple models from settings with full validation.

        Args:
            settings: Settings dictionary from UI

        Returns:
            Dictionary mapping model_path -> loaded_model_data
        """
        model_paths = self.path_resolver.resolve_model_paths(settings)

        if not model_paths:
            raise ModelLoadError("No valid model paths found")

        loaded_models = {}
        failed_models = []

        for model_path in model_paths:
            try:
                model_data = self.load_single_model(model_path)
                loaded_models[model_path] = model_data
                logger.info(f"Successfully loaded model: {Path(model_path).name}")

            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
                failed_models.append((model_path, str(e)))

        if not loaded_models:
            error_msg = "All models failed to load:\n" + "\n".join(
                f"  - {Path(path).name}: {error}"
                for path, error in failed_models
            )
            raise ModelLoadError(error_msg)

        if failed_models:
            logger.warning(f"Successfully loaded {len(loaded_models)} models, "
                         f"failed to load {len(failed_models)} models")

        return loaded_models

    def load_single_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a single model with metadata extraction.

        Args:
            model_path: Path to model file

        Returns:
            Dictionary with model data and metadata
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise ModelLoadError(f"Model file does not exist: {model_path}")

        if not model_path.is_file():
            raise ModelLoadError(f"Path is not a file: {model_path}")

        extension = model_path.suffix.lower()
        if extension not in self._loaders:
            raise ModelLoadError(f"Unsupported model format: {extension}")

        logger.debug(f"Loading model: {model_path}")

        try:
            # Load model using appropriate loader
            loader = self._loaders[extension]
            model_data = loader(model_path)

            # Extract and validate metadata
            metadata = self._extract_metadata(model_data, model_path)

            # Validate model compatibility
            validation_result = self._validate_model(model_data, metadata)

            return {
                'model': model_data.get('model'),
                'features': model_data.get('features', []),
                'metadata': metadata,
                'model_path': str(model_path),
                'model_type': self._determine_model_type(model_data),
                'validation': validation_result,
                'standardizer': model_data.get('standardizer'),
                'scaler': model_data.get('scaler', model_data.get('std')),  # Handle legacy naming
                'pca': model_data.get('pca'),  # PCA preprocessor for backward compatibility
                'encoder': model_data.get('encoder'),  # Generic encoder (PCA, VAE, Autoencoder)
                'preprocessor': model_data.get('preprocessor'),  # Alternative key for encoder
                'encoder_type': model_data.get('encoder_type', 'none'),  # Type of encoder used
                'raw_data': model_data  # Keep original for compatibility
            }

        except Exception as e:
            raise ModelLoadError(f"Failed to load {model_path}: {e}") from e

    def _load_pytorch(self, model_path: Path) -> Dict[str, Any]:
        """Load PyTorch model (.pt/.pth/.ckpt)."""
        try:
            # Try GPU first, fallback to CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                data = torch.load(model_path, map_location=device)
            except Exception:
                # Fallback to CPU if GPU loading fails
                data = torch.load(model_path, map_location='cpu')

            # Check if this is a Lightning checkpoint
            logger.debug(f"Loaded data type: {type(data)}, is dict: {isinstance(data, dict)}")
            if isinstance(data, dict):
                logger.debug(f"Dict keys: {list(data.keys())}")
            
            if isinstance(data, dict) and 'state_dict' in data:
                # This is a Lightning checkpoint - use LightningMultiHorizonPredictor
                logger.info(f"Lightning checkpoint detected: {model_path.name}, loading with LightningMultiHorizonPredictor")
                
                try:
                    from ..inference.lightning_predictor import LightningMultiHorizonPredictor
                    
                    # Create predictor instance (loads the model)
                    predictor = LightningMultiHorizonPredictor(
                        checkpoint_path=str(model_path),
                        device='cpu'  # Will be moved to GPU if needed
                    )
                    
                    # Return wrapped in standard format
                    return {
                        'model': predictor,  # The predictor has a predict() method!
                        'model_type': 'lightning',
                        'checkpoint_path': str(model_path),
                        'horizons': predictor.horizons,
                        'is_multi_horizon': predictor.is_multi_horizon,
                        'hyper_parameters': data.get('hyper_parameters', {}),
                        'epoch': data.get('epoch'),
                    }
                    
                except ImportError as e:
                    logger.error(f"Failed to import LightningMultiHorizonPredictor: {e}")
                    logger.warning("Lightning checkpoints require lightning_predictor module")
                    # Fallback to old behavior
                    return {
                        'model': None,
                        'state_dict': data.get('state_dict'),
                        'hyper_parameters': data.get('hyper_parameters', {}),
                        'model_type': 'lightning',
                        'checkpoint_path': str(model_path)
                    }
                except Exception as e:
                    logger.error(f"Failed to load Lightning model with predictor: {e}")
                    # Fallback to old behavior
                    return {
                        'model': None,
                        'state_dict': data.get('state_dict'),
                        'hyper_parameters': data.get('hyper_parameters', {}),
                        'model_type': 'lightning',
                        'checkpoint_path': str(model_path)
                    }

            if isinstance(data, dict):
                return data
            else:
                # If it's just a model, wrap in standard format
                return {'model': data}

        except Exception as e:
            raise ModelLoadError(f"PyTorch model loading failed: {e}") from e

    def _load_pickle(self, model_path: Path) -> Dict[str, Any]:
        """Load pickled model (.pkl/.pickle), handles both plain and compressed formats."""
        import gzip
        import zlib

        try:
            # Try joblib first (it handles compression automatically)
            try:
                data = joblib.load(model_path)
                logger.debug("Successfully loaded with joblib")
                if isinstance(data, dict):
                    return data
                else:
                    return {'model': data}
            except Exception as joblib_err:
                logger.debug(f"Joblib load failed: {joblib_err}, trying pickle methods")

            # Try loading directly (uncompressed pickle)
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                logger.debug("Successfully loaded uncompressed pickle")
            except (pickle.UnpicklingError, EOFError) as e:
                # If direct pickle fails, try decompressing first
                logger.debug(f"Direct pickle load failed, trying decompression: {e}")

                # Try gzip decompression
                try:
                    with gzip.open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.debug("Successfully loaded gzip-compressed pickle")
                except Exception as gzip_err:
                    logger.debug(f"Gzip load failed: {gzip_err}, trying zlib")
                    # Try zlib decompression
                    try:
                        with open(model_path, 'rb') as f:
                            compressed_data = f.read()
                        decompressed_data = zlib.decompress(compressed_data)
                        data = pickle.loads(decompressed_data)
                        logger.debug("Successfully loaded zlib-compressed pickle")
                    except Exception as zlib_err:
                        logger.error(f"All decompression methods failed. Joblib: {joblib_err}, Gzip: {gzip_err}, Zlib: {zlib_err}")
                        raise

            if isinstance(data, dict):
                return data
            else:
                # If it's just a model, wrap in standard format
                return {'model': data}

        except Exception as e:
            raise ModelLoadError(f"Pickle model loading failed: {e}") from e

    def _load_joblib(self, model_path: Path) -> Dict[str, Any]:
        """Load joblib model (.joblib)."""
        try:
            data = joblib.load(model_path)

            if isinstance(data, dict):
                return data
            else:
                # If it's just a model, wrap in standard format
                return {'model': data}

        except Exception as e:
            raise ModelLoadError(f"Joblib model loading failed: {e}") from e

    def _extract_metadata(self, model_data: Dict[str, Any], model_path: Path) -> ModelMetadata:
        """Extract metadata from loaded model data."""
        try:
            # Try to get metadata from model data first
            if 'metadata' in model_data:
                metadata_dict = model_data['metadata']
                if isinstance(metadata_dict, ModelMetadata):
                    # Ensure model_path is set
                    if not hasattr(metadata_dict, 'model_path') or not metadata_dict.model_path:
                        metadata_dict.model_path = str(model_path)
                    return metadata_dict
                elif isinstance(metadata_dict, dict):
                    return ModelMetadata.from_dict(metadata_dict, model_path=str(model_path))

            # Try to load metadata from companion file
            try:
                companion_metadata = self.metadata_manager.load_metadata(str(model_path))
                if companion_metadata:
                    return companion_metadata
            except Exception:
                pass

            # Create basic metadata from model info
            model_info = self.path_resolver.get_model_info(str(model_path))

            # Determine model type and extract basic info
            model_type = self._determine_model_type(model_data)

            # Extract training parameters if available
            training_params = self._extract_training_params(model_data)

            # Create metadata object and set attributes (ModelMetadata.__init__ takes no args)
            metadata = ModelMetadata()
            metadata.model_path = str(model_path)
            metadata.model_type = model_type
            metadata.file_size = model_info.get('size_bytes', 0)
            metadata.created_at = model_info.get('modified', 0)
            metadata.feature_names = model_data.get('features', [])
            metadata.training_params = training_params
            
            return metadata

        except Exception as e:
            logger.warning(f"Failed to extract metadata for {model_path}: {e}")
            # Return minimal metadata
            metadata = ModelMetadata()
            metadata.model_path = str(model_path)
            metadata.model_type = 'unknown'
            metadata.file_size = model_path.stat().st_size if model_path.exists() else 0
            
            return metadata

    def _determine_model_type(self, model_data: Dict[str, Any]) -> str:
        """Determine the type of model from loaded data."""
        model = model_data.get('model')

        if model is None:
            return 'unknown'

        # Check PyTorch models
        if hasattr(model, 'state_dict') or isinstance(model, torch.nn.Module):
            return 'pytorch'

        # Check for VAE/Diffusion models
        if 'vae' in model_data or 'diffusion' in model_data:
            return 'vae_diffusion'

        # Check sklearn models
        model_str = str(type(model))
        if 'sklearn' in model_str:
            return 'sklearn'
        elif 'xgboost' in model_str:
            return 'xgboost'
        elif 'lightgbm' in model_str:
            return 'lightgbm'
        elif 'catboost' in model_str:
            return 'catboost'

        return 'custom'

    def _extract_training_params(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract training parameters from model data."""
        params = {}

        # Common parameter names to look for
        param_keys = [
            'timeframe', 'horizon', 'features_config', 'standardizer_config',
            'n_estimators', 'max_depth', 'learning_rate', 'random_state',
            'train_size', 'val_size', 'test_size', 'epochs', 'batch_size'
        ]

        for key in param_keys:
            if key in model_data:
                params[key] = model_data[key]

        # Check for nested parameter structures
        if 'config' in model_data:
            config = model_data['config']
            if isinstance(config, dict):
                params.update(config)

        if 'training_info' in model_data:
            training_info = model_data['training_info']
            if isinstance(training_info, dict):
                params.update(training_info)

        return params

    def _validate_model(self, model_data: Dict[str, Any], metadata: ModelMetadata) -> Dict[str, Any]:
        """Validate loaded model for common issues."""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }

        model = model_data.get('model')

        # Check if model exists
        if model is None:
            validation['valid'] = False
            validation['errors'].append("No model found in loaded data")
            return validation

        # Check features consistency
        features = model_data.get('features', [])
        if not features:
            validation['warnings'].append("No feature list found in model data")

        # Check for standardizer/scaler
        standardizer = model_data.get('standardizer') or model_data.get('scaler') or model_data.get('std')
        if not standardizer:
            validation['warnings'].append("No standardizer/scaler found - predictions may be unstable")

        # Model-specific validations
        model_type = metadata.model_type

        if model_type == 'pytorch':
            try:
                # Check if model is in eval mode
                if hasattr(model, 'training') and model.training:
                    validation['warnings'].append("PyTorch model is in training mode - switching to eval mode is recommended")
            except Exception:
                pass

        elif model_type == 'sklearn':
            try:
                # Check if model has predict method
                if not hasattr(model, 'predict'):
                    validation['valid'] = False
                    validation['errors'].append("Sklearn model missing predict method")
            except Exception:
                pass

        # Check file integrity
        model_path = Path(metadata.model_path)
        if model_path.exists():
            actual_size = model_path.stat().st_size
            if actual_size == 0:
                validation['valid'] = False
                validation['errors'].append("Model file is empty")
            elif actual_size != metadata.file_size and metadata.file_size > 0:
                validation['warnings'].append("Model file size mismatch - file may have been modified")

        return validation

    def get_model_summary(self, model_paths: list) -> Dict[str, Any]:
        """Get summary information about multiple models without loading them."""
        summary = {
            'total_models': len(model_paths),
            'by_type': {},
            'total_size_mb': 0,
            'models': []
        }

        for model_path in model_paths:
            try:
                model_info = self.path_resolver.get_model_info(model_path)
                model_type = model_info.get('model_type', 'unknown')

                summary['by_type'][model_type] = summary['by_type'].get(model_type, 0) + 1
                summary['total_size_mb'] += model_info.get('size_mb', 0)
                summary['models'].append({
                    'path': model_path,
                    'name': model_info.get('filename', ''),
                    'type': model_type,
                    'size_mb': model_info.get('size_mb', 0),
                    'exists': model_info.get('exists', False)
                })

            except Exception as e:
                logger.warning(f"Failed to get info for {model_path}: {e}")
                summary['models'].append({
                    'path': model_path,
                    'name': Path(model_path).name,
                    'type': 'error',
                    'error': str(e)
                })

        return summary


# Global instance for convenience
_global_loader: Optional[StandardizedModelLoader] = None

def get_model_loader() -> StandardizedModelLoader:
    """Get the global standardized model loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = StandardizedModelLoader()
    return _global_loader