"""
Lazy Model Loading

Loads models on-demand rather than at startup.
Implements OPT-003 - reduces startup time from 30s to <1s, memory from 15GB to 1-2GB.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class LazyModelLoader:
    """
    Lazy loading for ML models - load on first use.
    
    Benefits:
    - Fast startup (<1s vs 30s)
    - Low memory footprint (1-2GB vs 15GB)
    - Load only models actually used
    - LRU cache eviction for memory management
    
    Usage:
        loader = LazyModelLoader('artifacts/models')
        model = loader.get_model('EUR_USD_15m_h4_ridge')
        predictions = model.predict(features)
    """
    
    def __init__(
        self,
        model_dir: str,
        max_cached_models: int = 10
    ):
        """
        Initialize lazy loader.
        
        Args:
            model_dir: Directory containing model files
            max_cached_models: Maximum models to keep in memory (LRU eviction)
        """
        self.model_dir = Path(model_dir)
        self.max_cached_models = max_cached_models
        
        # Scan for available models
        self.available_models = self._scan_models()
        
        # Loaded models cache (LRU)
        self.loaded_models: Dict[str, Any] = {}
        self.access_order: list = []  # For LRU tracking
        
        logger.info(
            f"LazyModelLoader initialized: {len(self.available_models)} models found, "
            f"max cached: {max_cached_models}"
        )
    
    def _scan_models(self) -> Dict[str, Path]:
        """Scan model directory for available models"""
        models = {}
        
        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")
            return models
        
        # Find all .pkl files
        for model_path in self.model_dir.rglob('*.pkl'):
            model_key = self._path_to_key(model_path)
            models[model_key] = model_path
        
        logger.info(f"Found {len(models)} models in {self.model_dir}")
        return models
    
    def _path_to_key(self, path: Path) -> str:
        """Convert path to model key"""
        # Remove model_dir prefix and .pkl suffix
        relative = path.relative_to(self.model_dir)
        key = str(relative).replace('\\', '_').replace('/', '_').replace('.pkl', '')
        return key
    
    def get_model(self, model_key: str) -> Optional[Any]:
        """
        Get model by key, loading if necessary.
        
        Args:
            model_key: Model identifier (e.g., 'EUR_USD_15m_h4_ridge')
            
        Returns:
            Loaded model or None if not found
        """
        # Check if already loaded
        if model_key in self.loaded_models:
            self._update_lru(model_key)
            logger.debug(f"Model cache hit: {model_key}")
            return self.loaded_models[model_key]
        
        # Check if model exists
        if model_key not in self.available_models:
            logger.error(f"Model not found: {model_key}")
            return None
        
        # Load model
        logger.info(f"Loading model: {model_key}")
        model = self._load_model(self.available_models[model_key])
        
        if model is not None:
            # Add to cache
            self._add_to_cache(model_key, model)
        
        return model
    
    def _load_model(self, path: Path) -> Optional[Any]:
        """Load model from disk"""
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.debug(f"Model loaded successfully: {path.name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}")
            return None
    
    def _add_to_cache(self, model_key: str, model: Any):
        """Add model to cache with LRU eviction"""
        # Evict if cache is full
        if len(self.loaded_models) >= self.max_cached_models:
            self._evict_lru()
        
        # Add to cache
        self.loaded_models[model_key] = model
        self.access_order.append(model_key)
        
        logger.debug(
            f"Model cached: {model_key} "
            f"(cache size: {len(self.loaded_models)}/{self.max_cached_models})"
        )
    
    def _update_lru(self, model_key: str):
        """Update LRU order on access"""
        if model_key in self.access_order:
            self.access_order.remove(model_key)
        self.access_order.append(model_key)
    
    def _evict_lru(self):
        """Evict least recently used model"""
        if not self.access_order:
            return
        
        # Remove oldest
        lru_key = self.access_order.pop(0)
        if lru_key in self.loaded_models:
            del self.loaded_models[lru_key]
            logger.debug(f"Evicted LRU model: {lru_key}")
    
    def preload_models(self, model_keys: list):
        """Preload specific models (optional optimization)"""
        logger.info(f"Preloading {len(model_keys)} models...")
        
        for key in model_keys:
            self.get_model(key)
    
    def clear_cache(self):
        """Clear all loaded models from memory"""
        count = len(self.loaded_models)
        self.loaded_models.clear()
        self.access_order.clear()
        logger.info(f"Cleared {count} models from cache")
    
    def get_statistics(self) -> Dict:
        """Get loader statistics"""
        return {
            'available_models': len(self.available_models),
            'loaded_models': len(self.loaded_models),
            'max_cached': self.max_cached_models,
            'cache_utilization': len(self.loaded_models) / self.max_cached_models
        }
    
    def list_available_models(self) -> list:
        """List all available model keys"""
        return list(self.available_models.keys())
    
    def is_loaded(self, model_key: str) -> bool:
        """Check if model is currently loaded"""
        return model_key in self.loaded_models
