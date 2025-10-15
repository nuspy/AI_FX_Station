"""
Configuration Loader for Training Pipeline

Loads and validates YAML configuration files for the two-phase training system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class ConfigLoader:
    """
    Configuration loader for training pipeline.

    Loads YAML configuration and provides typed access to settings.
    Validates configuration on load and provides sensible defaults.
    """

    DEFAULT_CONFIG_PATH = Path("configs/training_pipeline/default_config.yaml")

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to configuration file (default: configs/training_pipeline/default_config.yaml)
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                self.config = self._get_default_config()
                return

            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            logger.info(f"Loaded configuration from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            self.config = self._get_default_config()

    def _validate_config(self):
        """Validate configuration values."""
        # Ensure all required top-level keys exist
        required_keys = [
            'storage', 'queue', 'recovery', 'model_management',
            'performance', 'inference_grid', 'training_defaults',
            'regime_detection', 'metrics'
        ]

        for key in required_keys:
            if key not in self.config:
                logger.warning(f"Missing config section: {key}, using defaults")
                self.config[key] = self._get_default_section(key)

        # Validate numeric ranges
        self._validate_numeric_ranges()

        logger.info("Configuration validated successfully")

    def _validate_numeric_ranges(self):
        """Validate numeric configuration values are in valid ranges."""
        # Queue settings
        queue = self.config.get('queue', {})
        if queue.get('max_parallel_queues', 1) < 1:
            queue['max_parallel_queues'] = 1
        if queue.get('auto_checkpoint_interval', 10) < 1:
            queue['auto_checkpoint_interval'] = 10

        # Model management
        model_mgmt = self.config.get('model_management', {})
        if model_mgmt.get('keep_top_n_per_regime', 1) < 1:
            model_mgmt['keep_top_n_per_regime'] = 1
        if not 0 <= model_mgmt.get('min_improvement_threshold', 0.01) <= 1:
            model_mgmt['min_improvement_threshold'] = 0.01

        # Performance
        perf = self.config.get('performance', {})
        if perf.get('database_connection_pool_size', 5) < 1:
            perf['database_connection_pool_size'] = 5

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found."""
        return {
            'storage': {
                'artifacts_dir': './artifacts',
                'checkpoints_dir': './checkpoints/training_pipeline',
                'max_checkpoint_age_days': 30,
                'compress_old_models': True
            },
            'queue': {
                'max_parallel_queues': 1,
                'auto_checkpoint_interval': 10,
                'max_inference_workers': 4
            },
            'recovery': {
                'auto_resume_on_crash': False,
                'detect_crash_after_minutes': 5
            },
            'model_management': {
                'delete_non_best_models': True,
                'keep_top_n_per_regime': 1,
                'min_improvement_threshold': 0.01
            },
            'performance': {
                'model_cache_size_mb': 1000,
                'database_connection_pool_size': 5,
                'batch_insert_size': 100
            },
            'inference_grid': {
                'prediction_methods': ['direct', 'recursive', 'direct_multi'],
                'ensemble_methods': ['mean', 'weighted', 'stacking'],
                'confidence_thresholds': [0.0, 0.3, 0.5, 0.7, 0.9],
                'lookback_windows': [50, 100, 200]
            },
            'training_defaults': {
                'validation_split': 0.2,
                'test_split': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 1
            },
            'regime_detection': {
                'trend_window': 50,
                'volatility_window': 20,
                'returns_window': 10,
                'min_regime_duration': 10
            },
            'metrics': {
                'primary_metric': 'sharpe_ratio',
                'secondary_metrics': [
                    'max_drawdown', 'win_rate', 'profit_factor',
                    'sortino_ratio', 'calmar_ratio'
                ],
                'thresholds': {
                    'min_sharpe_ratio': 0.5,
                    'max_drawdown': -0.3,
                    'min_win_rate': 0.45,
                    'min_profit_factor': 1.2
                }
            }
        }

    def _get_default_section(self, section: str) -> Dict[str, Any]:
        """Get default values for a config section."""
        defaults = self._get_default_config()
        return defaults.get(section, {})

    # Property accessors for typed access

    @property
    def artifacts_dir(self) -> str:
        """Get artifacts directory path."""
        return self.config.get('storage', {}).get('artifacts_dir', './artifacts')

    @property
    def checkpoints_dir(self) -> str:
        """Get checkpoints directory path."""
        return self.config.get('storage', {}).get('checkpoints_dir', './checkpoints/training_pipeline')

    @property
    def max_checkpoint_age_days(self) -> int:
        """Get max checkpoint age in days."""
        return self.config.get('storage', {}).get('max_checkpoint_age_days', 30)

    @property
    def compress_old_models(self) -> bool:
        """Check if old models should be compressed."""
        return self.config.get('storage', {}).get('compress_old_models', True)

    @property
    def max_parallel_queues(self) -> int:
        """Get max parallel training queues."""
        return self.config.get('queue', {}).get('max_parallel_queues', 1)

    @property
    def auto_checkpoint_interval(self) -> int:
        """Get auto-checkpoint interval (models)."""
        return self.config.get('queue', {}).get('auto_checkpoint_interval', 10)

    @property
    def max_inference_workers(self) -> int:
        """Get max inference worker threads."""
        return self.config.get('queue', {}).get('max_inference_workers', 4)

    @property
    def auto_resume_on_crash(self) -> bool:
        """Check if auto-resume on crash is enabled."""
        return self.config.get('recovery', {}).get('auto_resume_on_crash', False)

    @property
    def detect_crash_after_minutes(self) -> int:
        """Get crash detection timeout in minutes."""
        return self.config.get('recovery', {}).get('detect_crash_after_minutes', 5)

    @property
    def delete_non_best_models(self) -> bool:
        """Check if non-best models should be deleted."""
        return self.config.get('model_management', {}).get('delete_non_best_models', True)

    @property
    def keep_top_n_per_regime(self) -> int:
        """Get number of top models to keep per regime."""
        return self.config.get('model_management', {}).get('keep_top_n_per_regime', 1)

    @property
    def min_improvement_threshold(self) -> float:
        """Get minimum improvement threshold for model replacement."""
        return self.config.get('model_management', {}).get('min_improvement_threshold', 0.01)

    @property
    def model_cache_size_mb(self) -> int:
        """Get model cache size in MB."""
        return self.config.get('performance', {}).get('model_cache_size_mb', 1000)

    @property
    def database_connection_pool_size(self) -> int:
        """Get database connection pool size."""
        return self.config.get('performance', {}).get('database_connection_pool_size', 5)

    @property
    def batch_insert_size(self) -> int:
        """Get batch insert size for database operations."""
        return self.config.get('performance', {}).get('batch_insert_size', 100)

    @property
    def prediction_methods(self) -> List[str]:
        """Get list of prediction methods for inference grid."""
        return self.config.get('inference_grid', {}).get('prediction_methods', ['direct'])

    @property
    def ensemble_methods(self) -> List[str]:
        """Get list of ensemble methods for inference grid."""
        return self.config.get('inference_grid', {}).get('ensemble_methods', ['mean'])

    @property
    def confidence_thresholds(self) -> List[float]:
        """Get confidence thresholds for inference grid."""
        return self.config.get('inference_grid', {}).get('confidence_thresholds', [0.0])

    @property
    def lookback_windows(self) -> List[int]:
        """Get lookback windows for inference grid."""
        return self.config.get('inference_grid', {}).get('lookback_windows', [100])

    @property
    def validation_split(self) -> float:
        """Get validation split ratio."""
        return self.config.get('training_defaults', {}).get('validation_split', 0.2)

    @property
    def test_split(self) -> float:
        """Get test split ratio."""
        return self.config.get('training_defaults', {}).get('test_split', 0.1)

    @property
    def random_state(self) -> int:
        """Get random state for reproducibility."""
        return self.config.get('training_defaults', {}).get('random_state', 42)

    @property
    def n_jobs(self) -> int:
        """Get number of parallel jobs (-1 = all cores)."""
        return self.config.get('training_defaults', {}).get('n_jobs', -1)

    @property
    def verbose(self) -> int:
        """Get verbosity level."""
        return self.config.get('training_defaults', {}).get('verbose', 1)

    @property
    def trend_window(self) -> int:
        """Get trend calculation window."""
        return self.config.get('regime_detection', {}).get('trend_window', 50)

    @property
    def volatility_window(self) -> int:
        """Get volatility calculation window."""
        return self.config.get('regime_detection', {}).get('volatility_window', 20)

    @property
    def returns_window(self) -> int:
        """Get returns calculation window."""
        return self.config.get('regime_detection', {}).get('returns_window', 10)

    @property
    def min_regime_duration(self) -> int:
        """Get minimum regime duration in bars."""
        return self.config.get('regime_detection', {}).get('min_regime_duration', 10)

    @property
    def primary_metric(self) -> str:
        """Get primary performance metric."""
        return self.config.get('metrics', {}).get('primary_metric', 'sharpe_ratio')

    @property
    def secondary_metrics(self) -> List[str]:
        """Get list of secondary performance metrics."""
        return self.config.get('metrics', {}).get('secondary_metrics', [
            'max_drawdown', 'win_rate', 'profit_factor'
        ])

    @property
    def metric_thresholds(self) -> Dict[str, float]:
        """Get performance metric thresholds."""
        return self.config.get('metrics', {}).get('thresholds', {})

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., 'storage.artifacts_dir')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def reload(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self._load_config()
        self._validate_config()


# Global config instance (singleton pattern)
_global_config: Optional[ConfigLoader] = None


def get_config(config_path: Optional[Path] = None) -> ConfigLoader:
    """
    Get global configuration instance.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigLoader(config_path)

    return _global_config


def reload_config():
    """Reload global configuration from file."""
    global _global_config

    if _global_config is not None:
        _global_config.reload()
