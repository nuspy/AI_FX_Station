"""
Configuration grid generation and hashing utilities.

Provides functions for generating Cartesian product of training configurations,
computing configuration hashes for deduplication, and validating configurations.
"""

import hashlib
import json
from typing import Dict, Any, List, Optional
from itertools import product
import logging

logger = logging.getLogger(__name__)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute SHA256 hash of a training configuration for deduplication.

    The hash is computed from a canonical JSON representation to ensure
    identical configurations produce identical hashes.

    Args:
        config: Training configuration dictionary

    Returns:
        SHA256 hash as hexadecimal string

    Example:
        >>> config = {"model_type": "random_forest", "symbol": "EURUSD"}
        >>> hash_val = compute_config_hash(config)
        >>> len(hash_val)
        64
    """
    # Extract only hashable fields (exclude metadata like UUIDs, timestamps)
    hashable_fields = [
        'model_type', 'encoder', 'symbol', 'base_timeframe',
        'days_history', 'horizon', 'indicator_tfs', 'additional_features',
        'preprocessing_params', 'model_hyperparams'
    ]

    hashable_config = {k: v for k, v in config.items() if k in hashable_fields}

    # Sort keys for canonical representation
    canonical_json = json.dumps(hashable_config, sort_keys=True, default=str)

    # Compute SHA256
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def validate_config(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate a training configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if config is valid
        - error_message: Error description if invalid, None if valid

    Example:
        >>> config = {"model_type": "random_forest", "symbol": "EURUSD"}
        >>> is_valid, error = validate_config(config)
        >>> is_valid
        True
    """
    # Required fields
    required_fields = [
        'model_type', 'encoder', 'symbol', 'base_timeframe',
        'days_history', 'horizon'
    ]

    # Check required fields
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"

    # Validate model_type
    valid_model_types = [
        'diffusion', 'random_forest', 'gradient_boosting',
        'linear_regression', 'ridge', 'lasso', 'elasticnet',
        'xgboost', 'lightgbm', 'lstm', 'transformer'
    ]
    if config['model_type'] not in valid_model_types:
        return False, f"Invalid model_type: {config['model_type']}. Must be one of {valid_model_types}"

    # Validate encoder
    valid_encoders = ['none', 'vae', 'autoencoder']
    if config['encoder'] not in valid_encoders:
        return False, f"Invalid encoder: {config['encoder']}. Must be one of {valid_encoders}"

    # Validate base_timeframe
    valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
    if config['base_timeframe'] not in valid_timeframes:
        return False, f"Invalid base_timeframe: {config['base_timeframe']}. Must be one of {valid_timeframes}"

    # Validate numeric ranges
    if config['days_history'] <= 0:
        return False, "days_history must be positive"

    if config['horizon'] <= 0:
        return False, "horizon must be positive"

    # Validate JSON fields if present
    if 'indicator_tfs' in config and config['indicator_tfs'] is not None:
        if not isinstance(config['indicator_tfs'], list):
            return False, "indicator_tfs must be a list"

    if 'additional_features' in config and config['additional_features'] is not None:
        if not isinstance(config['additional_features'], list):
            return False, "additional_features must be a list"

    if 'preprocessing_params' in config and config['preprocessing_params'] is not None:
        if not isinstance(config['preprocessing_params'], dict):
            return False, "preprocessing_params must be a dictionary"

    if 'model_hyperparams' in config and config['model_hyperparams'] is not None:
        if not isinstance(config['model_hyperparams'], dict):
            return False, "model_hyperparams must be a dictionary"

    return True, None


def generate_config_grid(grid_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate Cartesian product of configuration parameters.

    Creates all possible combinations of provided parameter lists.

    Args:
        grid_params: Dictionary mapping parameter names to lists of values
            Example:
            {
                'model_type': ['random_forest', 'gradient_boosting'],
                'symbol': ['EURUSD', 'GBPUSD'],
                'encoder': ['none'],
                'base_timeframe': ['H1'],
                'days_history': [30, 60],
                'horizon': [24]
            }

    Returns:
        List of configuration dictionaries, one for each combination

    Example:
        >>> grid_params = {
        ...     'model_type': ['random_forest', 'xgboost'],
        ...     'symbol': ['EURUSD'],
        ...     'encoder': ['none'],
        ...     'base_timeframe': ['H1'],
        ...     'days_history': [30],
        ...     'horizon': [24]
        ... }
        >>> configs = generate_config_grid(grid_params)
        >>> len(configs)
        2
        >>> configs[0]['model_type']
        'random_forest'
    """
    if not grid_params:
        return []

    # Ensure all required parameters are present
    required_params = ['model_type', 'encoder', 'symbol', 'base_timeframe', 'days_history', 'horizon']
    for param in required_params:
        if param not in grid_params:
            raise ValueError(f"Missing required parameter in grid: {param}")

    # Get parameter names and their value lists
    param_names = list(grid_params.keys())
    param_values = [grid_params[name] for name in param_names]

    # Generate Cartesian product
    configs = []
    for combination in product(*param_values):
        config = dict(zip(param_names, combination))
        configs.append(config)

    logger.info(f"Generated {len(configs)} configurations from grid")

    return configs


def generate_config_grid_with_validation(
    grid_params: Dict[str, List[Any]],
    validate: bool = True
) -> tuple[List[Dict[str, Any]], List[tuple[Dict[str, Any], str]]]:
    """
    Generate configuration grid with optional validation.

    Args:
        grid_params: Dictionary mapping parameter names to lists of values
        validate: If True, validate each configuration

    Returns:
        Tuple of (valid_configs, invalid_configs_with_errors)
        - valid_configs: List of valid configuration dictionaries
        - invalid_configs_with_errors: List of (config, error_message) tuples

    Example:
        >>> grid_params = {
        ...     'model_type': ['random_forest', 'invalid_model'],
        ...     'symbol': ['EURUSD'],
        ...     'encoder': ['none'],
        ...     'base_timeframe': ['H1'],
        ...     'days_history': [30],
        ...     'horizon': [24]
        ... }
        >>> valid, invalid = generate_config_grid_with_validation(grid_params)
        >>> len(valid)
        1
        >>> len(invalid)
        1
    """
    # Generate all combinations
    all_configs = generate_config_grid(grid_params)

    if not validate:
        return all_configs, []

    # Validate each configuration
    valid_configs = []
    invalid_configs = []

    for config in all_configs:
        is_valid, error_msg = validate_config(config)
        if is_valid:
            valid_configs.append(config)
        else:
            invalid_configs.append((config, error_msg))

    if invalid_configs:
        logger.warning(f"Found {len(invalid_configs)} invalid configurations")
        for config, error in invalid_configs:
            logger.warning(f"  Invalid: {config.get('model_type', 'unknown')} - {error}")

    logger.info(f"Generated {len(valid_configs)} valid configurations")

    return valid_configs, invalid_configs


def add_config_hashes(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add config_hash field to each configuration.

    Args:
        configs: List of configuration dictionaries

    Returns:
        Same list with config_hash field added to each config

    Example:
        >>> configs = [{'model_type': 'random_forest', 'symbol': 'EURUSD'}]
        >>> configs_with_hashes = add_config_hashes(configs)
        >>> 'config_hash' in configs_with_hashes[0]
        True
    """
    for config in configs:
        if 'config_hash' not in config:
            config['config_hash'] = compute_config_hash(config)

    return configs


def deduplicate_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate configurations based on config hash.

    Args:
        configs: List of configuration dictionaries (must have config_hash field)

    Returns:
        List of unique configurations

    Example:
        >>> configs = [
        ...     {'model_type': 'random_forest', 'config_hash': 'abc123'},
        ...     {'model_type': 'random_forest', 'config_hash': 'abc123'},
        ...     {'model_type': 'xgboost', 'config_hash': 'def456'}
        ... ]
        >>> unique = deduplicate_configs(configs)
        >>> len(unique)
        2
    """
    seen_hashes = set()
    unique_configs = []

    for config in configs:
        config_hash = config.get('config_hash') or compute_config_hash(config)
        if config_hash not in seen_hashes:
            seen_hashes.add(config_hash)
            unique_configs.append(config)

    removed_count = len(configs) - len(unique_configs)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate configurations")

    return unique_configs


def filter_already_trained(
    configs: List[Dict[str, Any]],
    trained_hashes: set[str]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter out configurations that have already been trained.

    Args:
        configs: List of configuration dictionaries (must have config_hash field)
        trained_hashes: Set of config hashes that have been trained

    Returns:
        Tuple of (untrained_configs, already_trained_configs)

    Example:
        >>> configs = [
        ...     {'model_type': 'random_forest', 'config_hash': 'abc123'},
        ...     {'model_type': 'xgboost', 'config_hash': 'def456'}
        ... ]
        >>> trained = {'abc123'}
        >>> untrained, already_trained = filter_already_trained(configs, trained)
        >>> len(untrained)
        1
        >>> len(already_trained)
        1
    """
    untrained = []
    already_trained = []

    for config in configs:
        config_hash = config.get('config_hash') or compute_config_hash(config)
        if config_hash in trained_hashes:
            already_trained.append(config)
        else:
            untrained.append(config)

    if already_trained:
        logger.info(f"Filtered out {len(already_trained)} already-trained configurations")

    return untrained, already_trained


def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of a configuration.

    Args:
        config: Configuration dictionary

    Returns:
        String summary

    Example:
        >>> config = {
        ...     'model_type': 'random_forest',
        ...     'symbol': 'EURUSD',
        ...     'base_timeframe': 'H1',
        ...     'encoder': 'none',
        ...     'days_history': 30,
        ...     'horizon': 24
        ... }
        >>> summary = get_config_summary(config)
        >>> 'EURUSD' in summary
        True
    """
    parts = [
        config.get('model_type', 'unknown'),
        config.get('symbol', 'unknown'),
        config.get('base_timeframe', 'unknown'),
        f"hist={config.get('days_history', 0)}d",
        f"hz={config.get('horizon', 0)}",
    ]

    if config.get('encoder') and config['encoder'] != 'none':
        parts.insert(1, f"enc={config['encoder']}")

    return ' | '.join(parts)


def estimate_training_time(
    config: Dict[str, Any],
    base_time_seconds: float = 300.0
) -> float:
    """
    Estimate training time for a configuration (rough estimate).

    Args:
        config: Configuration dictionary
        base_time_seconds: Base training time in seconds

    Returns:
        Estimated training time in seconds

    Note:
        This is a rough estimate. Actual time depends on hardware,
        data size, and model complexity.
    """
    time_multipliers = {
        # Model type multipliers
        'linear_regression': 0.1,
        'ridge': 0.1,
        'lasso': 0.1,
        'elasticnet': 0.1,
        'random_forest': 1.0,
        'gradient_boosting': 1.5,
        'xgboost': 1.2,
        'lightgbm': 1.0,
        'lstm': 3.0,
        'transformer': 4.0,
        'diffusion': 10.0,
    }

    model_type = config.get('model_type', 'random_forest')
    multiplier = time_multipliers.get(model_type, 1.0)

    # Encoder adds overhead
    if config.get('encoder') and config['encoder'] != 'none':
        multiplier *= 1.5

    # More history = more training time
    days_history = config.get('days_history', 30)
    if days_history > 60:
        multiplier *= 1.3
    elif days_history > 90:
        multiplier *= 1.5

    return base_time_seconds * multiplier


def estimate_grid_time(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Estimate total time to train all configurations in grid.

    Args:
        configs: List of configuration dictionaries

    Returns:
        Dictionary with time estimates

    Example:
        >>> configs = [
        ...     {'model_type': 'random_forest', 'days_history': 30},
        ...     {'model_type': 'xgboost', 'days_history': 30}
        ... ]
        >>> estimates = estimate_grid_time(configs)
        >>> estimates['total_configs']
        2
    """
    total_seconds = sum(estimate_training_time(config) for config in configs)
    total_hours = total_seconds / 3600
    total_days = total_hours / 24

    return {
        'total_configs': len(configs),
        'estimated_seconds': total_seconds,
        'estimated_hours': total_hours,
        'estimated_days': total_days,
        'average_seconds_per_config': total_seconds / len(configs) if configs else 0
    }
