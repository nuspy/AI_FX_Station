"""
Model Parameter Compatibility Matrix

Defines which parameters are supported by each training model.
"""

# Parameters supported by Lightning model (train.py)
LIGHTNING_SUPPORTED = {
    'symbol', 'timeframe', 'horizon', 'days_history', 'patch_len',
    'warmup_bars', 'epochs', 'batch_size', 'val_frac', 'num_workers',
    'artifacts_dir', 'indicator_tfs', 'min_feature_coverage',
    'atr_n', 'rsi_n', 'bb_n', 'hurst_window', 'rv_window',
    'returns_window', 'session_overlap', 'higher_tf', 'vp_bins',
    'seed', 'fast_dev_run', 'use_nvidia_opts', 'use_amp',
    'precision', 'compile_model', 'use_fused_optimizer',
    'use_flash_attention', 'gradient_accumulation_steps'
}

# Parameters supported by sklearn models (train_sklearn.py)
SKLEARN_SUPPORTED = {
    'symbol', 'timeframe', 'horizon', 'algo', 'encoder', 'latent_dim',
    'encoder_epochs', 'pca', 'artifacts_dir', 'warmup_bars', 'val_frac',
    'alpha', 'l1_ratio', 'random_state', 'n_estimators', 'days_history',
    'atr_n', 'rsi_n', 'bb_n', 'hurst_window', 'rv_window',
    'min_feature_coverage', 'indicator_tfs', 'use_relative_ohlc',
    'use_temporal_features', 'n_regimes', 'higher_tf', 'session_overlap',
    'returns_window', 'vp_window', 'vp_bins', 'use_vsa',
    'vsa_volume_ma', 'vsa_spread_ma'
}

# Parameter descriptions for user-friendly messages
PARAMETER_DESCRIPTIONS = {
    'vp_window': 'Volume Profile window size',
    'vp_bins': 'Volume Profile number of bins',
    'use_vsa': 'Volume Spread Analysis (VSA)',
    'vsa_volume_ma': 'VSA Volume MA period',
    'vsa_spread_ma': 'VSA Spread MA period',
    'encoder': 'Latent encoder type (VAE/AE)',
    'latent_dim': 'Latent space dimensions',
    'encoder_epochs': 'Encoder pre-training epochs',
    'pca': 'PCA dimensionality reduction',
    'algo': 'ML algorithm (ridge/lasso/xgboost/lgbm)',
    'alpha': 'Regularization alpha parameter',
    'l1_ratio': 'ElasticNet L1 ratio',
    'n_estimators': 'Number of estimators (trees)',
    'random_state': 'Random seed for reproducibility',
    'use_relative_ohlc': 'Use relative OHLC features',
    'use_temporal_features': 'Use temporal features',
    'n_regimes': 'Number of market regimes',
    'patch_len': 'Patch length for transformer',
    'use_nvidia_opts': 'NVIDIA GPU optimizations',
    'use_amp': 'Automatic Mixed Precision',
    'precision': 'Training precision (fp16/bf16/fp32)',
    'compile_model': 'torch.compile optimization',
    'use_fused_optimizer': 'Fused optimizer',
    'use_flash_attention': 'FlashAttention',
    'gradient_accumulation_steps': 'Gradient accumulation steps'
}


def get_unsupported_params(model_type: str, params: dict) -> list:
    """
    Check which parameters are not supported by the given model type.
    
    Args:
        model_type: "lightning" or "sklearn"
        params: Dictionary of parameter names
        
    Returns:
        List of tuples (param_name, description) for unsupported params
    """
    if model_type == "lightning":
        supported = LIGHTNING_SUPPORTED
    elif model_type == "sklearn":
        supported = SKLEARN_SUPPORTED
    else:
        return []
    
    unsupported = []
    for param_name in params.keys():
        if param_name not in supported:
            description = PARAMETER_DESCRIPTIONS.get(param_name, param_name)
            unsupported.append((param_name, description))
    
    return unsupported


def get_model_type_from_model_name(model_name: str) -> str:
    """
    Determine model type from model selection.
    
    Args:
        model_name: Model name from UI (e.g., "latents", "ridge", "xgboost")
        
    Returns:
        "lightning" or "sklearn"
    """
    lightning_models = {"latents"}
    sklearn_models = {"ridge", "lasso", "xgboost", "lgbm", "elasticnet"}
    
    if model_name.lower() in lightning_models:
        return "lightning"
    elif model_name.lower() in sklearn_models:
        return "sklearn"
    else:
        # Default to sklearn for unknown models
        return "sklearn"
