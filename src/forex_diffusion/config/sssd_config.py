"""
SSSD Configuration Management

Provides dataclasses and loading utilities for SSSD model configuration.
Supports YAML-based configs with validation and merging.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from loguru import logger


@dataclass
class S4Config:
    """S4 layer configuration."""
    state_dim: int = 128
    n_layers: int = 4
    dropout: float = 0.1
    kernel_init: str = "hippo"
    ffn_expansion: int = 4


@dataclass
class EncoderConfig:
    """Multi-scale encoder configuration."""
    timeframes: List[str] = field(default_factory=lambda: ["5m", "15m", "1h", "4h"])
    feature_dim: int = 200
    context_dim: int = 512
    attention_heads: int = 8
    attention_dropout: float = 0.1


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""
    steps_train: int = 1000
    steps_inference: int = 20
    schedule: str = "cosine"
    schedule_offset: float = 0.008
    sampler_train: str = "ddpm"
    sampler_inference: str = "ddim"
    clip_min: float = 1e-12


@dataclass
class DiffusionHeadConfig:
    """Diffusion head network configuration."""
    latent_dim: int = 256
    timestep_emb_dim: int = 128
    conditioning_dim: int = 640
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1


@dataclass
class HorizonConfig:
    """Forecast horizon configuration."""
    minutes: List[int] = field(default_factory=lambda: [5, 15, 60, 240])
    weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    embedding_dim: int = 128
    consistency_weight: float = 0.1


@dataclass
class ModelConfig:
    """Complete SSSD model configuration."""
    name: str = "sssd_v1"
    model_type: str = "sssd_diffusion"
    asset: str = "EURUSD"

    s4: S4Config = field(default_factory=S4Config)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    head: DiffusionHeadConfig = field(default_factory=DiffusionHeadConfig)
    horizons: HorizonConfig = field(default_factory=HorizonConfig)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer: str = "AdamW"
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    scheduler: str = "cosine_annealing"
    lr_warmup_steps: int = 1000
    lr_min: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 15
    min_delta: float = 0.0001
    monitor: str = "val_loss"


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    save_every_n_epochs: int = 10
    keep_best_only: bool = False
    save_optimizer_state: bool = True


@dataclass
class MixedPrecisionConfig:
    """Mixed precision training configuration."""
    enabled: bool = True
    opt_level: str = "O1"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_every_n_steps: int = 10
    tensorboard: bool = True
    wandb: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 64
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class DataConfig:
    """Data configuration."""
    train_start: str = "2019-01-01"
    train_end: str = "2023-06-30"
    val_start: str = "2023-07-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-12-31"

    feature_pipeline: str = "unified_pipeline_v2"

    lookback_bars: Dict[str, int] = field(default_factory=lambda: {
        "5m": 500,
        "15m": 166,
        "1h": 41,
        "4h": 10
    })

    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "noise_injection": 0.01,
        "time_warping": False
    })

    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class InferenceConfig:
    """Inference configuration."""
    num_samples: int = 100
    sampler: str = "ddim"
    ddim_eta: float = 0.0

    confidence_threshold: float = 0.7
    uncertainty_mode: str = "std"

    cache_predictions: bool = True
    cache_ttl_seconds: int = 300

    batch_size: int = 1
    compile_model: bool = False


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "cuda"
    deterministic: bool = False
    seed: int = 42

    empty_cache_every_n_epochs: int = 5
    max_memory_allocated_gb: int = 8

    checkpoint_dir: str = "artifacts/sssd/checkpoints"
    log_dir: str = "artifacts/sssd/logs"
    tensorboard_dir: str = "artifacts/sssd/tensorboard"
    plot_dir: str = "artifacts/sssd/plots"

    database_url: str = "sqlite:///data/forex.db"


@dataclass
class HyperoptConfig:
    """Hyperparameter optimization configuration."""
    search_space: Dict[str, List[Any]] = field(default_factory=lambda: {
        "s4_state_dim": [64, 128, 256],
        "s4_n_layers": [2, 3, 4, 5],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "batch_size": [32, 64, 128],
        "diffusion_steps_inference": [10, 15, 20, 25]
    })

    method: str = "optuna"
    n_trials: int = 50
    timeout_hours: int = 120

    objective: str = "val_rmse"
    direction: str = "minimize"

    pruner: str = "median"
    pruner_warmup_steps: int = 10


@dataclass
class ProductionConfig:
    """Production deployment overrides."""
    diffusion_steps_inference: int = 15
    tensorboard: bool = False
    wandb: bool = False
    compile_model: bool = True
    cache_ttl_seconds: int = 60
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 1


@dataclass
class SSSDConfig:
    """
    Complete SSSD configuration.

    This is the top-level configuration object that contains all
    sub-configurations for model, training, data, inference, etc.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    hyperopt: HyperoptConfig = field(default_factory=HyperoptConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str | Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to {path}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> SSSDConfig:
        """Create config from dictionary."""
        # Recursively create nested dataclass instances

        # Model config
        model_dict = config_dict.get("model", {})
        model_config = ModelConfig(
            name=model_dict.get("name", "sssd_v1"),
            model_type=model_dict.get("model_type", "sssd_diffusion"),
            asset=model_dict.get("asset", "EURUSD"),
            s4=S4Config(**model_dict.get("s4", {})),
            encoder=EncoderConfig(**model_dict.get("encoder", {})),
            diffusion=DiffusionConfig(**model_dict.get("diffusion", {})),
            head=DiffusionHeadConfig(**model_dict.get("head", {})),
            horizons=HorizonConfig(**model_dict.get("horizons", {}))
        )

        # Training config
        train_dict = config_dict.get("training", {})
        training_config = TrainingConfig(
            epochs=train_dict.get("epochs", 100),
            batch_size=train_dict.get("batch_size", 64),
            gradient_clip_norm=train_dict.get("gradient_clip_norm", 1.0),
            gradient_accumulation_steps=train_dict.get("gradient_accumulation_steps", 1),
            optimizer=OptimizerConfig(**train_dict.get("optimizer", {})) if "optimizer" in train_dict else OptimizerConfig(),
            scheduler=SchedulerConfig(**train_dict.get("scheduler", {})) if "scheduler" in train_dict else SchedulerConfig(),
            early_stopping=EarlyStoppingConfig(**train_dict.get("early_stopping", {})) if "early_stopping" in train_dict else EarlyStoppingConfig(),
            checkpoint=CheckpointConfig(**train_dict.get("checkpoint", {})) if "checkpoint" in train_dict else CheckpointConfig(),
            mixed_precision=MixedPrecisionConfig(**train_dict.get("mixed_precision", {})) if "mixed_precision" in train_dict else MixedPrecisionConfig(),
            logging=LoggingConfig(**train_dict.get("logging", {})) if "logging" in train_dict else LoggingConfig()
        )

        # Data config
        data_config = DataConfig(**config_dict.get("data", {}))

        # Inference config
        inference_config = InferenceConfig(**config_dict.get("inference", {}))

        # System config
        system_config = SystemConfig(**config_dict.get("system", {}))

        # Hyperopt config
        hyperopt_config = HyperoptConfig(**config_dict.get("hyperopt", {}))

        # Production config
        production_config = ProductionConfig(**config_dict.get("production", {}))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            inference=inference_config,
            system=system_config,
            hyperopt=hyperopt_config,
            production=production_config
        )


def load_sssd_config(
    config_path: str | Path,
    asset_config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> SSSDConfig:
    """
    Load SSSD configuration from YAML file(s).

    Args:
        config_path: Path to main config file (e.g., default_config.yaml)
        asset_config_path: Optional path to asset-specific config (e.g., eurusd_config.yaml)
        overrides: Optional dict of config overrides

    Returns:
        SSSDConfig object

    Example:
        >>> config = load_sssd_config(
        ...     "configs/sssd/default_config.yaml",
        ...     "configs/sssd/eurusd_config.yaml",
        ...     overrides={"training.epochs": 50}
        ... )
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load main config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    logger.info(f"Loaded main config from {config_path}")

    # Merge with asset-specific config if provided
    if asset_config_path is not None:
        asset_config_path = Path(asset_config_path)
        if asset_config_path.exists():
            with open(asset_config_path, 'r') as f:
                asset_config_dict = yaml.safe_load(f)

            # Deep merge
            config_dict = _deep_merge(config_dict, asset_config_dict)
            logger.info(f"Merged asset config from {asset_config_path}")
        else:
            logger.warning(f"Asset config file not found: {asset_config_path}")

    # Apply overrides
    if overrides is not None:
        config_dict = _apply_overrides(config_dict, overrides)
        logger.info(f"Applied {len(overrides)} config overrides")

    # Create SSSDConfig object
    config = SSSDConfig.from_dict(config_dict)

    # Validate config
    _validate_config(config)

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _apply_overrides(config_dict: Dict, overrides: Dict[str, Any]) -> Dict:
    """
    Apply dot-notation overrides to config dict.

    Example:
        overrides = {"training.epochs": 50, "model.s4.state_dim": 256}
    """
    for key, value in overrides.items():
        keys = key.split('.')
        d = config_dict

        # Navigate to nested dict
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]

        # Set value
        d[keys[-1]] = value

    return config_dict


def _validate_config(config: SSSDConfig):
    """Validate configuration values."""
    # Validate horizon weights sum to 1.0
    weights_sum = sum(config.model.horizons.weights)
    if abs(weights_sum - 1.0) > 0.01:
        logger.warning(
            f"Horizon weights sum to {weights_sum:.3f}, not 1.0. "
            f"Normalizing weights."
        )
        total = sum(config.model.horizons.weights)
        config.model.horizons.weights = [w / total for w in config.model.horizons.weights]

    # Validate number of horizons matches number of weights
    if len(config.model.horizons.minutes) != len(config.model.horizons.weights):
        raise ValueError(
            f"Number of horizons ({len(config.model.horizons.minutes)}) "
            f"must match number of weights ({len(config.model.horizons.weights)})"
        )

    # Validate timeframes in encoder match lookback_bars
    encoder_timeframes = set(config.model.encoder.timeframes)
    lookback_timeframes = set(config.data.lookback_bars.keys())

    if encoder_timeframes != lookback_timeframes:
        logger.warning(
            f"Encoder timeframes {encoder_timeframes} do not match "
            f"lookback_bars timeframes {lookback_timeframes}"
        )

    # Validate device
    if config.system.device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    f"CUDA device specified but not available. "
                    f"Falling back to CPU."
                )
                config.system.device = "cpu"
        except ImportError:
            logger.warning("PyTorch not installed. Cannot validate CUDA availability.")

    logger.info("Config validation complete")
