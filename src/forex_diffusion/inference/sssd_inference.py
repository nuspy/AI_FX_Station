"""
SSSD Inference Service

Real-time inference service for SSSD model with:
- Load model from checkpoint
- Predict with uncertainty quantification
- Caching (Redis, 5-min TTL)
- Integration with real-time pipeline
"""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from dataclasses import dataclass
import time
import json

from ..models.sssd import SSSDModel
from ..config.sssd_config import load_sssd_config, SSSDConfig
from ..features.unified_pipeline import unified_feature_pipeline, FeatureConfig, Standardizer


@dataclass
class SSSDPrediction:
    """SSSD prediction result with uncertainty."""

    asset: str
    timestamp: pd.Timestamp
    horizons: List[int]  # Minutes

    # Predictions for each horizon
    mean: Dict[int, float]  # Horizon -> mean prediction
    std: Dict[int, float]   # Horizon -> standard deviation
    q05: Dict[int, float]   # Horizon -> 5th percentile
    q50: Dict[int, float]   # Horizon -> median
    q95: Dict[int, float]   # Horizon -> 95th percentile

    # Metadata
    inference_time_ms: float
    model_name: str
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asset": self.asset,
            "timestamp": self.timestamp.isoformat(),
            "horizons": self.horizons,
            "predictions": {
                str(h): {
                    "mean": float(self.mean[h]),
                    "std": float(self.std[h]),
                    "q05": float(self.q05[h]),
                    "q50": float(self.q50[h]),
                    "q95": float(self.q95[h])
                }
                for h in self.horizons
            },
            "metadata": {
                "inference_time_ms": self.inference_time_ms,
                "model_name": self.model_name,
                "num_samples": self.num_samples
            }
        }


class SSSDInferenceService:
    """Inference service for SSSD model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: Optional[str | Path] = None,
        device: str = "cuda",
        compile_model: bool = False
    ):
        """
        Initialize inference service.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config (if None, load from checkpoint)
            device: Device to run inference on
            compile_model: Whether to compile model with torch.compile (faster)
        """
        checkpoint_path = Path(checkpoint_path)

        # Load configuration
        if config_path is not None:
            self.config = load_sssd_config(config_path)
        else:
            # Load config from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            from ..config.sssd_config import SSSDConfig
            self.config = SSSDConfig.from_dict(checkpoint["config"])

        # Override device
        self.config.system.device = device
        self.device = torch.device(device)

        # Load model
        logger.info(f"Loading SSSD model from {checkpoint_path}")
        self.model, self.epoch, _, self.metrics = SSSDModel.load_checkpoint(
            checkpoint_path,
            config=self.config,
            map_location=device
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Compile model for faster inference
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")

        # Load standardizer
        standardizer_path = checkpoint_path.parent / "standardizer.json"
        if standardizer_path.exists():
            with open(standardizer_path, "r") as f:
                std_dict = json.load(f)

            self.standardizer = Standardizer(
                cols=std_dict["cols"],
                mu=std_dict["mu"],
                sigma=std_dict["sigma"]
            )
            logger.info(f"Loaded standardizer with {len(self.standardizer.cols)} features")
        else:
            logger.warning("Standardizer not found, predictions may be unreliable")
            self.standardizer = None

        # Load feature config
        feature_config_path = checkpoint_path.parent / "feature_config.json"
        if feature_config_path.exists():
            from ..features.unified_pipeline import load_feature_config
            self.feature_config = load_feature_config(feature_config_path)
        else:
            # Use default config
            self.feature_config = FeatureConfig()

        # Caching (simple in-memory cache)
        self.cache = {}
        self.cache_ttl_seconds = self.config.inference.cache_ttl_seconds

        logger.info(
            f"SSSD inference service initialized: "
            f"asset={self.config.model.asset}, "
            f"horizons={self.config.model.horizons.minutes}, "
            f"device={self.device}"
        )

    @torch.no_grad()
    def predict(
        self,
        df: pd.DataFrame,
        num_samples: Optional[int] = None,
        sampler: str = "ddim",
        use_cache: bool = True
    ) -> SSSDPrediction:
        """
        Make prediction on recent OHLC data.

        Args:
            df: Recent OHLC data (at least lookback_bars rows)
            num_samples: Number of diffusion samples for uncertainty (default from config)
            sampler: Sampling algorithm ("ddim" or "ddpm")
            use_cache: Whether to use cache

        Returns:
            SSSDPrediction with mean, std, and quantiles for each horizon
        """
        start_time = time.time()

        if num_samples is None:
            num_samples = self.config.inference.num_samples

        # Check cache
        cache_key = self._get_cache_key(df)
        if use_cache and cache_key in self.cache:
            cached_pred, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl_seconds:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_pred

        # Compute features
        features_df, _, _ = unified_feature_pipeline(
            df,
            config=self.feature_config,
            timeframe=self.config.model.encoder.timeframes[0],
            standardizer=self.standardizer,
            fit_standardizer=False,
            output_format="multi_timeframe"
        )

        # Convert to tensors
        features_dict = {}
        for tf, tf_df in features_df.items():
            # Take last row (most recent)
            feature_values = tf_df.iloc[-1:][
                [col for col in tf_df.columns if col != "timestamp"]
            ].values

            # Convert to tensor (batch_size=1, seq_len=1, feature_dim)
            features_dict[tf] = torch.tensor(
                feature_values,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

        # Get horizons
        horizons = list(range(len(self.config.model.horizons.minutes)))

        # Inference
        predictions = self.model.inference_forward(
            features=features_dict,
            horizons=horizons,
            num_samples=num_samples,
            sampler=sampler,
            num_steps=self.config.model.diffusion.steps_inference
        )

        # Extract predictions for each horizon
        mean_dict = {}
        std_dict = {}
        q05_dict = {}
        q50_dict = {}
        q95_dict = {}

        for h_idx, h_minutes in enumerate(self.config.model.horizons.minutes):
            h_pred = predictions[h_idx]

            mean_dict[h_minutes] = h_pred["mean"].item()
            std_dict[h_minutes] = h_pred["std"].item()
            q05_dict[h_minutes] = h_pred["q05"].item()
            q50_dict[h_minutes] = h_pred["q50"].item()
            q95_dict[h_minutes] = h_pred["q95"].item()

        # Create prediction object
        inference_time_ms = (time.time() - start_time) * 1000

        prediction = SSSDPrediction(
            asset=self.config.model.asset,
            timestamp=pd.Timestamp.now(tz="UTC"),
            horizons=self.config.model.horizons.minutes,
            mean=mean_dict,
            std=std_dict,
            q05=q05_dict,
            q50=q50_dict,
            q95=q95_dict,
            inference_time_ms=inference_time_ms,
            model_name=self.config.model.name,
            num_samples=num_samples
        )

        # Cache result
        if use_cache:
            self.cache[cache_key] = (prediction, time.time())

        return prediction

    def predict_batch(
        self,
        dfs: List[pd.DataFrame],
        num_samples: Optional[int] = None,
        sampler: str = "ddim"
    ) -> List[SSSDPrediction]:
        """
        Batch prediction for multiple dataframes.

        Args:
            dfs: List of OHLC dataframes
            num_samples: Number of diffusion samples
            sampler: Sampling algorithm

        Returns:
            List of predictions
        """
        predictions = []
        for df in dfs:
            pred = self.predict(df, num_samples=num_samples, sampler=sampler, use_cache=False)
            predictions.append(pred)

        return predictions

    def get_directional_confidence(self, prediction: SSSDPrediction, horizon: int) -> float:
        """
        Get directional confidence for a specific horizon.

        Confidence is based on how far the mean is from zero relative to std.

        Args:
            prediction: SSSDPrediction object
            horizon: Forecast horizon in minutes

        Returns:
            Confidence score in [0, 1]
        """
        mean = prediction.mean[horizon]
        std = prediction.std[horizon]

        # Confidence = |mean| / (|mean| + std)
        # If mean is far from zero and std is small, confidence is high
        confidence = abs(mean) / (abs(mean) + std + 1e-8)

        return float(np.clip(confidence, 0.0, 1.0))

    def get_direction(self, prediction: SSSDPrediction, horizon: int) -> int:
        """
        Get predicted direction for a specific horizon.

        Args:
            prediction: SSSDPrediction object
            horizon: Forecast horizon in minutes

        Returns:
            1 for bullish, -1 for bearish, 0 for neutral
        """
        mean = prediction.mean[horizon]
        confidence = self.get_directional_confidence(prediction, horizon)

        threshold = self.config.inference.confidence_threshold

        if confidence < threshold:
            return 0  # Neutral (low confidence)
        elif mean > 0:
            return 1  # Bullish
        else:
            return -1  # Bearish

    def _get_cache_key(self, df: pd.DataFrame) -> str:
        """Generate cache key from dataframe."""
        # Use last timestamp and close price as key
        last_ts = df.iloc[-1]["ts_utc"]
        last_close = df.iloc[-1]["close"]
        return f"{self.config.model.asset}_{last_ts}_{last_close}"

    def clear_cache(self):
        """Clear prediction cache."""
        self.cache = {}
        logger.info("Prediction cache cleared")


class SSSDEnsembleInferenceService:
    """
    Inference service for ensemble of multiple SSSD models.

    Useful for multi-asset or multi-configuration ensembling.
    """

    def __init__(self, services: List[SSSDInferenceService], weights: Optional[List[float]] = None):
        """
        Initialize ensemble service.

        Args:
            services: List of SSSDInferenceService instances
            weights: Optional weights for each service (default: equal weights)
        """
        self.services = services

        if weights is None:
            weights = [1.0 / len(services)] * len(services)

        self.weights = np.array(weights)
        self.weights /= self.weights.sum()  # Normalize

        logger.info(f"Initialized SSSD ensemble with {len(services)} models")

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, num_samples: Optional[int] = None) -> SSSDPrediction:
        """
        Make ensemble prediction.

        Args:
            df: OHLC dataframe
            num_samples: Number of samples per model

        Returns:
            Weighted ensemble prediction
        """
        # Get predictions from all models
        predictions = [svc.predict(df, num_samples=num_samples) for svc in self.services]

        # Weighted average of means
        horizons = predictions[0].horizons
        mean_dict = {}
        std_dict = {}
        q05_dict = {}
        q50_dict = {}
        q95_dict = {}

        for h in horizons:
            means = np.array([pred.mean[h] for pred in predictions])
            stds = np.array([pred.std[h] for pred in predictions])
            q05s = np.array([pred.q05[h] for pred in predictions])
            q50s = np.array([pred.q50[h] for pred in predictions])
            q95s = np.array([pred.q95[h] for pred in predictions])

            mean_dict[h] = float(np.sum(means * self.weights))
            std_dict[h] = float(np.sqrt(np.sum((stds ** 2) * self.weights)))  # Pooled std
            q05_dict[h] = float(np.sum(q05s * self.weights))
            q50_dict[h] = float(np.sum(q50s * self.weights))
            q95_dict[h] = float(np.sum(q95s * self.weights))

        # Create ensemble prediction
        ensemble_pred = SSSDPrediction(
            asset=predictions[0].asset,
            timestamp=pd.Timestamp.now(tz="UTC"),
            horizons=horizons,
            mean=mean_dict,
            std=std_dict,
            q05=q05_dict,
            q50=q50_dict,
            q95=q95_dict,
            inference_time_ms=sum(p.inference_time_ms for p in predictions),
            model_name=f"ensemble_{len(self.services)}",
            num_samples=predictions[0].num_samples * len(self.services)
        )

        return ensemble_pred


def load_sssd_inference_service(
    asset: str,
    checkpoint_dir: str | Path = "artifacts/sssd/checkpoints",
    device: str = "cuda"
) -> SSSDInferenceService:
    """
    Load SSSD inference service for a specific asset.

    Args:
        asset: Asset symbol (e.g., "EURUSD")
        checkpoint_dir: Directory containing checkpoints
        device: Device to run inference on

    Returns:
        SSSDInferenceService instance
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Find best model checkpoint for asset
    best_checkpoint = checkpoint_dir / asset / "best_model.pt"

    if not best_checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found for {asset} at {best_checkpoint}")

    # Load service
    service = SSSDInferenceService(
        checkpoint_path=best_checkpoint,
        device=device,
        compile_model=True
    )

    return service
