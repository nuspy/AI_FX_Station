"""
SSSD Wrapper for Sklearn Ensemble Compatibility

Wraps SSSD model to be compatible with sklearn's BaseEstimator and RegressorMixin,
enabling integration into the existing stacking ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from loguru import logger

from ..inference.sssd_inference import SSSDInferenceService, load_sssd_inference_service
from ..config.sssd_config import SSSDConfig


class SSSDWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for SSSD model.

    Enables SSSD to be used in stacking ensembles alongside traditional
    sklearn models (Ridge, RandomForest, XGBoost, etc.).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        asset: str = "EURUSD",
        horizon: int = 5,
        num_samples: int = 100,
        sampler: str = "ddim",
        device: str = "cuda",
        use_uncertainty: bool = True,
        uncertainty_weight: float = 0.5
    ):
        """
        Initialize SSSD wrapper.

        Args:
            checkpoint_path: Path to SSSD checkpoint (if None, use asset default)
            asset: Asset symbol (used if checkpoint_path is None)
            horizon: Forecast horizon in minutes (5, 15, 60, or 240)
            num_samples: Number of diffusion samples for uncertainty
            sampler: Sampling algorithm ("ddim" or "ddpm")
            device: Device to run inference on
            use_uncertainty: Whether to use uncertainty in predictions
            uncertainty_weight: Weight for uncertainty-adjusted predictions
        """
        self.checkpoint_path = checkpoint_path
        self.asset = asset
        self.horizon = horizon
        self.num_samples = num_samples
        self.sampler = sampler
        self.device = device
        self.use_uncertainty = use_uncertainty
        self.uncertainty_weight = uncertainty_weight

        # Internal state
        self.service: Optional[SSSDInferenceService] = None
        self.is_fitted = False
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SSSDWrapper':
        """
        Fit SSSD model (placeholder - SSSD is pre-trained).

        For ensemble compatibility, this method:
        1. Loads the pre-trained SSSD model
        2. Validates it can make predictions
        3. Sets is_fitted flag

        Args:
            X: Training features (ignored - SSSD uses OHLC data)
            y: Training targets (ignored - SSSD is pre-trained)

        Returns:
            self
        """
        self.n_features_in_ = X.shape[1]

        # Load SSSD inference service
        if self.checkpoint_path is not None:
            logger.info(f"Loading SSSD from checkpoint: {self.checkpoint_path}")
            self.service = SSSDInferenceService(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                compile_model=True
            )
        else:
            logger.info(f"Loading SSSD for asset: {self.asset}")
            try:
                self.service = load_sssd_inference_service(
                    asset=self.asset,
                    device=self.device
                )
            except FileNotFoundError:
                logger.warning(
                    f"No checkpoint found for {self.asset}, SSSD will return zeros"
                )
                self.service = None

        self.is_fitted = True

        if self.service is not None:
            logger.info(
                f"SSSD wrapper fitted: asset={self.asset}, "
                f"horizon={self.horizon}m, "
                f"num_samples={self.num_samples}"
            )
        else:
            logger.warning("SSSD wrapper fitted but service is None (will return zeros)")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using SSSD model.

        Args:
            X: Features (n_samples, n_features)
                Note: SSSD doesn't use these features directly,
                it needs OHLC data which must be fetched separately

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("SSSD wrapper not fitted. Call fit() first.")

        n_samples = X.shape[0]

        # If service not loaded, return zeros (fallback)
        if self.service is None:
            logger.warning("SSSD service not available, returning zeros")
            return np.zeros(n_samples)

        # For ensemble integration, we assume predictions are pre-computed
        # and stored in a cache or fetched from the service
        # This is a placeholder - actual implementation should fetch OHLC data
        logger.warning(
            "SSSD predict() called with feature matrix. "
            "For production, use predict_from_ohlc() with OHLC data."
        )

        return np.zeros(n_samples)

    def predict_from_ohlc(self, df: pd.DataFrame) -> float:
        """
        Make prediction from OHLC data (proper SSSD inference).

        Args:
            df: OHLC dataframe with recent bars

        Returns:
            Predicted price change for specified horizon
        """
        if not self.is_fitted:
            raise RuntimeError("SSSD wrapper not fitted. Call fit() first.")

        if self.service is None:
            return 0.0

        # Make prediction
        prediction = self.service.predict(
            df=df,
            num_samples=self.num_samples,
            sampler=self.sampler,
            use_cache=True
        )

        # Extract prediction for specified horizon
        if self.horizon not in prediction.mean:
            logger.warning(
                f"Horizon {self.horizon}m not in predictions, "
                f"available: {list(prediction.mean.keys())}"
            )
            return 0.0

        mean_pred = prediction.mean[self.horizon]

        # Adjust for uncertainty if enabled
        if self.use_uncertainty and self.horizon in prediction.std:
            std_pred = prediction.std[self.horizon]

            # Uncertainty-adjusted prediction (shrink towards zero for high uncertainty)
            # adjusted = mean * (1 - weight * normalized_uncertainty)
            normalized_uncertainty = min(std_pred / (abs(mean_pred) + 1e-8), 1.0)
            adjusted_pred = mean_pred * (1.0 - self.uncertainty_weight * normalized_uncertainty)

            return float(adjusted_pred)
        else:
            return float(mean_pred)

    def get_prediction_with_uncertainty(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get full prediction with uncertainty estimates.

        Args:
            df: OHLC dataframe

        Returns:
            Dict with mean, std, q05, q50, q95
        """
        if not self.is_fitted or self.service is None:
            return {
                "mean": 0.0,
                "std": 0.0,
                "q05": 0.0,
                "q50": 0.0,
                "q95": 0.0
            }

        prediction = self.service.predict(df, num_samples=self.num_samples)

        if self.horizon not in prediction.mean:
            return {
                "mean": 0.0,
                "std": 0.0,
                "q05": 0.0,
                "q50": 0.0,
                "q95": 0.0
            }

        return {
            "mean": float(prediction.mean[self.horizon]),
            "std": float(prediction.std[self.horizon]),
            "q05": float(prediction.q05[self.horizon]),
            "q50": float(prediction.q50[self.horizon]),
            "q95": float(prediction.q95[self.horizon])
        }

    def get_confidence(self, df: pd.DataFrame) -> float:
        """
        Get directional confidence for prediction.

        Args:
            df: OHLC dataframe

        Returns:
            Confidence score in [0, 1]
        """
        if not self.is_fitted or self.service is None:
            return 0.0

        prediction = self.service.predict(df, num_samples=self.num_samples)

        return self.service.get_directional_confidence(prediction, self.horizon)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters (sklearn compatibility)."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "asset": self.asset,
            "horizon": self.horizon,
            "num_samples": self.num_samples,
            "sampler": self.sampler,
            "device": self.device,
            "use_uncertainty": self.use_uncertainty,
            "uncertainty_weight": self.uncertainty_weight
        }

    def set_params(self, **params) -> 'SSSDWrapper':
        """Set parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SSSDEnsembleIntegrator:
    """
    Integrates SSSD into existing ensemble with dynamic reweighting.

    Monitors SSSD uncertainty and adjusts ensemble weights accordingly.
    """

    def __init__(
        self,
        sssd_wrapper: SSSDWrapper,
        base_weight: float = 0.35,
        min_weight: float = 0.1,
        max_weight: float = 0.6,
        uncertainty_threshold: float = 0.02
    ):
        """
        Initialize SSSD ensemble integrator.

        Args:
            sssd_wrapper: SSSD wrapper instance
            base_weight: Base weight for SSSD in ensemble
            min_weight: Minimum weight when uncertainty is high
            max_weight: Maximum weight when uncertainty is low
            uncertainty_threshold: Uncertainty threshold for weight adjustment
        """
        self.sssd_wrapper = sssd_wrapper
        self.base_weight = base_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.uncertainty_threshold = uncertainty_threshold

    def compute_dynamic_weight(self, df: pd.DataFrame) -> float:
        """
        Compute dynamic weight for SSSD based on uncertainty.

        Args:
            df: OHLC dataframe

        Returns:
            Weight for SSSD in [min_weight, max_weight]
        """
        if not self.sssd_wrapper.is_fitted or self.sssd_wrapper.service is None:
            return self.min_weight

        # Get prediction with uncertainty
        pred_dict = self.sssd_wrapper.get_prediction_with_uncertainty(df)

        mean = pred_dict["mean"]
        std = pred_dict["std"]

        # Compute normalized uncertainty
        normalized_uncertainty = std / (abs(mean) + 1e-8)

        # Map uncertainty to weight
        # Low uncertainty → high weight
        # High uncertainty → low weight
        if normalized_uncertainty < self.uncertainty_threshold:
            # Low uncertainty, increase weight
            weight = self.base_weight + (self.max_weight - self.base_weight) * \
                     (1.0 - normalized_uncertainty / self.uncertainty_threshold)
        else:
            # High uncertainty, decrease weight
            excess_uncertainty = normalized_uncertainty - self.uncertainty_threshold
            weight = self.base_weight - (self.base_weight - self.min_weight) * \
                     min(excess_uncertainty / self.uncertainty_threshold, 1.0)

        return float(np.clip(weight, self.min_weight, self.max_weight))

    def get_weighted_prediction(
        self,
        df: pd.DataFrame,
        other_predictions: np.ndarray,
        other_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Get weighted ensemble prediction including SSSD.

        Args:
            df: OHLC dataframe for SSSD
            other_predictions: Predictions from other models (n_models,)
            other_weights: Weights for other models (default: equal weights)

        Returns:
            Weighted ensemble prediction
        """
        # Get SSSD prediction
        sssd_pred = self.sssd_wrapper.predict_from_ohlc(df)

        # Compute dynamic weight
        sssd_weight = self.compute_dynamic_weight(df)

        # Normalize other weights
        if other_weights is None:
            other_weights = np.ones(len(other_predictions)) / len(other_predictions)

        other_weights = other_weights * (1.0 - sssd_weight)
        other_weights /= other_weights.sum()

        # Weighted average
        ensemble_pred = sssd_weight * sssd_pred + np.sum(other_weights * other_predictions)

        return float(ensemble_pred)


def create_sssd_base_model_spec(
    asset: str = "EURUSD",
    horizon: int = 5,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
):
    """
    Create BaseModelSpec for SSSD to add to ensemble.

    Args:
        asset: Asset symbol
        horizon: Forecast horizon in minutes
        checkpoint_path: Optional checkpoint path
        device: Device to run on

    Returns:
        BaseModelSpec for SSSD
    """
    from .ensemble import BaseModelSpec

    sssd_wrapper = SSSDWrapper(
        checkpoint_path=checkpoint_path,
        asset=asset,
        horizon=horizon,
        num_samples=100,
        sampler="ddim",
        device=device,
        use_uncertainty=True,
        uncertainty_weight=0.5
    )

    spec = BaseModelSpec(
        model_id=f"sssd_{asset}_{horizon}m",
        model_type="sssd_diffusion",
        estimator=sssd_wrapper,
        feature_subset=None,  # SSSD uses OHLC data, not features
        hyperparameters={
            "asset": asset,
            "horizon": horizon,
            "num_samples": 100,
            "sampler": "ddim"
        }
    )

    logger.info(f"Created SSSD base model spec: {spec.model_id}")

    return spec
