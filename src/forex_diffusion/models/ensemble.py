"""
Ensemble Methods & Meta-Learning

Combines predictions from multiple base models using meta-learning.
Implements stacking with out-of-fold predictions for robust training.

Architecture:
- Level 1: Multiple base models (Ridge, RF, XGB, NN, etc.)
- Level 2: Meta-learner learns optimal weighting

Reference: "Stacked Generalization" by Wolpert (1992)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from loguru import logger


@dataclass
class EnsembleConfig:
    """Configuration for ensemble learning"""

    # Base models
    base_model_types: List[str] = field(
        default_factory=lambda: ["ridge", "lasso", "random_forest"]
    )

    # Meta-learner
    meta_learner_type: str = "ridge"  # ridge, lasso, linear
    meta_regularization: float = 1.0

    # Stacking configuration
    n_folds: int = 5
    shuffle_folds: bool = False  # Time-series: no shuffle

    # Feature engineering for meta-learner
    use_base_predictions: bool = True
    use_original_features: bool = False  # Can include original features
    use_prediction_variance: bool = True  # Add std of predictions

    # Model diversity
    feature_subsampling: bool = True
    feature_subsample_ratio: float = 0.8


@dataclass
class BaseModelSpec:
    """Specification for a base model"""
    model_id: str
    model_type: str
    estimator: BaseEstimator
    feature_subset: Optional[List[int]] = None  # Indices of features to use
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble with meta-learning.

    Level 1: Multiple base models make predictions
    Level 2: Meta-model learns optimal combination

    Uses out-of-fold predictions to prevent overfitting.
    """

    def __init__(
        self,
        base_models: Optional[List[BaseModelSpec]] = None,
        config: Optional[EnsembleConfig] = None,
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base model specifications
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.base_models = base_models or self._create_default_base_models()

        # Meta-learner
        self.meta_learner = self._create_meta_learner()

        # State
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None

    def _create_default_base_models(self) -> List[BaseModelSpec]:
        """Create default base model specifications"""
        models = []

        for model_type in self.config.base_model_types:
            if model_type == "ridge":
                estimator = Ridge(alpha=1.0)
            elif model_type == "lasso":
                estimator = Lasso(alpha=0.1)
            elif model_type == "random_forest":
                estimator = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    random_state=42,
                )
            elif model_type == "sssd_diffusion" or model_type.startswith("sssd_"):
                # SSSD model - skip default creation, must be added manually
                logger.info("SSSD model detected, skipping default creation")
                continue
            else:
                logger.warning(f"Unknown model type: {model_type}, skipping")
                continue

            model_spec = BaseModelSpec(
                model_id=f"{model_type}_{len(models)}",
                model_type=model_type,
                estimator=estimator,
            )

            models.append(model_spec)

        logger.info(f"Created {len(models)} default base models")

        return models

    def _create_meta_learner(self) -> BaseEstimator:
        """Create meta-learner model"""
        if self.config.meta_learner_type == "ridge":
            return Ridge(alpha=self.config.meta_regularization)
        elif self.config.meta_learner_type == "lasso":
            return Lasso(alpha=self.config.meta_regularization)
        else:
            return Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit stacking ensemble with out-of-fold predictions.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)

        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        logger.info(
            f"Training stacking ensemble: {len(self.base_models)} base models, "
            f"{n_samples} samples, {n_features} features"
        )

        # Step 1: Generate feature subsets for each base model (if enabled)
        if self.config.feature_subsampling:
            self._assign_feature_subsets(n_features)

        # Step 2: Train base models and collect out-of-fold predictions
        oof_predictions = self._train_base_models_with_oof(X, y)

        # Step 3: Build meta-features
        meta_X = self._build_meta_features(X, oof_predictions)

        # Step 4: Train meta-learner
        logger.info("Training meta-learner...")
        self.meta_learner.fit(meta_X, y)

        self.is_fitted = True

        logger.info("Stacking ensemble training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using stacking ensemble.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all base models
        base_predictions = self._get_base_predictions(X)

        # Build meta-features
        meta_X = self._build_meta_features(X, base_predictions)

        # Meta-learner prediction
        final_predictions = self.meta_learner.predict(meta_X)

        return final_predictions

    def _assign_feature_subsets(self, n_features: int) -> None:
        """
        Assign random feature subsets to each base model for diversity.

        Args:
            n_features: Total number of features
        """
        subset_size = int(n_features * self.config.feature_subsample_ratio)

        for model_spec in self.base_models:
            # Random subset of features
            feature_indices = np.random.choice(
                n_features,
                size=subset_size,
                replace=False,
            )
            model_spec.feature_subset = sorted(feature_indices.tolist())

        logger.debug(
            f"Assigned feature subsets: {subset_size}/{n_features} features per model"
        )

    def _train_base_models_with_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Train base models and generate out-of-fold predictions.

        This prevents overfitting: each sample's meta-feature is generated
        by a model that never saw that sample during training.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Out-of-fold predictions (n_samples, n_base_models)
        """
        n_samples = len(X)
        n_models = len(self.base_models)

        # Matrix to store OOF predictions
        oof_preds = np.zeros((n_samples, n_models))

        # K-fold split
        kf = KFold(
            n_splits=self.config.n_folds,
            shuffle=self.config.shuffle_folds,
            random_state=42 if self.config.shuffle_folds else None,
        )

        for model_idx, model_spec in enumerate(self.base_models):
            logger.info(f"Training base model {model_idx + 1}/{n_models}: {model_spec.model_id}")

            # Track fold predictions
            fold_predictions = np.zeros(n_samples)

            for fold_num, (train_idx, val_idx) in enumerate(kf.split(X)):
                # Split data
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]

                # Apply feature subset if specified
                if model_spec.feature_subset:
                    X_train_fold = X_train_fold[:, model_spec.feature_subset]
                    X_val_fold = X_val_fold[:, model_spec.feature_subset]

                # Clone and train model
                model_clone = clone(model_spec.estimator)
                model_clone.fit(X_train_fold, y_train_fold)

                # Predict on validation fold
                val_preds = model_clone.predict(X_val_fold)
                fold_predictions[val_idx] = val_preds

            # Store OOF predictions for this model
            oof_preds[:, model_idx] = fold_predictions

            # Now train final model on full dataset
            X_full = X
            if model_spec.feature_subset:
                X_full = X[:, model_spec.feature_subset]

            model_spec.estimator.fit(X_full, y)

            logger.debug(f"Model {model_spec.model_id} trained on full dataset")

        return oof_preds

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all base models.

        Args:
            X: Features

        Returns:
            Predictions matrix (n_samples, n_base_models)
        """
        n_samples = len(X)
        n_models = len(self.base_models)

        predictions = np.zeros((n_samples, n_models))

        for model_idx, model_spec in enumerate(self.base_models):
            X_subset = X
            if model_spec.feature_subset:
                X_subset = X[:, model_spec.feature_subset]

            predictions[:, model_idx] = model_spec.estimator.predict(X_subset)

        return predictions

    def _build_meta_features(
        self,
        X: np.ndarray,
        base_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Build features for meta-learner.

        Args:
            X: Original features
            base_predictions: Predictions from base models (n_samples, n_models)

        Returns:
            Meta-features (n_samples, n_meta_features)
        """
        meta_features = []

        # 1. Base model predictions
        if self.config.use_base_predictions:
            meta_features.append(base_predictions)

        # 2. Prediction variance (measure of agreement)
        if self.config.use_prediction_variance:
            pred_std = base_predictions.std(axis=1, keepdims=True)
            pred_mean = base_predictions.mean(axis=1, keepdims=True)
            pred_cv = pred_std / (np.abs(pred_mean) + 1e-10)  # Coefficient of variation

            meta_features.append(pred_std)
            meta_features.append(pred_cv)

        # 3. Original features (optional)
        if self.config.use_original_features:
            meta_features.append(X)

        # Concatenate all meta-features
        meta_X = np.hstack(meta_features)

        return meta_X

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get effective weights of base models in the ensemble.

        For linear meta-learner, these are the coefficients.

        Returns:
            Dictionary mapping model_id -> weight
        """
        if not self.is_fitted:
            return {}

        if not hasattr(self.meta_learner, 'coef_'):
            return {"note": "Meta-learner does not expose coefficients"}

        coefficients = self.meta_learner.coef_

        # Extract coefficients for base model predictions
        # (First n_models coefficients if using base predictions)
        n_models = len(self.base_models)

        if len(coefficients) < n_models:
            return {"error": "Coefficient mismatch"}

        weights = {}
        for i, model_spec in enumerate(self.base_models):
            weights[model_spec.model_id] = float(coefficients[i])

        return weights

    def get_prediction_breakdown(self, X: np.ndarray) -> pd.DataFrame:
        """
        Get detailed prediction breakdown for samples.

        Args:
            X: Features

        Returns:
            DataFrame with base model predictions and final prediction
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")

        base_preds = self._get_base_predictions(X)
        final_preds = self.predict(X)

        # Build DataFrame
        columns = [f"model_{spec.model_id}" for spec in self.base_models]
        df = pd.DataFrame(base_preds, columns=columns)

        df["ensemble_prediction"] = final_preds
        df["prediction_std"] = base_preds.std(axis=1)
        df["prediction_range"] = base_preds.max(axis=1) - base_preds.min(axis=1)

        return df

    def evaluate_base_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate individual base model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary mapping model_id -> metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")

        results = {}

        for model_spec in self.base_models:
            X_subset = X_test
            if model_spec.feature_subset:
                X_subset = X_test[:, model_spec.feature_subset]

            y_pred = model_spec.estimator.predict(X_subset)

            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(mse)

            # Directional accuracy (for regression)
            direction_correct = ((y_pred > 0) == (y_test > 0)).mean()

            results[model_spec.model_id] = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "directional_accuracy": float(direction_correct),
            }

        # Also evaluate ensemble
        y_pred_ensemble = self.predict(X_test)
        mse_ens = np.mean((y_test - y_pred_ensemble) ** 2)
        mae_ens = np.mean(np.abs(y_test - y_pred_ensemble))
        direction_ens = ((y_pred_ensemble > 0) == (y_test > 0)).mean()

        results["ensemble"] = {
            "mse": float(mse_ens),
            "mae": float(mae_ens),
            "rmse": float(np.sqrt(mse_ens)),
            "directional_accuracy": float(direction_ens),
        }

        return results


# Convenience functions
def create_stacking_ensemble(
    n_base_models: int = 3,
    include_original_features: bool = False,
    include_sssd: bool = False,
    sssd_asset: str = "EURUSD",
    sssd_horizon: int = 5,
) -> StackingEnsemble:
    """
    Create stacking ensemble with default configuration.

    Args:
        n_base_models: Number of base models to create
        include_original_features: Include original features in meta-learner
        include_sssd: Include SSSD diffusion model in ensemble
        sssd_asset: Asset for SSSD model
        sssd_horizon: Forecast horizon for SSSD (minutes)

    Returns:
        Configured StackingEnsemble
    """
    config = EnsembleConfig(
        use_original_features=include_original_features,
        feature_subsampling=True if n_base_models > 1 else False,
    )

    # Create diverse base models
    base_models = []

    model_types = ["ridge", "lasso", "random_forest"]

    for i in range(min(n_base_models, len(model_types))):
        model_type = model_types[i]

        if model_type == "ridge":
            estimator = Ridge(alpha=1.0)
        elif model_type == "lasso":
            estimator = Lasso(alpha=0.1)
        else:
            estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42 + i,
            )

        base_models.append(BaseModelSpec(
            model_id=f"{model_type}_{i}",
            model_type=model_type,
            estimator=estimator,
        ))

    # Add SSSD model if requested
    if include_sssd:
        try:
            from .sssd_wrapper import create_sssd_base_model_spec
            sssd_spec = create_sssd_base_model_spec(
                asset=sssd_asset,
                horizon=sssd_horizon,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            base_models.append(sssd_spec)
            logger.info(f"Added SSSD model to ensemble: {sssd_spec.model_id}")
        except Exception as e:
            logger.warning(f"Failed to add SSSD to ensemble: {e}")

    return StackingEnsemble(base_models=base_models, config=config)


def add_sssd_to_ensemble(
    ensemble: StackingEnsemble,
    asset: str = "EURUSD",
    horizon: int = 5,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> StackingEnsemble:
    """
    Add SSSD model to existing ensemble.

    Args:
        ensemble: Existing StackingEnsemble
        asset: Asset symbol
        horizon: Forecast horizon in minutes
        checkpoint_path: Optional checkpoint path
        device: Device to run on (default: cuda if available)

    Returns:
        Updated ensemble with SSSD added
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from .sssd_wrapper import create_sssd_base_model_spec

        sssd_spec = create_sssd_base_model_spec(
            asset=asset,
            horizon=horizon,
            checkpoint_path=checkpoint_path,
            device=device
        )

        ensemble.base_models.append(sssd_spec)

        logger.info(
            f"Added SSSD to ensemble: {sssd_spec.model_id}, "
            f"total base models: {len(ensemble.base_models)}"
        )

    except Exception as e:
        logger.error(f"Failed to add SSSD to ensemble: {e}")
        raise

    return ensemble
