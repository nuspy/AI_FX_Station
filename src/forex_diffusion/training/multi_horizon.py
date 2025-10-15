"""
Multi-Horizon Native Training

Trains models to predict multiple time horizons simultaneously:
- Single model outputs [h1, h2, h3, ...] predictions
- Shared feature representation
- Horizon-specific head layers
- More efficient than training N separate models

Reference: "Multi-Task Learning" by Caruana (1997)
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from loguru import logger


class MultiHorizonModel(BaseEstimator, RegressorMixin):
    """
    Multi-horizon prediction model.

    Single model that outputs predictions for multiple time horizons.
    """

    def __init__(
        self,
        horizons: List[int],
        base_estimator: Optional[BaseEstimator] = None,
        shared_features: bool = True,
    ):
        """
        Initialize multi-horizon model.

        Args:
            horizons: List of prediction horizons (in bars)
            base_estimator: Base estimator for each horizon
            shared_features: If True, use shared features for all horizons
        """
        self.horizons = sorted(horizons)
        self.base_estimator = base_estimator or Ridge(alpha=1.0)
        self.shared_features = shared_features

        # Multi-output wrapper
        self.model: Optional[MultiOutputRegressor] = None

    def _build_targets(
        self,
        y: pd.Series,
    ) -> np.ndarray:
        """
        Build multi-horizon targets.

        Args:
            y: Close prices

        Returns:
            Target matrix (n_samples, n_horizons) with returns for each horizon
        """
        targets = []

        for h in self.horizons:
            # Forward return for horizon h
            y_h = (y.shift(-h) / y) - 1.0
            targets.append(y_h.values)

        # Stack horizontally
        Y = np.column_stack(targets)

        return Y

    def fit(
        self,
        X: np.ndarray,
        y: pd.Series,
    ):
        """
        Fit multi-horizon model.

        Args:
            X: Feature matrix
            y: Close prices (not returns!)
        """
        # Build multi-horizon targets
        Y = self._build_targets(y)

        # Remove NaN rows (caused by forward shift)
        max_horizon = max(self.horizons)
        valid_mask = np.isfinite(Y).all(axis=1)

        # Also need to ensure we have enough future data
        valid_mask[-max_horizon:] = False

        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]

        logger.info(
            f"Training multi-horizon model: {len(self.horizons)} horizons, "
            f"{len(X_valid)} valid samples"
        )

        # Create multi-output model
        self.model = MultiOutputRegressor(self.base_estimator)

        # Fit
        self.model.fit(X_valid, Y_valid)

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict for all horizons.

        Args:
            X: Feature matrix

        Returns:
            Predictions matrix (n_samples, n_horizons)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted")

        return self.model.predict(X)

    def predict_single_horizon(
        self,
        X: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Predict for a specific horizon.

        Args:
            X: Feature matrix
            horizon: Target horizon

        Returns:
            Predictions for specified horizon
        """
        if horizon not in self.horizons:
            raise ValueError(
                f"Horizon {horizon} not in trained horizons {self.horizons}"
            )

        horizon_idx = self.horizons.index(horizon)
        predictions = self.predict(X)

        return predictions[:, horizon_idx]


class AdaptiveHorizonSelector:
    """
    Selects best horizon prediction based on market conditions.

    Uses regime detection or volatility to choose which horizon to trust.
    """

    def __init__(
        self,
        horizons: List[int],
        selection_strategy: str = "volatility",
    ):
        """
        Initialize adaptive horizon selector.

        Args:
            horizons: Available prediction horizons
            selection_strategy: "volatility", "regime", or "confidence"
        """
        self.horizons = sorted(horizons)
        self.selection_strategy = selection_strategy

    def select_horizon(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        current_idx: int,
    ) -> Tuple[int, float]:
        """
        Select best horizon for current market conditions.

        Args:
            df: DataFrame with OHLCV data
            predictions: Multi-horizon predictions (n_horizons,)
            current_idx: Current bar index

        Returns:
            (selected_horizon, selected_prediction)
        """
        if self.selection_strategy == "volatility":
            return self._select_by_volatility(df, predictions, current_idx)
        elif self.selection_strategy == "regime":
            return self._select_by_regime(df, predictions, current_idx)
        else:
            return self._select_by_confidence(predictions)

    def _select_by_volatility(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        current_idx: int,
        window: int = 20,
    ) -> Tuple[int, float]:
        """
        Select horizon based on recent volatility.

        Low volatility → longer horizons
        High volatility → shorter horizons
        """
        # Calculate recent volatility
        recent = df.iloc[max(0, current_idx - window) : current_idx + 1]
        returns = np.log(recent["close"] / recent["close"].shift(1)).dropna()
        volatility = returns.std()

        # Map volatility to horizon preference
        # High vol (>0.02) → shortest horizon
        # Low vol (<0.005) → longest horizon
        if volatility > 0.02:
            selected_idx = 0  # Shortest
        elif volatility < 0.005:
            selected_idx = len(self.horizons) - 1  # Longest
        else:
            # Interpolate
            vol_ratio = (volatility - 0.005) / (0.02 - 0.005)
            selected_idx = int(vol_ratio * (len(self.horizons) - 1))
            selected_idx = np.clip(selected_idx, 0, len(self.horizons) - 1)

        selected_horizon = self.horizons[selected_idx]
        selected_prediction = float(predictions[selected_idx])

        return selected_horizon, selected_prediction

    def _select_by_regime(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        current_idx: int,
    ) -> Tuple[int, float]:
        """
        Select horizon based on market regime.

        Trending → longer horizons
        Ranging → shorter horizons
        """
        # Check if regime features available
        if "regime_trending_up" in df.columns or "regime_trending_down" in df.columns:
            is_trending = df.iloc[current_idx].get("regime_trending_up", 0) or df.iloc[
                current_idx
            ].get("regime_trending_down", 0)

            if is_trending:
                # Trending: use longer horizon
                selected_idx = len(self.horizons) - 1
            else:
                # Ranging: use shorter horizon
                selected_idx = 0
        else:
            # Fallback to middle horizon
            selected_idx = len(self.horizons) // 2

        selected_horizon = self.horizons[selected_idx]
        selected_prediction = float(predictions[selected_idx])

        return selected_horizon, selected_prediction

    def _select_by_confidence(
        self,
        predictions: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Select horizon with highest confidence (largest absolute prediction).
        """
        # Simple heuristic: choose horizon with strongest signal
        abs_preds = np.abs(predictions)
        selected_idx = int(np.argmax(abs_preds))

        selected_horizon = self.horizons[selected_idx]
        selected_prediction = float(predictions[selected_idx])

        return selected_horizon, selected_prediction


def train_multi_horizon(
    X: np.ndarray,
    y: pd.Series,
    horizons: List[int] = [5, 10, 20, 50],
    base_estimator: Optional[BaseEstimator] = None,
) -> MultiHorizonModel:
    """
    Train multi-horizon model.

    Args:
        X: Feature matrix
        y: Close prices
        horizons: List of prediction horizons
        base_estimator: Base estimator (default: Ridge)

    Returns:
        Fitted MultiHorizonModel
    """
    model = MultiHorizonModel(
        horizons=horizons,
        base_estimator=base_estimator,
    )

    model.fit(X, y)

    return model


def evaluate_multi_horizon(
    model: MultiHorizonModel,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate multi-horizon model on test set.

    Args:
        model: Fitted MultiHorizonModel
        X_test: Test features
        y_test: Test close prices

    Returns:
        Dictionary mapping horizon -> metrics
    """
    predictions = model.predict(X_test)

    results = {}

    for i, horizon in enumerate(model.horizons):
        # Get predictions for this horizon
        y_pred_h = predictions[:, i]

        # Build actual returns for this horizon
        y_actual_h = (y_test.shift(-horizon) / y_test - 1.0).values

        # Remove NaN
        valid_mask = np.isfinite(y_actual_h)
        y_pred_valid = y_pred_h[valid_mask]
        y_actual_valid = y_actual_h[valid_mask]

        if len(y_pred_valid) == 0:
            continue

        # Calculate metrics
        mae = np.mean(np.abs(y_actual_valid - y_pred_valid))
        mse = np.mean((y_actual_valid - y_pred_valid) ** 2)
        rmse = np.sqrt(mse)

        # Directional accuracy
        direction_correct = ((y_pred_valid > 0) == (y_actual_valid > 0)).mean()

        results[horizon] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "directional_accuracy": float(direction_correct),
            "n_samples": len(y_pred_valid),
        }

    return results
