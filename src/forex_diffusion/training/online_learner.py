"""
Online Learning Implementation

Incremental model updates without full retraining.
Adapts to concept drift with weighted samples and forgetting mechanisms.

Supports:
- SGD-based models with partial_fit()
- Adaptive learning rate decay
- Concept drift handling
- Catastrophic forgetting prevention

Reference: "Online Learning and Stochastic Approximations" by Bottou (1998)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import StandardScaler
from loguru import logger


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning"""

    # Learning rate
    initial_learning_rate: float = 0.01
    learning_rate_decay: str = "invscaling"  # constant, optimal, invscaling, adaptive
    power_t: float = 0.25  # For invscaling decay

    # Sample weighting
    enable_sample_weighting: bool = True
    recency_weight_halflife: int = 100  # Samples until weight halves
    max_sample_weight: float = 10.0

    # Forgetting mechanism
    enable_forgetting: bool = True
    forgetting_factor: float = 0.999  # Weight decay per update
    sliding_window_size: Optional[int] = 5000  # Max samples to keep

    # Drift adaptation
    drift_adaptation_rate: float = 1.5  # Multiplier when drift detected
    performance_window: int = 50  # Window for performance tracking

    # Regularization
    alpha: float = 0.0001  # L2 regularization
    l1_ratio: float = 0.0  # ElasticNet mixing (0 = L2, 1 = L1)

    # Stability
    min_samples_before_update: int = 10
    max_updates_per_batch: int = 1
    validation_frequency: int = 100  # Validate every N updates


@dataclass
class OnlineMetrics:
    """Metrics for online learning performance"""

    total_updates: int = 0
    total_samples_seen: int = 0
    current_learning_rate: float = 0.01

    # Performance tracking
    recent_losses: List[float] = field(default_factory=list)
    recent_accuracies: List[float] = field(default_factory=list)

    # Drift detection
    drift_detected_count: int = 0
    last_drift_detection: Optional[datetime] = None

    # Forgetting
    samples_forgotten: int = 0

    def get_recent_performance(self, window: int = 50) -> Dict[str, float]:
        """Get recent performance metrics"""
        recent_loss = (
            np.mean(self.recent_losses[-window:]) if self.recent_losses else 0.0
        )
        recent_acc = (
            np.mean(self.recent_accuracies[-window:]) if self.recent_accuracies else 0.0
        )

        return {
            "recent_loss": float(recent_loss),
            "recent_accuracy": float(recent_acc),
            "total_updates": self.total_updates,
            "total_samples": self.total_samples_seen,
        }


class OnlineLearner:
    """
    Online learning wrapper for incremental model updates.

    Handles:
    - Incremental updates with partial_fit()
    - Adaptive learning rate
    - Sample weighting (recent samples weighted higher)
    - Concept drift detection and adaptation
    - Catastrophic forgetting prevention
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        config: Optional[OnlineLearningConfig] = None,
        task_type: str = "regression",
    ):
        """
        Initialize online learner.

        Args:
            base_estimator: Estimator supporting partial_fit() (default: SGD)
            config: Online learning configuration
            task_type: "regression" or "classification"
        """
        self.config = config or OnlineLearningConfig()
        self.task_type = task_type

        # Initialize base estimator
        if base_estimator is None:
            if task_type == "regression":
                self.model = SGDRegressor(
                    learning_rate=self.config.learning_rate_decay,
                    eta0=self.config.initial_learning_rate,
                    power_t=self.config.power_t,
                    alpha=self.config.alpha,
                    l1_ratio=self.config.l1_ratio,
                    max_iter=1,  # Single pass per partial_fit
                    tol=None,
                    warm_start=True,
                )
            else:
                self.model = SGDClassifier(
                    learning_rate=self.config.learning_rate_decay,
                    eta0=self.config.initial_learning_rate,
                    power_t=self.config.power_t,
                    alpha=self.config.alpha,
                    l1_ratio=self.config.l1_ratio,
                    max_iter=1,
                    tol=None,
                    warm_start=True,
                    loss="log_loss",  # For probabilistic output
                )
        else:
            self.model = base_estimator

        # Feature scaling
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Metrics
        self.metrics = OnlineMetrics(
            current_learning_rate=self.config.initial_learning_rate
        )

        # Sample buffer for sliding window
        self.sample_buffer: List[Tuple[np.ndarray, float, float]] = []  # (X, y, weight)

        # Validation holdout
        self.validation_X: Optional[np.ndarray] = None
        self.validation_y: Optional[np.ndarray] = None

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Incrementally update model with new samples.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            sample_weights: Optional sample weights

        Returns:
            Dictionary with update metrics
        """
        if len(X) == 0:
            return {}

        # Initial fit if needed
        if not self.is_fitted:
            logger.info("Performing initial fit...")
            self.scaler.fit(X)
            self.is_fitted = True

        # Transform features
        X_scaled = self.scaler.transform(X)

        # Calculate sample weights
        if sample_weights is None and self.config.enable_sample_weighting:
            sample_weights = self._calculate_recency_weights(len(X))

        # Update model
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X_scaled, y, sample_weight=sample_weights)
        else:
            logger.error("Model does not support partial_fit()")
            return {}

        # Update metrics
        self.metrics.total_updates += 1
        self.metrics.total_samples_seen += len(X)

        # Calculate loss
        y_pred = self.model.predict(X_scaled)

        if self.task_type == "regression":
            loss = np.mean((y - y_pred) ** 2)  # MSE
        else:
            loss = np.mean(y != y_pred)  # Error rate

        self.metrics.recent_losses.append(float(loss))

        # Trim metrics history
        if len(self.metrics.recent_losses) > 500:
            self.metrics.recent_losses = self.metrics.recent_losses[-500:]

        # Update learning rate (for adaptive decay)
        if hasattr(self.model, "learning_rate_"):
            self.metrics.current_learning_rate = self.model.learning_rate_

        # Add to sample buffer (for sliding window)
        if self.config.enable_forgetting and self.config.sliding_window_size:
            for i in range(len(X)):
                weight = sample_weights[i] if sample_weights is not None else 1.0
                self.sample_buffer.append((X_scaled[i], y[i], weight))

            # Trim buffer
            if len(self.sample_buffer) > self.config.sliding_window_size:
                n_to_remove = len(self.sample_buffer) - self.config.sliding_window_size
                self.sample_buffer = self.sample_buffer[n_to_remove:]
                self.metrics.samples_forgotten += n_to_remove

        # Periodic validation
        if (
            self.metrics.total_updates % self.config.validation_frequency == 0
            and self.validation_X is not None
        ):
            val_metrics = self._validate()
            return {**{"train_loss": loss}, **val_metrics}

        return {"train_loss": loss, "learning_rate": self.metrics.current_learning_rate}

    def update_with_drift_adaptation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        drift_detected: bool = False,
    ) -> Dict[str, float]:
        """
        Update with drift adaptation.

        If drift detected, increase learning rate temporarily.

        Args:
            X: Features
            y: Targets
            drift_detected: Whether concept drift was detected

        Returns:
            Update metrics
        """
        # Calculate sample weights
        sample_weights = self._calculate_recency_weights(len(X))

        if drift_detected:
            logger.warning("Concept drift detected, adapting learning rate")

            # Boost learning rate
            sample_weights *= self.config.drift_adaptation_rate

            # Cap weights
            sample_weights = np.clip(
                sample_weights,
                0.0,
                self.config.max_sample_weight,
            )

            self.metrics.drift_detected_count += 1
            self.metrics.last_drift_detection = datetime.now()

        return self.partial_fit(X, y, sample_weights=sample_weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new samples.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (classification only).

        Args:
            X: Feature matrix

        Returns:
            Class probabilities
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            raise AttributeError("Model does not support predict_proba")

    def set_validation_set(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Set validation set for periodic evaluation.

        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        self.validation_X = X_val
        self.validation_y = y_val
        logger.info(f"Validation set configured: {len(X_val)} samples")

    def _validate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if self.validation_X is None:
            return {}

        X_scaled = self.scaler.transform(self.validation_X)
        y_pred = self.model.predict(X_scaled)

        if self.task_type == "regression":
            mse = np.mean((self.validation_y - y_pred) ** 2)
            mae = np.mean(np.abs(self.validation_y - y_pred))

            return {
                "val_mse": float(mse),
                "val_mae": float(mae),
            }
        else:
            accuracy = np.mean(self.validation_y == y_pred)

            self.metrics.recent_accuracies.append(float(accuracy))

            if len(self.metrics.recent_accuracies) > 500:
                self.metrics.recent_accuracies = self.metrics.recent_accuracies[-500:]

            return {
                "val_accuracy": float(accuracy),
            }

    def _calculate_recency_weights(self, n_samples: int) -> np.ndarray:
        """
        Calculate recency-based sample weights.

        Recent samples get higher weight using exponential decay.

        Args:
            n_samples: Number of samples in current batch

        Returns:
            Sample weights array
        """
        # Time indices (most recent = n_samples - 1)
        time_indices = np.arange(n_samples)

        # Exponential decay: weight = exp(-lambda * (T - t))
        # lambda chosen so weight halves at halflife
        decay_lambda = np.log(2) / self.config.recency_weight_halflife

        # Calculate weights (most recent gets weight 1.0)
        weights = np.exp(-decay_lambda * (n_samples - 1 - time_indices))

        # Normalize so mean weight = 1.0
        weights = weights / weights.mean()

        return weights

    def replay_buffer(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Replay samples from buffer to prevent catastrophic forgetting.

        Args:
            batch_size: Number of samples to replay

        Returns:
            Replay metrics
        """
        if not self.sample_buffer:
            return {}

        # Sample randomly from buffer
        if len(self.sample_buffer) <= batch_size:
            replay_samples = self.sample_buffer
        else:
            indices = np.random.choice(
                len(self.sample_buffer),
                size=batch_size,
                replace=False,
            )
            replay_samples = [self.sample_buffer[i] for i in indices]

        # Extract X, y, weights
        X_replay = np.array([s[0] for s in replay_samples])
        y_replay = np.array([s[1] for s in replay_samples])
        weights_replay = np.array([s[2] for s in replay_samples])

        # Decay weights by forgetting factor
        weights_replay *= self.config.forgetting_factor

        # Update model
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X_replay, y_replay, sample_weight=weights_replay)

        logger.debug(f"Replayed {len(replay_samples)} samples from buffer")

        return {"replayed_samples": len(replay_samples)}

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of online learning metrics"""
        recent_perf = self.metrics.get_recent_performance()

        return {
            **recent_perf,
            "drift_detected_count": self.metrics.drift_detected_count,
            "last_drift": (
                self.metrics.last_drift_detection.isoformat()
                if self.metrics.last_drift_detection
                else None
            ),
            "samples_forgotten": self.metrics.samples_forgotten,
            "buffer_size": len(self.sample_buffer),
            "learning_rate": self.metrics.current_learning_rate,
        }

    def reset(self) -> None:
        """Reset online learner to initial state"""
        # Reinitialize model
        if self.task_type == "regression":
            self.model = SGDRegressor(
                learning_rate=self.config.learning_rate_decay,
                eta0=self.config.initial_learning_rate,
                power_t=self.config.power_t,
                alpha=self.config.alpha,
                max_iter=1,
                tol=None,
                warm_start=True,
            )
        else:
            self.model = SGDClassifier(
                learning_rate=self.config.learning_rate_decay,
                eta0=self.config.initial_learning_rate,
                power_t=self.config.power_t,
                alpha=self.config.alpha,
                max_iter=1,
                tol=None,
                warm_start=True,
                loss="log_loss",
            )

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.metrics = OnlineMetrics(
            current_learning_rate=self.config.initial_learning_rate
        )
        self.sample_buffer.clear()

        logger.info("Online learner reset to initial state")


# Convenience function
def create_online_learner(
    task_type: str = "regression",
    learning_rate: float = 0.01,
    enable_forgetting: bool = True,
) -> OnlineLearner:
    """
    Create online learner with default configuration.

    Args:
        task_type: "regression" or "classification"
        learning_rate: Initial learning rate
        enable_forgetting: Enable forgetting mechanism

    Returns:
        Configured OnlineLearner
    """
    config = OnlineLearningConfig(
        initial_learning_rate=learning_rate,
        enable_forgetting=enable_forgetting,
    )

    return OnlineLearner(config=config, task_type=task_type)
