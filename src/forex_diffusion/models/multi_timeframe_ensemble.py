"""
Multi-Timeframe Ensemble System

Ensemble of models trained on different timeframes for robust predictions.
Based on Renaissance Technologies' multi-scale trading approach.

Architecture:
- Multiple models: 1m, 5m, 15m, 1h, 4h, 1d
- Weighted voting by performance and confidence
- Regime-aware timeframe weighting
- Correlation tracking and penalty
- Consensus threshold for trade execution
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class Timeframe(Enum):
    """Supported timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class TimeframeModelPrediction:
    """Prediction from a single timeframe model."""
    timeframe: Timeframe
    signal: int  # -1 (sell), 0 (neutral), +1 (buy)
    confidence: float  # 0-1
    probability: float  # 0-1
    features_used: int
    model_accuracy: float  # Recent accuracy of this model


class MultiTimeframeEnsemble:
    """
    Ensemble of models trained on different timeframes.

    Each timeframe captures different aspects:
    - 1m: Microstructure, order flow, noise trading
    - 5m: Short-term momentum, quick reversals
    - 15m: Intraday patterns, session dynamics
    - 1h: Medium-term trends, multi-hour patterns
    - 4h: Macro patterns, daily cycles
    - 1d: Long-term trends, weekly/monthly patterns

    Based on Renaissance Technologies' multi-scale approach.

    Example:
        >>> ensemble = MultiTimeframeEnsemble(
        ...     consensus_threshold=0.60,
        ...     min_models_required=3
        ... )
        >>> ensemble.register_model(Timeframe.M5, model_5m)
        >>> ensemble.register_model(Timeframe.H1, model_1h)
        >>> result = ensemble.predict_ensemble(data_by_timeframe)
        >>> if result['final_signal'] == 1:
        ...     print(f"BUY with {result['confidence']:.1%} confidence")
    """

    def __init__(
        self,
        consensus_threshold: float = 0.60,
        min_models_required: int = 3,
        correlation_penalty_threshold: float = 0.70
    ):
        """
        Initialize multi-timeframe ensemble.

        Args:
            consensus_threshold: Minimum agreement for trade (default: 60%)
            min_models_required: Minimum models agreeing to trade (default: 3)
            correlation_penalty_threshold: Reduce weight if correlation > this (default: 0.70)
        """
        self.consensus_threshold = consensus_threshold
        self.min_models_required = min_models_required
        self.correlation_penalty = correlation_penalty_threshold

        # Store models for each timeframe
        self.models: Dict[Timeframe, Any] = {}

        # Track recent performance per timeframe
        self.performance_history: Dict[Timeframe, List[float]] = {
            tf: [] for tf in Timeframe
        }

        # Track prediction correlation
        self.prediction_correlation: Optional[np.ndarray] = None

    def register_model(
        self,
        timeframe: Timeframe,
        model: Any
    ):
        """
        Register a trained model for a timeframe.

        Args:
            timeframe: Timeframe for this model
            model: Trained sklearn-compatible model
        """
        self.models[timeframe] = model

    def predict_ensemble(
        self,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        current_regime: Optional[str] = None
    ) -> Dict:
        """
        Generate ensemble prediction across all timeframes.

        Args:
            data_by_timeframe: Dict mapping Timeframe to DataFrame with features
            current_regime: Optional current market regime

        Returns:
            Dict with:
                - final_signal: -1/0/+1
                - confidence: 0-1
                - consensus: 0-1
                - agreeing_models: int
                - total_models: int
                - individual_predictions: list of dict
                - reason: str
        """
        # 1. Get predictions from each timeframe
        predictions: List[TimeframeModelPrediction] = []

        for timeframe, model in self.models.items():
            if timeframe not in data_by_timeframe:
                continue

            data = data_by_timeframe[timeframe]

            # Ensure data is 2D array for sklearn
            if isinstance(data, pd.DataFrame):
                X = data.values
            else:
                X = data

            # Handle single sample
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Model prediction
            try:
                # Try predict_proba first (classification models)
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)
                    if len(pred_proba.shape) == 2 and pred_proba.shape[1] == 3:
                        # 3-class classification: SELL, NEUTRAL, BUY
                        pred_signal = np.argmax(pred_proba[-1]) - 1  # Map 0,1,2 to -1,0,+1
                        pred_confidence = pred_proba[-1][np.argmax(pred_proba[-1])]
                        pred_probability = pred_proba[-1][2]  # Probability of BUY
                    else:
                        # Binary classification
                        pred_signal = 1 if pred_proba[-1][1] > 0.5 else -1
                        pred_confidence = max(pred_proba[-1])
                        pred_probability = pred_proba[-1][1]
                else:
                    # Regression model
                    pred_value = model.predict(X)[-1]
                    pred_signal = 1 if pred_value > 0 else (-1 if pred_value < 0 else 0)
                    pred_confidence = min(abs(pred_value), 1.0)
                    pred_probability = (pred_value + 1) / 2  # Normalize to 0-1

            except Exception as e:
                print(f"Warning: Prediction failed for {timeframe.value}: {e}")
                continue

            # Get recent model accuracy
            recent_accuracy = self._get_recent_accuracy(timeframe)

            prediction = TimeframeModelPrediction(
                timeframe=timeframe,
                signal=pred_signal,
                confidence=pred_confidence,
                probability=pred_probability,
                features_used=X.shape[1],
                model_accuracy=recent_accuracy
            )
            predictions.append(prediction)

        if len(predictions) < self.min_models_required:
            return {
                'final_signal': 0,
                'confidence': 0.0,
                'consensus': 0.0,
                'agreeing_models': 0,
                'total_models': len(predictions),
                'individual_predictions': [],
                'reason': f"Insufficient models ({len(predictions)} < {self.min_models_required})"
            }

        # 2. Calculate weights based on:
        #    - Recent performance
        #    - Confidence
        #    - Regime appropriateness
        #    - Correlation penalty
        weights = self._calculate_weights(predictions, current_regime)

        # 3. Weighted voting
        weighted_votes = []
        for pred, weight in zip(predictions, weights):
            weighted_votes.append(pred.signal * weight)

        # 4. Calculate consensus
        total_weight = sum(weights)
        if total_weight == 0:
            return {
                'final_signal': 0,
                'confidence': 0.0,
                'consensus': 0.0,
                'agreeing_models': 0,
                'total_models': len(predictions),
                'individual_predictions': [],
                'reason': "Zero total weight"
            }

        weighted_sum = sum(weighted_votes)
        consensus_score = abs(weighted_sum) / total_weight

        # 5. Determine final signal
        if consensus_score >= self.consensus_threshold:
            final_signal = 1 if weighted_sum > 0 else -1
            confidence = consensus_score
        else:
            final_signal = 0  # No consensus
            confidence = 0.0

        # 6. Count agreeing models
        agreeing_models = sum(
            1 for pred in predictions
            if pred.signal == final_signal
        ) if final_signal != 0 else 0

        return {
            'final_signal': final_signal,
            'confidence': confidence,
            'consensus': consensus_score,
            'agreeing_models': agreeing_models,
            'total_models': len(predictions),
            'individual_predictions': [
                {
                    'timeframe': p.timeframe.value,
                    'signal': p.signal,
                    'confidence': p.confidence,
                    'probability': p.probability,
                    'accuracy': p.model_accuracy,
                    'weight': w
                }
                for p, w in zip(predictions, weights)
            ],
            'reason': f"Consensus {consensus_score:.1%}, {agreeing_models}/{len(predictions)} agree"
        }

    def _calculate_weights(
        self,
        predictions: List[TimeframeModelPrediction],
        current_regime: Optional[str]
    ) -> List[float]:
        """
        Calculate voting weights for each prediction.

        Factors:
        1. Recent model accuracy (higher = higher weight)
        2. Prediction confidence (higher = higher weight)
        3. Regime appropriateness (trending regimes favor higher TF)
        4. Correlation penalty (if predictions too correlated, reduce weight)

        Args:
            predictions: List of predictions from each timeframe
            current_regime: Current market regime (optional)

        Returns:
            List of weights (same length as predictions)
        """
        weights = []

        # Base weights from accuracy and confidence
        for pred in predictions:
            # Accuracy component (0.5-1.0)
            accuracy_weight = pred.model_accuracy if pred.model_accuracy > 0 else 0.5

            # Confidence component (0-1)
            confidence_weight = pred.confidence

            # Combine (geometric mean for balance)
            base_weight = np.sqrt(accuracy_weight * confidence_weight)

            # Regime adjustment
            regime_multiplier = self._get_regime_multiplier(pred.timeframe, current_regime)

            # Final weight
            weight = base_weight * regime_multiplier
            weights.append(weight)

        # Apply correlation penalty
        # If predictions too correlated, they're not adding diversity
        if len(predictions) >= 3:
            signals = np.array([p.signal for p in predictions])
            if np.std(signals) < 0.5:  # Very similar predictions
                # Reduce weights proportionally (lack of diversity)
                weights = [w * 0.8 for w in weights]

        return weights

    def _get_regime_multiplier(
        self,
        timeframe: Timeframe,
        regime: Optional[str]
    ) -> float:
        """
        Adjust timeframe weight based on current regime.

        Different regimes favor different timeframes:
        - Trending: Higher timeframes more reliable
        - Ranging: Lower timeframes better for reversals
        - High volatility: Medium timeframes balance

        Args:
            timeframe: Timeframe to weight
            regime: Current market regime

        Returns:
            Weight multiplier (0.7-1.4)
        """
        if regime is None:
            return 1.0

        regime_lower = regime.lower()

        # Trending regimes
        if any(keyword in regime_lower for keyword in ['trend', 'bull', 'bear', 'up', 'down']):
            multipliers = {
                Timeframe.M1: 0.7,
                Timeframe.M5: 0.9,
                Timeframe.M15: 1.0,
                Timeframe.H1: 1.2,
                Timeframe.H4: 1.3,
                Timeframe.D1: 1.4
            }

        # Ranging regimes
        elif 'rang' in regime_lower:
            multipliers = {
                Timeframe.M1: 1.2,
                Timeframe.M5: 1.3,
                Timeframe.M15: 1.1,
                Timeframe.H1: 0.9,
                Timeframe.H4: 0.8,
                Timeframe.D1: 0.7
            }

        # High volatility
        elif 'volatil' in regime_lower:
            multipliers = {
                Timeframe.M1: 0.8,
                Timeframe.M5: 0.9,
                Timeframe.M15: 1.1,
                Timeframe.H1: 1.2,
                Timeframe.H4: 1.0,
                Timeframe.D1: 0.9
            }

        else:
            # Default: equal weights
            multipliers = {tf: 1.0 for tf in Timeframe}

        return multipliers.get(timeframe, 1.0)

    def _get_recent_accuracy(
        self,
        timeframe: Timeframe,
        lookback: int = 50
    ) -> float:
        """
        Get recent accuracy for a timeframe model.

        Args:
            timeframe: Timeframe to query
            lookback: Number of recent trades to consider

        Returns:
            Average accuracy (0-1)
        """
        history = self.performance_history.get(timeframe, [])
        if not history:
            return 0.5  # Default neutral

        recent = history[-lookback:] if len(history) > lookback else history
        return np.mean(recent)

    def update_performance(
        self,
        timeframe: Timeframe,
        was_correct: bool
    ):
        """
        Update performance history after trade outcome known.

        Args:
            timeframe: Timeframe that made prediction
            was_correct: Whether prediction was correct
        """
        if timeframe not in self.performance_history:
            self.performance_history[timeframe] = []

        accuracy_point = 1.0 if was_correct else 0.0
        self.performance_history[timeframe].append(accuracy_point)

        # Keep only recent history (max 500 trades)
        if len(self.performance_history[timeframe]) > 500:
            self.performance_history[timeframe] = self.performance_history[timeframe][-500:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all timeframes.

        Returns:
            Dict with accuracy stats per timeframe
        """
        summary = {}

        for timeframe in Timeframe:
            history = self.performance_history.get(timeframe, [])

            if history:
                summary[timeframe.value] = {
                    'accuracy': np.mean(history),
                    'total_trades': len(history),
                    'recent_50_accuracy': np.mean(history[-50:]) if len(history) >= 50 else np.mean(history)
                }
            else:
                summary[timeframe.value] = {
                    'accuracy': 0.0,
                    'total_trades': 0,
                    'recent_50_accuracy': 0.0
                }

        return summary

    def get_model_count(self) -> int:
        """Get number of registered models."""
        return len(self.models)

    def has_timeframe(self, timeframe: Timeframe) -> bool:
        """Check if timeframe model is registered."""
        return timeframe in self.models
