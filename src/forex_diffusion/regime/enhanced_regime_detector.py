"""
Enhanced Regime Transition Detection

Extends regime detection from 4 to 6 states with transition detection.
Implements HMM entropy analysis and trading pause logic.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
from hmmlearn import hmm
from scipy import stats


class RegimeState(Enum):
    """Extended regime states"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    TRANSITION = "transition"  # New
    ACCUMULATION_DISTRIBUTION = "accumulation_distribution"  # New


@dataclass
class RegimeDetectionResult:
    """Regime detection result with probabilities"""
    current_regime: RegimeState
    probability: float
    all_probabilities: Dict[RegimeState, float]
    is_transition: bool
    entropy: float  # Probability distribution entropy
    duration: int  # Bars in current regime
    recommended_action: str  # trade, pause, tighten_stops, reduce_size


@dataclass
class TransitionIndicators:
    """Indicators of regime transition"""
    rapid_regime_switches: bool  # Frequent changes in short window
    high_entropy: bool  # Uncertain probability distribution
    cross_timeframe_disagreement: bool  # Different regimes across timeframes
    volatility_spike: bool  # Sudden volatility increase
    transition_score: float  # 0-1, higher = more transitional


class EnhancedRegimeDetector:
    """
    Enhanced HMM-based regime detector with 6 states and transition detection.

    Features:
    - 6 regime states (4 original + transition + accumulation/distribution)
    - Transition state detection via entropy analysis
    - Cross-timeframe regime validation
    - Automatic trading pause during transitions
    - Regime persistence tracking
    """

    def __init__(
        self,
        n_states: int = 6,
        entropy_threshold: float = 1.5,
        transition_lookback: int = 10,
        min_regime_duration: int = 5,
        volatility_spike_threshold: float = 2.0
    ):
        """
        Initialize enhanced regime detector.

        Args:
            n_states: Number of HMM states
            entropy_threshold: Entropy threshold for transition detection
            transition_lookback: Bars to look back for rapid switches
            min_regime_duration: Minimum bars in regime before considering change
            volatility_spike_threshold: Z-score threshold for volatility spike
        """
        self.n_states = n_states
        self.entropy_threshold = entropy_threshold
        self.transition_lookback = transition_lookback
        self.min_regime_duration = min_regime_duration
        self.volatility_spike_threshold = volatility_spike_threshold

        # HMM model
        self.model: Optional[hmm.GaussianHMM] = None
        self.is_fitted = False

        # State mapping (HMM states to regime labels)
        self.state_mapping: Dict[int, RegimeState] = {}

        # Tracking
        self.regime_history: List[Tuple[int, RegimeState, float]] = []  # (timestamp, regime, probability)
        self.current_regime: Optional[RegimeState] = None
        self.regime_start_time: Optional[int] = None

    def fit(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        volume: Optional[np.ndarray] = None,
        n_iter: int = 100
    ):
        """
        Fit HMM model to historical data.

        Args:
            returns: Price returns
            volatility: Volatility measurements
            volume: Optional volume data
            n_iter: Number of EM iterations
        """
        # Prepare features
        if volume is not None:
            features = np.column_stack([returns, volatility, volume])
        else:
            features = np.column_stack([returns, volatility])

        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42
        )

        self.model.fit(features)
        self.is_fitted = True

        # Map HMM states to regime labels based on characteristics
        self._map_states_to_regimes(features)

    def _map_states_to_regimes(self, features: np.ndarray):
        """
        Map HMM hidden states to interpretable regime labels.

        Args:
            features: Feature matrix used for fitting
        """
        if not self.is_fitted or self.model is None:
            return

        # Predict states for training data
        states = self.model.predict(features)

        # For each state, compute characteristics
        state_characteristics = {}

        for state_idx in range(self.n_states):
            mask = states == state_idx
            if np.sum(mask) == 0:
                continue

            state_returns = features[mask, 0]
            state_volatility = features[mask, 1]

            avg_return = np.mean(state_returns)
            avg_volatility = np.mean(state_volatility)
            return_std = np.std(state_returns)

            state_characteristics[state_idx] = {
                'avg_return': avg_return,
                'avg_volatility': avg_volatility,
                'return_std': return_std,
                'count': np.sum(mask)
            }

        # Sort states by characteristics and assign labels
        sorted_states = sorted(
            state_characteristics.items(),
            key=lambda x: (x[1]['avg_volatility'], abs(x[1]['avg_return'])),
            reverse=True
        )

        # Assignment logic:
        # Highest volatility -> HIGH_VOLATILITY
        # High positive returns -> TRENDING_UP
        # High negative returns -> TRENDING_DOWN
        # Low volatility, low returns -> RANGING
        # Medium volatility, mixed returns -> ACCUMULATION_DISTRIBUTION
        # Remaining -> TRANSITION

        if len(sorted_states) >= 6:
            self.state_mapping[sorted_states[0][0]] = RegimeState.HIGH_VOLATILITY
            self.state_mapping[sorted_states[1][0]] = RegimeState.TRANSITION
            self.state_mapping[sorted_states[2][0]] = RegimeState.TRENDING_UP if sorted_states[2][1]['avg_return'] > 0 else RegimeState.TRENDING_DOWN
            self.state_mapping[sorted_states[3][0]] = RegimeState.TRENDING_DOWN if sorted_states[3][1]['avg_return'] < 0 else RegimeState.TRENDING_UP
            self.state_mapping[sorted_states[4][0]] = RegimeState.ACCUMULATION_DISTRIBUTION
            self.state_mapping[sorted_states[5][0]] = RegimeState.RANGING
        else:
            # Fallback if fewer states
            for idx, (state_idx, chars) in enumerate(sorted_states):
                regimes = list(RegimeState)
                self.state_mapping[state_idx] = regimes[idx % len(regimes)]

    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        volume: Optional[np.ndarray] = None,
        timestamp: Optional[int] = None
    ) -> RegimeDetectionResult:
        """
        Detect current market regime.

        Args:
            returns: Recent returns
            volatility: Recent volatility
            volume: Optional volume
            timestamp: Current timestamp

        Returns:
            Regime detection result
        """
        if not self.is_fitted or self.model is None:
            return self._default_result()

        # Prepare features
        if volume is not None:
            features = np.column_stack([returns, volatility, volume])
        else:
            features = np.column_stack([returns, volatility])

        # Get state probabilities
        try:
            log_probs = self.model.score_samples(features)
            state_probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
            state_probs = state_probs / state_probs.sum(axis=1, keepdims=True)
        except Exception:
            return self._default_result()

        # Get most recent probabilities
        current_probs = state_probs[-1]
        most_likely_state = np.argmax(current_probs)
        max_prob = current_probs[most_likely_state]

        # Map to regime
        current_regime = self.state_mapping.get(most_likely_state, RegimeState.RANGING)

        # Calculate entropy
        entropy = self._calculate_entropy(current_probs)

        # Detect transition
        transition_indicators = self._detect_transition(
            features=features,
            state_probs=state_probs,
            entropy=entropy
        )

        # Override regime if transition detected
        if transition_indicators.transition_score > 0.6:
            current_regime = RegimeState.TRANSITION

        # Track regime duration
        if timestamp and (self.current_regime != current_regime or self.regime_start_time is None):
            self.regime_start_time = timestamp
            duration = 0
        elif timestamp and self.regime_start_time:
            duration = timestamp - self.regime_start_time
        else:
            duration = len(self.regime_history)

        # Update history
        if timestamp:
            self.regime_history.append((timestamp, current_regime, max_prob))
            if len(self.regime_history) > 1000:
                self.regime_history.pop(0)

        self.current_regime = current_regime

        # Determine recommended action
        recommended_action = self._determine_action(
            regime=current_regime,
            transition_indicators=transition_indicators,
            duration=duration
        )

        # Build probability dictionary
        all_probs = {}
        for state_idx, prob in enumerate(current_probs):
            regime_label = self.state_mapping.get(state_idx, RegimeState.RANGING)
            all_probs[regime_label] = float(prob)

        return RegimeDetectionResult(
            current_regime=current_regime,
            probability=float(max_prob),
            all_probabilities=all_probs,
            is_transition=transition_indicators.transition_score > 0.6,
            entropy=float(entropy),
            duration=int(duration),
            recommended_action=recommended_action
        )

    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy of probability distribution"""
        # Avoid log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    def _detect_transition(
        self,
        features: np.ndarray,
        state_probs: np.ndarray,
        entropy: float
    ) -> TransitionIndicators:
        """
        Detect if market is in transition state.

        Args:
            features: Feature matrix
            state_probs: State probabilities
            entropy: Current entropy

        Returns:
            Transition indicators
        """
        n_samples = len(features)

        # 1. Check for rapid regime switches
        if n_samples >= self.transition_lookback:
            recent_states = np.argmax(state_probs[-self.transition_lookback:], axis=1)
            n_switches = np.sum(np.diff(recent_states) != 0)
            rapid_switches = n_switches >= (self.transition_lookback * 0.4)
        else:
            rapid_switches = False

        # 2. Check for high entropy
        high_entropy = entropy > self.entropy_threshold

        # 3. Cross-timeframe disagreement (simulated here, would need multi-TF data)
        cross_tf_disagreement = False  # Placeholder

        # 4. Volatility spike
        if n_samples >= 20:
            recent_volatility = features[-self.transition_lookback:, 1]
            historical_volatility = features[-50:-self.transition_lookback, 1]
            vol_zscore = (np.mean(recent_volatility) - np.mean(historical_volatility)) / (np.std(historical_volatility) + 1e-10)
            volatility_spike = abs(vol_zscore) > self.volatility_spike_threshold
        else:
            volatility_spike = False

        # Calculate transition score
        indicators = [rapid_switches, high_entropy, cross_tf_disagreement, volatility_spike]
        transition_score = np.sum(indicators) / len(indicators)

        return TransitionIndicators(
            rapid_regime_switches=rapid_switches,
            high_entropy=high_entropy,
            cross_timeframe_disagreement=cross_tf_disagreement,
            volatility_spike=volatility_spike,
            transition_score=float(transition_score)
        )

    def _determine_action(
        self,
        regime: RegimeState,
        transition_indicators: TransitionIndicators,
        duration: int
    ) -> str:
        """
        Determine recommended trading action based on regime.

        Args:
            regime: Current regime
            transition_indicators: Transition indicators
            duration: Duration in current regime

        Returns:
            Recommended action string
        """
        # Transition state -> pause trading
        if regime == RegimeState.TRANSITION or transition_indicators.transition_score > 0.7:
            return "pause_trading"

        # Early in regime -> reduce size
        if duration < self.min_regime_duration:
            return "reduce_position_size"

        # Volatile regime -> tighten stops
        if regime == RegimeState.HIGH_VOLATILITY:
            return "tighten_stops"

        # Accumulation/Distribution -> monitor closely
        if regime == RegimeState.ACCUMULATION_DISTRIBUTION:
            return "monitor_closely"

        # Normal trending or ranging -> trade normally
        return "trade_normally"

    def _default_result(self) -> RegimeDetectionResult:
        """Return default result when model not fitted"""
        return RegimeDetectionResult(
            current_regime=RegimeState.RANGING,
            probability=0.5,
            all_probabilities={regime: 1.0/6 for regime in RegimeState},
            is_transition=False,
            entropy=0.0,
            duration=0,
            recommended_action="pause_trading"
        )

    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about regime history.

        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {}

        regimes = [r[1] for r in self.regime_history]
        regime_counts = {}
        for regime in RegimeState:
            regime_counts[regime.value] = regimes.count(regime)

        total = len(regimes)
        regime_percentages = {k: (v / total) * 100 for k, v in regime_counts.items()}

        return {
            'total_observations': total,
            'regime_counts': regime_counts,
            'regime_percentages': regime_percentages,
            'current_regime': self.current_regime.value if self.current_regime else None,
            'transition_count': regime_counts.get('transition', 0)
        }
