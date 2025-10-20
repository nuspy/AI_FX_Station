"""
Hidden Markov Model (HMM) Regime Detection

Identifies market regimes (trending, ranging, volatile) using HMM:
- Smooth regime transitions
- Probabilistic state assignment
- Adaptive to market conditions

Reference: "Regime Switching Models" by Hamilton (1989)
"""
from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from hmmlearn import hmm
from loguru import logger


class RegimeType(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state"""
    regime: RegimeType
    probability: float  # Confidence in regime assignment
    duration: int  # Bars in current regime
    transition_prob: float  # Probability of transition next bar


class HMMRegimeDetector:
    """
    HMM-based regime detection for market states.

    Uses Gaussian HMM to identify latent market regimes from observables.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        min_history: int = 100,
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            covariance_type: Type of covariance matrix ("full", "diag", "spherical")
            n_iter: Number of EM iterations
            random_state: Random seed
            min_history: Minimum bars needed for training
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.min_history = min_history

        # HMM model
        self.model: Optional[hmm.GaussianHMM] = None

        # Regime mapping (learned from data)
        self.regime_mapping: Dict[int, RegimeType] = {}

        # Training metadata
        self.is_fitted = False
        self.feature_names: List[str] = []

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for HMM from OHLCV data.

        Features:
        - Returns (log)
        - Volatility (rolling std)
        - Range (high-low)
        - Volume change
        - Directional movement
        """
        features = []

        # 1. Log returns
        returns = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        features.append(returns.values)

        # 2. Volatility (20-bar rolling std of returns)
        volatility = returns.rolling(20, min_periods=1).std().fillna(0)
        features.append(volatility.values)

        # 3. Normalized range
        range_norm = (df["high"] - df["low"]) / df["close"].fillna(0)
        features.append(range_norm.values)

        # 4. Volume change (log ratio)
        if "volume" in df.columns:
            vol_change = np.log(
                df["volume"] / df["volume"].shift(1).replace(0, 1)
            ).fillna(0)
            features.append(vol_change.values)

        # 5. Directional movement (ADX-like)
        plus_dm = (df["high"] - df["high"].shift(1)).clip(lower=0)
        minus_dm = (df["low"].shift(1) - df["low"]).clip(lower=0)
        tr = df["high"] - df["low"]
        directional = ((plus_dm - minus_dm) / (tr + 1e-10)).fillna(0)
        features.append(directional.values)

        # Stack features
        X = np.column_stack(features)

        self.feature_names = ["returns", "volatility", "range", "volume_change", "directional"]

        return X

    def fit(self, df: pd.DataFrame):
        """
        Fit HMM model to historical data.

        Args:
            df: DataFrame with OHLCV data
        """
        if len(df) < self.min_history:
            raise ValueError(
                f"Need at least {self.min_history} bars, got {len(df)}"
            )

        # Extract features
        X = self._extract_features(df)

        # Remove NaN/Inf
        mask = np.isfinite(X).all(axis=1)
        X_clean = X[mask]

        if len(X_clean) < self.min_history:
            raise ValueError("Not enough valid data after cleaning")

        # Initialize HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        # Fit model
        logger.info(f"Fitting HMM with {self.n_regimes} regimes on {len(X_clean)} samples")
        self.model.fit(X_clean)

        # Predict states for training data
        states = self.model.predict(X_clean)

        # Map states to regime types based on characteristics
        self._map_regimes(X_clean, states)

        self.is_fitted = True
        logger.info(f"HMM fitted successfully. Regime mapping: {self.regime_mapping}")

    def _map_regimes(self, X: np.ndarray, states: np.ndarray):
        """
        Map HMM states to regime types based on feature characteristics.

        Args:
            X: Feature matrix
            states: HMM state predictions
        """
        for state_id in range(self.n_regimes):
            # Get samples in this state
            mask = states == state_id
            if mask.sum() == 0:
                self.regime_mapping[state_id] = RegimeType.UNKNOWN
                continue

            X_state = X[mask]

            # Calculate state characteristics
            avg_returns = X_state[:, 0].mean()  # returns
            avg_vol = X_state[:, 1].mean()      # volatility
            avg_range = X_state[:, 2].mean()    # range

            # Classify regime based on characteristics
            if avg_vol > 0.015:  # High volatility
                regime = RegimeType.VOLATILE
            elif abs(avg_returns) > 0.002:  # Strong directional movement
                if avg_returns > 0:
                    regime = RegimeType.TRENDING_UP
                else:
                    regime = RegimeType.TRENDING_DOWN
            else:  # Low volatility, low returns
                regime = RegimeType.RANGING

            self.regime_mapping[state_id] = regime

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime states for data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with regime predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract features
        X = self._extract_features(df)

        # Remove NaN/Inf
        mask = np.isfinite(X).all(axis=1)
        X_clean = X[mask]

        if len(X_clean) == 0:
            raise ValueError("No valid data after cleaning")

        # Predict states
        states = self.model.predict(X_clean)

        # Get state probabilities
        posteriors = self.model.predict_proba(X_clean)

        # Map to regime types
        regimes = [self.regime_mapping.get(s, RegimeType.UNKNOWN) for s in states]

        # Build result DataFrame
        result = pd.DataFrame(index=df.index)
        result["regime_state"] = RegimeType.UNKNOWN.value
        result["regime_probability"] = 0.0

        # Fill valid indices
        valid_indices = df.index[mask]
        result.loc[valid_indices, "regime_state"] = [r.value for r in regimes]
        result.loc[valid_indices, "regime_probability"] = posteriors.max(axis=1)

        # Add binary regime indicators
        result["regime_trending_up"] = (result["regime_state"] == RegimeType.TRENDING_UP.value).astype(int)
        result["regime_trending_down"] = (result["regime_state"] == RegimeType.TRENDING_DOWN.value).astype(int)
        result["regime_ranging"] = (result["regime_state"] == RegimeType.RANGING.value).astype(int)
        result["regime_volatile"] = (result["regime_state"] == RegimeType.VOLATILE.value).astype(int)

        # Add transition probabilities
        result["regime_transition_prob"] = 0.0
        for i in range(len(states) - 1):
            if states[i] != states[i + 1]:
                # Transition occurred
                result.iloc[i, result.columns.get_loc("regime_transition_prob")] = 1.0 - posteriors[i, states[i]]

        return result

    def get_current_regime(
        self,
        df: pd.DataFrame,
        lookback: int = 5,
    ) -> RegimeState:
        """
        Get current regime state with statistics.

        Args:
            df: DataFrame with recent OHLCV data
            lookback: Number of recent bars to consider

        Returns:
            RegimeState with current regime info
        """
        if not self.is_fitted:
            return RegimeState(
                regime=RegimeType.UNKNOWN,
                probability=0.0,
                duration=0,
                transition_prob=0.0,
            )

        # Predict on recent data
        result = self.predict(df.tail(lookback * 2))

        # Get most recent regime
        recent_regimes = result["regime_state"].tail(lookback).values
        current_regime_str = recent_regimes[-1]
        current_regime = RegimeType(current_regime_str)

        # Calculate duration (consecutive bars in same regime)
        duration = 1
        for i in range(len(recent_regimes) - 2, -1, -1):
            if recent_regimes[i] == current_regime_str:
                duration += 1
            else:
                break

        # Get probability and transition prob
        probability = float(result["regime_probability"].iloc[-1])
        transition_prob = float(result["regime_transition_prob"].iloc[-1])

        return RegimeState(
            regime=current_regime,
            probability=probability,
            duration=duration,
            transition_prob=transition_prob,
        )

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get HMM state transition matrix.

        Returns:
            Transition probability matrix (n_regimes x n_regimes)
        """
        if not self.is_fitted:
            return np.zeros((self.n_regimes, self.n_regimes))

        return self.model.transmat_

    def get_regime_statistics(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Get statistics about regime distribution.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with regime statistics
        """
        if not self.is_fitted:
            return {}

        result = self.predict(df)

        # Count regime occurrences
        regime_counts = result["regime_state"].value_counts().to_dict()

        # Average duration per regime
        regime_durations = {}
        for regime_type in RegimeType:
            mask = result["regime_state"] == regime_type.value
            if mask.sum() == 0:
                continue

            # Find consecutive runs
            runs = []
            current_run = 0
            for val in mask.values:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)

            regime_durations[regime_type.value] = {
                "avg_duration": np.mean(runs) if runs else 0,
                "max_duration": max(runs) if runs else 0,
            }

        # Current regime
        current = self.get_current_regime(df)

        return {
            "regime_counts": regime_counts,
            "regime_durations": regime_durations,
            "current_regime": current.regime.value,
            "current_probability": current.probability,
            "current_duration": current.duration,
            "transition_matrix": self.get_transition_matrix().tolist(),
        }


# Convenience function
def detect_regimes(
    df: pd.DataFrame,
    n_regimes: int = 4,
) -> pd.DataFrame:
    """
    Quick regime detection on DataFrame.

    Usage:
        regime_features = detect_regimes(df)
        df = pd.concat([df, regime_features], axis=1)
    """
    detector = HMMRegimeDetector(n_regimes=n_regimes)
    detector.fit(df)
    return detector.predict(df)
