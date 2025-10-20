"""
Cross-Timeframe Regime Coherence Validation

Validates that regime classifications are coherent across multiple timeframes.
Higher timeframe regimes dominate lower timeframe behavior.

Example coherence rules:
- 4H trending + 1H ranging = OK (pullback within trend)
- 4H ranging + 1H trending = WARNING (possible breakout)
- 4H trending up + 1H trending down = CONFLICT (counter-trend)

Reference: "Multiple Time Frame Analysis" by Aronson (2011)
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

from .hmm_detector import HMMRegimeDetector, RegimeType


class CoherenceLevel(Enum):
    """Coherence quality levels"""
    HIGH = "high"          # >80% - Fully aligned, high confidence
    MEDIUM = "medium"      # 50-80% - Acceptable alignment
    LOW = "low"            # 20-50% - Weak alignment, caution
    CONFLICT = "conflict"  # <20% - Contradictory, avoid trading


@dataclass
class TimeframeRegime:
    """Regime information for a specific timeframe"""
    timeframe: str
    regime_type: RegimeType
    probability: float
    duration: int  # Bars in current regime
    transition_prob: float


@dataclass
class CoherenceResult:
    """Result of coherence validation across timeframes"""
    primary_timeframe: str
    regimes: Dict[str, TimeframeRegime]  # timeframe -> regime

    # Coherence metrics
    coherence_score: float  # 0-100
    coherence_level: CoherenceLevel

    # Compatibility matrix
    compatibility_matrix: Dict[Tuple[str, str], float]

    # Trading recommendation
    position_size_multiplier: float  # 0-1, based on coherence
    trade_recommendation: str  # "full", "reduced", "avoid"

    # Detailed analysis
    conflicts: List[str]
    warnings: List[str]
    notes: List[str]


class CoherenceValidator:
    """
    Validates cross-timeframe regime coherence.

    Maintains compatibility rules between regime types and enforces
    hierarchical timeframe constraints (higher TF dominates).
    """

    def __init__(self):
        """Initialize coherence validator with compatibility rules"""

        # Compatibility matrix: (higher_TF_regime, lower_TF_regime) -> score [0, 1]
        # 1.0 = fully compatible
        # 0.5 = neutral/acceptable
        # 0.0 = incompatible/conflict

        self.compatibility_rules = self._build_compatibility_rules()

        # Timeframe hierarchy (higher index = higher timeframe)
        self.timeframe_hierarchy = [
            "1m", "5m", "15m", "30m", "1H", "2H", "4H", "8H", "1D", "1W", "1M"
        ]

    def _build_compatibility_rules(self) -> Dict[Tuple[RegimeType, RegimeType], float]:
        """
        Build compatibility matrix for regime combinations.

        Returns:
            Dict mapping (higher_TF, lower_TF) -> compatibility score
        """
        rules = {}

        # Same regime = fully compatible
        for regime in RegimeType:
            if regime != RegimeType.UNKNOWN:
                rules[(regime, regime)] = 1.0

        # TRENDING_UP on higher TF
        rules[(RegimeType.TRENDING_UP, RegimeType.RANGING)] = 0.8    # Pullback OK
        rules[(RegimeType.TRENDING_UP, RegimeType.VOLATILE)] = 0.6   # Acceptable
        rules[(RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN)] = 0.1  # Conflict!

        # TRENDING_DOWN on higher TF
        rules[(RegimeType.TRENDING_DOWN, RegimeType.RANGING)] = 0.8
        rules[(RegimeType.TRENDING_DOWN, RegimeType.VOLATILE)] = 0.6
        rules[(RegimeType.TRENDING_DOWN, RegimeType.TRENDING_UP)] = 0.1  # Conflict!

        # RANGING on higher TF
        rules[(RegimeType.RANGING, RegimeType.TRENDING_UP)] = 0.5    # Possible breakout
        rules[(RegimeType.RANGING, RegimeType.TRENDING_DOWN)] = 0.5  # Possible breakout
        rules[(RegimeType.RANGING, RegimeType.VOLATILE)] = 0.7       # Expected

        # VOLATILE on higher TF
        rules[(RegimeType.VOLATILE, RegimeType.TRENDING_UP)] = 0.6
        rules[(RegimeType.VOLATILE, RegimeType.TRENDING_DOWN)] = 0.6
        rules[(RegimeType.VOLATILE, RegimeType.RANGING)] = 0.4

        # Reverse mappings (lower TF influences higher TF interpretation)
        # Generally lower weight

        rules[(RegimeType.RANGING, RegimeType.TRENDING_UP)] = 0.5
        rules[(RegimeType.RANGING, RegimeType.TRENDING_DOWN)] = 0.5
        rules[(RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN)] = 0.1
        rules[(RegimeType.TRENDING_DOWN, RegimeType.TRENDING_UP)] = 0.1

        # Fill symmetric cases
        for (r1, r2), score in list(rules.items()):
            if (r2, r1) not in rules:
                # Symmetric but slightly lower weight if going down hierarchy
                rules[(r2, r1)] = score * 0.9

        return rules

    def validate_coherence(
        self,
        regimes: Dict[str, TimeframeRegime],
        primary_timeframe: str,
    ) -> CoherenceResult:
        """
        Validate coherence across multiple timeframes.

        Args:
            regimes: Dictionary mapping timeframe -> TimeframeRegime
            primary_timeframe: The trading timeframe (focal point)

        Returns:
            CoherenceResult with validation details
        """
        if len(regimes) < 2:
            logger.warning("Need at least 2 timeframes for coherence validation")
            return self._create_neutral_result(regimes, primary_timeframe)

        # Sort timeframes by hierarchy
        sorted_timeframes = self._sort_timeframes(list(regimes.keys()))

        # Build compatibility matrix for all pairs
        compatibility_matrix = {}
        pairwise_scores = []

        for i, tf1 in enumerate(sorted_timeframes):
            for j, tf2 in enumerate(sorted_timeframes):
                if i == j:
                    continue

                regime1 = regimes[tf1].regime_type
                regime2 = regimes[tf2].regime_type

                # Higher TF should dominate
                if i > j:  # tf1 is higher timeframe
                    key = (regime1, regime2)
                else:  # tf2 is higher timeframe
                    key = (regime2, regime1)

                compatibility = self.compatibility_rules.get(key, 0.5)
                compatibility_matrix[(tf1, tf2)] = compatibility

                # Weight by probability confidence
                conf1 = regimes[tf1].probability
                conf2 = regimes[tf2].probability
                weighted_score = compatibility * conf1 * conf2

                pairwise_scores.append(weighted_score)

        # Calculate overall coherence score
        if pairwise_scores:
            coherence_score = np.mean(pairwise_scores) * 100  # Scale to 0-100
        else:
            coherence_score = 50.0  # Neutral

        # Classify coherence level
        if coherence_score >= 80:
            coherence_level = CoherenceLevel.HIGH
        elif coherence_score >= 50:
            coherence_level = CoherenceLevel.MEDIUM
        elif coherence_score >= 20:
            coherence_level = CoherenceLevel.LOW
        else:
            coherence_level = CoherenceLevel.CONFLICT

        # Position sizing multiplier based on coherence
        position_multiplier = np.clip(coherence_score / 100, 0.0, 1.0)

        # Trading recommendation
        if coherence_level == CoherenceLevel.HIGH:
            trade_rec = "full"
        elif coherence_level == CoherenceLevel.MEDIUM:
            trade_rec = "reduced"
        elif coherence_level == CoherenceLevel.LOW:
            trade_rec = "minimal"
        else:
            trade_rec = "avoid"

        # Identify conflicts and warnings
        conflicts = []
        warnings = []
        notes = []

        for (tf1, tf2), compat in compatibility_matrix.items():
            if compat < 0.3:
                conflicts.append(
                    f"Conflict between {tf1} ({regimes[tf1].regime_type.value}) "
                    f"and {tf2} ({regimes[tf2].regime_type.value})"
                )
            elif compat < 0.6:
                warnings.append(
                    f"Weak alignment: {tf1} ({regimes[tf1].regime_type.value}) "
                    f"vs {tf2} ({regimes[tf2].regime_type.value})"
                )

        # Check if higher TFs are aligned with primary
        primary_idx = sorted_timeframes.index(primary_timeframe)
        higher_tfs = sorted_timeframes[primary_idx + 1:]

        if higher_tfs:
            primary_regime = regimes[primary_timeframe].regime_type

            for higher_tf in higher_tfs:
                higher_regime = regimes[higher_tf].regime_type

                if higher_regime != primary_regime:
                    notes.append(
                        f"Higher TF {higher_tf} in {higher_regime.value}, "
                        f"primary {primary_timeframe} in {primary_regime.value}"
                    )

        result = CoherenceResult(
            primary_timeframe=primary_timeframe,
            regimes=regimes,
            coherence_score=float(coherence_score),
            coherence_level=coherence_level,
            compatibility_matrix=compatibility_matrix,
            position_size_multiplier=float(position_multiplier),
            trade_recommendation=trade_rec,
            conflicts=conflicts,
            warnings=warnings,
            notes=notes,
        )

        logger.info(
            f"Coherence validation: score={coherence_score:.1f}, "
            f"level={coherence_level.value}, recommendation={trade_rec}"
        )

        if conflicts:
            logger.warning(f"Conflicts detected: {conflicts}")

        return result

    def _sort_timeframes(self, timeframes: List[str]) -> List[str]:
        """
        Sort timeframes by hierarchy (lowest to highest).

        Args:
            timeframes: List of timeframe strings

        Returns:
            Sorted list of timeframes
        """
        def get_hierarchy_index(tf: str) -> int:
            if tf in self.timeframe_hierarchy:
                return self.timeframe_hierarchy.index(tf)
            else:
                # Unknown timeframe, try to parse
                logger.warning(f"Unknown timeframe {tf}, using default position")
                return len(self.timeframe_hierarchy) // 2

        return sorted(timeframes, key=get_hierarchy_index)

    def _create_neutral_result(
        self,
        regimes: Dict[str, TimeframeRegime],
        primary_timeframe: str,
    ) -> CoherenceResult:
        """Create neutral result when validation cannot be performed"""
        return CoherenceResult(
            primary_timeframe=primary_timeframe,
            regimes=regimes,
            coherence_score=50.0,
            coherence_level=CoherenceLevel.MEDIUM,
            compatibility_matrix={},
            position_size_multiplier=0.5,
            trade_recommendation="reduced",
            conflicts=[],
            warnings=["Insufficient timeframes for coherence validation"],
            notes=[],
        )

    def get_recommended_position_size(
        self,
        base_position_size: float,
        coherence_result: CoherenceResult,
    ) -> float:
        """
        Calculate recommended position size based on coherence.

        Args:
            base_position_size: Base position size (e.g., 1.0 lot)
            coherence_result: Coherence validation result

        Returns:
            Adjusted position size
        """
        adjusted_size = base_position_size * coherence_result.position_size_multiplier

        logger.debug(
            f"Position sizing: base={base_position_size}, "
            f"multiplier={coherence_result.position_size_multiplier:.2f}, "
            f"adjusted={adjusted_size}"
        )

        return adjusted_size


# Convenience function
def validate_multi_timeframe_regimes(
    hmm_detector: HMMRegimeDetector,
    data_by_timeframe: Dict[str, pd.DataFrame],
    primary_timeframe: str,
    lookback: int = 100,
) -> CoherenceResult:
    """
    Quick multi-timeframe coherence validation.

    Args:
        hmm_detector: Fitted HMM regime detector
        data_by_timeframe: Dict mapping timeframe -> OHLCV DataFrame
        primary_timeframe: Trading timeframe
        lookback: Lookback period for regime detection

    Returns:
        CoherenceResult
    """
    validator = CoherenceValidator()

    # Detect regime for each timeframe
    regimes = {}

    for timeframe, df in data_by_timeframe.items():
        # Get current regime
        regime_state = hmm_detector.get_current_regime(df.tail(lookback))

        regimes[timeframe] = TimeframeRegime(
            timeframe=timeframe,
            regime_type=regime_state.regime,
            probability=regime_state.probability,
            duration=regime_state.duration,
            transition_prob=regime_state.transition_prob,
        )

    # Validate coherence
    result = validator.validate_coherence(regimes, primary_timeframe)

    return result
