"""
Multi-Timeframe Pattern Detection and Composite Pattern Recognition.

Implements advanced pattern detection across multiple timeframes with
optimal combinations for maximum profit and minimum risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from ..cache import cache_decorator, get_pattern_cache


class PatternAlignment(Enum):
    """Pattern alignment across timeframes"""
    BULLISH_CONFLUENCE = "bullish_confluence"
    BEARISH_CONFLUENCE = "bearish_confluence"
    MIXED_SIGNALS = "mixed_signals"
    NEUTRAL = "neutral"


@dataclass
class TimeframePattern:
    """Pattern detected on specific timeframe"""
    pattern_key: str
    timeframe: str
    confidence: float
    direction: str
    strength: float
    target_price: Optional[float] = None
    failure_price: Optional[float] = None
    formation_time: Optional[datetime] = None
    pattern_type: str = "unknown"


@dataclass
class MultiTimeframeSignal:
    """Combined signal from multiple timeframes"""
    primary_pattern: TimeframePattern
    supporting_patterns: List[TimeframePattern] = field(default_factory=list)
    alignment: PatternAlignment = PatternAlignment.NEUTRAL
    combined_confidence: float = 0.0
    combined_strength: float = 0.0
    risk_reward_ratio: float = 0.0
    strategy_type: str = "unknown"  # scalping, intraday, swing
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class CompositePattern:
    """Complex pattern formed by multiple sub-patterns"""
    name: str
    primary_pattern: str
    secondary_pattern: str
    confidence: float
    description: str
    expected_move: float
    risk_level: str  # low, medium, high


class MultiTimeframeAnalyzer:
    """
    Analyzes patterns across multiple timeframes to identify optimal
    trading opportunities with best profit/risk characteristics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('patterns', {}).get('multi_timeframe', {})
        self.enabled = self.config.get('enabled', True)

        # Timeframe combinations optimized for performance
        self.combinations = {
            combo['name']: {
                'timeframes': combo['timeframes'],
                'weights': combo['weights']
            }
            for combo in self.config.get('combinations', [])
        }

        # Default combinations if config is empty
        if not self.combinations:
            self.combinations = {
                'scalping': {
                    'timeframes': ['1m', '5m', '15m'],
                    'weights': [0.2, 0.3, 0.5]
                },
                'intraday': {
                    'timeframes': ['15m', '1h', '4h'],
                    'weights': [0.2, 0.4, 0.4]
                },
                'swing': {
                    'timeframes': ['1h', '4h', '1d'],
                    'weights': [0.3, 0.4, 0.3]
                }
            }

        # Composite pattern definitions
        self.composite_patterns = self._initialize_composite_patterns()

        # Performance thresholds for optimal signals
        self.min_confluence_confidence = 0.7
        self.min_risk_reward_ratio = 2.0

    def _initialize_composite_patterns(self) -> Dict[str, CompositePattern]:
        """Initialize composite pattern definitions"""
        return {
            'head_shoulders_in_triangle': CompositePattern(
                name="Head & Shoulders in Triangle",
                primary_pattern="head_shoulders",
                secondary_pattern="triangle",
                confidence=0.85,
                description="H&S pattern forming within triangle consolidation - high reversal probability",
                expected_move=0.15,  # 15% move expected
                risk_level="medium"
            ),
            'flag_in_wedge': CompositePattern(
                name="Flag in Wedge",
                primary_pattern="flag",
                secondary_pattern="wedge",
                confidence=0.80,
                description="Flag continuation pattern within wedge structure - strong momentum continuation",
                expected_move=0.12,
                risk_level="low"
            ),
            'double_top_with_divergence': CompositePattern(
                name="Double Top with Divergence",
                primary_pattern="double_top",
                secondary_pattern="momentum_divergence",
                confidence=0.88,
                description="Double top with momentum divergence - very reliable reversal signal",
                expected_move=0.18,
                risk_level="low"
            ),
            'elliott_wave_with_harmonics': CompositePattern(
                name="Elliott Wave with Harmonic Confluence",
                primary_pattern="elliott_wave",
                secondary_pattern="harmonic",
                confidence=0.82,
                description="Elliott wave completion at harmonic ratio level - precise entry opportunity",
                expected_move=0.20,
                risk_level="medium"
            ),
            'triangle_breakout_flag': CompositePattern(
                name="Triangle Breakout with Flag",
                primary_pattern="triangle",
                secondary_pattern="flag",
                confidence=0.78,
                description="Triangle breakout followed by flag consolidation - trend continuation",
                expected_move=0.14,
                risk_level="low"
            ),
            'harmonic_at_structure': CompositePattern(
                name="Harmonic at Key Structure",
                primary_pattern="harmonic",
                secondary_pattern="support_resistance",
                confidence=0.85,
                description="Harmonic pattern completion at major support/resistance level",
                expected_move=0.16,
                risk_level="medium"
            )
        }

    @cache_decorator('multi_timeframe_analysis', ttl=1800)  # 30 minutes cache
    def analyze_multi_timeframe_patterns(self,
                                       patterns_by_timeframe: Dict[str, List[TimeframePattern]],
                                       asset: str,
                                       current_price: float) -> List[MultiTimeframeSignal]:
        """
        Analyze patterns across multiple timeframes and generate optimal signals.

        Args:
            patterns_by_timeframe: Dict mapping timeframe to detected patterns
            asset: Asset symbol
            current_price: Current market price

        Returns:
            List of multi-timeframe signals sorted by profit/risk potential
        """
        if not self.enabled or not patterns_by_timeframe:
            return []

        try:
            signals = []

            # Analyze each strategy type (scalping, intraday, swing)
            for strategy_name, strategy_config in self.combinations.items():
                timeframes = strategy_config['timeframes']
                weights = strategy_config['weights']

                # Get patterns for this strategy's timeframes
                strategy_patterns = {}
                for tf in timeframes:
                    strategy_patterns[tf] = patterns_by_timeframe.get(tf, [])

                # Generate signals for this strategy
                strategy_signals = self._generate_strategy_signals(
                    strategy_patterns, timeframes, weights, strategy_name, current_price
                )

                signals.extend(strategy_signals)

            # Look for composite patterns
            composite_signals = self._detect_composite_patterns(patterns_by_timeframe, current_price)
            signals.extend(composite_signals)

            # Filter and rank by profit/risk potential
            optimal_signals = self._rank_signals_by_performance(signals)

            # Cache results
            cache = get_pattern_cache()
            if cache:
                cache.cache_multi_timeframe_analysis(
                    asset, list(patterns_by_timeframe.keys()), 'signals', optimal_signals
                )

            return optimal_signals

        except Exception as e:
            logger.error(f"Error analyzing multi-timeframe patterns: {e}")
            return []

    def _generate_strategy_signals(self,
                                 patterns_by_tf: Dict[str, List[TimeframePattern]],
                                 timeframes: List[str],
                                 weights: List[float],
                                 strategy_name: str,
                                 current_price: float) -> List[MultiTimeframeSignal]:
        """Generate signals for specific strategy (scalping/intraday/swing)"""
        signals = []

        if not patterns_by_tf:
            return signals

        try:
            # Find the primary timeframe (highest weight)
            primary_tf_idx = weights.index(max(weights))
            primary_tf = timeframes[primary_tf_idx]
            primary_patterns = patterns_by_tf.get(primary_tf, [])

            for primary_pattern in primary_patterns:
                # Look for supporting patterns in other timeframes
                supporting_patterns = []
                alignment_scores = []

                for i, tf in enumerate(timeframes):
                    if tf != primary_tf:
                        tf_patterns = patterns_by_tf.get(tf, [])
                        supporting, alignment_score = self._find_supporting_patterns(
                            primary_pattern, tf_patterns, weights[i]
                        )
                        supporting_patterns.extend(supporting)
                        alignment_scores.append(alignment_score)

                # Calculate confluence
                if alignment_scores:
                    alignment = self._determine_alignment(primary_pattern, supporting_patterns)
                    combined_confidence = self._calculate_combined_confidence(
                        primary_pattern, supporting_patterns, weights
                    )
                    combined_strength = self._calculate_combined_strength(
                        primary_pattern, supporting_patterns, weights
                    )

                    # Only create signal if meets minimum confluence requirements
                    if combined_confidence >= self.min_confluence_confidence:
                        # Calculate risk/reward
                        entry_price, stop_loss, take_profit = self._calculate_entry_levels(
                            primary_pattern, supporting_patterns, current_price
                        )

                        risk_reward = self._calculate_risk_reward_ratio(
                            entry_price, stop_loss, take_profit
                        )

                        # Only include signals with good risk/reward
                        if risk_reward >= self.min_risk_reward_ratio:
                            signal = MultiTimeframeSignal(
                                primary_pattern=primary_pattern,
                                supporting_patterns=supporting_patterns,
                                alignment=alignment,
                                combined_confidence=combined_confidence,
                                combined_strength=combined_strength,
                                risk_reward_ratio=risk_reward,
                                strategy_type=strategy_name,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit
                            )
                            signals.append(signal)

        except Exception as e:
            logger.debug(f"Error generating strategy signals: {e}")

        return signals

    def _find_supporting_patterns(self,
                                primary_pattern: TimeframePattern,
                                tf_patterns: List[TimeframePattern],
                                weight: float) -> Tuple[List[TimeframePattern], float]:
        """Find patterns that support the primary pattern direction"""
        supporting = []
        alignment_score = 0.0

        try:
            primary_direction = primary_pattern.direction.lower()

            for pattern in tf_patterns:
                pattern_direction = pattern.direction.lower()

                # Check direction alignment
                if self._directions_align(primary_direction, pattern_direction):
                    supporting.append(pattern)
                    # Score based on confidence and weight
                    alignment_score += pattern.confidence * weight

            return supporting, alignment_score

        except Exception:
            return [], 0.0

    def _directions_align(self, dir1: str, dir2: str) -> bool:
        """Check if two pattern directions align"""
        bullish_terms = ['bull', 'bullish', 'up', 'long', 'buy']
        bearish_terms = ['bear', 'bearish', 'down', 'short', 'sell']

        dir1_bullish = any(term in dir1 for term in bullish_terms)
        dir1_bearish = any(term in dir1 for term in bearish_terms)

        dir2_bullish = any(term in dir2 for term in bullish_terms)
        dir2_bearish = any(term in dir2 for term in bearish_terms)

        # Alignment if both bullish or both bearish
        return (dir1_bullish and dir2_bullish) or (dir1_bearish and dir2_bearish)

    def _determine_alignment(self,
                           primary: TimeframePattern,
                           supporting: List[TimeframePattern]) -> PatternAlignment:
        """Determine overall pattern alignment"""
        if not supporting:
            return PatternAlignment.NEUTRAL

        try:
            primary_bullish = any(term in primary.direction.lower()
                                for term in ['bull', 'bullish', 'up', 'long', 'buy'])

            supporting_bullish = sum(1 for p in supporting
                                   if any(term in p.direction.lower()
                                         for term in ['bull', 'bullish', 'up', 'long', 'buy']))

            supporting_bearish = len(supporting) - supporting_bullish

            if primary_bullish:
                if supporting_bullish >= supporting_bearish:
                    return PatternAlignment.BULLISH_CONFLUENCE
                else:
                    return PatternAlignment.MIXED_SIGNALS
            else:
                if supporting_bearish >= supporting_bullish:
                    return PatternAlignment.BEARISH_CONFLUENCE
                else:
                    return PatternAlignment.MIXED_SIGNALS

        except Exception:
            return PatternAlignment.NEUTRAL

    def _calculate_combined_confidence(self,
                                     primary: TimeframePattern,
                                     supporting: List[TimeframePattern],
                                     weights: List[float]) -> float:
        """Calculate weighted combined confidence"""
        try:
            # Primary pattern gets the highest weight
            max_weight = max(weights) if weights else 0.5
            combined_confidence = primary.confidence * max_weight

            # Add supporting patterns with their respective weights
            for i, pattern in enumerate(supporting):
                if i < len(weights):
                    combined_confidence += pattern.confidence * weights[i]

            # Normalize by total weights
            total_weight = max_weight + sum(weights[:len(supporting)])
            if total_weight > 0:
                combined_confidence /= total_weight

            return min(combined_confidence, 1.0)

        except Exception:
            return primary.confidence

    def _calculate_combined_strength(self,
                                   primary: TimeframePattern,
                                   supporting: List[TimeframePattern],
                                   weights: List[float]) -> float:
        """Calculate weighted combined strength"""
        try:
            max_weight = max(weights) if weights else 0.5
            combined_strength = primary.strength * max_weight

            for i, pattern in enumerate(supporting):
                if i < len(weights):
                    combined_strength += pattern.strength * weights[i]

            total_weight = max_weight + sum(weights[:len(supporting)])
            if total_weight > 0:
                combined_strength /= total_weight

            return min(combined_strength, 1.0)

        except Exception:
            return primary.strength

    def _calculate_entry_levels(self,
                              primary: TimeframePattern,
                              supporting: List[TimeframePattern],
                              current_price: float) -> Tuple[float, float, float]:
        """Calculate optimal entry, stop loss, and take profit levels"""
        try:
            # Use primary pattern's target and failure prices as base
            entry_price = current_price

            # Stop loss based on failure price or technical levels
            if primary.failure_price:
                stop_loss = primary.failure_price
            else:
                # Default stop loss based on pattern type and direction
                if 'bull' in primary.direction.lower():
                    stop_loss = current_price * 0.98  # 2% stop for bullish
                else:
                    stop_loss = current_price * 1.02  # 2% stop for bearish

            # Take profit based on target price or risk/reward ratio
            if primary.target_price:
                take_profit = primary.target_price
            else:
                # Calculate take profit for minimum 2:1 risk/reward
                risk = abs(entry_price - stop_loss)
                if 'bull' in primary.direction.lower():
                    take_profit = entry_price + (risk * 2.5)  # 2.5:1 risk/reward
                else:
                    take_profit = entry_price - (risk * 2.5)

            return entry_price, stop_loss, take_profit

        except Exception:
            # Fallback levels
            return current_price, current_price * 0.98, current_price * 1.04

    def _calculate_risk_reward_ratio(self,
                                   entry: float,
                                   stop_loss: float,
                                   take_profit: float) -> float:
        """Calculate risk/reward ratio"""
        try:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)

            if risk > 0:
                return reward / risk
            else:
                return 0.0

        except Exception:
            return 0.0

    def _detect_composite_patterns(self,
                                 patterns_by_tf: Dict[str, List[TimeframePattern]],
                                 current_price: float) -> List[MultiTimeframeSignal]:
        """Detect composite patterns across timeframes"""
        composite_signals = []

        try:
            all_patterns = []
            for tf_patterns in patterns_by_tf.values():
                all_patterns.extend(tf_patterns)

            # Look for each defined composite pattern
            for composite_name, composite_def in self.composite_patterns.items():
                primary_patterns = [p for p in all_patterns
                                  if composite_def.primary_pattern.lower() in p.pattern_key.lower()]

                for primary in primary_patterns:
                    # Look for secondary pattern that forms the composite
                    secondary_patterns = [p for p in all_patterns
                                        if (composite_def.secondary_pattern.lower() in p.pattern_key.lower() and
                                            p.timeframe != primary.timeframe)]

                    if secondary_patterns:
                        # Create composite signal
                        best_secondary = max(secondary_patterns, key=lambda x: x.confidence)

                        # Calculate composite confidence
                        composite_confidence = (primary.confidence * 0.6 +
                                              best_secondary.confidence * 0.4) * composite_def.confidence

                        if composite_confidence >= self.min_confluence_confidence:
                            entry_price, stop_loss, take_profit = self._calculate_entry_levels(
                                primary, [best_secondary], current_price
                            )

                            risk_reward = self._calculate_risk_reward_ratio(
                                entry_price, stop_loss, take_profit
                            )

                            if risk_reward >= self.min_risk_reward_ratio:
                                composite_signal = MultiTimeframeSignal(
                                    primary_pattern=primary,
                                    supporting_patterns=[best_secondary],
                                    alignment=self._determine_alignment(primary, [best_secondary]),
                                    combined_confidence=composite_confidence,
                                    combined_strength=(primary.strength + best_secondary.strength) / 2,
                                    risk_reward_ratio=risk_reward,
                                    strategy_type=f"composite_{composite_name}",
                                    entry_price=entry_price,
                                    stop_loss=stop_loss,
                                    take_profit=take_profit
                                )
                                composite_signals.append(composite_signal)

        except Exception as e:
            logger.debug(f"Error detecting composite patterns: {e}")

        return composite_signals

    def _rank_signals_by_performance(self, signals: List[MultiTimeframeSignal]) -> List[MultiTimeframeSignal]:
        """Rank signals by profit/risk potential"""
        try:
            # Score each signal based on multiple factors
            scored_signals = []

            for signal in signals:
                # Performance score combines confidence, strength, and risk/reward
                performance_score = (
                    signal.combined_confidence * 0.3 +
                    signal.combined_strength * 0.2 +
                    min(signal.risk_reward_ratio / 5.0, 1.0) * 0.3 +  # Cap at 5:1 for scoring
                    self._get_strategy_bonus(signal.strategy_type) * 0.2
                )

                scored_signals.append((signal, performance_score))

            # Sort by performance score (highest first)
            scored_signals.sort(key=lambda x: x[1], reverse=True)

            # Return top signals (limit to best opportunities)
            return [signal for signal, score in scored_signals[:10]]

        except Exception as e:
            logger.debug(f"Error ranking signals: {e}")
            return signals[:5]  # Return first 5 as fallback

    def _get_strategy_bonus(self, strategy_type: str) -> float:
        """Get bonus score based on strategy type performance characteristics"""
        strategy_bonuses = {
            'scalping': 0.7,        # Lower bonus due to higher risk
            'intraday': 0.9,        # Good balance
            'swing': 0.8,           # Longer timeframe, good reliability
            'composite_head_shoulders_in_triangle': 1.0,  # High reliability composite
            'composite_double_top_with_divergence': 1.0,
            'composite_flag_in_wedge': 0.9,
            'composite_elliott_wave_with_harmonics': 0.8,
            'composite_triangle_breakout_flag': 0.9,
            'composite_harmonic_at_structure': 0.9
        }

        return strategy_bonuses.get(strategy_type, 0.5)

    def get_signal_summary(self, signal: MultiTimeframeSignal) -> Dict[str, Any]:
        """Get formatted signal summary for display"""
        return {
            'strategy': signal.strategy_type.replace('_', ' ').title(),
            'primary_pattern': signal.primary_pattern.pattern_key,
            'primary_timeframe': signal.primary_pattern.timeframe,
            'supporting_count': len(signal.supporting_patterns),
            'alignment': signal.alignment.value.replace('_', ' ').title(),
            'confidence': f"{signal.combined_confidence:.1%}",
            'strength': f"{signal.combined_strength:.1%}",
            'risk_reward': f"{signal.risk_reward_ratio:.1f}:1",
            'entry': f"{signal.entry_price:.5f}" if signal.entry_price else "N/A",
            'stop_loss': f"{signal.stop_loss:.5f}" if signal.stop_loss else "N/A",
            'take_profit': f"{signal.take_profit:.5f}" if signal.take_profit else "N/A",
            'expected_move': self._estimate_expected_move(signal),
            'risk_level': self._assess_risk_level(signal)
        }

    def _estimate_expected_move(self, signal: MultiTimeframeSignal) -> str:
        """Estimate expected price move percentage"""
        try:
            if signal.entry_price and signal.take_profit:
                move_percent = abs(signal.take_profit - signal.entry_price) / signal.entry_price * 100
                return f"{move_percent:.1f}%"
            else:
                # Default estimates based on strategy type
                if 'scalping' in signal.strategy_type:
                    return "0.5-2.0%"
                elif 'intraday' in signal.strategy_type:
                    return "2.0-5.0%"
                elif 'swing' in signal.strategy_type:
                    return "5.0-15.0%"
                else:
                    return "Variable"
        except Exception:
            return "Unknown"

    def _assess_risk_level(self, signal: MultiTimeframeSignal) -> str:
        """Assess risk level of the signal"""
        try:
            # Base risk on multiple factors
            risk_score = 0

            # Risk from strategy type
            if 'scalping' in signal.strategy_type:
                risk_score += 2
            elif 'intraday' in signal.strategy_type:
                risk_score += 1
            # swing = 0 (lower risk)

            # Risk from confluence
            if signal.alignment == PatternAlignment.MIXED_SIGNALS:
                risk_score += 2
            elif signal.alignment == PatternAlignment.NEUTRAL:
                risk_score += 1

            # Risk from confidence
            if signal.combined_confidence < 0.7:
                risk_score += 1
            elif signal.combined_confidence < 0.8:
                risk_score += 0.5

            # Risk from risk/reward ratio
            if signal.risk_reward_ratio < 2.5:
                risk_score += 1

            # Classify risk level
            if risk_score <= 1:
                return "Low"
            elif risk_score <= 3:
                return "Medium"
            else:
                return "High"

        except Exception:
            return "Medium"