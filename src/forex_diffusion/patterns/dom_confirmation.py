"""
DOM Confirmation Module for Pattern Detection

Enhances pattern confidence scores by validating with real-time order book data.
For bullish patterns, checks if bid depth > ask depth (buying pressure).
For bearish patterns, checks if ask depth > bid depth (selling pressure).
"""
from __future__ import annotations

from typing import Dict, Any, List
from loguru import logger


class DOMPatternConfirmation:
    """
    Validates pattern signals against DOM (Depth of Market) data.

    Provides confidence adjustments based on order flow alignment with pattern direction.
    """

    def __init__(self, dom_service=None):
        """
        Initialize DOM confirmation module.

        Args:
            dom_service: DOM aggregator service for fetching real-time data
        """
        self.dom_service = dom_service

        # Confidence adjustment parameters
        self.strong_confirmation_threshold = 0.3  # >30% imbalance
        self.moderate_confirmation_threshold = 0.15  # >15% imbalance
        self.strong_boost = 0.20  # +20% score boost
        self.moderate_boost = 0.10  # +10% score boost
        self.weak_penalty = -0.05  # -5% score penalty for opposing flow

    def confirm_pattern(
        self,
        pattern_direction: str,
        symbol: str,
        original_score: float = 0.5
    ) -> Dict[str, Any]:
        """
        Confirm pattern with DOM data and calculate adjusted confidence.

        Args:
            pattern_direction: Pattern direction ('bull' or 'bear')
            symbol: Trading symbol (e.g., 'EURUSD')
            original_score: Original pattern score (0-1)

        Returns:
            Dictionary with:
            - adjusted_score: Score adjusted for DOM confirmation
            - dom_aligned: Whether DOM aligns with pattern
            - confidence_boost: Amount added/subtracted from score
            - imbalance: Depth imbalance value
            - reasoning: Explanation of adjustment
        """
        if not self.dom_service:
            return {
                'adjusted_score': original_score,
                'dom_aligned': None,
                'confidence_boost': 0.0,
                'imbalance': 0.0,
                'reasoning': 'DOM service not available'
            }

        try:
            # Fetch latest DOM snapshot
            dom_snapshot = self.dom_service.get_latest_dom_snapshot(symbol)
            if not dom_snapshot:
                return {
                    'adjusted_score': original_score,
                    'dom_aligned': None,
                    'confidence_boost': 0.0,
                    'imbalance': 0.0,
                    'reasoning': f'No DOM data available for {symbol}'
                }

            # Extract depth imbalance
            depth_imbalance = dom_snapshot.get('depth_imbalance', 0.0)
            bid_depth = dom_snapshot.get('bid_depth', 0)
            ask_depth = dom_snapshot.get('ask_depth', 0)

            # Determine if DOM aligns with pattern direction
            # Bullish: expect positive imbalance (bid > ask)
            # Bearish: expect negative imbalance (ask > bid)
            is_bullish_pattern = pattern_direction.lower() in ['bull', 'bullish', 'buy', 'long']
            is_bearish_pattern = pattern_direction.lower() in ['bear', 'bearish', 'sell', 'short']

            if is_bullish_pattern:
                dom_aligned = depth_imbalance > 0
                imbalance_strength = depth_imbalance
            elif is_bearish_pattern:
                dom_aligned = depth_imbalance < 0
                imbalance_strength = abs(depth_imbalance)
            else:
                # Neutral or unknown pattern direction
                return {
                    'adjusted_score': original_score,
                    'dom_aligned': None,
                    'confidence_boost': 0.0,
                    'imbalance': depth_imbalance,
                    'reasoning': f'Unknown pattern direction: {pattern_direction}'
                }

            # Calculate confidence adjustment
            confidence_boost = 0.0
            reasoning = ""

            if dom_aligned:
                if imbalance_strength > self.strong_confirmation_threshold:
                    confidence_boost = self.strong_boost
                    reasoning = (
                        f"✅ STRONG DOM confirmation: {pattern_direction} pattern with "
                        f"{imbalance_strength*100:.1f}% imbalance "
                        f"(bid={bid_depth:,.0f}, ask={ask_depth:,.0f})"
                    )
                elif imbalance_strength > self.moderate_confirmation_threshold:
                    confidence_boost = self.moderate_boost
                    reasoning = (
                        f"✅ Moderate DOM confirmation: {pattern_direction} pattern with "
                        f"{imbalance_strength*100:.1f}% imbalance"
                    )
                else:
                    confidence_boost = 0.0
                    reasoning = (
                        f"→ Weak DOM alignment: {pattern_direction} pattern with "
                        f"{imbalance_strength*100:.1f}% imbalance (no adjustment)"
                    )
            else:
                # DOM opposes pattern direction - apply penalty
                confidence_boost = self.weak_penalty
                reasoning = (
                    f"⚠️  DOM contradiction: {pattern_direction} pattern but "
                    f"imbalance is {depth_imbalance*100:.1f}% (opposing flow)"
                )

            # Apply adjustment (clamp to 0-1 range)
            adjusted_score = max(0.0, min(1.0, original_score + confidence_boost))

            return {
                'adjusted_score': adjusted_score,
                'dom_aligned': dom_aligned,
                'confidence_boost': confidence_boost,
                'imbalance': depth_imbalance,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'reasoning': reasoning
            }

        except Exception as e:
            logger.error(f"DOM confirmation error for {symbol}: {e}", exc_info=True)
            return {
                'adjusted_score': original_score,
                'dom_aligned': None,
                'confidence_boost': 0.0,
                'imbalance': 0.0,
                'reasoning': f'Error: {e}'
            }

    def batch_confirm_patterns(
        self,
        patterns: List[Dict[str, Any]],
        symbol: str
    ) -> List[Dict[str, Any]]:
        """
        Confirm multiple patterns with DOM data.

        Args:
            patterns: List of pattern dictionaries with 'direction' and 'score' fields
            symbol: Trading symbol

        Returns:
            List of patterns with added DOM confirmation fields
        """
        confirmed_patterns = []

        for pattern in patterns:
            direction = pattern.get('direction', 'neutral')
            original_score = pattern.get('score', 0.5)

            confirmation = self.confirm_pattern(direction, symbol, original_score)

            # Add confirmation data to pattern
            pattern_copy = pattern.copy()
            pattern_copy.update({
                'score': confirmation['adjusted_score'],
                'original_score': original_score,
                'dom_aligned': confirmation['dom_aligned'],
                'dom_boost': confirmation['confidence_boost'],
                'dom_imbalance': confirmation['imbalance'],
                'dom_reasoning': confirmation['reasoning']
            })

            confirmed_patterns.append(pattern_copy)

        return confirmed_patterns

    def get_confirmation_summary(
        self,
        symbol: str,
        direction: str
    ) -> str:
        """
        Get human-readable summary of DOM confirmation for a direction.

        Args:
            symbol: Trading symbol
            direction: Pattern direction

        Returns:
            Summary string
        """
        confirmation = self.confirm_pattern(direction, symbol, 0.5)
        return confirmation['reasoning']


def enrich_patterns_with_dom(
    patterns: List[Any],
    symbol: str,
    dom_service=None
) -> List[Any]:
    """
    Enrich pattern events with DOM confirmation.

    This function can be called after pattern detection to add DOM-based
    confidence adjustments.

    Args:
        patterns: List of PatternEvent objects or pattern dictionaries
        symbol: Trading symbol
        dom_service: DOM aggregator service

    Returns:
        List of patterns with adjusted scores
    """
    if not dom_service or not patterns:
        return patterns

    confirmer = DOMPatternConfirmation(dom_service)
    enriched = []

    for pattern in patterns:
        # Extract direction and score (handle both dict and object formats)
        if isinstance(pattern, dict):
            direction = pattern.get('direction', 'neutral')
            original_score = pattern.get('score', 0.5)
        else:
            direction = getattr(pattern, 'direction', 'neutral')
            original_score = getattr(pattern, 'score', 0.5)

            # Convert direction enum to string if needed
            if hasattr(direction, 'value'):
                direction = direction.value

        # Get DOM confirmation
        confirmation = confirmer.confirm_pattern(direction, symbol, original_score)

        # Apply adjustment
        if isinstance(pattern, dict):
            pattern_copy = pattern.copy()
            pattern_copy['score'] = confirmation['adjusted_score']
            pattern_copy['dom_confirmation'] = confirmation
            enriched.append(pattern_copy)
        else:
            # Modify object in place
            try:
                setattr(pattern, 'score', confirmation['adjusted_score'])
                setattr(pattern, 'dom_confirmation', confirmation)
            except Exception as e:
                logger.warning(f"Could not set DOM confirmation on pattern: {e}")

            enriched.append(pattern)

    logger.debug(
        f"DOM enrichment: {len(enriched)} patterns processed for {symbol}, "
        f"{sum(1 for p in enriched if (p.get('dom_confirmation') if isinstance(p, dict) else getattr(p, 'dom_confirmation', None)) and (p['dom_confirmation']['dom_aligned'] if isinstance(p, dict) else p.dom_confirmation['dom_aligned']))} aligned"
    )

    return enriched
