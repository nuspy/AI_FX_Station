"""
Pattern Strength Calculator without Volume Data.

Calculates pattern strength using:
- Detection confidence
- Price volatility during formation
- Historical success rate
- Price action quality metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger

from ..cache import cache_decorator, get_pattern_cache


@dataclass
class StrengthComponents:
    """Individual components of pattern strength calculation"""
    confidence: float = 0.0          # Raw detector confidence
    volatility: float = 0.0          # Volatility-based score
    historical_success: float = 0.0  # Historical performance score
    price_action: float = 0.0        # Price action quality score
    final_strength: float = 0.0      # Combined weighted score
    star_rating: int = 1             # 1-5 star rating


class PatternStrengthCalculator:
    """
    Calculates pattern strength without volume data using multiple factors:

    1. Confidence: Raw confidence from pattern detector
    2. Volatility: Normalized volatility during pattern formation
    3. Historical Success: Success rate of this pattern type historically
    4. Price Action: Quality of price movements (range, momentum, consistency)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('patterns', {}).get('strength', {})

        # Weights for different components
        self.weights = self.config.get('weights', {
            'confidence': 0.35,
            'volatility': 0.25,
            'historical_success': 0.25,
            'price_action': 0.15
        })

        # Scaling parameters
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.volatility_window = self.config.get('volatility_window', 20)

        # Historical success cache
        self._historical_cache: Dict[str, float] = {}

    def calculate_strength(self,
                          pattern_event: Any,
                          price_data: pd.DataFrame,
                          pattern_type: str,
                          asset: str,
                          timeframe: str) -> StrengthComponents:
        """
        Calculate overall pattern strength.

        Args:
            pattern_event: Pattern detection event object
            price_data: OHLC price data during pattern formation
            pattern_type: Type of pattern (e.g., 'head_shoulders', 'triangle')
            asset: Asset symbol
            timeframe: Timeframe string

        Returns:
            StrengthComponents with individual scores and final rating
        """
        try:
            # 1. Get raw confidence from detector
            confidence_score = self._calculate_confidence_score(pattern_event)

            # 2. Calculate volatility-based score
            volatility_score = self._calculate_volatility_score(price_data, pattern_event)

            # 3. Get historical success rate
            historical_score = self._get_historical_success_score(pattern_type, asset, timeframe)

            # 4. Calculate price action quality
            price_action_score = self._calculate_price_action_score(price_data, pattern_event)

            # 5. Combine with weights
            final_strength = (
                self.weights['confidence'] * confidence_score +
                self.weights['volatility'] * volatility_score +
                self.weights['historical_success'] * historical_score +
                self.weights['price_action'] * price_action_score
            )

            # Convert to star rating (1-5)
            star_rating = self._strength_to_stars(final_strength)

            return StrengthComponents(
                confidence=confidence_score,
                volatility=volatility_score,
                historical_success=historical_score,
                price_action=price_action_score,
                final_strength=final_strength,
                star_rating=star_rating
            )

        except Exception as e:
            logger.error(f"Error calculating pattern strength: {e}")
            return StrengthComponents(star_rating=1)

    def _calculate_confidence_score(self, pattern_event: Any) -> float:
        """Calculate normalized confidence score from detector"""
        try:
            # Get confidence from pattern event
            raw_confidence = getattr(pattern_event, 'confidence', 0.5)

            # Normalize to 0-1 range with minimum threshold
            if raw_confidence < self.min_confidence:
                return 0.0

            # Scale from min_confidence to 1.0 -> 0.0 to 1.0
            normalized = (raw_confidence - self.min_confidence) / (1.0 - self.min_confidence)
            return min(1.0, max(0.0, normalized))

        except Exception:
            return 0.5  # Default moderate confidence

    def _calculate_volatility_score(self, price_data: pd.DataFrame, pattern_event: Any) -> float:
        """
        Calculate volatility-based score.

        Higher volatility during formation generally means stronger patterns,
        but extreme volatility might indicate noise.
        """
        try:
            if price_data.empty or len(price_data) < 5:
                return 0.5  # Default score for insufficient data

            # Get formation period if available
            formation_data = self._get_formation_period_data(price_data, pattern_event)

            if formation_data.empty:
                formation_data = price_data.tail(self.volatility_window)

            # Calculate True Range based volatility
            high = formation_data['high']
            low = formation_data['low']
            close = formation_data['close']
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=min(14, len(formation_data))).mean()

            # Normalize by average price
            avg_price = formation_data['close'].mean()
            normalized_volatility = (atr.iloc[-1] / avg_price) if avg_price > 0 else 0

            # Score volatility (moderate volatility = good, extreme = bad)
            if normalized_volatility < 0.01:  # Very low volatility
                return 0.3
            elif normalized_volatility < 0.03:  # Good volatility
                return 0.9
            elif normalized_volatility < 0.06:  # High but acceptable
                return 0.7
            else:  # Too high volatility
                return 0.4

        except Exception as e:
            logger.debug(f"Error calculating volatility score: {e}")
            return 0.5

    def _calculate_price_action_score(self, price_data: pd.DataFrame, pattern_event: Any) -> float:
        """
        Calculate price action quality score based on:
        - Momentum consistency
        - Range expansion/contraction
        - Trend clarity
        """
        try:
            if price_data.empty or len(price_data) < 5:
                return 0.5

            formation_data = self._get_formation_period_data(price_data, pattern_event)
            if formation_data.empty:
                formation_data = price_data.tail(min(20, len(price_data)))

            scores = []

            # 1. Momentum consistency (how smooth is the trend)
            momentum_score = self._calculate_momentum_consistency(formation_data)
            scores.append(momentum_score)

            # 2. Range quality (proper expansion/contraction)
            range_score = self._calculate_range_quality(formation_data)
            scores.append(range_score)

            # 3. Trend clarity (how clear is the direction)
            trend_score = self._calculate_trend_clarity(formation_data)
            scores.append(trend_score)

            # 4. Pattern geometry (how well-formed is the pattern)
            geometry_score = self._calculate_pattern_geometry(formation_data, pattern_event)
            scores.append(geometry_score)

            # Average all components
            return np.mean(scores)

        except Exception as e:
            logger.debug(f"Error calculating price action score: {e}")
            return 0.5

    def _calculate_momentum_consistency(self, data: pd.DataFrame) -> float:
        """Calculate momentum consistency score"""
        try:
            if len(data) < 3:
                return 0.5

            # Calculate price changes
            price_changes = data['close'].diff().dropna()

            if len(price_changes) < 2:
                return 0.5

            # Check momentum consistency (fewer direction changes = better)
            direction_changes = (price_changes.shift(1) * price_changes < 0).sum()
            total_periods = len(price_changes)

            # Score: fewer direction changes = higher score
            consistency_ratio = 1.0 - (direction_changes / total_periods)
            return max(0.0, min(1.0, consistency_ratio))

        except Exception:
            return 0.5

    def _calculate_range_quality(self, data: pd.DataFrame) -> float:
        """Calculate range expansion/contraction quality"""
        try:
            if len(data) < 5:
                return 0.5

            # Calculate true ranges
            high = data['high']
            low = data['low']
            close = data['close']
            prev_close = close.shift(1)

            tr = pd.concat([
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            ], axis=1).max(axis=1)

            # Check if ranges are expanding or contracting appropriately
            range_trend = np.polyfit(range(len(tr)), tr, 1)[0]

            # Score based on appropriate range behavior
            # Positive slope (expanding ranges) can be good for breakouts
            # Negative slope (contracting ranges) can be good for consolidations
            normalized_slope = abs(range_trend) / tr.mean() if tr.mean() > 0 else 0

            if normalized_slope < 0.01:  # Very stable ranges
                return 0.7
            elif normalized_slope < 0.05:  # Good range behavior
                return 0.9
            else:  # Too volatile ranges
                return 0.4

        except Exception:
            return 0.5

    def _calculate_trend_clarity(self, data: pd.DataFrame) -> float:
        """Calculate trend clarity score"""
        try:
            if len(data) < 3:
                return 0.5

            # Calculate trend using linear regression
            prices = data['close'].values
            x = np.arange(len(prices))

            # Fit linear trend
            slope, intercept = np.polyfit(x, prices, 1)

            # Calculate R-squared (trend clarity)
            y_pred = slope * x + intercept
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)

            if ss_tot == 0:
                return 0.5

            r_squared = 1 - (ss_res / ss_tot)

            # Higher R-squared = clearer trend
            return max(0.0, min(1.0, r_squared))

        except Exception:
            return 0.5

    def _calculate_pattern_geometry(self, data: pd.DataFrame, pattern_event: Any) -> float:
        """Calculate pattern geometry quality"""
        try:
            # This is pattern-specific and would need pattern type info
            # For now, use a general geometry score based on price structure

            if len(data) < 3:
                return 0.5

            # Check for proper high/low relationships
            highs = data['high']
            lows = data['low']

            # Calculate structure quality (smooth transitions)
            high_smoothness = self._calculate_smoothness(highs)
            low_smoothness = self._calculate_smoothness(lows)

            return (high_smoothness + low_smoothness) / 2.0

        except Exception:
            return 0.5

    def _calculate_smoothness(self, series: pd.Series) -> float:
        """Calculate smoothness of a price series"""
        try:
            if len(series) < 3:
                return 0.5

            # Calculate second derivative (acceleration)
            first_diff = series.diff()
            second_diff = first_diff.diff()

            # Smoothness = inverse of average absolute second derivative
            avg_acceleration = abs(second_diff).mean()
            price_range = series.max() - series.min()

            if price_range == 0:
                return 0.5

            # Normalize acceleration by price range
            normalized_accel = avg_acceleration / price_range

            # Convert to smoothness score (lower acceleration = higher smoothness)
            smoothness = 1.0 / (1.0 + normalized_accel * 100)
            return max(0.0, min(1.0, smoothness))

        except Exception:
            return 0.5

    def _get_formation_period_data(self, price_data: pd.DataFrame, pattern_event: Any) -> pd.DataFrame:
        """Extract price data for pattern formation period"""
        try:
            # Try to get formation period from pattern event
            start_ts = getattr(pattern_event, 'formation_start_ts', None) or \
                      getattr(pattern_event, 'start_ts', None)
            end_ts = getattr(pattern_event, 'formation_end_ts', None) or \
                    getattr(pattern_event, 'end_ts', None) or \
                    getattr(pattern_event, 'confirm_ts', None)

            if start_ts and end_ts:
                # Filter data by timestamp if available
                if 'timestamp' in price_data.columns:
                    mask = (price_data['timestamp'] >= start_ts) & (price_data['timestamp'] <= end_ts)
                    formation_data = price_data[mask]
                    if not formation_data.empty:
                        return formation_data

            # Fallback: use recent data
            lookback = getattr(pattern_event, 'lookback', self.volatility_window)
            return price_data.tail(min(lookback, len(price_data)))

        except Exception:
            return pd.DataFrame()

    @cache_decorator('historical_success', ttl=86400)  # Cache for 24 hours
    def _get_historical_success_score(self, pattern_type: str, asset: str, timeframe: str) -> float:
        """
        Get historical success rate for this pattern type.

        This would typically query the database for historical performance,
        but for now returns reasonable defaults based on pattern research.
        """
        try:
            # Try to get from pattern cache first
            cache = get_pattern_cache()
            if cache:
                cached_score = cache.get_parameter_optimization(asset, timeframe, pattern_type, 'historical_success')
                if cached_score is not None:
                    return cached_score.get('success_rate', 0.5)

            # Default success rates based on pattern research
            # These would be replaced with actual database queries
            default_success_rates = {
                'head_shoulders': 0.84,      # 84% meet price target
                'double_top': 0.79,          # 79% meet price target
                'double_bottom': 0.79,
                'triangle_ascending': 0.68,
                'triangle_descending': 0.64,
                'triangle_symmetrical': 0.71,
                'flag_bull': 0.87,           # Very reliable
                'flag_bear': 0.83,
                'engulfing_bullish': 0.58,
                'engulfing_bearish': 0.54,
                'hammer': 0.42,              # Needs confirmation
                'doji': 0.35,                # Low reliability
                'gartley': 0.78,             # Harmonic patterns
                'butterfly': 0.81,
                'elliott_wave_impulse': 0.72,
                'broadening_bottom': 0.50,   # Variable
            }

            # Get success rate or default to 50%
            success_rate = default_success_rates.get(pattern_type.lower(), 0.50)

            # Cache the result
            if cache:
                cache.cache_parameter_optimization(
                    asset, timeframe, pattern_type, 'historical_success',
                    {'success_rate': success_rate}
                )

            return success_rate

        except Exception as e:
            logger.debug(f"Error getting historical success score: {e}")
            return 0.5

    def _strength_to_stars(self, strength: float) -> int:
        """Convert strength score (0-1) to star rating (1-5)"""
        if strength >= 0.9:
            return 5  # ★★★★★
        elif strength >= 0.75:
            return 4  # ★★★★☆
        elif strength >= 0.6:
            return 3  # ★★★☆☆
        elif strength >= 0.4:
            return 2  # ★★☆☆☆
        else:
            return 1  # ★☆☆☆☆

    def get_strength_description(self, stars: int) -> str:
        """Get textual description of strength rating"""
        descriptions = {
            5: "Excellent - Very high confidence pattern",
            4: "Good - Strong pattern with good setup",
            3: "Average - Decent pattern, moderate confidence",
            2: "Weak - Low confidence, proceed with caution",
            1: "Poor - Very weak pattern, high risk"
        }
        return descriptions.get(stars, "Unknown")

    def format_strength_display(self, components: StrengthComponents) -> Dict[str, str]:
        """Format strength components for GUI display"""
        return {
            'stars': '★' * components.star_rating + '☆' * (5 - components.star_rating),
            'percentage': f"{components.final_strength:.1%}",
            'description': self.get_strength_description(components.star_rating),
            'components': {
                'Confidence': f"{components.confidence:.1%}",
                'Volatility': f"{components.volatility:.1%}",
                'Historical': f"{components.historical_success:.1%}",
                'Price Action': f"{components.price_action:.1%}"
            }
        }