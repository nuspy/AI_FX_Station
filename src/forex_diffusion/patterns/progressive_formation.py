"""
Progressive Pattern Formation Detection.

Detects patterns in formation (60% confidence threshold) with real-time updates
every minute candle and visual indication through dashed lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from ..cache import cache_decorator, get_pattern_cache


class FormationStage(Enum):
    """Stages of pattern formation"""
    EARLY = "early"           # < 40% confidence
    DEVELOPING = "developing" # 40-60% confidence
    FORMING = "forming"       # 60-80% confidence - show with dashed lines
    MATURE = "mature"         # 80-95% confidence
    COMPLETED = "completed"   # > 95% confidence


@dataclass
class ProgressivePattern:
    """Pattern in progressive formation"""
    pattern_key: str
    pattern_type: str
    stage: FormationStage
    confidence: float
    completion_percent: float
    direction: str
    timeframe: str
    formation_data: Dict[str, Any]
    visual_elements: Dict[str, Any]
    estimated_completion_time: Optional[datetime] = None
    key_levels: Dict[str, float] = None
    formation_quality: float = 0.0


@dataclass
class FormationUpdate:
    """Update event for pattern formation progress"""
    pattern_id: str
    previous_stage: FormationStage
    current_stage: FormationStage
    confidence_change: float
    completion_change: float
    timestamp: datetime
    trigger_reason: str


class ProgressivePatternDetector:
    """
    Detects and tracks patterns in progressive formation stages.

    Features:
    - 60% confidence threshold for showing patterns in formation
    - Real-time updates on every 1-minute candle
    - Dashed line visualization for forming patterns
    - Stage progression tracking
    - Formation quality assessment
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('patterns', {}).get('progressive', {})
        self.enabled = self.config.get('enabled', True)

        # Configuration parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 60) / 100.0  # 60%
        self.update_frequency = self.config.get('update_frequency', '1min')

        # Visual configuration
        visual_config = self.config.get('visual', {})
        self.dashed_line_style = visual_config.get('line_style', 'dashed')
        self.forming_alpha = visual_config.get('alpha', 0.6)

        # Pattern tracking
        self.active_patterns: Dict[str, ProgressivePattern] = {}
        # Pattern history with bounded size (max 100 updates per pattern)
        self.pattern_history: Dict[str, deque] = {}
        self.max_history_per_pattern = 100

        # Formation thresholds for each stage
        self.stage_thresholds = {
            FormationStage.EARLY: (0.0, 0.4),
            FormationStage.DEVELOPING: (0.4, 0.6),
            FormationStage.FORMING: (0.6, 0.8),
            FormationStage.MATURE: (0.8, 0.95),
            FormationStage.COMPLETED: (0.95, 1.0)
        }

    def update_patterns(self,
                       current_data: pd.DataFrame,
                       asset: str,
                       timeframe: str,
                       pattern_detectors: List[Any]) -> List[ProgressivePattern]:
        """
        Update progressive pattern formation based on latest data.

        Called every minute candle to track formation progress.
        """
        if not self.enabled or current_data.empty:
            return []

        try:
            updated_patterns = []
            current_time = datetime.now()

            # Run pattern detectors with partial formation detection
            forming_patterns = self._detect_forming_patterns(
                current_data, asset, timeframe, pattern_detectors
            )

            # Update existing patterns and detect new ones
            for pattern_info in forming_patterns:
                pattern_id = self._generate_pattern_id(pattern_info, asset, timeframe)

                if pattern_id in self.active_patterns:
                    # Update existing pattern
                    updated_pattern = self._update_existing_pattern(
                        pattern_id, pattern_info, current_time
                    )
                else:
                    # New pattern detected
                    updated_pattern = self._create_new_pattern(
                        pattern_id, pattern_info, asset, timeframe, current_time
                    )

                if updated_pattern:
                    updated_patterns.append(updated_pattern)

            # Remove patterns that are no longer forming
            self._cleanup_inactive_patterns(current_time)

            # Cache results
            cache = get_pattern_cache()
            if cache:
                cache.cache_pattern_detection(
                    asset, timeframe, 'progressive_formation', {}, updated_patterns
                )

            return updated_patterns

        except Exception as e:
            logger.error(f"Error updating progressive patterns: {e}")
            return []

    def _detect_forming_patterns(self,
                               data: pd.DataFrame,
                               asset: str,
                               timeframe: str,
                               pattern_detectors: List[Any]) -> List[Dict[str, Any]]:
        """Detect patterns in formation using modified detectors"""
        forming_patterns = []

        try:
            for detector in pattern_detectors:
                try:
                    # Run detector with partial formation mode
                    patterns = self._run_progressive_detection(detector, data)

                    for pattern in patterns:
                        # Only include patterns above confidence threshold
                        if pattern.get('confidence', 0.0) >= self.confidence_threshold:
                            forming_patterns.append({
                                'detector': detector,
                                'pattern_data': pattern,
                                'pattern_key': getattr(detector, 'pattern_key', 'unknown'),
                                'pattern_type': getattr(detector, 'pattern_type', 'chart')
                            })

                except Exception as e:
                    logger.debug(f"Error in progressive detection for {detector}: {e}")

        except Exception as e:
            logger.debug(f"Error detecting forming patterns: {e}")

        return forming_patterns

    def _run_progressive_detection(self, detector: Any, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run detector in progressive mode to detect partial formations"""
        try:
            # Check if detector supports progressive mode
            if hasattr(detector, 'detect_progressive'):
                return detector.detect_progressive(data)

            # Fallback: run normal detection and estimate formation progress
            elif hasattr(detector, 'detect'):
                normal_patterns = detector.detect(data)
                progressive_patterns = []

                for pattern in normal_patterns:
                    # Estimate formation progress based on pattern characteristics
                    formation_progress = self._estimate_formation_progress(pattern, data)

                    if formation_progress:
                        pattern.update(formation_progress)
                        progressive_patterns.append(pattern)

                return progressive_patterns

            return []

        except Exception as e:
            logger.debug(f"Error running progressive detection: {e}")
            return []

    def _estimate_formation_progress(self, pattern: Dict[str, Any], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Estimate formation progress for patterns without native progressive support"""
        try:
            # Get pattern characteristics
            confidence = pattern.get('confidence', 0.5)

            # Estimate completion based on time and structure
            formation_start = pattern.get('formation_start_ts')
            formation_end = pattern.get('formation_end_ts')

            if formation_start and formation_end:
                # Calculate time-based completion
                total_formation_time = formation_end - formation_start
                elapsed_time = datetime.now().timestamp() - formation_start
                time_completion = min(elapsed_time / total_formation_time, 1.0) if total_formation_time > 0 else 0.5
            else:
                time_completion = 0.5

            # Estimate structure completion based on pattern geometry
            structure_completion = self._estimate_structure_completion(pattern, data)

            # Combined completion estimate
            completion_percent = (confidence * 0.4 + time_completion * 0.3 + structure_completion * 0.3)

            return {
                'completion_percent': completion_percent,
                'time_completion': time_completion,
                'structure_completion': structure_completion,
                'formation_quality': self._assess_formation_quality(pattern, data)
            }

        except Exception:
            return None

    def _estimate_structure_completion(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """Estimate structural completion of pattern formation"""
        try:
            pattern_type = pattern.get('pattern_key', '').lower()

            # Pattern-specific structure analysis
            if 'triangle' in pattern_type:
                return self._estimate_triangle_completion(pattern, data)
            elif 'head' in pattern_type and 'shoulders' in pattern_type:
                return self._estimate_hns_completion(pattern, data)
            elif 'double' in pattern_type:
                return self._estimate_double_completion(pattern, data)
            elif 'flag' in pattern_type:
                return self._estimate_flag_completion(pattern, data)
            else:
                # Generic structure completion based on price action
                return self._estimate_generic_completion(pattern, data)

        except Exception:
            return 0.5

    def _estimate_triangle_completion(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """Estimate triangle pattern completion"""
        try:
            # Check convergence of trend lines
            if len(data) < 10:
                return 0.3

            highs = data['high'].tail(20)
            lows = data['low'].tail(20)

            # Simple convergence measure
            high_range = highs.max() - highs.min()
            low_range = lows.max() - lows.min()
            recent_range = data['high'].tail(5).max() - data['low'].tail(5).min()

            if high_range > 0 and low_range > 0:
                convergence = 1.0 - (recent_range / max(high_range, low_range))
                return max(0.3, min(0.9, convergence))

            return 0.5

        except Exception:
            return 0.5

    def _estimate_hns_completion(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """Estimate head and shoulders completion"""
        try:
            # Look for three peaks structure
            if len(data) < 15:
                return 0.2

            highs = data['high'].rolling(5).max()
            peaks = []

            # Find local peaks
            for i in range(5, len(highs) - 5):
                if highs.iloc[i] == highs.iloc[i-5:i+6].max():
                    peaks.append((i, highs.iloc[i]))

            # Check for H&S structure (3 peaks with middle highest)
            if len(peaks) >= 3:
                # Check if middle peak is highest (head)
                recent_peaks = peaks[-3:]
                if len(recent_peaks) == 3:
                    left_shoulder, head, right_shoulder = recent_peaks
                    if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
                        # Good H&S structure
                        return 0.85
                    else:
                        return 0.6
                return 0.4
            elif len(peaks) >= 2:
                return 0.6
            else:
                return 0.3

        except Exception:
            return 0.5

    def _estimate_double_completion(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """Estimate double top/bottom completion"""
        try:
            if len(data) < 10:
                return 0.3

            # Look for two similar peaks/troughs
            if 'top' in pattern.get('pattern_key', '').lower():
                extremes = data['high'].rolling(5).max()
            else:
                extremes = data['low'].rolling(5).min()

            # Find extreme points
            extreme_points = []
            for i in range(5, len(extremes) - 5):
                if 'top' in pattern.get('pattern_key', '').lower():
                    if extremes.iloc[i] == extremes.iloc[i-5:i+6].max():
                        extreme_points.append(extremes.iloc[i])
                else:
                    if extremes.iloc[i] == extremes.iloc[i-5:i+6].min():
                        extreme_points.append(extremes.iloc[i])

            if len(extreme_points) >= 2:
                # Check similarity of last two extremes (with division by zero protection)
                last_two = extreme_points[-2:]
                denominator = max(abs(last_two[0]), abs(last_two[1]), 1e-10)
                similarity = 1.0 - abs(last_two[0] - last_two[1]) / denominator

                if similarity > 0.95:  # Very similar levels
                    return 0.9
                elif similarity > 0.9:
                    return 0.8
                else:
                    return 0.6
            elif len(extreme_points) == 1:
                return 0.5
            else:
                return 0.3

        except Exception:
            return 0.5

    def _estimate_flag_completion(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """Estimate flag pattern completion"""
        try:
            if len(data) < 8:
                return 0.3

            # Flag is a brief consolidation after strong move
            # Check for consolidation phase
            recent_range = data['high'].tail(5).max() - data['low'].tail(5).min()
            prior_range = data['high'].tail(15).head(10).max() - data['low'].tail(15).head(10).min()

            if prior_range > 0:
                consolidation_ratio = recent_range / prior_range

                # Good flag has tight consolidation (10-30% of prior range)
                if 0.1 <= consolidation_ratio <= 0.3:
                    return 0.85
                elif 0.05 <= consolidation_ratio <= 0.5:
                    return 0.7
                else:
                    return 0.4

            return 0.5

        except Exception:
            return 0.5

    def _estimate_generic_completion(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """Generic structure completion estimation"""
        try:
            # Based on price action quality and trend consistency
            if len(data) < 5:
                return 0.3

            # Check trend consistency
            closes = data['close']
            price_trend = (closes.iloc[-1] - closes.iloc[-10]) / closes.iloc[-10] if len(closes) >= 10 else 0

            # Check volatility stability
            returns = closes.pct_change().dropna()
            volatility = returns.rolling(5).std().iloc[-1] if len(returns) >= 5 else 0.02

            # Higher completion for stable trends with moderate volatility
            if abs(price_trend) > 0.02 and volatility < 0.03:  # Good trend, low volatility
                return 0.75
            elif abs(price_trend) > 0.01:  # Moderate trend
                return 0.6
            else:
                return 0.4

        except Exception:
            return 0.5

    def _assess_formation_quality(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """Assess overall quality of pattern formation"""
        try:
            quality_factors = []

            # 1. Price action smoothness
            if len(data) >= 10:
                closes = data['close']
                smoothness = self._calculate_price_smoothness(closes)
                quality_factors.append(smoothness)

            # 2. Volume confirmation (if available)
            if 'volume' in data.columns:
                volume_quality = self._assess_volume_quality(data)
                quality_factors.append(volume_quality)

            # 3. Timeframe appropriateness
            tf_quality = self._assess_timeframe_quality(pattern, data)
            quality_factors.append(tf_quality)

            # 4. Market context
            context_quality = self._assess_market_context(data)
            quality_factors.append(context_quality)

            # Average quality factors
            return np.mean(quality_factors) if quality_factors else 0.5

        except Exception:
            return 0.5

    def _calculate_price_smoothness(self, prices: pd.Series) -> float:
        """Calculate price action smoothness"""
        try:
            # Measure how smooth the price transitions are
            first_diff = prices.diff().dropna()
            second_diff = first_diff.diff().dropna()

            if len(second_diff) > 0:
                # Lower second derivative indicates smoother price action
                avg_acceleration = abs(second_diff).mean()
                price_range = prices.max() - prices.min()

                if price_range > 0:
                    normalized_acceleration = avg_acceleration / price_range
                    smoothness = 1.0 / (1.0 + normalized_acceleration * 1000)
                    return max(0.0, min(1.0, smoothness))

            return 0.5

        except Exception:
            return 0.5

    def _assess_volume_quality(self, data: pd.DataFrame) -> float:
        """Assess volume quality for pattern formation"""
        try:
            # Volume should generally decrease during consolidation patterns
            # and increase during breakout patterns
            volumes = data['volume'].tail(10)

            if len(volumes) >= 5:
                recent_vol = volumes.tail(5).mean()
                prior_vol = volumes.head(5).mean()

                if prior_vol > 0:
                    vol_ratio = recent_vol / prior_vol

                    # Ideal volume behavior depends on pattern type
                    # For consolidation patterns, decreasing volume is good
                    if 0.7 <= vol_ratio <= 1.3:  # Stable volume
                        return 0.8
                    elif vol_ratio < 0.7:  # Decreasing volume (good for consolidation)
                        return 0.9
                    else:  # Increasing volume (may indicate early breakout)
                        return 0.6

            return 0.5

        except Exception:
            return 0.5

    def _assess_timeframe_quality(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """Assess if pattern is appropriate for the timeframe"""
        try:
            # Pattern duration should match timeframe expectations
            formation_start = pattern.get('formation_start_ts')
            if formation_start:
                duration_hours = (datetime.now().timestamp() - formation_start) / 3600

                timeframe = pattern.get('timeframe', '1h')

                # Expected duration ranges by timeframe
                expected_durations = {
                    '1m': (0.1, 2),      # 5 minutes to 2 hours
                    '5m': (0.5, 8),      # 30 minutes to 8 hours
                    '15m': (2, 24),      # 2 to 24 hours
                    '1h': (8, 72),       # 8 hours to 3 days
                    '4h': (24, 168),     # 1 to 7 days
                    '1d': (168, 720)     # 1 week to 1 month
                }

                min_duration, max_duration = expected_durations.get(timeframe, (4, 48))

                if min_duration <= duration_hours <= max_duration:
                    return 0.9
                elif duration_hours < min_duration:
                    return 0.6  # Too early, but could develop
                else:
                    return 0.3  # Too long, pattern may be invalid

            return 0.5

        except Exception:
            return 0.5

    def _assess_market_context(self, data: pd.DataFrame) -> float:
        """Assess broader market context for pattern formation"""
        try:
            # Simple market context based on recent price action
            if len(data) < 20:
                return 0.5

            # Check for trending vs ranging market
            closes = data['close']

            # Long-term trend
            long_trend = (closes.iloc[-1] - closes.iloc[-20]) / closes.iloc[-20]

            # Short-term volatility
            short_vol = closes.pct_change().tail(5).std()

            # Patterns form better in appropriate market conditions
            # Consolidation patterns in trending markets are high quality
            # Reversal patterns at extremes are high quality

            if abs(long_trend) > 0.05:  # Strong trend
                if short_vol < 0.02:  # Low volatility
                    return 0.9  # Great for consolidation patterns
                else:
                    return 0.7
            else:  # Ranging market
                return 0.6  # Moderate quality

        except Exception:
            return 0.5

    def _generate_pattern_id(self, pattern_info: Dict[str, Any], asset: str, timeframe: str) -> str:
        """Generate unique pattern ID"""
        pattern_key = pattern_info.get('pattern_key', 'unknown')
        timestamp = int(datetime.now().timestamp())
        return f"{asset}_{timeframe}_{pattern_key}_{timestamp}"

    def _update_existing_pattern(self,
                               pattern_id: str,
                               pattern_info: Dict[str, Any],
                               current_time: datetime) -> Optional[ProgressivePattern]:
        """Update existing progressive pattern"""
        try:
            existing_pattern = self.active_patterns[pattern_id]
            new_confidence = pattern_info['pattern_data'].get('confidence', 0.0)
            new_completion = pattern_info['pattern_data'].get('completion_percent', 0.0)

            # Determine new stage
            new_stage = self._determine_stage(new_confidence)

            # Create update event if stage changed
            if new_stage != existing_pattern.stage:
                update = FormationUpdate(
                    pattern_id=pattern_id,
                    previous_stage=existing_pattern.stage,
                    current_stage=new_stage,
                    confidence_change=new_confidence - existing_pattern.confidence,
                    completion_change=new_completion - existing_pattern.completion_percent,
                    timestamp=current_time,
                    trigger_reason="stage_progression"
                )

                if pattern_id not in self.pattern_history:
                    self.pattern_history[pattern_id] = []
                self.pattern_history[pattern_id].append(update)

            # Update pattern
            existing_pattern.stage = new_stage
            existing_pattern.confidence = new_confidence
            existing_pattern.completion_percent = new_completion
            existing_pattern.formation_quality = pattern_info['pattern_data'].get('formation_quality', 0.5)

            # Update visual elements
            existing_pattern.visual_elements = self._create_visual_elements(existing_pattern)

            self.active_patterns[pattern_id] = existing_pattern
            return existing_pattern

        except Exception as e:
            logger.debug(f"Error updating existing pattern {pattern_id}: {e}")
            return None

    def _create_new_pattern(self,
                          pattern_id: str,
                          pattern_info: Dict[str, Any],
                          asset: str,
                          timeframe: str,
                          current_time: datetime) -> Optional[ProgressivePattern]:
        """Create new progressive pattern"""
        try:
            pattern_data = pattern_info['pattern_data']
            confidence = pattern_data.get('confidence', 0.0)
            completion = pattern_data.get('completion_percent', 0.0)

            stage = self._determine_stage(confidence)

            # Only track patterns at forming stage or higher
            if stage.value not in ['forming', 'mature', 'completed']:
                return None

            progressive_pattern = ProgressivePattern(
                pattern_key=pattern_info['pattern_key'],
                pattern_type=pattern_info['pattern_type'],
                stage=stage,
                confidence=confidence,
                completion_percent=completion,
                direction=pattern_data.get('direction', 'neutral'),
                timeframe=timeframe,
                formation_data=pattern_data,
                visual_elements=self._create_visual_elements_for_stage(stage),
                estimated_completion_time=self._estimate_completion_time(pattern_data),
                key_levels=self._extract_key_levels(pattern_data),
                formation_quality=pattern_data.get('formation_quality', 0.5)
            )

            self.active_patterns[pattern_id] = progressive_pattern
            return progressive_pattern

        except Exception as e:
            logger.debug(f"Error creating new pattern {pattern_id}: {e}")
            return None

    def _determine_stage(self, confidence: float) -> FormationStage:
        """Determine formation stage based on confidence"""
        for stage, (min_conf, max_conf) in self.stage_thresholds.items():
            if min_conf <= confidence < max_conf:
                return stage

        # Fallback
        if confidence >= 0.95:
            return FormationStage.COMPLETED
        else:
            return FormationStage.EARLY

    def _create_visual_elements(self, pattern: ProgressivePattern) -> Dict[str, Any]:
        """Create visual elements for pattern display"""
        return self._create_visual_elements_for_stage(pattern.stage)

    def _create_visual_elements_for_stage(self, stage: FormationStage) -> Dict[str, Any]:
        """Create visual elements based on formation stage"""
        visual_elements = {
            'line_style': 'solid',
            'alpha': 1.0,
            'color_modifier': 1.0,
            'show_formation_lines': False,
            'show_confidence_indicator': False
        }

        if stage == FormationStage.FORMING:
            # Dashed lines for forming patterns
            visual_elements.update({
                'line_style': self.dashed_line_style,
                'alpha': self.forming_alpha,
                'color_modifier': 0.8,
                'show_formation_lines': True,
                'show_confidence_indicator': True
            })
        elif stage == FormationStage.MATURE:
            # Semi-transparent solid lines
            visual_elements.update({
                'line_style': 'solid',
                'alpha': 0.8,
                'color_modifier': 0.9,
                'show_formation_lines': True,
                'show_confidence_indicator': True
            })
        elif stage == FormationStage.COMPLETED:
            # Full opacity solid lines
            visual_elements.update({
                'line_style': 'solid',
                'alpha': 1.0,
                'color_modifier': 1.0,
                'show_formation_lines': True,
                'show_confidence_indicator': False
            })

        return visual_elements

    def _estimate_completion_time(self, pattern_data: Dict[str, Any]) -> Optional[datetime]:
        """Estimate when pattern formation will complete"""
        try:
            formation_start = pattern_data.get('formation_start_ts')
            completion_percent = pattern_data.get('completion_percent', 0.0)

            if formation_start and completion_percent > 0:
                elapsed_time = datetime.now().timestamp() - formation_start
                total_estimated_time = elapsed_time / completion_percent

                completion_timestamp = formation_start + total_estimated_time
                return datetime.fromtimestamp(completion_timestamp)

            return None

        except Exception:
            return None

    def _extract_key_levels(self, pattern_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key price levels from pattern data"""
        key_levels = {}

        try:
            # Standard pattern levels
            if 'target_price' in pattern_data:
                key_levels['target'] = pattern_data['target_price']

            if 'failure_price' in pattern_data:
                key_levels['invalidation'] = pattern_data['failure_price']

            if 'support_level' in pattern_data:
                key_levels['support'] = pattern_data['support_level']

            if 'resistance_level' in pattern_data:
                key_levels['resistance'] = pattern_data['resistance_level']

            # Pattern-specific levels
            pattern_key = pattern_data.get('pattern_key', '').lower()

            if 'neckline' in pattern_data:
                key_levels['neckline'] = pattern_data['neckline']

            if 'breakout_level' in pattern_data:
                key_levels['breakout'] = pattern_data['breakout_level']

        except Exception as e:
            logger.debug(f"Error extracting key levels: {e}")

        return key_levels

    def _cleanup_inactive_patterns(self, current_time: datetime):
        """Remove patterns that are no longer active or relevant"""
        try:
            patterns_to_remove = []

            for pattern_id, pattern in self.active_patterns.items():
                # Remove patterns older than 24 hours without completion
                if pattern.estimated_completion_time:
                    if current_time > pattern.estimated_completion_time + timedelta(hours=24):
                        patterns_to_remove.append(pattern_id)
                        continue

                # Remove patterns with very low confidence that haven't progressed
                if pattern.confidence < 0.3:
                    patterns_to_remove.append(pattern_id)
                    continue

                # Remove completed patterns after some time
                if pattern.stage == FormationStage.COMPLETED:
                    if pattern_id in self.pattern_history:
                        last_update = max(update.timestamp for update in self.pattern_history[pattern_id])
                        if current_time > last_update + timedelta(hours=6):
                            patterns_to_remove.append(pattern_id)

            # Remove identified patterns
            for pattern_id in patterns_to_remove:
                if pattern_id in self.active_patterns:
                    del self.active_patterns[pattern_id]

                # Keep history for a while longer
                if pattern_id in self.pattern_history:
                    last_update = max(update.timestamp for update in self.pattern_history[pattern_id])
                    if current_time > last_update + timedelta(days=7):
                        del self.pattern_history[pattern_id]

            if patterns_to_remove:
                logger.debug(f"Cleaned up {len(patterns_to_remove)} inactive patterns")

        except Exception as e:
            logger.error(f"Error cleaning up inactive patterns: {e}")

    def get_forming_patterns(self) -> List[ProgressivePattern]:
        """Get all patterns currently in forming stage"""
        return [pattern for pattern in self.active_patterns.values()
                if pattern.stage == FormationStage.FORMING]

    def get_pattern_updates(self, since: datetime) -> List[FormationUpdate]:
        """Get all pattern updates since specified time"""
        updates = []

        for pattern_history in self.pattern_history.values():
            updates.extend([update for update in pattern_history if update.timestamp > since])

        return sorted(updates, key=lambda x: x.timestamp)

    def get_pattern_summary(self, pattern: ProgressivePattern) -> Dict[str, Any]:
        """Get formatted pattern summary for display"""
        return {
            'pattern_name': pattern.pattern_key.replace('_', ' ').title(),
            'stage': pattern.stage.value.title(),
            'confidence': f"{pattern.confidence:.1%}",
            'completion': f"{pattern.completion_percent:.1%}",
            'direction': pattern.direction.title(),
            'timeframe': pattern.timeframe,
            'quality': f"{pattern.formation_quality:.1%}",
            'estimated_completion': pattern.estimated_completion_time.strftime("%H:%M") if pattern.estimated_completion_time else "Unknown",
            'key_levels': {k: f"{v:.5f}" for k, v in pattern.key_levels.items()} if pattern.key_levels else {},
            'visual_style': pattern.visual_elements.get('line_style', 'solid'),
            'opacity': pattern.visual_elements.get('alpha', 1.0)
        }