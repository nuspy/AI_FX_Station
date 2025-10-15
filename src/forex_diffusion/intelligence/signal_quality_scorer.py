"""
Signal Quality Scoring System

Provides unified quality assessment for all trading signals across multiple dimensions.
Implements the quality gate before execution decisions.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import json


class SignalSource(Enum):
    """Signal source types"""
    PATTERN = "pattern"
    HARMONIC = "harmonic"
    ORDERFLOW = "orderflow"
    NEWS = "news"
    CORRELATION = "correlation"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"


@dataclass
class QualityDimensions:
    """Individual quality dimension scores (0-1 range)"""
    pattern_strength: float = 0.5
    mtf_agreement: float = 0.5
    regime_confidence: float = 0.5
    volume_confirmation: float = 0.5
    sentiment_alignment: float = 0.5
    correlation_safety: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            'pattern_strength': self.pattern_strength,
            'mtf_agreement': self.mtf_agreement,
            'regime_confidence': self.regime_confidence,
            'volume_confirmation': self.volume_confirmation,
            'sentiment_alignment': self.sentiment_alignment,
            'correlation_safety': self.correlation_safety
        }


@dataclass
class QualityWeights:
    """Configurable weights for quality dimensions"""
    pattern_strength: float = 0.25
    mtf_agreement: float = 0.20
    regime_confidence: float = 0.20
    volume_confirmation: float = 0.15
    sentiment_alignment: float = 0.10
    correlation_safety: float = 0.10

    def normalize(self) -> 'QualityWeights':
        """Ensure weights sum to 1.0"""
        total = sum([
            self.pattern_strength, self.mtf_agreement, self.regime_confidence,
            self.volume_confirmation, self.sentiment_alignment, self.correlation_safety
        ])
        if total == 0:
            return self
        return QualityWeights(
            pattern_strength=self.pattern_strength / total,
            mtf_agreement=self.mtf_agreement / total,
            regime_confidence=self.regime_confidence / total,
            volume_confirmation=self.volume_confirmation / total,
            sentiment_alignment=self.sentiment_alignment / total,
            correlation_safety=self.correlation_safety / total
        )


@dataclass
class SignalQualityScore:
    """Complete quality assessment for a signal"""
    composite_score: float
    dimensions: QualityDimensions
    weights_used: QualityWeights
    threshold: float
    passed: bool
    source: SignalSource
    regime: Optional[str] = None
    timestamp: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SignalQualityScorer:
    """
    Unified signal quality scoring system.

    Evaluates signals across 6 dimensions and produces a composite quality score.
    Quality gates can be applied before execution.
    """

    def __init__(
        self,
        default_threshold: float = 0.65,
        regime_specific_thresholds: Optional[Dict[str, float]] = None,
        regime_specific_weights: Optional[Dict[str, QualityWeights]] = None
    ):
        """
        Initialize quality scorer.

        Args:
            default_threshold: Minimum composite score for execution (0-1)
            regime_specific_thresholds: Override thresholds per regime
            regime_specific_weights: Override dimension weights per regime
        """
        self.default_threshold = default_threshold
        self.regime_thresholds = regime_specific_thresholds or {}
        self.regime_weights = regime_specific_weights or {}

        # Default weights (equal importance initially)
        self.default_weights = QualityWeights().normalize()

    def compute_composite_score(
        self,
        dimensions: QualityDimensions,
        weights: Optional[QualityWeights] = None
    ) -> float:
        """
        Compute weighted composite quality score.

        Args:
            dimensions: Individual dimension scores
            weights: Dimension weights (uses default if None)

        Returns:
            Composite score (0-1)
        """
        if weights is None:
            weights = self.default_weights

        scores = dimensions.to_dict()
        weight_dict = {
            'pattern_strength': weights.pattern_strength,
            'mtf_agreement': weights.mtf_agreement,
            'regime_confidence': weights.regime_confidence,
            'volume_confirmation': weights.volume_confirmation,
            'sentiment_alignment': weights.sentiment_alignment,
            'correlation_safety': weights.correlation_safety
        }

        # Weighted sum
        composite = sum(scores[k] * weight_dict[k] for k in scores.keys())
        return np.clip(composite, 0.0, 1.0)

    def score_signal(
        self,
        dimensions: QualityDimensions,
        source: SignalSource,
        regime: Optional[str] = None,
        override_weights: Optional[QualityWeights] = None,
        override_threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SignalQualityScore:
        """
        Score a signal and determine if it passes quality gate.

        Args:
            dimensions: Individual quality dimensions
            source: Signal source type
            regime: Current market regime
            override_weights: Optional weight override
            override_threshold: Optional threshold override
            metadata: Additional metadata

        Returns:
            Complete quality assessment
        """
        # Determine weights to use
        if override_weights:
            weights = override_weights.normalize()
        elif regime and regime in self.regime_weights:
            weights = self.regime_weights[regime].normalize()
        else:
            weights = self.default_weights

        # Compute composite score
        composite = self.compute_composite_score(dimensions, weights)

        # Determine threshold to use
        if override_threshold is not None:
            threshold = override_threshold
        elif regime and regime in self.regime_thresholds:
            threshold = self.regime_thresholds[regime]
        else:
            threshold = self.default_threshold

        # Check if passed
        passed = composite >= threshold

        return SignalQualityScore(
            composite_score=composite,
            dimensions=dimensions,
            weights_used=weights,
            threshold=threshold,
            passed=passed,
            source=source,
            regime=regime,
            timestamp=int(datetime.now().timestamp() * 1000),
            metadata=metadata or {}
        )

    def score_pattern_signal(
        self,
        pattern_confidence: float,
        mtf_confirmations: List[bool],
        regime_probability: float,
        volume_ratio: float,
        sentiment_score: Optional[float] = None,
        correlation_risk: float = 0.0,
        regime: Optional[str] = None,
        **kwargs
    ) -> SignalQualityScore:
        """
        Score a pattern-based signal.
        
        HIGH-003: Uses pattern confidence scores in quality assessment

        Args:
            pattern_confidence: Pattern detector confidence (0-1) from PatternEvent.score
            mtf_confirmations: List of boolean confirmations per timeframe
            regime_probability: HMM regime probability (0-1)
            volume_ratio: Volume relative to average (0-2+)
            sentiment_score: Optional sentiment alignment (-1 to +1)
            correlation_risk: Portfolio correlation risk (0-1, 0=safe)
            regime: Current regime

        Returns:
            Quality assessment with composite score weighted by pattern confidence
        """
        # HIGH-003: Pattern confidence directly influences quality score
        # Convert inputs to dimension scores
        pattern_strength = np.clip(pattern_confidence, 0.0, 1.0)

        # MTF agreement: ratio of confirmations
        mtf_agreement = np.mean(mtf_confirmations) if mtf_confirmations else 0.5

        # Regime confidence
        regime_confidence = np.clip(regime_probability, 0.0, 1.0)

        # Volume confirmation: higher volume = more confidence
        # Volume ratio of 1.5+ is strong, below 0.8 is weak
        if volume_ratio >= 1.5:
            volume_confirmation = 1.0
        elif volume_ratio >= 1.2:
            volume_confirmation = 0.8
        elif volume_ratio >= 0.8:
            volume_confirmation = 0.6
        else:
            volume_confirmation = 0.3

        # Sentiment alignment: convert -1/+1 to 0/1 (distance from neutral)
        if sentiment_score is not None:
            sentiment_alignment = abs(sentiment_score)  # Strong sentiment either way
        else:
            sentiment_alignment = 0.5  # Neutral if unavailable

        # Correlation safety: inverse of correlation risk
        correlation_safety = 1.0 - np.clip(correlation_risk, 0.0, 1.0)

        dimensions = QualityDimensions(
            pattern_strength=pattern_strength,
            mtf_agreement=mtf_agreement,
            regime_confidence=regime_confidence,
            volume_confirmation=volume_confirmation,
            sentiment_alignment=sentiment_alignment,
            correlation_safety=correlation_safety
        )

        return self.score_signal(
            dimensions=dimensions,
            source=SignalSource.PATTERN,
            regime=regime,
            metadata=kwargs
        )

    def score_ensemble_signal(
        self,
        model_predictions: List[float],
        model_confidences: List[float],
        model_agreement: float,
        regime_probability: float,
        volume_ratio: float,
        sentiment_score: Optional[float] = None,
        correlation_risk: float = 0.0,
        regime: Optional[str] = None,
        **kwargs
    ) -> SignalQualityScore:
        """
        Score an ensemble prediction signal.

        Args:
            model_predictions: List of model predictions
            model_confidences: Confidence scores per model (0-1)
            model_agreement: Agreement between models (0-1)
            regime_probability: HMM regime probability (0-1)
            volume_ratio: Volume relative to average
            sentiment_score: Optional sentiment alignment
            correlation_risk: Portfolio correlation risk
            regime: Current regime

        Returns:
            Quality assessment
        """
        # Pattern strength = average model confidence
        pattern_strength = np.mean(model_confidences) if model_confidences else 0.5

        # MTF agreement = model agreement
        mtf_agreement = np.clip(model_agreement, 0.0, 1.0)

        # Regime confidence
        regime_confidence = np.clip(regime_probability, 0.0, 1.0)

        # Volume confirmation
        if volume_ratio >= 1.5:
            volume_confirmation = 1.0
        elif volume_ratio >= 1.2:
            volume_confirmation = 0.8
        elif volume_ratio >= 0.8:
            volume_confirmation = 0.6
        else:
            volume_confirmation = 0.3

        # Sentiment alignment
        if sentiment_score is not None:
            sentiment_alignment = abs(sentiment_score)
        else:
            sentiment_alignment = 0.5

        # Correlation safety
        correlation_safety = 1.0 - np.clip(correlation_risk, 0.0, 1.0)

        dimensions = QualityDimensions(
            pattern_strength=pattern_strength,
            mtf_agreement=mtf_agreement,
            regime_confidence=regime_confidence,
            volume_confirmation=volume_confirmation,
            sentiment_alignment=sentiment_alignment,
            correlation_safety=correlation_safety
        )

        return self.score_signal(
            dimensions=dimensions,
            source=SignalSource.ENSEMBLE,
            regime=regime,
            metadata={
                'model_count': len(model_predictions),
                'prediction_std': np.std(model_predictions) if len(model_predictions) > 1 else 0.0,
                **kwargs
            }
        )

    def update_regime_weights(self, regime: str, weights: QualityWeights):
        """Update dimension weights for a specific regime"""
        self.regime_weights[regime] = weights.normalize()

    def update_regime_threshold(self, regime: str, threshold: float):
        """Update quality threshold for a specific regime"""
        self.regime_thresholds[regime] = np.clip(threshold, 0.0, 1.0)

    def get_quality_statistics(
        self,
        quality_scores: List[SignalQualityScore]
    ) -> Dict[str, Any]:
        """
        Compute statistics over a set of quality scores.

        Args:
            quality_scores: List of quality assessments

        Returns:
            Statistics dictionary
        """
        if not quality_scores:
            return {}

        composites = [s.composite_score for s in quality_scores]
        passed = [s.passed for s in quality_scores]

        # Aggregate dimension scores
        dim_scores = {
            'pattern_strength': [s.dimensions.pattern_strength for s in quality_scores],
            'mtf_agreement': [s.dimensions.mtf_agreement for s in quality_scores],
            'regime_confidence': [s.dimensions.regime_confidence for s in quality_scores],
            'volume_confirmation': [s.dimensions.volume_confirmation for s in quality_scores],
            'sentiment_alignment': [s.dimensions.sentiment_alignment for s in quality_scores],
            'correlation_safety': [s.dimensions.correlation_safety for s in quality_scores]
        }

        stats = {
            'count': len(quality_scores),
            'pass_rate': np.mean(passed),
            'composite_mean': np.mean(composites),
            'composite_std': np.std(composites),
            'composite_min': np.min(composites),
            'composite_max': np.max(composites),
            'dimension_means': {k: np.mean(v) for k, v in dim_scores.items()},
            'dimension_stds': {k: np.std(v) for k, v in dim_scores.items()}
        }

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary"""
        return {
            'default_threshold': self.default_threshold,
            'regime_thresholds': self.regime_thresholds,
            'regime_weights': {
                k: {
                    'pattern_strength': v.pattern_strength,
                    'mtf_agreement': v.mtf_agreement,
                    'regime_confidence': v.regime_confidence,
                    'volume_confirmation': v.volume_confirmation,
                    'sentiment_alignment': v.sentiment_alignment,
                    'correlation_safety': v.correlation_safety
                }
                for k, v in self.regime_weights.items()
            },
            'default_weights': {
                'pattern_strength': self.default_weights.pattern_strength,
                'mtf_agreement': self.default_weights.mtf_agreement,
                'regime_confidence': self.default_weights.regime_confidence,
                'volume_confirmation': self.default_weights.volume_confirmation,
                'sentiment_alignment': self.default_weights.sentiment_alignment,
                'correlation_safety': self.default_weights.correlation_safety
            }
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'SignalQualityScorer':
        """Create scorer from configuration dictionary"""
        regime_weights = {}
        if 'regime_weights' in config:
            for regime, weights in config['regime_weights'].items():
                regime_weights[regime] = QualityWeights(**weights)

        scorer = cls(
            default_threshold=config.get('default_threshold', 0.65),
            regime_specific_thresholds=config.get('regime_thresholds', {}),
            regime_specific_weights=regime_weights
        )

        if 'default_weights' in config:
            scorer.default_weights = QualityWeights(**config['default_weights']).normalize()

        return scorer
