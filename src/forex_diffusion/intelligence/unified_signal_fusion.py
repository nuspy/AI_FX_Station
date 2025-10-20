"""
Unified Signal Fusion System

Integrates all signal sources (patterns, harmonics, order flow, correlations, events, ensemble)
with quality scoring and regime-aware filtering.

This is the central hub that connects all new components to the existing trading workflow.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

# Import all signal components
from .signal_quality_scorer import (
    SignalQualityScorer,
    QualityDimensions,
    SignalSource,
    SignalQualityScore
)
from .enhanced_calibration import EnhancedConformalCalibrator
from .event_signal_processor import EventSignalProcessor, EventSignal
from .adaptive_parameter_system import AdaptiveParameterSystem
from ..analysis.order_flow_analyzer import OrderFlowAnalyzer, OrderFlowSignal
from ..analysis.correlation_analyzer import CrossAssetCorrelationAnalyzer, CorrelationSignal
from ..regime.enhanced_regime_detector import EnhancedRegimeDetector, RegimeState


@dataclass
class FusedSignal:
    """Unified signal with quality assessment"""
    signal_id: str
    source: SignalSource
    symbol: str
    timeframe: str
    direction: str  # bull, bear, neutral
    strength: float  # 0-1

    # Quality assessment
    quality_score: SignalQualityScore

    # Regime context
    regime: Optional[str] = None
    regime_confidence: float = 0.5

    # Price levels
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Timing
    timestamp: int = 0
    valid_until: Optional[int] = None

    # Metadata
    original_signal: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class UnifiedSignalFusion:
    """
    Central signal fusion system integrating all components.

    Workflow:
    1. Collect signals from all sources (patterns, order flow, correlations, events, ensemble)
    2. Score each signal with quality dimensions
    3. Filter by regime and quality threshold
    4. Rank signals by composite quality score
    5. Apply correlation safety checks
    6. Output fused signals for execution
    """

    def __init__(
        self,
        quality_scorer: Optional[SignalQualityScorer] = None,
        calibrator: Optional[EnhancedConformalCalibrator] = None,
        regime_detector: Optional[EnhancedRegimeDetector] = None,
        event_processor: Optional[EventSignalProcessor] = None,
        orderflow_analyzer: Optional[OrderFlowAnalyzer] = None,
        correlation_analyzer: Optional[CrossAssetCorrelationAnalyzer] = None,
        adaptive_params: Optional[AdaptiveParameterSystem] = None,
        default_quality_threshold: float = 0.65,
        max_signals_per_regime: int = 5
    ):
        """
        Initialize unified signal fusion.

        Args:
            quality_scorer: Signal quality scorer
            calibrator: Enhanced calibrator
            regime_detector: Regime detector
            event_processor: Event signal processor
            orderflow_analyzer: Order flow analyzer
            correlation_analyzer: Correlation analyzer
            adaptive_params: Adaptive parameter system
            default_quality_threshold: Default quality threshold
            max_signals_per_regime: Maximum signals per regime
        """
        self.quality_scorer = quality_scorer or SignalQualityScorer()
        self.calibrator = calibrator
        self.regime_detector = regime_detector
        self.event_processor = event_processor
        self.orderflow_analyzer = orderflow_analyzer
        self.correlation_analyzer = correlation_analyzer
        self.adaptive_params = adaptive_params

        self.default_quality_threshold = default_quality_threshold
        self.max_signals_per_regime = max_signals_per_regime

        # State
        self.current_regime: Optional[str] = None
        self.current_regime_confidence: float = 0.5
        self.open_positions: List[str] = []

    def set_current_regime(self, regime: str, confidence: float):
        """Update current market regime"""
        self.current_regime = regime
        self.current_regime_confidence = confidence

    def set_open_positions(self, positions: List[str]):
        """Update list of open positions"""
        self.open_positions = positions

    def get_active_parameters(self) -> Dict[str, float]:
        """
        Get currently active parameters (from adaptive system if available).

        Returns:
            Dictionary of parameter values
        """
        if self.adaptive_params:
            params = self.adaptive_params.get_current_parameters()
            # Use adaptive parameters if available, otherwise use defaults
            return {
                'quality_threshold': params.get('quality_threshold', self.default_quality_threshold),
                'max_signals_per_regime': params.get('max_signals_per_regime', self.max_signals_per_regime),
                'position_size_multiplier': params.get('position_size_multiplier', 1.0),
                'stop_loss_distance': params.get('stop_loss_distance', 1.5),
                'take_profit_distance': params.get('take_profit_distance', 2.0),
            }
        else:
            return {
                'quality_threshold': self.default_quality_threshold,
                'max_signals_per_regime': self.max_signals_per_regime,
                'position_size_multiplier': 1.0,
                'stop_loss_distance': 1.5,
                'take_profit_distance': 2.0,
            }

    def record_signal_outcome(
        self,
        signal_id: str,
        symbol: str,
        pnl: float,
        outcome: str,
        timestamp: int
    ):
        """
        Record signal outcome for adaptive parameter system.

        Args:
            signal_id: Signal identifier
            symbol: Trading symbol
            pnl: Profit/loss
            outcome: 'win', 'loss', or 'breakeven'
            timestamp: Timestamp of outcome
        """
        if self.adaptive_params:
            params = self.get_active_parameters()
            self.adaptive_params.record_trade(
                timestamp=timestamp,
                symbol=symbol,
                regime=self.current_regime or 'unknown',
                pnl=pnl,
                outcome=outcome,
                parameters_used=params
            )

            # Check if adaptation should be triggered
            should_trigger, trigger_reason, metrics = self.adaptive_params.should_trigger_adaptation()
            if should_trigger:
                # Run adaptation cycle
                adaptations = self.adaptive_params.run_adaptation_cycle()
                if adaptations:
                    # Update quality threshold if adapted
                    for adaptation in adaptations:
                        if adaptation.parameter_name == 'quality_threshold' and adaptation.deployed:
                            self.default_quality_threshold = adaptation.new_value

    def _collect_ldm4ts_signals(
        self,
        symbol: str,
        timeframe: str,
        current_ohlcv: np.ndarray,
        horizons: List[int] = [15, 60, 240]
    ) -> List[FusedSignal]:
        """
        Collect LDM4TS forecast signals.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            current_ohlcv: Latest OHLCV data [L, 5]
            horizons: Forecast horizons in minutes
            
        Returns:
            List of FusedSignals
        """
        signals = []
        
        try:
            from ..inference.ldm4ts_inference import LDM4TSInferenceService
            
            # Get service instance
            service = LDM4TSInferenceService.get_instance()
            
            if not service._initialized:
                logger.debug("LDM4TS not initialized, skipping")
                return signals
            
            # Run inference
            prediction = service.predict(
                ohlcv=current_ohlcv,
                horizons=horizons,
                num_samples=50,
                symbol=symbol
            )
            
            current_price = float(current_ohlcv[-1, 3])
            
            # Create signal for each horizon
            for horizon in prediction.horizons:
                mean_pred = prediction.mean[horizon]
                uncertainty = prediction.std[horizon]
                uncertainty_pct = uncertainty / current_price
                
                # Direction
                price_change = mean_pred - current_price
                direction = "bull" if price_change > 0 else ("bear" if price_change < 0 else "neutral")
                
                # Strength (Sharpe-like: change / uncertainty)
                if uncertainty > 0:
                    strength = min(abs(price_change) / uncertainty, 1.0)
                else:
                    strength = 0.5
                
                # Price levels
                entry_price = current_price
                target_price = mean_pred
                
                if direction == "bull":
                    stop_price = prediction.q05[horizon]
                elif direction == "bear":
                    stop_price = prediction.q95[horizon]
                else:
                    stop_price = current_price * 0.999  # 0.1% stop
                
                signal = FusedSignal(
                    signal_id=f"ldm4ts_{symbol}_{timeframe}_{horizon}m_{int(time.time())}",
                    source=SignalSource.LDM4TS_FORECAST,
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    strength=strength,
                    quality_score=None,  # Filled later
                    regime=self.current_regime,
                    regime_confidence=self.current_regime_confidence,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_price=stop_price,
                    timestamp=int(prediction.timestamp.timestamp() * 1000),
                    valid_until=int((prediction.timestamp + pd.Timedelta(minutes=horizon)).timestamp() * 1000),
                    metadata={
                        "horizon_minutes": horizon,
                        "mean_pred": mean_pred,
                        "uncertainty": uncertainty,
                        "uncertainty_pct": uncertainty_pct,
                        "q05": prediction.q05[horizon],
                        "q50": prediction.q50[horizon],
                        "q95": prediction.q95[horizon],
                        "price_change": price_change,
                        "price_change_pct": price_change / current_price,
                        "model_name": "LDM4TS",
                        "inference_time_ms": prediction.inference_time_ms,
                        "num_samples": prediction.num_samples
                    }
                )
                
                signals.append(signal)
            
            logger.debug(f"Generated {len(signals)} LDM4TS signals for {symbol}")
            
        except Exception as e:
            logger.error(f"LDM4TS signal collection failed: {e}")
        
        return signals

    def fuse_signals(
        self,
        pattern_signals: Optional[List[Any]] = None,
        ensemble_predictions: Optional[List[Any]] = None,
        orderflow_signals: Optional[List[OrderFlowSignal]] = None,
        correlation_signals: Optional[List[CorrelationSignal]] = None,
        event_signals: Optional[List[EventSignal]] = None,
        ldm4ts_ohlcv: Optional[np.ndarray] = None,  # NEW: OHLCV for LDM4TS
        market_data: Optional[pd.DataFrame] = None,
        sentiment_score: Optional[float] = None
    ) -> List[FusedSignal]:
        """
        Fuse signals from all sources with quality scoring.

        Args:
            pattern_signals: Signals from pattern detectors
            ensemble_predictions: Ensemble model predictions
            orderflow_signals: Order flow signals
            correlation_signals: Correlation-based signals
            event_signals: Event-driven signals
            market_data: Current market data for context
            sentiment_score: Current sentiment score

        Returns:
            List of fused signals sorted by quality
        """
        fused_signals: List[FusedSignal] = []

        # Process pattern signals
        if pattern_signals:
            fused_signals.extend(self._process_pattern_signals(
                pattern_signals, market_data, sentiment_score
            ))

        # Process ensemble predictions
        if ensemble_predictions:
            fused_signals.extend(self._process_ensemble_predictions(
                ensemble_predictions, market_data, sentiment_score
            ))

        # Process order flow signals
        if orderflow_signals:
            fused_signals.extend(self._process_orderflow_signals(
                orderflow_signals, sentiment_score
            ))

        # Process correlation signals
        if correlation_signals:
            fused_signals.extend(self._process_correlation_signals(
                correlation_signals, sentiment_score
            ))

        # Process event signals
        if event_signals:
            fused_signals.extend(self._process_event_signals(
                event_signals, sentiment_score
            ))
        
        # NEW: Process LDM4TS forecasts
        if ldm4ts_ohlcv is not None and len(ldm4ts_ohlcv) >= 100:
            try:
                # Extract symbol/timeframe from market_data
                symbol = market_data.get('symbol', 'EUR/USD') if isinstance(market_data, dict) else 'EUR/USD'
                timeframe = market_data.get('timeframe', '1m') if isinstance(market_data, dict) else '1m'
                
                ldm4ts_signals = self._collect_ldm4ts_signals(
                    symbol=symbol,
                    timeframe=timeframe,
                    current_ohlcv=ldm4ts_ohlcv,
                    horizons=[15, 60, 240]
                )
                
                if ldm4ts_signals:
                    logger.info(f"✅ Added {len(ldm4ts_signals)} LDM4TS forecast signals for {symbol}")
                    fused_signals.extend(ldm4ts_signals)
                    
            except Exception as e:
                logger.error(f"Failed to collect LDM4TS signals: {e}", exc_info=True)

        # Filter by regime and quality
        filtered_signals = self._filter_by_regime_and_quality(fused_signals)

        # Apply correlation safety checks
        safe_signals = self._apply_correlation_safety(filtered_signals)

        # Rank by quality score
        ranked_signals = sorted(
            safe_signals,
            key=lambda s: s.quality_score.composite_score,
            reverse=True
        )

        # Limit to max signals per regime (use adaptive parameter if available)
        active_params = self.get_active_parameters()
        max_signals = int(active_params.get('max_signals_per_regime', self.max_signals_per_regime))
        return ranked_signals[:max_signals]

    def _process_pattern_signals(
        self,
        pattern_signals: List[Any],
        market_data: Optional[pd.DataFrame],
        sentiment_score: Optional[float]
    ) -> List[FusedSignal]:
        """
        Process pattern detection signals.
        
        HIGH-003: Uses pattern confidence scores from PatternEvent.score field
        """
        fused = []

        for signal in pattern_signals:
            # Extract pattern information
            # HIGH-003: PatternEvent uses 'score' field for confidence
            pattern_confidence = getattr(signal, 'score', getattr(signal, 'confidence', 0.7))
            direction = getattr(signal, 'direction', 'neutral')
            
            # Convert direction to string if it's an enum
            if hasattr(direction, 'value'):
                direction = direction.value

            # Calculate MTF confirmations (simplified)
            mtf_confirmations = [True] * 2 + [False] * 1  # Example

            # Get volume ratio (simplified)
            volume_ratio = 1.2

            # Get correlation risk
            correlation_risk = self._get_correlation_risk(getattr(signal, 'symbol', ''))

            # Score the signal
            quality_score = self.quality_scorer.score_pattern_signal(
                pattern_confidence=pattern_confidence,
                mtf_confirmations=mtf_confirmations,
                regime_probability=self.current_regime_confidence,
                volume_ratio=volume_ratio,
                sentiment_score=sentiment_score,
                correlation_risk=correlation_risk,
                regime=self.current_regime
            )

            # Create fused signal
            fused_signal = FusedSignal(
                signal_id=f"pattern_{id(signal)}",
                source=SignalSource.PATTERN,
                symbol=getattr(signal, 'symbol', 'UNKNOWN'),
                timeframe=getattr(signal, 'timeframe', '1H'),
                direction=direction,
                strength=pattern_confidence,
                quality_score=quality_score,
                regime=self.current_regime,
                regime_confidence=self.current_regime_confidence,
                entry_price=getattr(signal, 'entry_price', None),
                target_price=getattr(signal, 'target_price', None),
                stop_price=getattr(signal, 'stop_price', None),
                timestamp=int(datetime.now().timestamp() * 1000),
                original_signal=signal
            )

            fused.append(fused_signal)

        return fused

    def _process_ensemble_predictions(
        self,
        ensemble_predictions: List[Any],
        market_data: Optional[pd.DataFrame],
        sentiment_score: Optional[float]
    ) -> List[FusedSignal]:
        """Process ensemble model predictions"""
        fused = []

        # Implementation would process ensemble predictions
        # For now, return empty list as placeholder

        return fused

    def _process_orderflow_signals(
        self,
        orderflow_signals: List[OrderFlowSignal],
        sentiment_score: Optional[float]
    ) -> List[FusedSignal]:
        """Process order flow signals"""
        fused = []

        for signal in orderflow_signals:
            # Create quality dimensions
            dimensions = QualityDimensions(
                pattern_strength=signal.strength,
                mtf_agreement=0.7,  # Order flow doesn't have MTF
                regime_confidence=self.current_regime_confidence,
                volume_confirmation=signal.confidence,  # Use signal confidence as volume proxy
                sentiment_alignment=abs(sentiment_score) if sentiment_score else 0.5,
                correlation_safety=1.0 - self._get_correlation_risk(signal.symbol)
            )

            quality_score = self.quality_scorer.score_signal(
                dimensions=dimensions,
                source=SignalSource.ORDERFLOW,
                regime=self.current_regime
            )

            fused_signal = FusedSignal(
                signal_id=f"orderflow_{signal.symbol}_{signal.timestamp}",
                source=SignalSource.ORDERFLOW,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                direction=signal.direction,
                strength=signal.strength,
                quality_score=quality_score,
                regime=self.current_regime,
                regime_confidence=self.current_regime_confidence,
                entry_price=signal.entry_price,
                target_price=signal.target_price,
                stop_price=signal.stop_price,
                timestamp=signal.timestamp,
                original_signal=signal
            )

            fused.append(fused_signal)

        return fused

    def _process_correlation_signals(
        self,
        correlation_signals: List[CorrelationSignal],
        sentiment_score: Optional[float]
    ) -> List[FusedSignal]:
        """Process correlation-based signals"""
        fused = []

        for signal in correlation_signals:
            dimensions = QualityDimensions(
                pattern_strength=signal.strength,
                mtf_agreement=0.6,
                regime_confidence=self.current_regime_confidence,
                volume_confirmation=0.7,
                sentiment_alignment=abs(sentiment_score) if sentiment_score else 0.5,
                correlation_safety=signal.confidence  # Correlation signals have inherent safety
            )

            quality_score = self.quality_scorer.score_signal(
                dimensions=dimensions,
                source=SignalSource.CORRELATION,
                regime=self.current_regime
            )

            fused_signal = FusedSignal(
                signal_id=f"correlation_{signal.primary_asset}_{signal.timestamp}",
                source=SignalSource.CORRELATION,
                symbol=signal.primary_asset,
                timeframe='1H',  # Default
                direction=signal.direction,
                strength=signal.strength,
                quality_score=quality_score,
                regime=self.current_regime,
                regime_confidence=self.current_regime_confidence,
                timestamp=signal.timestamp,
                original_signal=signal
            )

            fused.append(fused_signal)

        return fused

    def _process_event_signals(
        self,
        event_signals: List[EventSignal],
        sentiment_score: Optional[float]
    ) -> List[FusedSignal]:
        """Process event-driven signals"""
        fused = []

        for signal in event_signals:
            dimensions = QualityDimensions(
                pattern_strength=signal.signal_strength,
                mtf_agreement=0.5,  # Events don't have MTF
                regime_confidence=self.current_regime_confidence,
                volume_confirmation=0.8 if signal.event.impact_level.value == 'high' else 0.5,
                sentiment_alignment=abs(signal.sentiment_score) if signal.sentiment_score else 0.5,
                correlation_safety=0.7  # Events affect multiple assets
            )

            quality_score = self.quality_scorer.score_signal(
                dimensions=dimensions,
                source=SignalSource.NEWS,
                regime=self.current_regime
            )

            # Process each affected symbol
            for symbol in signal.affected_symbols[:3]:  # Limit to 3 symbols
                fused_signal = FusedSignal(
                    signal_id=f"event_{signal.event.event_id}_{symbol}",
                    source=SignalSource.NEWS,
                    symbol=symbol,
                    timeframe='1H',
                    direction=signal.signal_direction,
                    strength=signal.signal_strength,
                    quality_score=quality_score,
                    regime=self.current_regime,
                    regime_confidence=self.current_regime_confidence,
                    timestamp=signal.event.timestamp,
                    valid_until=signal.valid_until,
                    original_signal=signal
                )

                fused.append(fused_signal)

        return fused

    def _filter_by_regime_and_quality(
        self,
        signals: List[FusedSignal]
    ) -> List[FusedSignal]:
        """Filter signals by regime and quality threshold (using adaptive params if available)"""
        filtered = []

        # Get active parameters (may be adapted)
        active_params = self.get_active_parameters()
        quality_threshold = active_params['quality_threshold']

        for signal in signals:
            # Check regime action
            if self.regime_detector and hasattr(self.regime_detector, 'current_regime'):
                # If in transition regime, skip new signals
                if self.regime_detector.current_regime == RegimeState.TRANSITION:
                    continue

            # Check quality threshold (use adaptive threshold)
            if signal.quality_score.composite_score >= quality_threshold:
                filtered.append(signal)

        return filtered

    def _apply_correlation_safety(
        self,
        signals: List[FusedSignal]
    ) -> List[FusedSignal]:
        """Apply correlation safety checks"""
        if not self.correlation_analyzer:
            return signals

        safe_signals = []

        for signal in signals:
            # Check correlation with open positions
            # For now, simple check
            if signal.quality_score.dimensions.correlation_safety >= 0.5:
                safe_signals.append(signal)

        return safe_signals

    def _get_correlation_risk(self, symbol: str) -> float:
        """Get correlation risk for a symbol given open positions"""
        if not self.open_positions or symbol in self.open_positions:
            return 0.0

        # Simplified: assume moderate risk if we have open positions
        return 0.3 if len(self.open_positions) > 0 else 0.0
