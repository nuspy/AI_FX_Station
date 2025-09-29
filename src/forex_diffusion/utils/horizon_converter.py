"""
Enhanced Horizon conversion utilities for multi-horizon predictions.

Handles conversion between:
- Training format: horizon in bars (e.g., 5)
- Inference format: time labels (e.g., ["1m", "5m", "15m"])
- Smart scaling for multi-horizon predictions from single models
- Scenario-based configurations for different trading styles
"""
from __future__ import annotations

from typing import List, Union, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


def timeframe_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    tf = str(tf).strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    elif tf.endswith("h"):
        return int(tf[:-1]) * 60
    elif tf.endswith("d"):
        return int(tf[:-1]) * 24 * 60
    elif tf.endswith("w"):
        return int(tf[:-1]) * 7 * 24 * 60
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")


def minutes_to_timeframe(minutes: int) -> str:
    """Convert minutes to timeframe string."""
    if minutes < 60:
        return f"{minutes}m"
    elif minutes < 24 * 60:
        hours = minutes // 60
        if minutes % 60 == 0:
            return f"{hours}h"
        else:
            return f"{minutes}m"
    else:
        days = minutes // (24 * 60)
        if minutes % (24 * 60) == 0:
            return f"{days}d"
        else:
            return f"{minutes}m"


def horizon_bars_to_time_labels(
    horizon_bars: int,
    base_timeframe: str,
    target_timeframes: List[str] = None
) -> List[str]:
    """
    Convert horizon in bars to time labels.

    Args:
        horizon_bars: Number of bars to predict ahead
        base_timeframe: Base timeframe of the model (e.g., "1m")
        target_timeframes: Desired output timeframes, if None uses standard progression

    Returns:
        List of time labels corresponding to the horizon
    """
    if target_timeframes is None:
        # Standard progression based on base timeframe
        base_minutes = timeframe_to_minutes(base_timeframe)

        # Create progression: 1x, 5x, 15x, 60x of base timeframe
        multipliers = [1, 5, 15, 60] if base_minutes == 1 else [1, 3, 5, 15]
        target_timeframes = []

        for mult in multipliers:
            target_minutes = base_minutes * mult
            tf_label = minutes_to_timeframe(target_minutes)
            target_timeframes.append(tf_label)
            if len(target_timeframes) >= horizon_bars:
                break

    # Limit to the number of horizon bars
    return target_timeframes[:horizon_bars]


def time_labels_to_horizon_bars(
    time_labels: List[str],
    base_timeframe: str
) -> List[int]:
    """
    Convert time labels to horizon bars relative to base timeframe.

    Args:
        time_labels: List of time labels (e.g., ["1m", "5m", "15m"])
        base_timeframe: Base timeframe of the model

    Returns:
        List of horizon bars corresponding to each time label
    """
    base_minutes = timeframe_to_minutes(base_timeframe)
    horizon_bars = []

    for label in time_labels:
        label_minutes = timeframe_to_minutes(label)
        bars = max(1, label_minutes // base_minutes)
        horizon_bars.append(bars)

    return horizon_bars


def convert_horizons_for_inference(
    horizons: Union[List[str], List[int], int],
    base_timeframe: str,
    model_horizon_bars: int = None
) -> Tuple[List[str], List[int]]:
    """
    Convert various horizon formats to consistent inference format.

    Args:
        horizons: Horizons in various formats
        base_timeframe: Base timeframe for conversion
        model_horizon_bars: Original model horizon in bars (for validation)

    Returns:
        Tuple of (time_labels, horizon_bars)
    """
    if isinstance(horizons, int):
        # Single horizon bar value
        time_labels = horizon_bars_to_time_labels(horizons, base_timeframe)
        horizon_bars = [horizons]

    elif isinstance(horizons, list):
        if not horizons:
            # Default horizons
            time_labels = ["1m", "5m", "15m"]
            horizon_bars = time_labels_to_horizon_bars(time_labels, base_timeframe)

        elif isinstance(horizons[0], str):
            # Time labels format
            time_labels = horizons
            horizon_bars = time_labels_to_horizon_bars(time_labels, base_timeframe)

        elif isinstance(horizons[0], (int, float)):
            # Bars format
            horizon_bars = [int(h) for h in horizons]
            time_labels = []
            for bars in horizon_bars:
                labels = horizon_bars_to_time_labels(bars, base_timeframe)
                time_labels.extend(labels)
            time_labels = time_labels[:len(horizon_bars)]

        else:
            raise ValueError(f"Unsupported horizon format: {type(horizons[0])}")
    else:
        raise ValueError(f"Unsupported horizon type: {type(horizons)}")

    # Validation if model horizon is known
    if model_horizon_bars is not None:
        max_bars = max(horizon_bars) if horizon_bars else 1
        if max_bars > model_horizon_bars:
            logger.warning(
                f"Requested horizon {max_bars} bars exceeds model training horizon "
                f"{model_horizon_bars} bars. Results may be unreliable."
            )

    return time_labels, horizon_bars


def create_future_timestamps(
    last_timestamp_ms: int,
    base_timeframe: str,
    time_labels: List[str]
) -> List[int]:
    """
    Create future timestamps for predictions.

    Args:
        last_timestamp_ms: Last timestamp in milliseconds
        base_timeframe: Base timeframe for the data
        time_labels: Target time labels for predictions

    Returns:
        List of future timestamps in milliseconds
    """
    base_dt = pd.to_datetime(last_timestamp_ms, unit="ms", utc=True)
    future_timestamps = []

    for label in time_labels:
        try:
            future_dt = base_dt + pd.to_timedelta(label)
            future_timestamps.append(int(future_dt.value // 1_000_000))
        except Exception:
            # Fallback to base timeframe increment
            base_minutes = timeframe_to_minutes(base_timeframe)
            future_dt = base_dt + pd.to_timedelta(f"{base_minutes}m")
            future_timestamps.append(int(future_dt.value // 1_000_000))

    return future_timestamps


def validate_horizon_compatibility(
    inference_horizons: List[str],
    training_horizon_bars: int,
    base_timeframe: str
) -> dict:
    """
    Validate compatibility between inference horizons and training horizon.

    Returns:
        Dict with validation results
    """
    _, inference_bars = convert_horizons_for_inference(
        inference_horizons, base_timeframe
    )

    max_inference_bars = max(inference_bars) if inference_bars else 1

    results = {
        "compatible": True,
        "warnings": [],
        "max_horizon_bars": max_inference_bars,
        "training_horizon_bars": training_horizon_bars
    }

    if max_inference_bars > training_horizon_bars:
        results["compatible"] = False
        results["warnings"].append(
            f"Maximum inference horizon ({max_inference_bars} bars) exceeds "
            f"training horizon ({training_horizon_bars} bars)"
        )

    return results


# ===== ENHANCED MULTI-HORIZON SYSTEM =====

class ScalingMode(Enum):
    """Available scaling modes for multi-horizon predictions."""
    LINEAR = "linear"
    SQRT = "sqrt"
    LOG = "log"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    REGIME_AWARE = "regime_aware"
    SMART_ADAPTIVE = "smart_adaptive"


class MarketRegime(Enum):
    """Market regime types for regime-aware scaling."""
    TRENDING = "trending"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class TradingScenario:
    """Trading scenario configuration for multi-horizon predictions."""
    name: str
    horizons: List[str]
    scaling_mode: ScalingMode
    max_horizon: str
    focus: str
    exclude_weekends: bool = False
    exclude_holidays: bool = False
    session_aware: bool = False
    session_overlap: bool = False
    weekend_adjustment: bool = False
    business_days_only: bool = False
    market_hours_aware: bool = False
    session_gap_adjustment: bool = False


# Predefined trading scenarios
TRADING_SCENARIOS = {
    "scalping": TradingScenario(
        name="Scalping (High Frequency)",
        horizons=["1m", "3m", "5m", "10m", "15m"],
        scaling_mode=ScalingMode.SQRT,
        max_horizon="15m",
        focus="micro-movements"
    ),

    "intraday_4h": TradingScenario(
        name="Intraday 4h",
        horizons=["5m", "15m", "30m", "1h", "2h", "4h"],
        scaling_mode=ScalingMode.VOLATILITY_ADJUSTED,
        max_horizon="4h",
        focus="trend intraday",
        session_aware=True
    ),

    "intraday_8h": TradingScenario(
        name="Intraday 8h",
        horizons=["15m", "30m", "1h", "2h", "4h", "6h", "8h"],
        scaling_mode=ScalingMode.REGIME_AWARE,
        max_horizon="8h",
        focus="full trading session",
        session_overlap=True
    ),

    "intraday_2d": TradingScenario(
        name="Intraday 2 Days",
        horizons=["1h", "4h", "8h", "12h", "1d", "2d"],
        scaling_mode=ScalingMode.SMART_ADAPTIVE,
        max_horizon="2d",
        focus="short-term trends"
    ),

    "intraday_3d": TradingScenario(
        name="Intraday 3 Days",
        horizons=["4h", "8h", "12h", "1d", "2d", "3d"],
        scaling_mode=ScalingMode.SMART_ADAPTIVE,
        max_horizon="3d",
        focus="medium-short trends",
        weekend_adjustment=True
    ),

    "intraday_5d": TradingScenario(
        name="Intraday 5 Days",
        horizons=["8h", "12h", "1d", "2d", "3d", "4d", "5d"],
        scaling_mode=ScalingMode.LOG,
        max_horizon="5d",
        focus="weekly trends",
        exclude_weekends=True,
        business_days_only=True
    ),

    "intraday_10d": TradingScenario(
        name="Intraday 10 Days",
        horizons=["1d", "2d", "3d", "5d", "7d", "10d"],
        scaling_mode=ScalingMode.REGIME_AWARE,
        max_horizon="10d",
        focus="bi-weekly trends",
        exclude_weekends=True,
        exclude_holidays=True,
        market_hours_aware=True
    ),

    "intraday_15d": TradingScenario(
        name="Intraday 15 Days",
        horizons=["2d", "3d", "5d", "7d", "10d", "15d"],
        scaling_mode=ScalingMode.SMART_ADAPTIVE,
        max_horizon="15d",
        focus="monthly trends",
        exclude_weekends=True,
        exclude_holidays=True,
        session_gap_adjustment=True
    )
}


class MarketRegimeDetector:
    """Simple market regime detection for regime-aware scaling."""

    @staticmethod
    def detect_regime(market_data: Optional[pd.DataFrame]) -> MarketRegime:
        """Detect current market regime from recent data."""
        if market_data is None or len(market_data) < 20:
            return MarketRegime.UNKNOWN

        try:
            # Calculate volatility (rolling std of returns)
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]

            # Calculate trend strength (correlation with time)
            prices = market_data['close'].tail(20)
            time_index = np.arange(len(prices))
            correlation = np.corrcoef(prices, time_index)[0, 1]

            # Regime classification
            vol_threshold = returns.std() * 1.5
            trend_threshold = 0.3

            if volatility > vol_threshold:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < vol_threshold * 0.5:
                return MarketRegime.LOW_VOLATILITY
            elif abs(correlation) > trend_threshold:
                return MarketRegime.TRENDING
            else:
                return MarketRegime.RANGING

        except Exception as e:
            logger.warning(f"Failed to detect market regime: {e}")
            return MarketRegime.UNKNOWN


class EnhancedHorizonScaler:
    """Enhanced scaling system for multi-horizon predictions."""

    # Scaling functions
    SCALING_FUNCTIONS: Dict[ScalingMode, Callable] = {
        ScalingMode.LINEAR: lambda base, ratio, **kwargs: base * ratio,
        ScalingMode.SQRT: lambda base, ratio, **kwargs: base * np.sqrt(ratio),
        ScalingMode.LOG: lambda base, ratio, **kwargs: base * np.log1p(ratio),
    }

    # Regime adjustment factors
    REGIME_FACTORS = {
        MarketRegime.TRENDING: 1.1,      # Trends persist longer
        MarketRegime.RANGING: 0.9,       # Mean reversion stronger
        MarketRegime.HIGH_VOLATILITY: 1.2,  # Larger moves possible
        MarketRegime.LOW_VOLATILITY: 0.8,   # Smaller moves expected
        MarketRegime.UNKNOWN: 1.0
    }

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.calibration_cache = {}

    def scale_prediction(
        self,
        base_prediction: float,
        base_timeframe: str,
        target_timeframe: str,
        scaling_mode: ScalingMode = ScalingMode.SMART_ADAPTIVE,
        market_data: Optional[pd.DataFrame] = None,
        volatility: Optional[float] = None,
        regime: Optional[MarketRegime] = None
    ) -> Dict[str, float]:
        """Scale prediction from base timeframe to target timeframe."""

        # Calculate time ratio
        base_minutes = timeframe_to_minutes(base_timeframe)
        target_minutes = timeframe_to_minutes(target_timeframe)
        time_ratio = target_minutes / base_minutes

        # Detect regime if not provided
        if regime is None:
            regime = self.regime_detector.detect_regime(market_data)

        # Estimate volatility if not provided
        if volatility is None:
            volatility = self._estimate_volatility(market_data)

        # Apply scaling based on mode
        if scaling_mode == ScalingMode.VOLATILITY_ADJUSTED:
            scaled_pred = self._volatility_adjusted_scaling(
                base_prediction, time_ratio, volatility
            )
        elif scaling_mode == ScalingMode.REGIME_AWARE:
            scaled_pred = self._regime_aware_scaling(
                base_prediction, time_ratio, regime
            )
        elif scaling_mode == ScalingMode.SMART_ADAPTIVE:
            scaled_pred = self._smart_adaptive_scaling(
                base_prediction, time_ratio, volatility, regime, market_data
            )
        else:
            # Use basic scaling functions
            scaling_func = self.SCALING_FUNCTIONS.get(
                scaling_mode, self.SCALING_FUNCTIONS[ScalingMode.LINEAR]
            )
            scaled_pred = scaling_func(base_prediction, time_ratio)

        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(
            base_prediction, time_ratio, volatility, regime
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            target_timeframe, time_ratio, volatility, regime
        )

        return {
            "prediction": scaled_pred,
            "lower": scaled_pred - uncertainty,
            "upper": scaled_pred + uncertainty,
            "confidence": confidence,
            "regime": regime.value,
            "volatility": volatility,
            "time_ratio": time_ratio,
            "scaling_mode": scaling_mode.value
        }

    def _volatility_adjusted_scaling(
        self, base_pred: float, time_ratio: float, volatility: float
    ) -> float:
        """Scale with volatility adjustment."""
        vol_factor = 1.0 + (volatility - 0.01) * 2.0  # Normalize around 1% volatility
        vol_factor = np.clip(vol_factor, 0.5, 2.0)  # Reasonable bounds
        return base_pred * np.sqrt(time_ratio) * vol_factor

    def _regime_aware_scaling(
        self, base_pred: float, time_ratio: float, regime: MarketRegime
    ) -> float:
        """Scale with regime awareness."""
        regime_factor = self.REGIME_FACTORS.get(regime, 1.0)
        return base_pred * np.sqrt(time_ratio) * regime_factor

    def _smart_adaptive_scaling(
        self,
        base_pred: float,
        time_ratio: float,
        volatility: float,
        regime: MarketRegime,
        market_data: Optional[pd.DataFrame]
    ) -> float:
        """Adaptive scaling combining multiple factors."""

        # Base scaling (sqrt for non-linear time decay)
        base_scaled = base_pred * np.sqrt(time_ratio)

        # Volatility adjustment
        vol_factor = 1.0 + (volatility - 0.01) * 1.5
        vol_factor = np.clip(vol_factor, 0.6, 1.8)

        # Regime adjustment
        regime_factor = self.REGIME_FACTORS.get(regime, 1.0)

        # Time decay factor (predictions become less reliable over time)
        decay_factor = 1.0 / (1.0 + 0.1 * np.log1p(time_ratio))

        # Session factor (if crossing major sessions)
        session_factor = self._calculate_session_factor(time_ratio)

        final_prediction = (
            base_scaled * vol_factor * regime_factor * decay_factor * session_factor
        )

        return final_prediction

    def _estimate_volatility(self, market_data: Optional[pd.DataFrame]) -> float:
        """Estimate current market volatility."""
        if market_data is None or len(market_data) < 10:
            return 0.01  # Default 1% volatility

        try:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(20, min_periods=5).std().iloc[-1]
            return float(volatility) if not np.isnan(volatility) else 0.01
        except Exception:
            return 0.01

    def _calculate_uncertainty(
        self,
        base_pred: float,
        time_ratio: float,
        volatility: float,
        regime: MarketRegime
    ) -> float:
        """Calculate prediction uncertainty based on horizon and market conditions."""

        # Base uncertainty grows with time horizon
        base_uncertainty = abs(base_pred) * 0.02 * np.sqrt(time_ratio)

        # Volatility adjustment
        vol_multiplier = 1.0 + volatility * 50  # Scale volatility impact

        # Regime adjustment
        regime_multipliers = {
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.LOW_VOLATILITY: 0.7,
            MarketRegime.RANGING: 0.8,
            MarketRegime.TRENDING: 1.2,
            MarketRegime.UNKNOWN: 1.0
        }
        regime_multiplier = regime_multipliers.get(regime, 1.0)

        uncertainty = base_uncertainty * vol_multiplier * regime_multiplier
        return uncertainty

    def _calculate_confidence(
        self,
        target_timeframe: str,
        time_ratio: float,
        volatility: float,
        regime: MarketRegime
    ) -> float:
        """Calculate confidence score for the prediction."""

        # Base confidence decreases with time horizon
        base_confidence = 0.85 / (1.0 + 0.1 * time_ratio)

        # Volatility penalty
        vol_penalty = volatility * 10  # High volatility reduces confidence

        # Regime bonus/penalty
        regime_adjustments = {
            MarketRegime.TRENDING: 0.05,      # More confident in trends
            MarketRegime.RANGING: -0.02,      # Less confident in ranges
            MarketRegime.HIGH_VOLATILITY: -0.08,  # Much less confident
            MarketRegime.LOW_VOLATILITY: 0.03,    # Slightly more confident
            MarketRegime.UNKNOWN: -0.05
        }
        regime_adjustment = regime_adjustments.get(regime, 0.0)

        confidence = base_confidence - vol_penalty + regime_adjustment
        return np.clip(confidence, 0.1, 0.95)  # Keep in reasonable bounds

    def _calculate_session_factor(self, time_ratio: float) -> float:
        """Calculate session transition factor."""
        # If prediction crosses major session boundaries, reduce scaling
        if time_ratio > 8:  # More than 8 hours
            return 0.95  # Slight reduction for session transitions
        return 1.0


class MultiHorizonPredictor:
    """Main class for multi-horizon predictions from single models."""

    def __init__(self):
        self.scaler = EnhancedHorizonScaler()
        self.regime_detector = MarketRegimeDetector()

    def predict_multi_horizon(
        self,
        base_prediction: float,
        base_timeframe: str,
        target_horizons: Union[List[str], str],
        scenario: Optional[str] = None,
        scaling_mode: ScalingMode = ScalingMode.SMART_ADAPTIVE,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict]:
        """Generate multi-horizon predictions from single base prediction."""

        # Handle scenario-based configurations
        if scenario and scenario in TRADING_SCENARIOS:
            scenario_config = TRADING_SCENARIOS[scenario]
            target_horizons = scenario_config.horizons
            scaling_mode = scenario_config.scaling_mode
            logger.info(f"Using {scenario_config.name} scenario with {len(target_horizons)} horizons")

        # Ensure target_horizons is a list
        if isinstance(target_horizons, str):
            target_horizons = [target_horizons]

        # Detect current regime once
        current_regime = self.regime_detector.detect_regime(market_data)
        current_volatility = self.scaler._estimate_volatility(market_data)

        results = {}

        for target_horizon in target_horizons:
            try:
                scaled_result = self.scaler.scale_prediction(
                    base_prediction=base_prediction,
                    base_timeframe=base_timeframe,
                    target_timeframe=target_horizon,
                    scaling_mode=scaling_mode,
                    market_data=market_data,
                    volatility=current_volatility,
                    regime=current_regime
                )
                results[target_horizon] = scaled_result

            except Exception as e:
                logger.error(f"Failed to scale prediction for {target_horizon}: {e}")
                # Fallback to linear scaling
                base_minutes = timeframe_to_minutes(base_timeframe)
                target_minutes = timeframe_to_minutes(target_horizon)
                ratio = target_minutes / base_minutes

                results[target_horizon] = {
                    "prediction": base_prediction * ratio,
                    "lower": base_prediction * ratio * 0.95,
                    "upper": base_prediction * ratio * 1.05,
                    "confidence": 0.5,
                    "regime": "unknown",
                    "volatility": current_volatility,
                    "time_ratio": ratio,
                    "scaling_mode": "linear_fallback"
                }

        return results

    def get_available_scenarios(self) -> Dict[str, TradingScenario]:
        """Get all available trading scenarios."""
        return TRADING_SCENARIOS.copy()

    def get_scenario_info(self, scenario_name: str) -> Optional[TradingScenario]:
        """Get information about a specific scenario."""
        return TRADING_SCENARIOS.get(scenario_name)


# Enhanced convenience functions maintaining backward compatibility

def convert_single_to_multi_horizon(
    base_prediction: float,
    base_timeframe: str,
    target_horizons: Union[List[str], str],
    scenario: Optional[str] = None,
    scaling_mode: str = "smart_adaptive",
    market_data: Optional[pd.DataFrame] = None,
    uncertainty_bands: bool = True
) -> Dict[str, Dict]:
    """Convert single prediction to multi-horizon predictions."""

    predictor = MultiHorizonPredictor()

    # Convert string scaling mode to enum
    try:
        scaling_enum = ScalingMode(scaling_mode)
    except ValueError:
        logger.warning(f"Unknown scaling mode {scaling_mode}, using smart_adaptive")
        scaling_enum = ScalingMode.SMART_ADAPTIVE

    return predictor.predict_multi_horizon(
        base_prediction=base_prediction,
        base_timeframe=base_timeframe,
        target_horizons=target_horizons,
        scenario=scenario,
        scaling_mode=scaling_enum,
        market_data=market_data
    )


def get_trading_scenarios() -> Dict[str, str]:
    """Get available trading scenarios for UI selection."""
    return {key: scenario.name for key, scenario in TRADING_SCENARIOS.items()}


def validate_multi_horizon_request(
    base_prediction: float,
    base_timeframe: str,
    target_horizons: List[str],
    scenario: Optional[str] = None
) -> Dict[str, any]:
    """Enhanced validation for multi-horizon prediction requests."""
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "scenario_info": None
    }

    # Validate base prediction
    if not isinstance(base_prediction, (int, float)) or np.isnan(base_prediction):
        results["valid"] = False
        results["errors"].append("Invalid base prediction value")

    # Validate timeframes
    try:
        timeframe_to_minutes(base_timeframe)
        for horizon in target_horizons:
            timeframe_to_minutes(horizon)
    except ValueError as e:
        results["valid"] = False
        results["errors"].append(f"Invalid timeframe: {e}")

    # Validate scenario
    if scenario:
        if scenario not in TRADING_SCENARIOS:
            results["warnings"].append(f"Unknown scenario '{scenario}', using custom horizons")
        else:
            results["scenario_info"] = TRADING_SCENARIOS[scenario]

    # Check for very long horizons
    try:
        base_minutes = timeframe_to_minutes(base_timeframe)
        for horizon in target_horizons:
            horizon_minutes = timeframe_to_minutes(horizon)
            ratio = horizon_minutes / base_minutes
            if ratio > 1000:  # More than 1000x base timeframe
                results["warnings"].append(
                    f"Very long horizon {horizon} may have unreliable predictions"
                )
    except Exception:
        pass

    return results


def convert_horizons_for_enhanced_inference(
    horizons: Union[List[str], List[int], int, str],
    base_timeframe: str,
    scenario: Optional[str] = None,
    model_horizon_bars: int = None
) -> Tuple[List[str], List[int], Optional[TradingScenario]]:
    """Enhanced version with scenario support and better validation."""

    # Handle scenario-based configurations
    scenario_config = None
    if scenario and scenario in TRADING_SCENARIOS:
        scenario_config = TRADING_SCENARIOS[scenario]
        horizons = scenario_config.horizons
        logger.info(f"Using {scenario_config.name} scenario")

    # Use existing conversion logic
    time_labels, horizon_bars = convert_horizons_for_inference(
        horizons, base_timeframe, model_horizon_bars
    )

    return time_labels, horizon_bars, scenario_config