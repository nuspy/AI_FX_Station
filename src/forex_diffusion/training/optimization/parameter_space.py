"""
Parameter space management for pattern optimization.

This module defines parameter ranges, distributions, and sampling strategies
for both form parameters (detector configuration) and action parameters (execution logic).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
import numpy as np
from scipy.stats import qmc
from loguru import logger

class ParameterType(str, Enum):
    """Types of parameters"""
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

class DistributionType(str, Enum):
    """Parameter distribution types"""
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    CHOICE = "choice"

@dataclass
class ParameterDefinition:
    """Definition of a single parameter"""
    name: str
    param_type: ParameterType
    distribution: DistributionType

    # Range/choices
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None

    # Distribution parameters
    mean: Optional[float] = None
    std: Optional[float] = None

    # Metadata
    description: str = ""
    suggested_range: Optional[str] = None
    pattern_specific: bool = False
    timeframe_specific: bool = False

@dataclass
class ParameterSpace:
    """
    Manages parameter spaces for pattern optimization.

    Handles both form parameters (detector configuration) and action parameters
    (execution logic) with pattern-specific overrides and intelligent sampling.
    """

    def __init__(self):
        self.form_parameters: Dict[str, ParameterDefinition] = {}
        self.action_parameters: Dict[str, ParameterDefinition] = {}
        self.pattern_overrides: Dict[str, Dict[str, ParameterDefinition]] = {}
        self.timeframe_overrides: Dict[str, Dict[str, ParameterDefinition]] = {}

        # Initialize default parameter spaces
        self._initialize_default_parameters()
        self._initialize_timeframe_overrides()

    def _initialize_default_parameters(self):
        """Initialize default parameter definitions"""

        # Form parameters for chart patterns (detector configuration)
        chart_form_params = {
            "min_touches": ParameterDefinition(
                name="min_touches",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=2,
                max_value=8,
                description="Minimum number of price touches for pattern formation",
                suggested_range="3-6 for most patterns"
            ),
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=5,
                max_value=50,
                description="Minimum bars for pattern span",
                suggested_range="10-30 bars typical"
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=20,
                max_value=200,
                description="Maximum bars for pattern span",
                suggested_range="50-150 bars typical"
            ),
            "tolerance": ParameterDefinition(
                name="tolerance",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.LOG_UNIFORM,
                min_value=0.001,
                max_value=0.1,
                description="Price tolerance for pattern matching (as fraction)",
                suggested_range="0.005-0.05 common"
            ),
            "impulse_multiplier": ParameterDefinition(
                name="impulse_multiplier",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.5,
                max_value=3.0,
                description="Multiplier for impulse detection sensitivity",
                suggested_range="1.0-2.0 typical"
            ),
            "window": ParameterDefinition(
                name="window",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=5,
                max_value=100,
                description="Rolling window size for calculations",
                suggested_range="10-50 bars"
            ),
            "tightness": ParameterDefinition(
                name="tightness",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.1,
                max_value=0.9,
                description="Pattern tightness requirement (0=loose, 1=tight)",
                suggested_range="0.3-0.7 balanced"
            ),
            "atr_period": ParameterDefinition(
                name="atr_period",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=5,
                max_value=50,
                description="ATR calculation period",
                suggested_range="14-21 standard"
            )
        }

        # Form parameters for candlestick patterns
        candle_form_params = {
            "body_threshold": ParameterDefinition(
                name="body_threshold",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.1,
                max_value=0.9,
                description="Minimum body size as fraction of total range",
                suggested_range="0.3-0.7 typical"
            ),
            "wick_threshold": ParameterDefinition(
                name="wick_threshold",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.05,
                max_value=0.5,
                description="Maximum wick size as fraction of body",
                suggested_range="0.1-0.3 common"
            ),
            "gap_percentage": ParameterDefinition(
                name="gap_percentage",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.LOG_UNIFORM,
                min_value=0.001,
                max_value=0.05,
                description="Minimum gap size as percentage",
                suggested_range="0.005-0.02 typical"
            ),
            "volume_confirmation": ParameterDefinition(
                name="volume_confirmation",
                param_type=ParameterType.BOOLEAN,
                distribution=DistributionType.CHOICE,
                choices=[True, False],
                description="Require volume confirmation for pattern"
            ),
            "volume_multiplier": ParameterDefinition(
                name="volume_multiplier",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=1.0,
                max_value=3.0,
                description="Volume spike multiplier for confirmation",
                suggested_range="1.2-2.0 typical"
            ),
            "confluence_mtf": ParameterDefinition(
                name="confluence_mtf",
                param_type=ParameterType.BOOLEAN,
                distribution=DistributionType.CHOICE,
                choices=[True, False],
                description="Require multi-timeframe confluence"
            )
        }

        # Action parameters (execution logic)
        action_params = {
            "target_mode": ParameterDefinition(
                name="target_mode",
                param_type=ParameterType.CATEGORICAL,
                distribution=DistributionType.CHOICE,
                choices=["Altezza figura", "Flag pole", "Ampiezza canale", "Custom"],
                description="Target calculation method",
                suggested_range="Altezza figura most common"
            ),
            "risk_reward_ratio": ParameterDefinition(
                name="risk_reward_ratio",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.5,
                max_value=5.0,
                description="Risk to reward ratio",
                suggested_range="1.0-3.0 typical"
            ),
            "buffer_atr": ParameterDefinition(
                name="buffer_atr",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.0,
                max_value=2.0,
                description="ATR buffer for entry/exit points",
                suggested_range="0.25-1.0 common"
            ),
            "horizon_bars": ParameterDefinition(
                name="horizon_bars",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=5,
                max_value=100,
                description="Maximum bars to hold position",
                suggested_range="20-50 bars typical"
            ),
            "trailing_stop": ParameterDefinition(
                name="trailing_stop",
                param_type=ParameterType.BOOLEAN,
                distribution=DistributionType.CHOICE,
                choices=[True, False],
                description="Enable trailing stop loss"
            ),
            "trailing_distance": ParameterDefinition(
                name="trailing_distance",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.5,
                max_value=3.0,
                description="Trailing stop distance in ATR",
                suggested_range="1.0-2.0 typical"
            ),
            "partial_profits": ParameterDefinition(
                name="partial_profits",
                param_type=ParameterType.BOOLEAN,
                distribution=DistributionType.CHOICE,
                choices=[True, False],
                description="Take partial profits at intermediate levels"
            ),
            "profit_levels": ParameterDefinition(
                name="profit_levels",
                param_type=ParameterType.CATEGORICAL,
                distribution=DistributionType.CHOICE,
                choices=["25%@50%,75%@100%", "50%@75%", "33%@50%,67%@100%"],
                description="Partial profit taking levels"
            )
        }

        # Store default parameters
        self.form_parameters.update(chart_form_params)
        self.form_parameters.update(candle_form_params)
        self.action_parameters.update(action_params)

        # Pattern-specific overrides
        self._initialize_pattern_overrides()

    def _initialize_timeframe_overrides(self):
        """Initialize timeframe-specific parameter overrides"""

        # 1-minute timeframe: High frequency, noise-sensitive
        self.timeframe_overrides["1m"] = {
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=3,
                max_value=15,
                description="1m: Very short patterns, high noise",
                timeframe_specific=True
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=10,
                max_value=60,
                description="1m: Patterns don't last long",
                timeframe_specific=True
            ),
            "tolerance": ParameterDefinition(
                name="tolerance",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.LOG_UNIFORM,
                min_value=0.0005,
                max_value=0.005,
                description="1m: Tight tolerance for noise filtering",
                timeframe_specific=True
            ),
            "volume_required": ParameterDefinition(
                name="volume_required",
                param_type=ParameterType.BOOLEAN,
                distribution=DistributionType.CHOICE,
                choices=[True],  # Always require volume on 1m
                description="1m: Volume confirmation essential",
                timeframe_specific=True
            ),
            "atr_period": ParameterDefinition(
                name="atr_period",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=5,
                max_value=20,
                description="1m: Shorter ATR for volatility",
                timeframe_specific=True
            )
        }

        # 5-minute timeframe: Scalping, reduced noise
        self.timeframe_overrides["5m"] = {
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=5,
                max_value=25,
                description="5m: Short-term patterns",
                timeframe_specific=True
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=20,
                max_value=100,
                description="5m: Medium duration patterns",
                timeframe_specific=True
            ),
            "tolerance": ParameterDefinition(
                name="tolerance",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.LOG_UNIFORM,
                min_value=0.001,
                max_value=0.01,
                description="5m: Moderate tolerance",
                timeframe_specific=True
            )
        }

        # 15-minute timeframe: Intraday trading
        self.timeframe_overrides["15m"] = {
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=8,
                max_value=40,
                description="15m: Intraday patterns",
                timeframe_specific=True
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=30,
                max_value=150,
                description="15m: Longer intraday patterns",
                timeframe_specific=True
            ),
            "risk_reward_ratio": ParameterDefinition(
                name="risk_reward_ratio",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=1.0,
                max_value=3.0,
                description="15m: Conservative R:R for intraday",
                timeframe_specific=True
            )
        }

        # 1-hour timeframe: Swing trading
        self.timeframe_overrides["1h"] = {
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=12,
                max_value=60,
                description="1h: Swing patterns need time",
                timeframe_specific=True
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=50,
                max_value=300,
                description="1h: Long swing patterns",
                timeframe_specific=True
            ),
            "tolerance": ParameterDefinition(
                name="tolerance",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.LOG_UNIFORM,
                min_value=0.005,
                max_value=0.05,
                description="1h: Higher tolerance for larger moves",
                timeframe_specific=True
            ),
            "tightness": ParameterDefinition(
                name="tightness",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.2,
                max_value=0.7,
                description="1h: More flexible patterns",
                timeframe_specific=True
            ),
            "confluence_mtf": ParameterDefinition(
                name="confluence_mtf",
                param_type=ParameterType.BOOLEAN,
                distribution=DistributionType.CHOICE,
                choices=[True],  # Always require MTF on 1h+
                description="1h: Multi-timeframe confirmation important",
                timeframe_specific=True
            )
        }

        # 4-hour timeframe: Position trading
        self.timeframe_overrides["4h"] = {
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=20,
                max_value=80,
                description="4h: Position patterns develop slowly",
                timeframe_specific=True
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=80,
                max_value=500,
                description="4h: Very long patterns possible",
                timeframe_specific=True
            ),
            "min_touches": ParameterDefinition(
                name="min_touches",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=3,
                max_value=8,
                description="4h: More touches for reliability",
                timeframe_specific=True
            ),
            "risk_reward_ratio": ParameterDefinition(
                name="risk_reward_ratio",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=1.5,
                max_value=5.0,
                description="4h: Higher R:R for position trades",
                timeframe_specific=True
            )
        }

        # Daily timeframe: Long-term analysis
        self.timeframe_overrides["1d"] = {
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=30,
                max_value=120,
                description="1d: Long-term patterns, weeks/months",
                timeframe_specific=True
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=100,
                max_value=1000,
                description="1d: Multi-year patterns possible",
                timeframe_specific=True
            ),
            "tolerance": ParameterDefinition(
                name="tolerance",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.LOG_UNIFORM,
                min_value=0.01,
                max_value=0.15,
                description="1d: Very high tolerance for major moves",
                timeframe_specific=True
            ),
            "tightness": ParameterDefinition(
                name="tightness",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.1,
                max_value=0.5,
                description="1d: Very flexible, broad patterns",
                timeframe_specific=True
            ),
            "atr_period": ParameterDefinition(
                name="atr_period",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=14,
                max_value=50,
                description="1d: Longer ATR for trend analysis",
                timeframe_specific=True
            ),
            "volume_required": ParameterDefinition(
                name="volume_required",
                param_type=ParameterType.BOOLEAN,
                distribution=DistributionType.CHOICE,
                choices=[False],  # Less critical on daily
                description="1d: Volume less critical on long-term",
                timeframe_specific=True
            )
        }

    def _initialize_pattern_overrides(self):
        """Initialize pattern-specific parameter overrides"""

        # Head and Shoulders patterns need longer spans
        self.pattern_overrides["head_and_shoulders"] = {
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=30,
                max_value=100,
                description="H&S needs longer formation period",
                pattern_specific=True
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=80,
                max_value=300,
                description="H&S can be very long patterns",
                pattern_specific=True
            )
        }

        # Triangle patterns have specific touch requirements
        self.pattern_overrides["triangle"] = {
            "min_touches": ParameterDefinition(
                name="min_touches",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=4,
                max_value=8,
                description="Triangles need multiple touches",
                pattern_specific=True
            ),
            "tightness": ParameterDefinition(
                name="tightness",
                param_type=ParameterType.FLOAT,
                distribution=DistributionType.UNIFORM,
                min_value=0.4,
                max_value=0.8,
                description="Triangles require tighter formation",
                pattern_specific=True
            )
        }

        # Flag patterns are usually tight and short
        self.pattern_overrides["flag"] = {
            "min_span": ParameterDefinition(
                name="min_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=3,
                max_value=20,
                description="Flags are short-term patterns",
                pattern_specific=True
            ),
            "max_span": ParameterDefinition(
                name="max_span",
                param_type=ParameterType.INTEGER,
                distribution=DistributionType.UNIFORM,
                min_value=8,
                max_value=50,
                description="Flags don't last long",
                pattern_specific=True
            ),
            "target_mode": ParameterDefinition(
                name="target_mode",
                param_type=ParameterType.CATEGORICAL,
                distribution=DistributionType.CHOICE,
                choices=["Flag pole"],
                description="Flags use pole height for targets",
                pattern_specific=True
            )
        }

        # Double/Triple patterns
        for pattern in ["double_top", "double_bottom", "triple_top", "triple_bottom"]:
            self.pattern_overrides[pattern] = {
                "min_touches": ParameterDefinition(
                    name="min_touches",
                    param_type=ParameterType.INTEGER,
                    distribution=DistributionType.UNIFORM,
                    min_value=2 if "double" in pattern else 3,
                    max_value=3 if "double" in pattern else 4,
                    description=f"{pattern} requires specific touch count",
                    pattern_specific=True
                )
            }

        # Harmonic patterns need tight tolerances
        harmonic_patterns = ["gartley", "butterfly", "bat", "crab"]
        for pattern in harmonic_patterns:
            self.pattern_overrides[pattern] = {
                "tolerance": ParameterDefinition(
                    name="tolerance",
                    param_type=ParameterType.FLOAT,
                    distribution=DistributionType.LOG_UNIFORM,
                    min_value=0.001,
                    max_value=0.03,
                    description="Harmonic patterns need tight Fibonacci ratios",
                    pattern_specific=True
                ),
                "tightness": ParameterDefinition(
                    name="tightness",
                    param_type=ParameterType.FLOAT,
                    distribution=DistributionType.UNIFORM,
                    min_value=0.6,
                    max_value=0.95,
                    description="Harmonics require precise formation",
                    pattern_specific=True
                )
            }

    def initialize_ranges(self, pattern_key: str, timeframe: str = None,
                         custom_ranges: Optional[Dict[str, Any]] = None):
        """
        Initialize parameter ranges for a specific pattern and timeframe.

        Args:
            pattern_key: Pattern identifier
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '4h', '1d')
            custom_ranges: Custom parameter range overrides
        """
        logger.info(f"Initializing parameter ranges for pattern: {pattern_key}, timeframe: {timeframe}")

        # Apply timeframe-specific overrides first (lower priority)
        if timeframe and timeframe in self.timeframe_overrides:
            tf_overrides = self.timeframe_overrides[timeframe]
            self._apply_timeframe_overrides(tf_overrides)
            logger.info(f"Applied {len(tf_overrides)} timeframe-specific overrides for {timeframe}")

        # Apply pattern-specific overrides (higher priority, can override timeframe)
        if pattern_key in self.pattern_overrides:
            overrides = self.pattern_overrides[pattern_key]
            self._apply_pattern_overrides(pattern_key, overrides)
            logger.info(f"Applied {len(overrides)} pattern-specific overrides for {pattern_key}")

        # Apply custom ranges if provided (highest priority)
        if custom_ranges:
            self._apply_custom_ranges(custom_ranges)
            logger.info(f"Applied {len(custom_ranges)} custom range overrides")

    def _apply_timeframe_overrides(self, tf_overrides: Dict[str, ParameterDefinition]):
        """Apply timeframe-specific parameter overrides"""
        for param_name, param_def in tf_overrides.items():
            if param_name in self.form_parameters:
                self.form_parameters[param_name] = param_def
            elif param_name in self.action_parameters:
                self.action_parameters[param_name] = param_def

    def _apply_pattern_overrides(self, pattern_key: str, overrides: Dict[str, ParameterDefinition]):
        """Apply pattern-specific parameter overrides"""
        for param_name, param_def in overrides.items():
            if param_name in self.form_parameters:
                self.form_parameters[param_name] = param_def
            elif param_name in self.action_parameters:
                self.action_parameters[param_name] = param_def

    def _apply_custom_ranges(self, custom_ranges: Dict[str, Any]):
        """Apply custom parameter range overrides"""
        for param_name, range_config in custom_ranges.items():
            if param_name in self.form_parameters:
                self._update_parameter_range(self.form_parameters[param_name], range_config)
            elif param_name in self.action_parameters:
                self._update_parameter_range(self.action_parameters[param_name], range_config)

    def _update_parameter_range(self, param_def: ParameterDefinition, range_config: Dict[str, Any]):
        """Update a parameter definition with new range configuration"""
        if "min_value" in range_config:
            param_def.min_value = range_config["min_value"]
        if "max_value" in range_config:
            param_def.max_value = range_config["max_value"]
        if "choices" in range_config:
            param_def.choices = range_config["choices"]

    def generate_sobol_samples(self, pattern_key: str, n_samples: int,
                              seed: int = 42) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generate quasi-random parameter samples using Sobol sequences.

        Args:
            pattern_key: Pattern identifier for context
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            List of (form_parameters, action_parameters) tuples
        """
        # Get relevant parameters for this pattern
        form_params = self._get_form_parameters_for_pattern(pattern_key)
        action_params = list(self.action_parameters.values())

        all_params = form_params + action_params
        n_params = len(all_params)

        if n_params == 0:
            return []

        # Generate Sobol samples
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
        samples = sampler.random(n_samples)

        results = []
        for sample in samples:
            form_param_values = {}
            action_param_values = {}

            param_idx = 0

            # Sample form parameters
            for param in form_params:
                value = self._sample_parameter_value(param, sample[param_idx])
                form_param_values[param.name] = value
                param_idx += 1

            # Sample action parameters
            for param in action_params:
                value = self._sample_parameter_value(param, sample[param_idx])
                action_param_values[param.name] = value
                param_idx += 1

            results.append((form_param_values, action_param_values))

        logger.info(f"Generated {len(results)} Sobol samples for {pattern_key}")
        return results

    def generate_random_samples(self, pattern_key: str, n_samples: int,
                               seed: int = 42) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generate random parameter samples for comparison/initialization.

        Args:
            pattern_key: Pattern identifier
            n_samples: Number of samples
            seed: Random seed

        Returns:
            List of parameter combinations
        """
        np.random.seed(seed)

        form_params = self._get_form_parameters_for_pattern(pattern_key)
        action_params = list(self.action_parameters.values())

        results = []
        for _ in range(n_samples):
            form_param_values = {}
            action_param_values = {}

            # Sample form parameters
            for param in form_params:
                value = self._sample_parameter_value(param, np.random.random())
                form_param_values[param.name] = value

            # Sample action parameters
            for param in action_params:
                value = self._sample_parameter_value(param, np.random.random())
                action_param_values[param.name] = value

            results.append((form_param_values, action_param_values))

        return results

    def _get_form_parameters_for_pattern(self, pattern_key: str) -> List[ParameterDefinition]:
        """Get form parameters relevant to a specific pattern"""
        base_params = []

        # Determine if this is a chart or candle pattern
        if any(candle_word in pattern_key.lower()
               for candle_word in ["doji", "hammer", "engulfing", "star", "harami"]):
            # Candlestick pattern - use candle-specific parameters
            candle_param_names = ["body_threshold", "wick_threshold", "gap_percentage",
                                "volume_confirmation", "volume_multiplier", "confluence_mtf"]
            base_params = [self.form_parameters[name] for name in candle_param_names
                          if name in self.form_parameters]
        else:
            # Chart pattern - use chart-specific parameters
            chart_param_names = ["min_touches", "min_span", "max_span", "tolerance",
                               "impulse_multiplier", "window", "tightness", "atr_period"]
            base_params = [self.form_parameters[name] for name in chart_param_names
                          if name in self.form_parameters]

        # Apply pattern-specific overrides
        if pattern_key in self.pattern_overrides:
            overrides = self.pattern_overrides[pattern_key]
            for i, param in enumerate(base_params):
                if param.name in overrides:
                    base_params[i] = overrides[param.name]

        return base_params

    def _sample_parameter_value(self, param: ParameterDefinition,
                               uniform_sample: float) -> Any:
        """Convert uniform sample to parameter value based on distribution"""

        if param.distribution == DistributionType.UNIFORM:
            if param.param_type == ParameterType.INTEGER:
                return int(param.min_value + uniform_sample * (param.max_value - param.min_value))
            elif param.param_type == ParameterType.FLOAT:
                return param.min_value + uniform_sample * (param.max_value - param.min_value)

        elif param.distribution == DistributionType.LOG_UNIFORM:
            log_min = np.log(param.min_value)
            log_max = np.log(param.max_value)
            log_value = log_min + uniform_sample * (log_max - log_min)
            return float(np.exp(log_value))

        elif param.distribution == DistributionType.CHOICE:
            if param.choices:
                idx = int(uniform_sample * len(param.choices))
                idx = min(idx, len(param.choices) - 1)  # Clamp to valid range
                return param.choices[idx]

        elif param.distribution == DistributionType.NORMAL:
            # Convert uniform to normal using inverse CDF approximation
            # This is a simplified version - could use scipy.stats.norm.ppf for precision
            z_score = np.sqrt(2) * np.sqrt(-np.log(max(1e-10, min(1-1e-10, uniform_sample))))
            if uniform_sample > 0.5:
                z_score = -z_score
            value = param.mean + z_score * param.std

            # Clamp to bounds if specified
            if param.min_value is not None:
                value = max(value, param.min_value)
            if param.max_value is not None:
                value = min(value, param.max_value)

            if param.param_type == ParameterType.INTEGER:
                return int(value)
            return float(value)

        # Fallback for unknown distributions
        logger.warning(f"Unknown distribution {param.distribution} for parameter {param.name}")
        return param.min_value if param.min_value is not None else 0

    def get_parameter_bounds(self, pattern_key: str) -> Dict[str, Tuple[Any, Any]]:
        """
        Get parameter bounds for optimization algorithms.

        Args:
            pattern_key: Pattern identifier

        Returns:
            Dictionary mapping parameter names to (min, max) bounds
        """
        bounds = {}

        # Get all relevant parameters
        form_params = self._get_form_parameters_for_pattern(pattern_key)
        action_params = list(self.action_parameters.values())

        for param in form_params + action_params:
            if param.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                bounds[param.name] = (param.min_value, param.max_value)
            elif param.param_type == ParameterType.CATEGORICAL and param.choices:
                bounds[param.name] = (0, len(param.choices) - 1)
            elif param.param_type == ParameterType.BOOLEAN:
                bounds[param.name] = (0, 1)

        return bounds

    def validate_parameters(self, pattern_key: str, form_params: Dict[str, Any],
                           action_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameter values against constraints.

        Args:
            pattern_key: Pattern identifier
            form_params: Form parameter values
            action_params: Action parameter values

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Get parameter definitions
        form_param_defs = {p.name: p for p in self._get_form_parameters_for_pattern(pattern_key)}
        action_param_defs = {p.name: p for p in self.action_parameters.values()}

        # Validate form parameters
        for name, value in form_params.items():
            if name in form_param_defs:
                param_violations = self._validate_single_parameter(
                    form_param_defs[name], value)
                violations.extend(param_violations)

        # Validate action parameters
        for name, value in action_params.items():
            if name in action_param_defs:
                param_violations = self._validate_single_parameter(
                    action_param_defs[name], value)
                violations.extend(param_violations)

        # Cross-parameter validation
        cross_violations = self._validate_cross_constraints(
            pattern_key, form_params, action_params)
        violations.extend(cross_violations)

        return len(violations) == 0, violations

    def _validate_single_parameter(self, param_def: ParameterDefinition,
                                  value: Any) -> List[str]:
        """Validate a single parameter value"""
        violations = []

        # Type validation
        if param_def.param_type == ParameterType.INTEGER:
            if not isinstance(value, (int, np.integer)):
                violations.append(f"{param_def.name} must be integer, got {type(value)}")
                return violations
        elif param_def.param_type == ParameterType.FLOAT:
            if not isinstance(value, (int, float, np.number)):
                violations.append(f"{param_def.name} must be numeric, got {type(value)}")
                return violations
        elif param_def.param_type == ParameterType.BOOLEAN:
            if not isinstance(value, (bool, np.bool_)):
                violations.append(f"{param_def.name} must be boolean, got {type(value)}")
                return violations

        # Range validation
        if param_def.min_value is not None and value < param_def.min_value:
            violations.append(f"{param_def.name}={value} below minimum {param_def.min_value}")

        if param_def.max_value is not None and value > param_def.max_value:
            violations.append(f"{param_def.name}={value} above maximum {param_def.max_value}")

        # Choice validation
        if param_def.choices is not None and value not in param_def.choices:
            violations.append(f"{param_def.name}={value} not in allowed choices {param_def.choices}")

        return violations

    def _validate_cross_constraints(self, pattern_key: str,
                                   form_params: Dict[str, Any],
                                   action_params: Dict[str, Any]) -> List[str]:
        """Validate cross-parameter constraints"""
        violations = []

        # min_span < max_span
        if "min_span" in form_params and "max_span" in form_params:
            if form_params["min_span"] >= form_params["max_span"]:
                violations.append("min_span must be less than max_span")

        # Trailing stop requires trailing distance
        if action_params.get("trailing_stop") and "trailing_distance" not in action_params:
            violations.append("trailing_stop=True requires trailing_distance parameter")

        # Partial profits requires profit levels
        if action_params.get("partial_profits") and "profit_levels" not in action_params:
            violations.append("partial_profits=True requires profit_levels parameter")

        # Pattern-specific constraints
        if "triangle" in pattern_key.lower():
            if form_params.get("min_touches", 0) < 4:
                violations.append("Triangle patterns require at least 4 touches")

        return violations

    def get_suggested_ranges_text(self, pattern_key: str) -> str:
        """Get human-readable suggested parameter ranges for UI display"""

        form_params = self._get_form_parameters_for_pattern(pattern_key)
        action_params = list(self.action_parameters.values())

        suggestions = []

        suggestions.append("=== Form Parameters ===")
        for param in form_params:
            range_text = param.suggested_range or f"{param.min_value}-{param.max_value}"
            suggestions.append(f"{param.name}: {range_text}")

        suggestions.append("\n=== Action Parameters ===")
        for param in action_params:
            range_text = param.suggested_range or f"{param.min_value}-{param.max_value}"
            suggestions.append(f"{param.name}: {range_text}")

        return "\n".join(suggestions)

    def get_suggested_ranges(self, pattern_type: str = None) -> Dict[str, Any]:
        """Get suggested parameter ranges for optimization"""
        try:
            form_params = self.get_form_parameters(pattern_type or "generic")
            action_params = self.get_action_parameters(pattern_type or "generic")

            ranges = {
                "form_parameters": {},
                "action_parameters": {}
            }

            # Convert form parameters to ranges
            for param in form_params:
                ranges["form_parameters"][param.name] = {
                    "type": param.param_type.value,
                    "min": param.min_value,
                    "max": param.max_value,
                    "default": param.default_value,
                    "suggested_range": param.suggested_range
                }

            # Convert action parameters to ranges
            for param in action_params:
                ranges["action_parameters"][param.name] = {
                    "type": param.param_type.value,
                    "min": param.min_value,
                    "max": param.max_value,
                    "default": param.default_value,
                    "suggested_range": param.suggested_range
                }

            return ranges
        except Exception as e:
            logger.warning(f"Failed to get suggested ranges: {e}")
            return {"form_parameters": {}, "action_parameters": {}}