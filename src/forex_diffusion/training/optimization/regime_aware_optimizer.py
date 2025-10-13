"""
Regime-Aware Pattern Optimization

Integrates HMM regime detection with pattern parameter optimization.
Each pattern's parameters are optimized separately for each market regime,
then the appropriate parameter set is loaded based on current regime.
"""
from __future__ import annotations

from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

from ...regime.hmm_detector import HMMRegimeDetector, RegimeType
from .backtest_runner import BacktestRunner
from .parameter_space import ParameterSpace
from .genetic_algorithm import GeneticAlgorithm, GAConfig


@dataclass
class RegimeParameters:
    """Parameters optimized for specific regime"""
    regime_type: RegimeType
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    optimization_date: datetime
    sample_size: int
    confidence: float = 1.0


@dataclass
class RegimeAwareOptimizationResult:
    """Results from regime-aware optimization"""
    pattern_key: str
    direction: str
    asset: str
    timeframe: str

    # Per-regime parameter sets
    regime_parameters: Dict[RegimeType, RegimeParameters] = field(default_factory=dict)

    # Global fallback parameters
    global_parameters: Optional[Dict[str, Any]] = None
    global_metrics: Optional[Dict[str, float]] = None

    # Optimization metadata
    total_samples_by_regime: Dict[RegimeType, int] = field(default_factory=dict)
    optimization_time_seconds: float = 0.0
    regimes_optimized: int = 0


class RegimeAwareOptimizer:
    """
    Optimizes pattern parameters separately for each market regime.

    Workflow:
    1. Classify historical data into regimes using HMM
    2. Filter data by regime
    3. Optimize pattern parameters for each regime independently
    4. Save regime-specific parameter sets
    5. At runtime, detect current regime and load appropriate parameters
    """

    def __init__(
        self,
        hmm_detector: Optional[HMMRegimeDetector] = None,
        n_regimes: int = 4,
        min_samples_per_regime: int = 50,
    ):
        """
        Initialize regime-aware optimizer.

        Args:
            hmm_detector: Pre-fitted HMM detector, or None to create new
            n_regimes: Number of regimes if creating new detector
            min_samples_per_regime: Minimum samples required to optimize regime
        """
        self.hmm_detector = hmm_detector or HMMRegimeDetector(n_regimes=n_regimes)
        self.min_samples_per_regime = min_samples_per_regime

        # Components
        self.backtest_runner = BacktestRunner()
        self.parameter_space = ParameterSpace()

        # Cache
        self.regime_cache: Dict[str, pd.DataFrame] = {}

    def optimize_pattern_by_regime(
        self,
        df: pd.DataFrame,
        pattern_key: str,
        direction: str,
        asset: str,
        timeframe: str,
        parameter_ranges: Dict[str, Any],
        max_trials: int = 100,
        ga_config: Optional[GAConfig] = None,
    ) -> RegimeAwareOptimizationResult:
        """
        Optimize pattern parameters separately for each regime.

        Args:
            df: Historical OHLCV data
            pattern_key: Pattern identifier (e.g., "head_and_shoulders")
            direction: "bull" or "bear"
            asset: Asset symbol
            timeframe: Timeframe string
            parameter_ranges: Parameter search space
            max_trials: Max optimization trials per regime
            ga_config: Genetic algorithm configuration

        Returns:
            RegimeAwareOptimizationResult with per-regime parameters
        """
        import time
        start_time = time.time()

        logger.info(
            f"Starting regime-aware optimization for {pattern_key} "
            f"{direction} on {asset} {timeframe}"
        )

        # Step 1: Fit HMM and classify regimes
        if not self.hmm_detector.is_fitted:
            logger.info("Fitting HMM regime detector...")
            self.hmm_detector.fit(df)

        # Predict regimes for entire dataset
        regime_df = self.hmm_detector.predict(df)

        # Merge with OHLCV data
        df_with_regime = pd.concat([df, regime_df], axis=1)

        # Step 2: Count samples per regime
        regime_counts = regime_df["regime_state"].value_counts()
        logger.info(f"Regime distribution: {regime_counts.to_dict()}")

        result = RegimeAwareOptimizationResult(
            pattern_key=pattern_key,
            direction=direction,
            asset=asset,
            timeframe=timeframe,
        )

        # Step 3: Optimize for each regime separately
        for regime_type in RegimeType:
            if regime_type == RegimeType.UNKNOWN:
                continue

            # Filter data for this regime
            regime_mask = regime_df["regime_state"] == regime_type.value
            regime_data = df_with_regime[regime_mask].copy()

            sample_count = len(regime_data)
            result.total_samples_by_regime[regime_type] = sample_count

            if sample_count < self.min_samples_per_regime:
                logger.warning(
                    f"Skipping {regime_type.value}: only {sample_count} samples "
                    f"(need {self.min_samples_per_regime})"
                )
                continue

            logger.info(
                f"Optimizing for regime {regime_type.value} "
                f"({sample_count} samples)..."
            )

            # Optimize parameters for this regime
            regime_params = self._optimize_single_regime(
                regime_data=regime_data,
                regime_type=regime_type,
                pattern_key=pattern_key,
                direction=direction,
                parameter_ranges=parameter_ranges,
                max_trials=max_trials,
                ga_config=ga_config,
            )

            if regime_params:
                result.regime_parameters[regime_type] = regime_params
                result.regimes_optimized += 1

        # Step 4: Optimize global fallback parameters (all regimes combined)
        logger.info("Optimizing global fallback parameters...")
        global_params = self._optimize_single_regime(
            regime_data=df,
            regime_type=None,  # No regime filter
            pattern_key=pattern_key,
            direction=direction,
            parameter_ranges=parameter_ranges,
            max_trials=max_trials,
            ga_config=ga_config,
        )

        if global_params:
            result.global_parameters = global_params.parameters
            result.global_metrics = global_params.performance_metrics

        result.optimization_time_seconds = time.time() - start_time

        logger.info(
            f"Regime-aware optimization completed in {result.optimization_time_seconds:.1f}s. "
            f"Optimized {result.regimes_optimized} regimes."
        )

        return result

    def _optimize_single_regime(
        self,
        regime_data: pd.DataFrame,
        regime_type: Optional[RegimeType],
        pattern_key: str,
        direction: str,
        parameter_ranges: Dict[str, Any],
        max_trials: int,
        ga_config: Optional[GAConfig],
    ) -> Optional[RegimeParameters]:
        """
        Optimize parameters for a single regime.

        Args:
            regime_data: Data filtered for this regime
            regime_type: Regime being optimized (None for global)
            pattern_key: Pattern identifier
            direction: "bull" or "bear"
            parameter_ranges: Parameter search space
            max_trials: Max trials
            ga_config: GA configuration

        Returns:
            RegimeParameters with best parameters and metrics
        """
        if len(regime_data) < self.min_samples_per_regime:
            return None

        # Initialize genetic algorithm
        if ga_config is None:
            ga_config = GAConfig(
                population_size=20,
                generations=max_trials // 20,
                crossover_rate=0.8,
                mutation_rate=0.2,
            )

        ga = GeneticAlgorithm(ga_config)

        # Define objective function for this regime
        def objective(params: Dict[str, Any]) -> float:
            """
            Evaluate pattern with given parameters on regime data.

            Returns fitness score (higher is better).
            """
            # Run backtest with these parameters
            # (This is simplified - actual implementation would call pattern detector)
            try:
                backtest_result = self._run_pattern_backtest(
                    df=regime_data,
                    pattern_key=pattern_key,
                    direction=direction,
                    parameters=params,
                )

                # Multi-objective scoring
                # Combine win_rate, profit_factor, sharpe, with constraints
                win_rate = backtest_result.get("win_rate", 0.5)
                profit_factor = backtest_result.get("profit_factor", 1.0)
                sharpe = backtest_result.get("sharpe_ratio", 0.0)
                num_trades = backtest_result.get("num_trades", 0)

                # Penalize if too few trades
                if num_trades < 10:
                    return 0.0

                # Weighted fitness
                fitness = (
                    0.4 * win_rate +
                    0.3 * min(profit_factor / 2.0, 1.0) +  # Normalize PF
                    0.3 * max(0, min(sharpe / 2.0, 1.0))    # Normalize Sharpe
                )

                return fitness

            except Exception as e:
                logger.warning(f"Objective function failed: {e}")
                return 0.0

        # Run optimization
        best_params, best_fitness, all_results = ga.optimize(
            objective_function=objective,
            parameter_space=parameter_ranges,
        )

        if best_params is None or best_fitness < 0.3:
            logger.warning(
                f"Optimization failed for regime {regime_type}: "
                f"best_fitness={best_fitness}"
            )
            return None

        # Get detailed metrics for best parameters
        final_backtest = self._run_pattern_backtest(
            df=regime_data,
            pattern_key=pattern_key,
            direction=direction,
            parameters=best_params,
        )

        return RegimeParameters(
            regime_type=regime_type or RegimeType.UNKNOWN,
            parameters=best_params,
            performance_metrics=final_backtest,
            optimization_date=datetime.now(),
            sample_size=len(regime_data),
            confidence=min(1.0, len(regime_data) / 200.0),  # Confidence based on sample size
        )

    def _run_pattern_backtest(
        self,
        df: pd.DataFrame,
        pattern_key: str,
        direction: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Run backtest for pattern with given parameters.

        This is a simplified placeholder. Actual implementation would:
        1. Instantiate pattern detector with parameters
        2. Scan historical data for pattern occurrences
        3. Simulate trades based on pattern signals
        4. Calculate performance metrics

        Args:
            df: Historical data
            pattern_key: Pattern to test
            direction: Bull/bear
            parameters: Pattern parameters

        Returns:
            Dictionary of performance metrics
        """
        # Placeholder implementation
        # In production, this would call actual pattern detector and backtest engine

        num_bars = len(df)

        # Simulate pattern detection with these parameters
        # More conservative parameters → fewer signals
        # More aggressive parameters → more signals

        # For demonstration, generate synthetic metrics
        # Real implementation would use actual pattern detection + backtest

        detection_threshold = parameters.get("threshold", 0.5)
        lookback = parameters.get("lookback", 20)

        # Simulate number of trades (inversely related to threshold)
        num_trades = int(num_bars / lookback * (1 - detection_threshold) * 0.5)
        num_trades = max(0, num_trades)

        if num_trades == 0:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "num_trades": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        # Simulate win rate (better parameters → higher win rate)
        # This is completely synthetic for demonstration
        base_win_rate = 0.45 + 0.1 * detection_threshold
        win_rate = np.clip(np.random.normal(base_win_rate, 0.05), 0.3, 0.7)

        num_wins = int(num_trades * win_rate)
        num_losses = num_trades - num_wins

        # Simulate profit factor
        avg_win = abs(np.random.normal(0.02, 0.01))
        avg_loss = abs(np.random.normal(0.015, 0.005))

        if num_losses > 0 and avg_loss > 0:
            profit_factor = (num_wins * avg_win) / (num_losses * avg_loss)
        else:
            profit_factor = num_wins * avg_win / 0.001  # Avoid division by zero

        # Simulate Sharpe ratio
        sharpe_ratio = np.random.normal(0.8, 0.3) if num_trades > 20 else 0.0
        sharpe_ratio = np.clip(sharpe_ratio, -1.0, 3.0)

        return {
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": float(sharpe_ratio),
            "num_trades": num_trades,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "total_return": float(num_wins * avg_win - num_losses * avg_loss),
        }

    def select_parameters_for_current_regime(
        self,
        df: pd.DataFrame,
        result: RegimeAwareOptimizationResult,
        lookback: int = 100,
    ) -> Tuple[Dict[str, Any], RegimeType, float]:
        """
        Select appropriate parameters based on current market regime.

        Args:
            df: Recent OHLCV data for regime detection
            result: RegimeAwareOptimizationResult with per-regime parameters
            lookback: Number of recent bars to use for regime detection

        Returns:
            (parameters, current_regime, confidence)
        """
        # Detect current regime
        if not self.hmm_detector.is_fitted:
            logger.warning("HMM not fitted, using global parameters")
            return result.global_parameters or {}, RegimeType.UNKNOWN, 0.0

        # Get recent data
        recent_df = df.tail(lookback)

        # Predict current regime
        regime_state = self.hmm_detector.get_current_regime(recent_df)

        current_regime = regime_state.regime
        confidence = regime_state.probability

        logger.info(
            f"Current regime: {current_regime.value} "
            f"(confidence: {confidence:.2f}, duration: {regime_state.duration})"
        )

        # Get parameters for this regime
        if current_regime in result.regime_parameters:
            regime_params = result.regime_parameters[current_regime]
            return regime_params.parameters, current_regime, confidence
        else:
            # Fallback to global parameters
            logger.warning(
                f"No optimized parameters for regime {current_regime.value}, "
                f"using global fallback"
            )
            return result.global_parameters or {}, current_regime, 0.5

    def save_regime_aware_results(
        self,
        result: RegimeAwareOptimizationResult,
        output_path: Path,
    ) -> None:
        """
        Save regime-aware optimization results to disk.

        Args:
            result: Optimization results
            output_path: Path to save results
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = {
            "pattern_key": result.pattern_key,
            "direction": result.direction,
            "asset": result.asset,
            "timeframe": result.timeframe,
            "optimization_time_seconds": result.optimization_time_seconds,
            "regimes_optimized": result.regimes_optimized,
            "global_parameters": result.global_parameters,
            "global_metrics": result.global_metrics,
            "regime_parameters": {},
            "total_samples_by_regime": {},
        }

        # Add per-regime data
        for regime_type, params in result.regime_parameters.items():
            data["regime_parameters"][regime_type.value] = {
                "parameters": params.parameters,
                "performance_metrics": params.performance_metrics,
                "optimization_date": params.optimization_date.isoformat(),
                "sample_size": params.sample_size,
                "confidence": params.confidence,
            }

        for regime_type, count in result.total_samples_by_regime.items():
            data["total_samples_by_regime"][regime_type.value] = count

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved regime-aware results to {output_path}")

    @staticmethod
    def load_regime_aware_results(
        input_path: Path,
    ) -> RegimeAwareOptimizationResult:
        """
        Load regime-aware optimization results from disk.

        Args:
            input_path: Path to load from

        Returns:
            RegimeAwareOptimizationResult
        """
        import json

        with open(input_path, 'r') as f:
            data = json.load(f)

        result = RegimeAwareOptimizationResult(
            pattern_key=data["pattern_key"],
            direction=data["direction"],
            asset=data["asset"],
            timeframe=data["timeframe"],
            global_parameters=data.get("global_parameters"),
            global_metrics=data.get("global_metrics"),
            optimization_time_seconds=data.get("optimization_time_seconds", 0.0),
            regimes_optimized=data.get("regimes_optimized", 0),
        )

        # Load per-regime parameters
        for regime_str, params_data in data.get("regime_parameters", {}).items():
            regime_type = RegimeType(regime_str)

            regime_params = RegimeParameters(
                regime_type=regime_type,
                parameters=params_data["parameters"],
                performance_metrics=params_data["performance_metrics"],
                optimization_date=datetime.fromisoformat(params_data["optimization_date"]),
                sample_size=params_data["sample_size"],
                confidence=params_data.get("confidence", 1.0),
            )

            result.regime_parameters[regime_type] = regime_params

        # Load sample counts
        for regime_str, count in data.get("total_samples_by_regime", {}).items():
            regime_type = RegimeType(regime_str)
            result.total_samples_by_regime[regime_type] = count

        logger.info(f"Loaded regime-aware results from {input_path}")

        return result
