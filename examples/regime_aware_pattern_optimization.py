"""
Example: Regime-Aware Pattern Optimization

Demonstrates how to optimize pattern parameters separately for each market regime.

This example shows:
1. Loading historical data
2. Fitting HMM regime detector
3. Optimizing pattern parameters for each regime
4. Runtime regime detection and parameter selection
5. Saving/loading regime-aware results
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from forex_diffusion.regime.hmm_detector import HMMRegimeDetector, RegimeType
from forex_diffusion.training.optimization.regime_aware_optimizer import (
    RegimeAwareOptimizer,
    RegimeAwareOptimizationResult,
)


def generate_sample_data(n_bars: int = 1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demonstration"""

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="1H")

    # Generate synthetic price data with regime changes
    price = 1.2000
    prices = []
    volumes = []

    for i in range(n_bars):
        # Simulate regime changes
        if i < 300:
            # Trending up
            price += np.random.normal(0.0002, 0.0005)
            volume = np.random.lognormal(10, 0.5)
        elif i < 600:
            # Ranging
            price += np.random.normal(0, 0.0002)
            volume = np.random.lognormal(9.5, 0.3)
        elif i < 850:
            # Trending down
            price += np.random.normal(-0.0002, 0.0005)
            volume = np.random.lognormal(10.2, 0.6)
        else:
            # Volatile
            price += np.random.normal(0, 0.0008)
            volume = np.random.lognormal(11, 0.8)

        prices.append(price)
        volumes.append(volume)

    # Generate OHLC from prices
    df = pd.DataFrame(index=dates)
    df["close"] = prices

    # Add some noise for OHLC
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) + abs(np.random.normal(0, 0.0001, n_bars))
    df["low"] = df[["open", "close"]].min(axis=1) - abs(np.random.normal(0, 0.0001, n_bars))
    df["volume"] = volumes

    return df


def main():
    """Main example workflow"""

    print("="*80)
    print("Regime-Aware Pattern Optimization Example")
    print("="*80)

    # Step 1: Generate sample data
    print("\n1. Generating sample OHLCV data...")
    df = generate_sample_data(n_bars=1000)
    print(f"   Generated {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Step 2: Initialize HMM regime detector
    print("\n2. Initializing HMM Regime Detector...")
    hmm_detector = HMMRegimeDetector(n_regimes=4, min_history=100)
    print("   Fitting HMM on historical data...")
    hmm_detector.fit(df)

    # Show regime statistics
    regime_stats = hmm_detector.get_regime_statistics(df)
    print(f"\n   Regime Distribution:")
    for regime, count in regime_stats["regime_counts"].items():
        pct = count / len(df) * 100
        print(f"     {regime}: {count} bars ({pct:.1f}%)")

    # Step 3: Initialize regime-aware optimizer
    print("\n3. Initializing Regime-Aware Optimizer...")
    optimizer = RegimeAwareOptimizer(
        hmm_detector=hmm_detector,
        min_samples_per_regime=50,
    )

    # Step 4: Define parameter ranges for pattern
    print("\n4. Defining parameter search space...")
    parameter_ranges = {
        "threshold": {"type": "float", "min": 0.3, "max": 0.9},
        "lookback": {"type": "int", "min": 10, "max": 50},
        "min_pattern_bars": {"type": "int", "min": 5, "max": 20},
        "confirmation_bars": {"type": "int", "min": 1, "max": 5},
    }

    # Step 5: Run regime-aware optimization
    print("\n5. Running regime-aware optimization...")
    print("   (This will optimize pattern parameters separately for each regime)")

    result = optimizer.optimize_pattern_by_regime(
        df=df,
        pattern_key="head_and_shoulders",
        direction="bull",
        asset="EURUSD",
        timeframe="1H",
        parameter_ranges=parameter_ranges,
        max_trials=50,  # Reduced for demo
    )

    # Step 6: Display results
    print("\n6. Optimization Results:")
    print(f"   Total optimization time: {result.optimization_time_seconds:.1f}s")
    print(f"   Regimes optimized: {result.regimes_optimized}/4")

    print("\n   Per-Regime Parameters:")
    for regime_type, regime_params in result.regime_parameters.items():
        print(f"\n   {regime_type.value}:")
        print(f"     Sample size: {regime_params.sample_size}")
        print(f"     Confidence: {regime_params.confidence:.2f}")
        print(f"     Parameters: {regime_params.parameters}")
        print(f"     Metrics:")
        for metric, value in regime_params.performance_metrics.items():
            if isinstance(value, (int, float)):
                print(f"       {metric}: {value:.4f}")

    print("\n   Global Fallback Parameters:")
    print(f"     {result.global_parameters}")
    print(f"     Metrics: {result.global_metrics}")

    # Step 7: Runtime regime detection and parameter selection
    print("\n7. Runtime Regime Detection and Parameter Selection:")

    # Simulate detecting current regime from recent data
    recent_df = df.tail(100)  # Last 100 bars

    params, current_regime, confidence = optimizer.select_parameters_for_current_regime(
        df=recent_df,
        result=result,
        lookback=100,
    )

    print(f"   Current regime: {current_regime.value}")
    print(f"   Detection confidence: {confidence:.2f}")
    print(f"   Selected parameters: {params}")

    # Step 8: Save and load results
    print("\n8. Saving and Loading Results:")

    output_path = Path("./regime_aware_results.json")
    optimizer.save_regime_aware_results(result, output_path)
    print(f"   Saved results to {output_path}")

    loaded_result = RegimeAwareOptimizer.load_regime_aware_results(output_path)
    print(f"   Loaded results successfully")
    print(f"   Regimes loaded: {len(loaded_result.regime_parameters)}")

    # Step 9: Simulate regime transition
    print("\n9. Simulating Regime Transition:")

    # Generate new data with different regime
    print("   Simulating market transition to volatile regime...")
    new_data = df.copy()

    # Add volatile bars
    volatile_bars = generate_sample_data(n_bars=100)
    # Make them more volatile
    volatile_bars["high"] = volatile_bars["high"] + abs(np.random.normal(0, 0.0005, 100))
    volatile_bars["low"] = volatile_bars["low"] - abs(np.random.normal(0, 0.0005, 100))
    volatile_bars["volume"] = volatile_bars["volume"] * 2

    extended_df = pd.concat([df, volatile_bars], ignore_index=False)

    # Detect new regime
    params_new, regime_new, conf_new = optimizer.select_parameters_for_current_regime(
        df=extended_df,
        result=result,
        lookback=100,
    )

    print(f"   New regime detected: {regime_new.value}")
    print(f"   Confidence: {conf_new:.2f}")
    print(f"   Switched to parameters: {params_new}")

    if regime_new != current_regime:
        print(f"\n   ✓ Regime transition detected: {current_regime.value} → {regime_new.value}")
        print(f"   Parameters automatically switched to match new regime")

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
