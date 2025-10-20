"""
Forecast Backtesting Engine
Specialized backtesting system for generative models (diffusion, VAE, etc.)
Focuses on probabilistic forecast accuracy and multi-horizon validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
from enum import Enum

# Statistical imports for probabilistic metrics
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

class ForecastHorizon(Enum):
    """Forecast horizons for multi-horizon evaluation"""
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    M1 = "1m"

@dataclass
class ForecastRecord:
    """Individual forecast record for backtesting"""
    forecast_time: datetime
    horizon: ForecastHorizon
    target_time: datetime
    predicted_quantiles: Dict[str, float]  # e.g., {'q05': 1.1000, 'q50': 1.1050, 'q95': 1.1100}
    actual_price: Optional[float] = None
    pair: str = "EURUSD"
    model_name: str = "diffusion_v1"

    # Additional metadata
    regime: Optional[str] = None
    volatility: Optional[float] = None
    confidence_score: Optional[float] = None

@dataclass
class ForecastMetrics:
    """Comprehensive forecast evaluation metrics"""
    # Basic accuracy metrics
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0

    # Probabilistic metrics
    crps: float = 0.0  # Continuous Ranked Probability Score
    pit_uniformity: float = 0.0  # Probability Integral Transform uniformity test
    interval_coverage: Dict[str, float] = field(default_factory=dict)  # Coverage rates for prediction intervals

    # Directional accuracy
    directional_accuracy: float = 0.0
    hit_rate: float = 0.0

    # Multi-horizon metrics
    horizon_performance: Dict[str, float] = field(default_factory=dict)

    # Model comparison
    relative_improvement: float = 0.0  # vs baseline
    significance_test: Optional[float] = None

    # Metadata
    total_forecasts: int = 0
    evaluation_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))

class ForecastBacktestEngine:
    """
    Advanced backtesting engine specifically designed for probabilistic forecasts.

    Features:
    - Multi-horizon evaluation (1h to 1m)
    - Probabilistic metrics (CRPS, PIT, interval coverage)
    - Walk-forward validation
    - Regime-aware evaluation
    - Model comparison and significance testing
    """

    def __init__(self,
                 models_dir: str = "models/",
                 results_dir: str = "backtest_results/forecast/"):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.forecast_records: List[ForecastRecord] = []
        self.baseline_forecasts: List[ForecastRecord] = []

        # Configuration
        self.quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        self.prediction_intervals = [(0.05, 0.95), (0.25, 0.75), (0.1, 0.9)]

        logger.info("ForecastBacktestEngine initialized")

    def add_forecast_record(self,
                          forecast_time: datetime,
                          horizon: Union[ForecastHorizon, str],
                          predicted_quantiles: Dict[str, float],
                          model_name: str = "diffusion",
                          **metadata) -> None:
        """Add a forecast record to the backtesting dataset"""

        if isinstance(horizon, str):
            horizon = ForecastHorizon(horizon)

        target_time = self._calculate_target_time(forecast_time, horizon)

        record = ForecastRecord(
            forecast_time=forecast_time,
            horizon=horizon,
            target_time=target_time,
            predicted_quantiles=predicted_quantiles,
            model_name=model_name,
            **metadata
        )

        self.forecast_records.append(record)
        logger.debug(f"Added forecast record: {model_name} at {forecast_time} for {horizon.value}")

    def _calculate_target_time(self, forecast_time: datetime, horizon: ForecastHorizon) -> datetime:
        """Calculate target time based on forecast horizon"""
        horizon_deltas = {
            ForecastHorizon.H1: timedelta(hours=1),
            ForecastHorizon.H4: timedelta(hours=4),
            ForecastHorizon.D1: timedelta(days=1),
            ForecastHorizon.W1: timedelta(weeks=1),
            ForecastHorizon.M1: timedelta(days=30)
        }

        return forecast_time + horizon_deltas[horizon]

    def update_actuals(self, market_data: pd.DataFrame, pair: str = "EURUSD") -> int:
        """Update forecast records with actual prices from market data"""

        updated_count = 0

        for record in self.forecast_records:
            if record.pair != pair or record.actual_price is not None:
                continue

            # Find closest actual price to target time
            target_timestamp = pd.Timestamp(record.target_time)

            if target_timestamp in market_data.index:
                record.actual_price = market_data.loc[target_timestamp, 'close']
                updated_count += 1
            else:
                # Find nearest timestamp
                nearest_idx = market_data.index.get_indexer([target_timestamp], method='nearest')[0]
                if nearest_idx != -1:
                    record.actual_price = market_data.iloc[nearest_idx]['close']
                    updated_count += 1

        logger.info(f"Updated {updated_count} forecast records with actual prices")
        return updated_count

    def calculate_crps(self, predicted_quantiles: Dict[str, float], actual: float) -> float:
        """
        Calculate Continuous Ranked Probability Score (CRPS)
        CRPS measures the accuracy of probabilistic forecasts
        """
        try:
            # Convert quantiles to arrays for CRPS calculation
            quantile_values = np.array([predicted_quantiles.get(f'q{int(q*100):02d}', 0)
                                      for q in self.quantile_levels])

            # Simple CRPS approximation using quantiles
            # More sophisticated implementations would use full distribution

            crps_sum = 0.0
            for i, q in enumerate(self.quantile_levels):
                forecast_val = quantile_values[i]
                indicator = 1.0 if actual <= forecast_val else 0.0
                crps_sum += (indicator - q) * (forecast_val - actual)

            return abs(crps_sum / len(self.quantile_levels))

        except Exception as e:
            logger.warning(f"CRPS calculation failed: {e}")
            return float('inf')

    def calculate_pit_uniformity(self, records: List[ForecastRecord]) -> float:
        """
        Calculate Probability Integral Transform (PIT) uniformity test
        Tests if forecast distributions are well-calibrated
        """
        pit_values = []

        for record in records:
            if record.actual_price is None:
                continue

            # Calculate PIT value: P(Y <= y) where Y ~ forecast distribution
            actual = record.actual_price
            pit_value = 0.0

            # Approximate PIT using quantiles
            quantiles = record.predicted_quantiles
            for q_name, q_val in quantiles.items():
                if 'q' in q_name:
                    q_level = float(q_name[1:]) / 100.0
                    if actual <= q_val:
                        pit_value = q_level
                        break

            pit_values.append(pit_value)

        if len(pit_values) < 10:
            return 0.0

        # Test uniformity using Kolmogorov-Smirnov test
        pit_array = np.array(pit_values)
        ks_stat, p_value = stats.kstest(pit_array, 'uniform')

        # Return p-value (higher = more uniform = better calibrated)
        return p_value

    def calculate_interval_coverage(self, records: List[ForecastRecord]) -> Dict[str, float]:
        """Calculate empirical coverage rates for prediction intervals"""

        coverage_rates = {}

        for lower_q, upper_q in self.prediction_intervals:
            lower_name = f'q{int(lower_q*100):02d}'
            upper_name = f'q{int(upper_q*100):02d}'

            covered_count = 0
            total_count = 0

            for record in records:
                if record.actual_price is None:
                    continue

                quantiles = record.predicted_quantiles
                if lower_name in quantiles and upper_name in quantiles:
                    lower_bound = quantiles[lower_name]
                    upper_bound = quantiles[upper_name]

                    if lower_bound <= record.actual_price <= upper_bound:
                        covered_count += 1
                    total_count += 1

            if total_count > 0:
                empirical_coverage = covered_count / total_count
                nominal_coverage = upper_q - lower_q
                coverage_rates[f'{int(lower_q*100)}-{int(upper_q*100)}%'] = {
                    'empirical': empirical_coverage,
                    'nominal': nominal_coverage,
                    'error': abs(empirical_coverage - nominal_coverage)
                }

        return coverage_rates

    def run_backtest(self,
                     model_names: Optional[List[str]] = None,
                     horizons: Optional[List[ForecastHorizon]] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> Dict[str, ForecastMetrics]:
        """
        Run comprehensive forecast backtesting

        Returns metrics for each model/horizon combination
        """

        if model_names is None:
            model_names = list(set(record.model_name for record in self.forecast_records))

        if horizons is None:
            horizons = list(ForecastHorizon)

        results = {}

        logger.info(f"Running forecast backtest for {len(model_names)} models, {len(horizons)} horizons")

        for model_name in model_names:
            for horizon in horizons:
                # Filter records
                filtered_records = [
                    r for r in self.forecast_records
                    if (r.model_name == model_name and
                        r.horizon == horizon and
                        r.actual_price is not None and
                        (start_date is None or r.forecast_time >= start_date) and
                        (end_date is None or r.forecast_time <= end_date))
                ]

                if len(filtered_records) < 10:
                    logger.warning(f"Insufficient data for {model_name}_{horizon.value}: {len(filtered_records)} records")
                    continue

                # Calculate metrics
                metrics = self._calculate_comprehensive_metrics(filtered_records)
                results[f"{model_name}_{horizon.value}"] = metrics

        logger.info(f"Backtest completed. Results for {len(results)} model/horizon combinations")
        return results

    def _calculate_comprehensive_metrics(self, records: List[ForecastRecord]) -> ForecastMetrics:
        """Calculate all forecast evaluation metrics for a set of records"""

        actuals = [r.actual_price for r in records]
        point_forecasts = [r.predicted_quantiles.get('q50', 0) for r in records]

        # Basic accuracy metrics
        mae = mean_absolute_error(actuals, point_forecasts)
        rmse = np.sqrt(mean_squared_error(actuals, point_forecasts))
        mape = np.mean(np.abs((np.array(actuals) - np.array(point_forecasts)) / np.array(actuals))) * 100

        # Probabilistic metrics
        crps_scores = [self.calculate_crps(r.predicted_quantiles, r.actual_price) for r in records]
        avg_crps = np.mean([score for score in crps_scores if not np.isinf(score)])

        pit_uniformity = self.calculate_pit_uniformity(records)
        interval_coverage = self.calculate_interval_coverage(records)

        # Directional accuracy
        directions_actual = np.diff(actuals)
        directions_forecast = np.diff(point_forecasts)
        directional_accuracy = np.mean(np.sign(directions_actual) == np.sign(directions_forecast))

        # Hit rate (forecast within 10% of actual)
        hit_rate = np.mean(np.abs(np.array(actuals) - np.array(point_forecasts)) / np.array(actuals) < 0.1)

        return ForecastMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            crps=avg_crps,
            pit_uniformity=pit_uniformity,
            interval_coverage=interval_coverage,
            directional_accuracy=directional_accuracy,
            hit_rate=hit_rate,
            total_forecasts=len(records),
            evaluation_period=(min(r.forecast_time for r in records),
                             max(r.forecast_time for r in records))
        )

    def compare_models(self,
                      model_results: Dict[str, ForecastMetrics],
                      baseline_model: str = "random_walk") -> Dict[str, Dict[str, float]]:
        """Compare models against baseline and each other"""

        comparisons = {}

        if baseline_model not in model_results:
            logger.warning(f"Baseline model {baseline_model} not found in results")
            return comparisons

        baseline_metrics = model_results[baseline_model]

        for model_name, metrics in model_results.items():
            if model_name == baseline_model:
                continue

            # Calculate relative improvements
            comparison = {
                'mae_improvement': (baseline_metrics.mae - metrics.mae) / baseline_metrics.mae * 100,
                'rmse_improvement': (baseline_metrics.rmse - metrics.rmse) / baseline_metrics.rmse * 100,
                'crps_improvement': (baseline_metrics.crps - metrics.crps) / baseline_metrics.crps * 100,
                'directional_accuracy_improvement': metrics.directional_accuracy - baseline_metrics.directional_accuracy,
                'hit_rate_improvement': metrics.hit_rate - baseline_metrics.hit_rate
            }

            comparisons[model_name] = comparison

        return comparisons

    def export_results(self,
                      results: Dict[str, ForecastMetrics],
                      filename: Optional[str] = None) -> str:
        """Export backtesting results to JSON file"""

        if filename is None:
            filename = f"forecast_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        export_path = self.results_dir / filename

        # Convert results to serializable format
        serializable_results = {}
        for model_name, metrics in results.items():
            serializable_results[model_name] = {
                'mae': metrics.mae,
                'rmse': metrics.rmse,
                'mape': metrics.mape,
                'crps': metrics.crps,
                'pit_uniformity': metrics.pit_uniformity,
                'directional_accuracy': metrics.directional_accuracy,
                'hit_rate': metrics.hit_rate,
                'total_forecasts': metrics.total_forecasts,
                'interval_coverage': metrics.interval_coverage,
                'evaluation_start': metrics.evaluation_period[0].isoformat(),
                'evaluation_end': metrics.evaluation_period[1].isoformat()
            }

        with open(export_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results exported to {export_path}")
        return str(export_path)

    def generate_report(self, results: Dict[str, ForecastMetrics]) -> str:
        """Generate comprehensive backtesting report"""

        report = []
        report.append("=" * 80)
        report.append("FORECAST BACKTESTING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Model/Horizon Combinations: {len(results)}")
        report.append("")

        # Summary statistics
        all_maes = [m.mae for m in results.values()]
        all_crps = [m.crps for m in results.values()]

        report.append("SUMMARY STATISTICS:")
        report.append(f"  Average MAE: {np.mean(all_maes):.6f}")
        report.append(f"  Average CRPS: {np.mean(all_crps):.6f}")
        report.append(f"  Best MAE: {min(all_maes):.6f}")
        report.append(f"  Best CRPS: {min(all_crps):.6f}")
        report.append("")

        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)

        for model_name, metrics in sorted(results.items()):
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  MAE: {metrics.mae:.6f}")
            report.append(f"  RMSE: {metrics.rmse:.6f}")
            report.append(f"  MAPE: {metrics.mape:.2f}%")
            report.append(f"  CRPS: {metrics.crps:.6f}")
            report.append(f"  PIT Uniformity: {metrics.pit_uniformity:.4f}")
            report.append(f"  Directional Accuracy: {metrics.directional_accuracy:.4f}")
            report.append(f"  Hit Rate: {metrics.hit_rate:.4f}")
            report.append(f"  Total Forecasts: {metrics.total_forecasts}")

            # Interval coverage details
            if metrics.interval_coverage:
                report.append("  Prediction Intervals:")
                for interval, coverage in metrics.interval_coverage.items():
                    report.append(f"    {interval}: {coverage['empirical']:.3f} (nominal: {coverage['nominal']:.3f})")

        return "\n".join(report)


# Utility functions for integration with existing codebase

def create_forecast_record_from_prediction(prediction_result: Dict[str, Any],
                                         model_name: str = "diffusion") -> ForecastRecord:
    """Create ForecastRecord from existing prediction system output"""

    return ForecastRecord(
        forecast_time=datetime.now(),
        horizon=ForecastHorizon.H1,  # Default to 1 hour
        target_time=datetime.now() + timedelta(hours=1),
        predicted_quantiles=prediction_result.get('quantiles', {}),
        model_name=model_name,
        pair=prediction_result.get('pair', 'EURUSD'),
        confidence_score=prediction_result.get('confidence', None)
    )

def run_forecast_validation_study(engine: ForecastBacktestEngine,
                                models: List[str],
                                data_start: datetime,
                                data_end: datetime) -> Dict[str, Any]:
    """Run comprehensive forecast validation study"""

    logger.info("Starting forecast validation study...")

    # Run main backtesting
    results = engine.run_backtest(
        model_names=models,
        start_date=data_start,
        end_date=data_end
    )

    # Model comparison
    if len(results) > 1:
        comparisons = engine.compare_models(results)
    else:
        comparisons = {}

    # Generate report
    report = engine.generate_report(results)

    # Export results
    results_file = engine.export_results(results)

    return {
        'results': results,
        'comparisons': comparisons,
        'report': report,
        'results_file': results_file,
        'summary': {
            'total_models': len(set(r.split('_')[0] for r in results.keys())),
            'total_horizons': len(set(r.split('_')[1] for r in results.keys())),
            'best_model': min(results.keys(), key=lambda x: results[x].mae),
            'study_period': f"{data_start} to {data_end}"
        }
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Forecast Backtesting Engine...")

    engine = ForecastBacktestEngine()

    # Add sample forecast records
    base_time = datetime.now() - timedelta(days=30)

    for i in range(100):
        forecast_time = base_time + timedelta(hours=i)

        # Simulate quantile predictions
        base_price = 1.1000 + np.random.normal(0, 0.01)
        quantiles = {
            'q05': base_price - 0.005,
            'q25': base_price - 0.002,
            'q50': base_price,
            'q75': base_price + 0.002,
            'q95': base_price + 0.005
        }

        engine.add_forecast_record(
            forecast_time=forecast_time,
            horizon=ForecastHorizon.H1,
            predicted_quantiles=quantiles,
            model_name="test_model"
        )

    # Simulate actual prices (add some random walk)
    market_data = pd.DataFrame({
        'close': 1.1000 + np.cumsum(np.random.normal(0, 0.001, 100))
    }, index=pd.date_range(start=base_time + timedelta(hours=1), periods=100, freq='h'))

    engine.update_actuals(market_data)

    # Run backtest
    results = engine.run_backtest()

    # Generate report
    report = engine.generate_report(results)
    print(report)