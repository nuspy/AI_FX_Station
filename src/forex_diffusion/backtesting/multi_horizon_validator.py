"""
Multi-Horizon Walk-Forward Validation System
Advanced validation framework for generative models across multiple forecast horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
from enum import Enum
import json
import pickle
from collections import defaultdict

# Import our backtesting components
from .forecast_backtest_engine import ForecastBacktestEngine, ForecastRecord, ForecastHorizon, ForecastMetrics
from .probabilistic_metrics import ProbabilisticMetrics, ProbabilisticResult
from ..ml.model_cache import get_model_cache

logger = logging.getLogger(__name__)

class ValidationScheme(Enum):
    """Different validation schemes"""
    EXPANDING_WINDOW = "expanding"
    ROLLING_WINDOW = "rolling"
    BLOCKED_CV = "blocked_cv"
    PURGED_CV = "purged_cv"

@dataclass
class ValidationConfig:
    """Configuration for multi-horizon validation"""

    # Time windows
    initial_train_months: int = 12
    validation_months: int = 3
    test_months: int = 1

    # Walk-forward parameters
    step_months: int = 1
    max_windows: Optional[int] = None

    # Purging and embargo (for financial data)
    purge_days: int = 1
    embargo_days: int = 2

    # Validation scheme
    scheme: ValidationScheme = ValidationScheme.EXPANDING_WINDOW

    # Model retraining
    retrain_frequency: int = 3  # Retrain every N months

    # Parallel processing
    n_processes: int = 4

    # Output
    save_predictions: bool = True
    save_models: bool = False

@dataclass
class ValidationWindow:
    """Single validation window"""
    window_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime

    # Data splits
    train_data: Optional[pd.DataFrame] = None
    val_data: Optional[pd.DataFrame] = None
    test_data: Optional[pd.DataFrame] = None

    # Model info
    model_path: Optional[str] = None
    retrained: bool = False

@dataclass
class HorizonResults:
    """Results for a specific forecast horizon"""
    horizon: ForecastHorizon

    # Aggregate metrics across all windows
    avg_metrics: ForecastMetrics
    metrics_by_window: Dict[int, ForecastMetrics]
    probabilistic_results: ProbabilisticResult

    # Forecast records
    all_forecasts: List[ForecastRecord] = field(default_factory=list)

    # Performance trends
    performance_trend: Dict[str, List[float]] = field(default_factory=dict)

@dataclass
class ValidationResults:
    """Complete multi-horizon validation results"""

    # Results by horizon
    horizon_results: Dict[str, HorizonResults] = field(default_factory=dict)

    # Summary statistics
    best_horizon: Optional[str] = None
    worst_horizon: Optional[str] = None
    stability_metrics: Dict[str, float] = field(default_factory=dict)

    # Validation metadata
    config: Optional[ValidationConfig] = None
    total_windows: int = 0
    models_evaluated: List[str] = field(default_factory=list)
    validation_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))

class MultiHorizonValidator:
    """
    Advanced multi-horizon walk-forward validation system.

    Features:
    - Multiple forecast horizons (1h to 1m)
    - Various validation schemes (expanding, rolling, blocked CV)
    - Automatic model retraining
    - Parallel processing
    - Comprehensive performance tracking
    - Regime-aware evaluation
    """

    def __init__(self,
                 models_dir: str = "models/",
                 results_dir: str = "validation_results/",
                 cache_dir: str = "validation_cache/"):

        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.cache_dir = Path(cache_dir)

        # Create directories
        for dir_path in [self.models_dir, self.results_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.forecast_engine = ForecastBacktestEngine()
        self.prob_metrics = ProbabilisticMetrics()

        # Model cache for efficient loading
        self.model_cache = get_model_cache()

        # Validation state
        self.validation_windows: List[ValidationWindow] = []
        self.current_results: Optional[ValidationResults] = None

        logger.info("MultiHorizonValidator initialized")

    def create_validation_windows(self,
                                data: pd.DataFrame,
                                config: ValidationConfig) -> List[ValidationWindow]:
        """Create time-based validation windows"""

        windows = []

        # Get data date range
        start_date = data.index[0]
        end_date = data.index[-1]

        logger.info(f"Creating validation windows from {start_date} to {end_date}")

        # Calculate initial windows
        current_date = start_date
        window_id = 0

        while current_date < end_date:

            # Training period
            train_start = current_date
            train_end = train_start + timedelta(days=config.initial_train_months * 30)

            # Validation period (with purging)
            val_start = train_end + timedelta(days=config.purge_days)
            val_end = val_start + timedelta(days=config.validation_months * 30)

            # Test period (with embargo)
            test_start = val_end + timedelta(days=config.embargo_days)
            test_end = test_start + timedelta(days=config.test_months * 30)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Create window
            window = ValidationWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end
            )

            # Extract data for this window
            train_mask = (data.index >= train_start) & (data.index <= train_end)
            val_mask = (data.index >= val_start) & (data.index <= val_end)
            test_mask = (data.index >= test_start) & (data.index <= test_end)

            window.train_data = data[train_mask]
            window.val_data = data[val_mask]
            window.test_data = data[test_mask]

            windows.append(window)

            # Move to next window
            current_date += timedelta(days=config.step_months * 30)
            window_id += 1

            # Check max windows limit
            if config.max_windows and len(windows) >= config.max_windows:
                break

        logger.info(f"Created {len(windows)} validation windows")
        self.validation_windows = windows
        return windows

    def run_single_window_validation(self,
                                   window: ValidationWindow,
                                   model_name: str,
                                   horizons: List[ForecastHorizon],
                                   model_factory: Callable) -> Dict[str, Any]:
        """Run validation for a single window"""

        logger.info(f"Running validation for window {window.window_id}")

        window_results = {}

        try:
            # Load or train model
            model = self._get_model_for_window(window, model_name, model_factory)

            # Generate forecasts for each horizon
            for horizon in horizons:
                horizon_forecasts = self._generate_forecasts_for_horizon(
                    model, window.test_data, horizon, model_name
                )

                # Store forecasts
                window_results[horizon.value] = {
                    'forecasts': horizon_forecasts,
                    'actual_data': window.test_data,
                    'window_id': window.window_id
                }

        except Exception as e:
            logger.error(f"Error in window {window.window_id}: {e}")
            window_results = {'error': str(e)}

        return window_results

    def _get_model_for_window(self,
                            window: ValidationWindow,
                            model_name: str,
                            model_factory: Callable):
        """Get model for validation window (cached or retrained)"""

        # Check if model exists in cache
        model_cache_key = f"{model_name}_window_{window.window_id}"

        cached_model = self.model_cache.get_model(model_cache_key, 'diffusion')
        if cached_model is not None:
            logger.debug(f"Using cached model for window {window.window_id}")
            return cached_model

        # Train new model
        logger.info(f"Training new model for window {window.window_id}")
        model = model_factory()

        # Train on window training data
        if hasattr(model, 'fit'):
            model.fit(window.train_data)
        elif hasattr(model, 'train'):
            model.train(window.train_data)

        # Cache the model
        self.model_cache.cache_model(model_cache_key, model, 'diffusion')

        window.model_path = model_cache_key
        window.retrained = True

        return model

    def _generate_forecasts_for_horizon(self,
                                      model,
                                      test_data: pd.DataFrame,
                                      horizon: ForecastHorizon,
                                      model_name: str) -> List[ForecastRecord]:
        """Generate forecasts for specific horizon"""

        forecasts = []

        # Generate forecasts at regular intervals
        forecast_interval = timedelta(hours=1)  # Generate forecast every hour

        current_time = test_data.index[0]
        end_time = test_data.index[-1]

        while current_time < end_time:

            try:
                # Get available data up to current time
                available_data = test_data[test_data.index <= current_time]

                if len(available_data) < 24:  # Need at least 24 hours of data
                    current_time += forecast_interval
                    continue

                # Generate forecast using model
                forecast_result = self._generate_single_forecast(
                    model, available_data, horizon, current_time
                )

                if forecast_result:
                    forecast_record = ForecastRecord(
                        forecast_time=current_time,
                        horizon=horizon,
                        target_time=current_time + self._horizon_to_timedelta(horizon),
                        predicted_quantiles=forecast_result,
                        model_name=model_name
                    )
                    forecasts.append(forecast_record)

            except Exception as e:
                logger.warning(f"Failed to generate forecast at {current_time}: {e}")

            current_time += forecast_interval

        return forecasts

    def _generate_single_forecast(self,
                                model,
                                data: pd.DataFrame,
                                horizon: ForecastHorizon,
                                forecast_time: datetime) -> Optional[Dict[str, float]]:
        """Generate a single forecast using the model"""

        try:
            # This is a placeholder - actual implementation depends on model type
            # For diffusion models, you would call the model's predict method

            if hasattr(model, 'predict_quantiles'):
                quantiles = model.predict_quantiles(data, horizon=horizon.value)
                return quantiles
            elif hasattr(model, 'predict'):
                # For models that only return point forecasts
                point_forecast = model.predict(data)

                # Create quantiles around point forecast (simplified)
                std_estimate = data['close'].pct_change().std() * np.sqrt(self._horizon_to_hours(horizon))

                return {
                    'q05': point_forecast - 2 * std_estimate,
                    'q25': point_forecast - 0.67 * std_estimate,
                    'q50': point_forecast,
                    'q75': point_forecast + 0.67 * std_estimate,
                    'q95': point_forecast + 2 * std_estimate
                }

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")

        return None

    def _horizon_to_timedelta(self, horizon: ForecastHorizon) -> timedelta:
        """Convert forecast horizon to timedelta"""
        mapping = {
            ForecastHorizon.H1: timedelta(hours=1),
            ForecastHorizon.H4: timedelta(hours=4),
            ForecastHorizon.D1: timedelta(days=1),
            ForecastHorizon.W1: timedelta(weeks=1),
            ForecastHorizon.M1: timedelta(days=30)
        }
        return mapping[horizon]

    def _horizon_to_hours(self, horizon: ForecastHorizon) -> float:
        """Convert forecast horizon to hours"""
        mapping = {
            ForecastHorizon.H1: 1,
            ForecastHorizon.H4: 4,
            ForecastHorizon.D1: 24,
            ForecastHorizon.W1: 168,
            ForecastHorizon.M1: 720
        }
        return mapping[horizon]

    def run_multi_horizon_validation(self,
                                   data: pd.DataFrame,
                                   model_factory: Callable,
                                   model_name: str = "diffusion",
                                   horizons: Optional[List[ForecastHorizon]] = None,
                                   config: Optional[ValidationConfig] = None) -> ValidationResults:
        """
        Run complete multi-horizon validation study.

        Args:
            data: Market data for validation
            model_factory: Function that creates and returns a model instance
            model_name: Name identifier for the model
            horizons: List of horizons to validate
            config: Validation configuration

        Returns:
            Complete validation results
        """

        if config is None:
            config = ValidationConfig()

        if horizons is None:
            horizons = [ForecastHorizon.H1, ForecastHorizon.H4, ForecastHorizon.D1]

        logger.info(f"Starting multi-horizon validation for {model_name}")
        logger.info(f"Horizons: {[h.value for h in horizons]}")
        logger.info(f"Data period: {data.index[0]} to {data.index[-1]}")

        # Create validation windows
        windows = self.create_validation_windows(data, config)

        # Initialize results container
        results = ValidationResults(
            config=config,
            total_windows=len(windows),
            models_evaluated=[model_name],
            validation_period=(data.index[0], data.index[-1])
        )

        # Parallel processing of windows
        all_window_results = []

        if config.n_processes > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=config.n_processes) as executor:
                future_to_window = {
                    executor.submit(
                        self.run_single_window_validation,
                        window, model_name, horizons, model_factory
                    ): window
                    for window in windows
                }

                for future in as_completed(future_to_window):
                    window = future_to_window[future]
                    try:
                        window_result = future.result()
                        all_window_results.append((window.window_id, window_result))
                    except Exception as e:
                        logger.error(f"Window {window.window_id} failed: {e}")
        else:
            # Sequential execution
            for window in windows:
                window_result = self.run_single_window_validation(
                    window, model_name, horizons, model_factory
                )
                all_window_results.append((window.window_id, window_result))

        # Process results for each horizon
        for horizon in horizons:
            horizon_key = horizon.value

            # Collect all forecasts and actuals for this horizon
            all_forecasts = []
            metrics_by_window = {}

            for window_id, window_result in all_window_results:
                if 'error' in window_result:
                    continue

                horizon_data = window_result.get(horizon_key, {})
                forecasts = horizon_data.get('forecasts', [])
                actual_data = horizon_data.get('actual_data')

                # Update forecasts with actual prices
                for forecast in forecasts:
                    if forecast.target_time in actual_data.index:
                        forecast.actual_price = actual_data.loc[forecast.target_time, 'close']

                all_forecasts.extend(forecasts)

                # Calculate window-specific metrics
                if forecasts:
                    valid_forecasts = [f for f in forecasts if f.actual_price is not None]
                    if len(valid_forecasts) >= 5:
                        window_metrics = self._calculate_window_metrics(valid_forecasts)
                        metrics_by_window[window_id] = window_metrics

            # Calculate aggregate metrics for this horizon
            valid_forecasts = [f for f in all_forecasts if f.actual_price is not None]

            if len(valid_forecasts) >= 10:
                avg_metrics = self._calculate_window_metrics(valid_forecasts)

                # Calculate probabilistic metrics
                forecast_dicts = [f.predicted_quantiles for f in valid_forecasts]
                observations = [f.actual_price for f in valid_forecasts]

                prob_result = self.prob_metrics.comprehensive_evaluation(
                    forecast_dicts,
                    observations,
                    forecast_horizon=horizon.value
                )

                # Calculate performance trends
                performance_trend = self._calculate_performance_trends(metrics_by_window)

                # Store horizon results
                horizon_results = HorizonResults(
                    horizon=horizon,
                    avg_metrics=avg_metrics,
                    metrics_by_window=metrics_by_window,
                    probabilistic_results=prob_result,
                    all_forecasts=all_forecasts,
                    performance_trend=performance_trend
                )

                results.horizon_results[horizon_key] = horizon_results
            else:
                logger.warning(f"Insufficient valid forecasts for horizon {horizon_key}: {len(valid_forecasts)}")

        # Calculate summary statistics
        self._calculate_summary_statistics(results)

        # Save results
        self._save_validation_results(results, model_name)

        self.current_results = results
        logger.info("Multi-horizon validation completed")

        return results

    def _calculate_window_metrics(self, forecasts: List[ForecastRecord]) -> ForecastMetrics:
        """Calculate metrics for a single window"""

        # Extract actual and predicted values
        actuals = [f.actual_price for f in forecasts]
        predictions = [f.predicted_quantiles.get('q50', 0) for f in forecasts]

        # Basic metrics
        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions)) ** 2))
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100

        # Directional accuracy
        actual_directions = np.sign(np.diff(actuals))
        pred_directions = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_directions == pred_directions) if len(actual_directions) > 0 else 0

        return ForecastMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            directional_accuracy=directional_accuracy,
            total_forecasts=len(forecasts)
        )

    def _calculate_performance_trends(self, metrics_by_window: Dict[int, ForecastMetrics]) -> Dict[str, List[float]]:
        """Calculate performance trends across windows"""

        trends = {
            'mae': [],
            'rmse': [],
            'directional_accuracy': []
        }

        for window_id in sorted(metrics_by_window.keys()):
            metrics = metrics_by_window[window_id]
            trends['mae'].append(metrics.mae)
            trends['rmse'].append(metrics.rmse)
            trends['directional_accuracy'].append(metrics.directional_accuracy)

        return trends

    def _calculate_summary_statistics(self, results: ValidationResults):
        """Calculate summary statistics across all horizons"""

        if not results.horizon_results:
            return

        # Find best and worst horizons by CRPS
        best_crps = float('inf')
        worst_crps = 0
        best_horizon = None
        worst_horizon = None

        stability_metrics = {}

        for horizon_key, horizon_result in results.horizon_results.items():
            crps = horizon_result.probabilistic_results.crps

            if crps < best_crps:
                best_crps = crps
                best_horizon = horizon_key

            if crps > worst_crps:
                worst_crps = crps
                worst_horizon = horizon_key

            # Calculate stability (coefficient of variation of MAE across windows)
            mae_values = [m.mae for m in horizon_result.metrics_by_window.values()]
            if len(mae_values) > 1:
                stability_metrics[f'{horizon_key}_mae_cv'] = np.std(mae_values) / np.mean(mae_values)

        results.best_horizon = best_horizon
        results.worst_horizon = worst_horizon
        results.stability_metrics = stability_metrics

    def _save_validation_results(self, results: ValidationResults, model_name: str):
        """Save validation results to disk"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"validation_{model_name}_{timestamp}.pkl"

        try:
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)

            logger.info(f"Validation results saved to {results_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def generate_validation_report(self, results: Optional[ValidationResults] = None) -> str:
        """Generate comprehensive validation report"""

        if results is None:
            results = self.current_results

        if results is None:
            return "No validation results available."

        report = []
        report.append("=" * 80)
        report.append("MULTI-HORIZON VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models: {', '.join(results.models_evaluated)}")
        report.append(f"Validation Period: {results.validation_period[0]} to {results.validation_period[1]}")
        report.append(f"Total Windows: {results.total_windows}")
        report.append("")

        # Summary statistics
        report.append("SUMMARY:")
        report.append(f"  Best Horizon: {results.best_horizon}")
        report.append(f"  Worst Horizon: {results.worst_horizon}")
        report.append("")

        # Detailed results by horizon
        report.append("HORIZON RESULTS:")
        report.append("-" * 40)

        for horizon_key, horizon_result in results.horizon_results.items():
            report.append(f"\n{horizon_key.upper()}:")

            # Basic metrics
            metrics = horizon_result.avg_metrics
            report.append(f"  MAE: {metrics.mae:.6f}")
            report.append(f"  RMSE: {metrics.rmse:.6f}")
            report.append(f"  MAPE: {metrics.mape:.2f}%")
            report.append(f"  Directional Accuracy: {metrics.directional_accuracy:.4f}")

            # Probabilistic metrics
            prob = horizon_result.probabilistic_results
            report.append(f"  CRPS: {prob.crps:.6f}")
            report.append(f"  Log Score: {prob.log_score:.6f}")
            report.append(f"  PIT p-value: {prob.pit_uniformity_pvalue:.4f}")

            # Stability
            stability_key = f'{horizon_key}_mae_cv'
            if stability_key in results.stability_metrics:
                report.append(f"  MAE Stability (CV): {results.stability_metrics[stability_key]:.4f}")

            report.append(f"  Total Forecasts: {len(horizon_result.all_forecasts)}")
            report.append(f"  Windows Evaluated: {len(horizon_result.metrics_by_window)}")

        return "\n".join(report)


# Example model factory for testing
def create_dummy_diffusion_model():
    """Create a dummy model for testing purposes"""

    class DummyModel:
        def fit(self, data):
            self.data_mean = data['close'].mean()
            self.data_std = data['close'].std()

        def predict_quantiles(self, data, horizon='1h'):
            # Simple random walk with increasing uncertainty
            last_price = data['close'].iloc[-1]

            horizon_hours = {'1h': 1, '4h': 4, '1d': 24, '1w': 168}
            h = horizon_hours.get(horizon, 1)

            std_scaling = np.sqrt(h)
            base_std = self.data_std * std_scaling

            return {
                'q05': last_price - 2 * base_std,
                'q25': last_price - 0.67 * base_std,
                'q50': last_price,
                'q75': last_price + 0.67 * base_std,
                'q95': last_price + 2 * base_std
            }

    return DummyModel()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Multi-Horizon Validator...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='h')
    prices = 1.1 + np.cumsum(np.random.randn(len(dates)) * 0.001)

    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)

    # Initialize validator
    validator = MultiHorizonValidator()

    # Configuration
    config = ValidationConfig(
        initial_train_months=6,
        validation_months=1,
        test_months=1,
        step_months=1,
        max_windows=3,
        n_processes=1
    )

    # Run validation
    horizons = [ForecastHorizon.H1, ForecastHorizon.H4]

    results = validator.run_multi_horizon_validation(
        data=data,
        model_factory=create_dummy_diffusion_model,
        model_name="dummy_diffusion",
        horizons=horizons,
        config=config
    )

    # Generate report
    report = validator.generate_validation_report(results)
    print(report)