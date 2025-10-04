"""
Multi-horizon model validation with expanding window methodology.

Validates model performance across multiple prediction horizons to ensure
robustness and identify optimal forecasting windows.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from loguru import logger

from ..postproc.conformal import SplitConformalPredictor, evaluate_coverage
from ..backtest.engine import BacktestEngine


@dataclass
class HorizonResult:
    """Results for a single horizon."""
    horizon: int  # Forecast horizon in bars
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    directional_accuracy: float  # Percentage of correct direction predictions
    sharpe_ratio: float  # Sharpe ratio from backtest
    max_drawdown: float  # Maximum drawdown from backtest
    coverage_95: float  # Coverage of 95% prediction intervals
    interval_width: float  # Average interval width
    n_samples: int  # Number of test samples


class MultiHorizonValidator:
    """
    Multi-horizon validation with expanding window.

    Methodology:
    1. Train on initial window
    2. Validate on multiple horizons (1h, 4h, 12h, 24h, etc.)
    3. Expand training window incrementally
    4. Track performance degradation over time
    5. Identify optimal horizon per symbol/timeframe
    """

    def __init__(
        self,
        horizons: List[int],
        initial_train_bars: int = 2000,
        test_bars: int = 500,
        expand_step: int = 250,
        n_expansions: int = 4,
        alpha: float = 0.05
    ):
        """
        Initialize multi-horizon validator.

        Args:
            horizons: List of forecast horizons to test (in bars)
            initial_train_bars: Initial training window size
            test_bars: Test window size
            expand_step: Number of bars to expand training window
            n_expansions: Number of window expansions
            alpha: Significance level for prediction intervals
        """
        self.horizons = sorted(horizons)
        self.initial_train_bars = initial_train_bars
        self.test_bars = test_bars
        self.expand_step = expand_step
        self.n_expansions = n_expansions
        self.alpha = alpha

        self.results: Dict[int, List[HorizonResult]] = {h: [] for h in horizons}

    def validate_model(
        self,
        model,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        device: str = "cpu"
    ) -> Dict[int, HorizonResult]:
        """
        Validate model across multiple horizons.

        Args:
            model: Trained model (ForexDiffusionLit)
            data: Historical OHLCV data
            symbol: Trading pair
            timeframe: Timeframe
            device: Device for inference

        Returns:
            Dictionary mapping horizon -> aggregated results
        """
        logger.info(f"Starting multi-horizon validation: {len(self.horizons)} horizons")

        model = model.to(device)
        model.eval()

        # Expanding window validation
        for expansion in range(self.n_expansions):
            train_end = self.initial_train_bars + (expansion * self.expand_step)
            test_start = train_end
            test_end = test_start + self.test_bars

            if test_end > len(data):
                logger.warning(f"Insufficient data for expansion {expansion}")
                break

            logger.info(f"Expansion {expansion+1}/{self.n_expansions}: train=[0:{train_end}], test=[{test_start}:{test_end}]")

            # Validate each horizon
            for horizon in self.horizons:
                result = self._validate_horizon(
                    model=model,
                    data=data,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    horizon=horizon,
                    symbol=symbol,
                    timeframe=timeframe,
                    device=device
                )

                self.results[horizon].append(result)

                logger.info(
                    f"  Horizon {horizon}h: MAE={result.mae:.6f}, "
                    f"Sharpe={result.sharpe_ratio:.2f}, "
                    f"Coverage={result.coverage_95:.3f}"
                )

        # Aggregate results per horizon
        aggregated = {}
        for horizon in self.horizons:
            if self.results[horizon]:
                aggregated[horizon] = self._aggregate_results(self.results[horizon])

        return aggregated

    def _validate_horizon(
        self,
        model,
        data: pd.DataFrame,
        train_end: int,
        test_start: int,
        test_end: int,
        horizon: int,
        symbol: str,
        timeframe: str,
        device: str
    ) -> HorizonResult:
        """Validate single horizon."""
        # Split data
        calibration_data = data.iloc[train_end - 500:train_end]  # Last 500 bars for calibration
        test_data = data.iloc[test_start:test_end]

        # Generate predictions
        cal_predictions, cal_actuals = self._generate_predictions(
            model, calibration_data, horizon, device
        )
        test_predictions, test_actuals = self._generate_predictions(
            model, test_data, horizon, device
        )

        # Calculate metrics
        mae = np.mean(np.abs(test_predictions - test_actuals))
        rmse = np.sqrt(np.mean((test_predictions - test_actuals) ** 2))
        mape = np.mean(np.abs((test_predictions - test_actuals) / (test_actuals + 1e-8))) * 100

        # Directional accuracy
        pred_direction = np.sign(test_predictions - test_data['close'].values[:-horizon])
        actual_direction = np.sign(test_actuals - test_data['close'].values[:-horizon])
        directional_accuracy = np.mean(pred_direction == actual_direction) * 100

        # Conformal prediction intervals
        conf_predictor = SplitConformalPredictor(alpha=self.alpha)
        conf_result = conf_predictor.calibrate(cal_predictions, cal_actuals)
        _, intervals = conf_predictor.predict(test_predictions, return_intervals=True)

        coverage_metrics = evaluate_coverage(test_predictions, intervals, test_actuals)

        # Backtest
        backtest_metrics = self._run_backtest(
            predictions=test_predictions,
            data=test_data,
            horizon=horizon,
            symbol=symbol,
            timeframe=timeframe
        )

        return HorizonResult(
            horizon=horizon,
            mae=mae,
            rmse=rmse,
            mape=mape,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=backtest_metrics.get('sharpe_ratio', 0),
            max_drawdown=backtest_metrics.get('max_drawdown', 0),
            coverage_95=coverage_metrics['coverage'],
            interval_width=coverage_metrics['avg_width'],
            n_samples=len(test_predictions)
        )

    def _generate_predictions(
        self,
        model,
        data: pd.DataFrame,
        horizon: int,
        device: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for given data and horizon."""
        from ..training.train import _add_time_features, CHANNEL_ORDER

        # Prepare data
        df = data.copy()
        df = _add_time_features(df)

        # Get normalization stats from model
        mu = np.array(model.dataset_stats['mu'])
        sigma = np.array(model.dataset_stats['sigma'])

        # Extract features
        values = df[CHANNEL_ORDER].values
        closes = df['close'].values

        patch_len = 64  # Default patch length

        predictions = []
        actuals = []

        with torch.no_grad():
            for i in range(patch_len, len(df) - horizon):
                # Extract patch
                patch = values[i - patch_len:i].T  # (C, L)

                # Normalize
                patch_norm = (patch - mu[:, None]) / (sigma[:, None] + 1e-8)

                # Convert to tensor
                x = torch.from_numpy(patch_norm.astype(np.float32)).unsqueeze(0).to(device)

                # Forward pass (VAE mean for deterministic prediction)
                mu_z, _ = model.vae.encode(x)
                x_rec = model.vae.decode(mu_z)

                # Extract predicted close (channel 3)
                pred_close = x_rec[0, 3, -1].item()

                # Denormalize
                pred_close = pred_close * sigma[3] + mu[3]

                # Get actual future close
                actual_close = closes[i + horizon]

                predictions.append(pred_close)
                actuals.append(actual_close)

        return np.array(predictions), np.array(actuals)

    def _run_backtest(
        self,
        predictions: np.ndarray,
        data: pd.DataFrame,
        horizon: int,
        symbol: str,
        timeframe: str
    ) -> Dict:
        """Run simple backtest with predictions."""
        # Simple strategy: buy if prediction > current, sell if prediction < current
        returns = []
        equity = [1.0]

        for i in range(len(predictions)):
            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[min(i + horizon, len(data) - 1)]
            pred_price = predictions[i]

            # Trading decision
            if pred_price > current_price * 1.001:  # 0.1% threshold
                # Long
                ret = (future_price - current_price) / current_price
            elif pred_price < current_price * 0.999:
                # Short
                ret = (current_price - future_price) / current_price
            else:
                # No trade
                ret = 0

            returns.append(ret)
            equity.append(equity[-1] * (1 + ret))

        returns = np.array(returns)

        # Calculate metrics
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # Annualized
        max_dd = self._max_drawdown(np.array(equity))

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'total_return': (equity[-1] - 1) * 100
        }

    def _max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / (peak + 1e-12)
        return np.max(dd)

    def _aggregate_results(self, results: List[HorizonResult]) -> HorizonResult:
        """Aggregate multiple results for same horizon."""
        return HorizonResult(
            horizon=results[0].horizon,
            mae=np.mean([r.mae for r in results]),
            rmse=np.mean([r.rmse for r in results]),
            mape=np.mean([r.mape for r in results]),
            directional_accuracy=np.mean([r.directional_accuracy for r in results]),
            sharpe_ratio=np.mean([r.sharpe_ratio for r in results]),
            max_drawdown=np.mean([r.max_drawdown for r in results]),
            coverage_95=np.mean([r.coverage_95 for r in results]),
            interval_width=np.mean([r.interval_width for r in results]),
            n_samples=sum([r.n_samples for r in results])
        )

    def get_optimal_horizon(self, criterion: str = 'sharpe') -> int:
        """
        Find optimal horizon based on criterion.

        Args:
            criterion: 'sharpe', 'mae', 'directional_accuracy', 'coverage'

        Returns:
            Optimal horizon
        """
        aggregated = {h: self._aggregate_results(results) for h, results in self.results.items() if results}

        if not aggregated:
            raise ValueError("No validation results available")

        if criterion == 'sharpe':
            best_horizon = max(aggregated.keys(), key=lambda h: aggregated[h].sharpe_ratio)
        elif criterion == 'mae':
            best_horizon = min(aggregated.keys(), key=lambda h: aggregated[h].mae)
        elif criterion == 'directional_accuracy':
            best_horizon = max(aggregated.keys(), key=lambda h: aggregated[h].directional_accuracy)
        elif criterion == 'coverage':
            # Best horizon with coverage closest to 95%
            best_horizon = min(aggregated.keys(), key=lambda h: abs(aggregated[h].coverage_95 - 0.95))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return best_horizon

    def export_results(self, output_path: Path):
        """Export validation results to CSV."""
        rows = []
        for horizon, results in self.results.items():
            for i, result in enumerate(results):
                rows.append({
                    'horizon': horizon,
                    'expansion': i,
                    'mae': result.mae,
                    'rmse': result.rmse,
                    'mape': result.mape,
                    'directional_accuracy': result.directional_accuracy,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'coverage_95': result.coverage_95,
                    'interval_width': result.interval_width,
                    'n_samples': result.n_samples
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported validation results to {output_path}")


def validate_model_across_horizons(
    checkpoint_path: Path,
    symbol: str,
    timeframe: str,
    days_history: int = 90,
    horizons: Optional[List[int]] = None,
    device: str = "cpu"
) -> Dict[int, HorizonResult]:
    """
    Convenience function to validate model across multiple horizons.

    Args:
        checkpoint_path: Path to model checkpoint
        symbol: Trading pair
        timeframe: Timeframe
        days_history: Days of historical data
        horizons: List of horizons (default: [1, 4, 12, 24, 48])
        device: Device for inference

    Returns:
        Dictionary mapping horizon -> results
    """
    from ..train.loop import ForexDiffusionLit
    from ..training.train_sklearn import fetch_candles_from_db

    # Default horizons
    if horizons is None:
        horizons = [1, 4, 12, 24, 48]  # 1h, 4h, 12h, 24h, 48h

    # Load model
    model = ForexDiffusionLit.load_from_checkpoint(str(checkpoint_path))
    model = model.to(device)
    model.eval()

    # Load data
    data = fetch_candles_from_db(symbol, timeframe, days_history)

    # Create validator
    validator = MultiHorizonValidator(
        horizons=horizons,
        initial_train_bars=2000,
        test_bars=500,
        n_expansions=4
    )

    # Run validation
    results = validator.validate_model(model, data, symbol, timeframe, device)

    # Log summary
    logger.info("=" * 60)
    logger.info("Multi-Horizon Validation Results")
    logger.info("=" * 60)
    for horizon, result in results.items():
        logger.info(f"Horizon {horizon}h:")
        logger.info(f"  MAE: {result.mae:.6f}")
        logger.info(f"  Directional Accuracy: {result.directional_accuracy:.1f}%")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Coverage (95%): {result.coverage_95:.3f}")
    logger.info("=" * 60)

    # Find optimal
    optimal = validator.get_optimal_horizon(criterion='sharpe')
    logger.info(f"Optimal horizon (Sharpe): {optimal}h")

    return results
