"""
Inference backtesting for trained models (Internal Loop).

Generates inference configurations, runs backtests, calculates metrics,
and breaks down performance by market regime.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from itertools import product
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from .database import (
    InferenceBacktest, create_inference_backtest,
    update_inference_backtest_results
)
from .regime_manager import RegimeManager
from .config_loader import get_config

logger = logging.getLogger(__name__)


class InferenceBacktester:
    """
    Runs inference backtests on trained models with various configurations.

    The internal loop generates all inference parameter combinations and
    backtests each one to find the best performing inference method.
    """

    def __init__(
        self,
        session: Session,
        regime_manager: RegimeManager,
        max_parallel_workers: Optional[int] = None
    ):
        """
        Initialize InferenceBacktester.

        Args:
            session: SQLAlchemy session
            regime_manager: RegimeManager instance for regime classification
            max_parallel_workers: Maximum parallel backtest workers (default: from config)
        """
        self.config = get_config()
        self.session = session
        self.regime_manager = regime_manager
        self.max_parallel_workers = max_parallel_workers or self.config.max_inference_workers

    def generate_inference_grid(
        self,
        prediction_methods: Optional[List[str]] = None,
        ensemble_methods: Optional[List[str]] = None,
        confidence_thresholds: Optional[List[float]] = None,
        lookback_windows: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate Cartesian product of inference configurations.

        Args:
            prediction_methods: Methods for prediction (direct, recursive, direct_multi)
            ensemble_methods: Ensemble combination methods (mean, weighted, stacking)
            confidence_thresholds: Confidence thresholds (0.0 to 1.0)
            lookback_windows: Lookback window sizes in bars

        Returns:
            List of inference configuration dictionaries
        """
        # Defaults from config
        if prediction_methods is None:
            prediction_methods = self.config.prediction_methods

        if ensemble_methods is None:
            ensemble_methods = self.config.ensemble_methods

        if confidence_thresholds is None:
            confidence_thresholds = self.config.confidence_thresholds

        if lookback_windows is None:
            lookback_windows = self.config.lookback_windows

        # Generate all combinations
        configs = []

        for pred_method in prediction_methods:
            for conf_thresh in confidence_thresholds:
                for lookback in lookback_windows:
                    # For non-ensemble methods
                    config = {
                        'prediction_method': pred_method,
                        'ensemble_method': None,
                        'confidence_threshold': conf_thresh,
                        'lookback_window': lookback,
                        'inference_params': {}
                    }
                    configs.append(config)

                    # For ensemble methods
                    if pred_method in ['direct', 'recursive']:
                        for ens_method in ensemble_methods:
                            config_ens = {
                                'prediction_method': pred_method,
                                'ensemble_method': ens_method,
                                'confidence_threshold': conf_thresh,
                                'lookback_window': lookback,
                                'inference_params': {}
                            }
                            configs.append(config_ens)

        logger.info(f"Generated {len(configs)} inference configurations")

        return configs

    def backtest_single_inference(
        self,
        model,
        model_config: Dict[str, Any],
        inference_config: Dict[str, Any],
        ohlc_data: pd.DataFrame,
        training_run_id: int
    ) -> Tuple[int, Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Run backtest for a single inference configuration.

        Args:
            model: Trained model object
            model_config: Training configuration
            inference_config: Inference configuration
            ohlc_data: OHLC DataFrame for backtesting
            training_run_id: Training run ID

        Returns:
            Tuple of (backtest_id, overall_metrics, regime_metrics)
        """
        start_time = datetime.utcnow()

        try:
            # Create backtest record
            backtest = create_inference_backtest(
                self.session,
                training_run_id=training_run_id,
                prediction_method=inference_config['prediction_method'],
                ensemble_method=inference_config.get('ensemble_method'),
                confidence_threshold=inference_config.get('confidence_threshold'),
                lookback_window=inference_config.get('lookback_window'),
                inference_params=inference_config.get('inference_params')
            )
            self.session.commit()

            # Generate predictions using model
            predictions = self._generate_predictions(
                model,
                ohlc_data,
                inference_config
            )

            # Run backtest with predictions
            backtest_results = self._run_backtest(
                predictions,
                ohlc_data,
                inference_config
            )

            # Calculate overall metrics
            overall_metrics = self.calculate_metrics(backtest_results)

            # Calculate regime-specific metrics
            regime_metrics = self.regime_manager.calculate_regime_metrics(
                ohlc_data,
                backtest_results
            )

            # Calculate duration
            duration_seconds = (datetime.utcnow() - start_time).total_seconds()

            # Update backtest record with results
            update_inference_backtest_results(
                self.session,
                backtest_id=backtest.id,
                backtest_metrics=overall_metrics,
                regime_metrics=regime_metrics,
                backtest_duration_seconds=duration_seconds
            )
            self.session.commit()

            logger.info(
                f"Completed inference backtest {backtest.id}: "
                f"Sharpe={overall_metrics.get('sharpe_ratio', 0):.3f}"
            )

            return backtest.id, overall_metrics, regime_metrics

        except Exception as e:
            logger.error(f"Inference backtest failed: {e}")
            self.session.rollback()
            raise

    def _generate_predictions(
        self,
        model,
        ohlc_data: pd.DataFrame,
        inference_config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            model: Trained model
            ohlc_data: OHLC data
            inference_config: Inference configuration

        Returns:
            Array of predictions
        """
        prediction_method = inference_config['prediction_method']
        lookback_window = inference_config.get('lookback_window', 100)

        # Prepare features (simplified - actual implementation would use feature engineering)
        features = self._prepare_features(ohlc_data, lookback_window)

        # Generate predictions based on method
        if prediction_method == 'direct':
            predictions = model.predict(features)
        elif prediction_method == 'recursive':
            predictions = self._recursive_prediction(model, features)
        elif prediction_method == 'direct_multi':
            predictions = self._direct_multi_prediction(model, features)
        else:
            raise ValueError(f"Unknown prediction method: {prediction_method}")

        # Apply ensemble if specified
        ensemble_method = inference_config.get('ensemble_method')
        if ensemble_method:
            predictions = self._apply_ensemble(predictions, ensemble_method)

        # Apply confidence threshold
        confidence_threshold = inference_config.get('confidence_threshold', 0.0)
        if confidence_threshold > 0:
            predictions = self._apply_confidence_filter(
                predictions,
                confidence_threshold,
                features=features
            )

        return predictions

    def _prepare_features(
        self,
        ohlc_data: pd.DataFrame,
        lookback_window: int
    ) -> np.ndarray:
        """
        Prepare features for prediction.

        Note: This is a simplified version. Actual implementation would use
        the full feature engineering pipeline from training.
        """
        # Basic features: returns, volatility, momentum
        close_prices = ohlc_data['close'].values

        # Rolling returns
        returns = np.diff(close_prices) / close_prices[:-1]

        # Pad to match original length
        returns = np.concatenate([[0], returns])

        # Rolling volatility
        volatility = pd.Series(returns).rolling(window=20).std().fillna(0).values

        # Momentum (rate of change)
        momentum = pd.Series(close_prices).pct_change(periods=10).fillna(0).values

        # Stack features
        features = np.column_stack([returns, volatility, momentum])

        return features

    def _recursive_prediction(self, model, features: np.ndarray) -> np.ndarray:
        """
        Recursive multi-step prediction.

        Each prediction is fed back as input for the next step, allowing
        the model to predict multiple steps ahead recursively.
        """
        predictions = []
        current_features = features.copy()
        n_features = current_features.shape[1]

        for i in range(len(features)):
            # Predict next value
            pred = model.predict(current_features[i:i+1])[0]
            predictions.append(pred)

            # Update features with prediction for next iteration
            if i < len(features) - 1:
                # Shift features and append prediction
                # This creates a recursive feedback loop
                next_features = current_features[i+1].copy()

                # Update return feature with predicted return
                next_features[0] = pred

                # Recalculate derived features based on new return
                if n_features >= 2:
                    # Update volatility (exponential moving average)
                    prev_vol = current_features[i, 1]
                    next_features[1] = 0.9 * prev_vol + 0.1 * abs(pred)

                if n_features >= 3:
                    # Update momentum (rate of change)
                    next_features[2] = pred - current_features[max(0, i-10), 0]

                current_features[i+1] = next_features

        return np.array(predictions)

    def _direct_multi_prediction(self, model, features: np.ndarray) -> np.ndarray:
        """
        Direct multi-step prediction with horizon-specific adjustments.

        Instead of recursive feedback, this method adjusts features for
        multi-step-ahead prediction directly. Better for longer horizons
        where recursive errors don't compound.
        """
        predictions = []
        n_samples = len(features)

        # Generate predictions for each time step
        for i in range(n_samples):
            # Use current features
            current_feature = features[i:i+1]

            # Make prediction
            pred = model.predict(current_feature)[0]
            predictions.append(pred)

        # Apply smoothing for multi-step predictions
        # This reduces noise in longer-horizon predictions
        predictions = np.array(predictions)

        # Exponential moving average smoothing
        alpha = 0.3  # Smoothing factor
        smoothed_predictions = np.zeros_like(predictions)
        smoothed_predictions[0] = predictions[0]

        for i in range(1, len(predictions)):
            smoothed_predictions[i] = (
                alpha * predictions[i] +
                (1 - alpha) * smoothed_predictions[i-1]
            )

        return smoothed_predictions

    def _apply_ensemble(
        self,
        predictions: np.ndarray,
        ensemble_method: str,
        prediction_history: Optional[List[np.ndarray]] = None,
        performance_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply ensemble method to combine multiple predictions.

        Args:
            predictions: Current predictions
            ensemble_method: Method to use ('mean', 'weighted', 'stacking')
            prediction_history: Historical predictions from multiple methods
            performance_weights: Weights based on historical performance

        Returns:
            Ensembled predictions
        """
        # If no history provided, return predictions as-is
        if prediction_history is None or len(prediction_history) == 0:
            return predictions

        if ensemble_method == 'mean':
            # Simple average of all predictions
            all_predictions = np.array([predictions] + prediction_history)
            return np.mean(all_predictions, axis=0)

        elif ensemble_method == 'weighted':
            # Performance-weighted ensemble
            if performance_weights is None:
                # Default to equal weights
                performance_weights = np.ones(len(prediction_history) + 1)

            # Normalize weights
            weights = performance_weights / np.sum(performance_weights)

            # Weighted average
            all_predictions = np.array([predictions] + prediction_history)
            weighted_sum = np.zeros_like(predictions)

            for i, pred in enumerate(all_predictions):
                weighted_sum += weights[i] * pred

            return weighted_sum

        elif ensemble_method == 'stacking':
            # Meta-model stacking with adaptive weighting
            # Uses recent performance to dynamically adjust weights

            all_predictions = np.array([predictions] + prediction_history)
            n_models = len(all_predictions)

            # Calculate adaptive weights based on prediction variance
            # Lower variance predictions get higher weight
            variances = np.var(all_predictions, axis=1)
            inverse_var = 1.0 / (variances + 1e-8)  # Avoid division by zero
            stacking_weights = inverse_var / np.sum(inverse_var)

            # Weighted combination
            stacked_pred = np.zeros_like(predictions)
            for i in range(n_models):
                stacked_pred += stacking_weights[i] * all_predictions[i]

            return stacked_pred

        return predictions

    def _apply_confidence_filter(
        self,
        predictions: np.ndarray,
        threshold: float,
        features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Filter predictions by confidence threshold.

        Uses multiple confidence indicators:
        1. Prediction magnitude (stronger signals = higher confidence)
        2. Feature volatility (lower volatility = higher confidence)
        3. Prediction consistency (similar to recent predictions = higher confidence)

        Args:
            predictions: Array of predictions
            threshold: Confidence threshold (0.0 to 1.0)
            features: Optional feature array for additional confidence metrics

        Returns:
            Filtered predictions (low-confidence set to 0)
        """
        if threshold == 0.0:
            # No filtering
            return predictions

        filtered_predictions = predictions.copy()
        n = len(predictions)

        for i in range(n):
            # Confidence metric 1: Prediction magnitude
            # Stronger signals are more confident
            magnitude_confidence = min(abs(predictions[i]) * 2, 1.0)

            # Confidence metric 2: Prediction consistency
            # Check if prediction aligns with recent trend
            if i >= 5:
                recent_preds = predictions[max(0, i-5):i]
                if len(recent_preds) > 0:
                    # Calculate agreement with recent predictions
                    same_direction = np.sum(np.sign(recent_preds) == np.sign(predictions[i]))
                    consistency_confidence = same_direction / len(recent_preds)
                else:
                    consistency_confidence = 0.5
            else:
                consistency_confidence = 0.5

            # Confidence metric 3: Feature-based confidence
            feature_confidence = 0.5
            if features is not None and i < len(features):
                # Lower feature volatility = higher confidence
                if i >= 20:
                    recent_volatility = np.std(features[max(0, i-20):i, 0])
                    # Normalize to 0-1 range (assume volatility rarely exceeds 0.1)
                    feature_confidence = 1.0 - min(recent_volatility * 10, 1.0)

            # Combined confidence score (weighted average)
            combined_confidence = (
                0.4 * magnitude_confidence +
                0.4 * consistency_confidence +
                0.2 * feature_confidence
            )

            # Filter out low-confidence predictions
            if combined_confidence < threshold:
                filtered_predictions[i] = 0.0

        return filtered_predictions

    def _run_backtest(
        self,
        predictions: np.ndarray,
        ohlc_data: pd.DataFrame,
        inference_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run backtest simulation with predictions.

        Args:
            predictions: Model predictions
            ohlc_data: OHLC data
            inference_config: Inference configuration

        Returns:
            Dictionary with backtest results including trades
        """
        # Simple backtest logic
        trades = []
        positions = []
        equity_curve = [100000.0]  # Starting capital
        current_position = 0.0

        close_prices = ohlc_data['close'].values

        for i in range(1, len(predictions)):
            pred = predictions[i]

            # Trading logic: enter position based on prediction
            target_position = np.sign(pred) if abs(pred) > 0.001 else 0.0

            if target_position != current_position:
                # Close existing position
                if current_position != 0:
                    exit_price = close_prices[i]
                    pnl = current_position * (exit_price - entry_price)
                    equity_curve.append(equity_curve[-1] + pnl)

                    trades.append({
                        'entry_index': entry_idx,
                        'exit_index': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'long' if current_position > 0 else 'short',
                        'pnl': pnl,
                        'return_pct': pnl / entry_price * 100
                    })

                # Open new position
                if target_position != 0:
                    entry_idx = i
                    entry_price = close_prices[i]
                    current_position = target_position

            positions.append(current_position)

        return {
            'trades': trades,
            'positions': positions,
            'equity_curve': equity_curve,
            'predictions': predictions
        }

    def calculate_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.

        Args:
            backtest_results: Backtest results dictionary

        Returns:
            Dictionary of performance metrics
        """
        trades = backtest_results.get('trades', [])
        equity_curve = backtest_results.get('equity_curve', [100000.0])

        if not trades:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'total_pnl': 0.0,
                'avg_trade_pnl': 0.0
            }

        # Extract trade PNLs
        pnls = [t['pnl'] for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        # Calculate metrics
        total_pnl = sum(pnls)
        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (from trade returns)
        if len(pnls) > 1:
            sharpe_ratio = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / (running_max + 1e-8)
        max_drawdown = np.max(drawdown)

        # Sortino ratio (downside deviation)
        negative_returns = [p for p in pnls if p < 0]
        downside_std = np.std(negative_returns) if negative_returns else 0.0
        sortino_ratio = np.mean(pnls) / (downside_std + 1e-8) * np.sqrt(252) if downside_std > 0 else 0.0

        # Calmar ratio (return / max drawdown)
        calmar_ratio = (total_pnl / 100000.0) / (max_drawdown + 1e-8) if max_drawdown > 0 else 0.0

        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_pnl': float(total_pnl),
            'avg_trade_pnl': float(np.mean(pnls)) if pnls else 0.0,
            'avg_win': float(np.mean(winning_trades)) if winning_trades else 0.0,
            'avg_loss': float(np.mean(losing_trades)) if losing_trades else 0.0
        }

    def backtest_all_inference_configs(
        self,
        model,
        model_config: Dict[str, Any],
        ohlc_data: pd.DataFrame,
        training_run_id: int,
        inference_grid: Optional[List[Dict[str, Any]]] = None
    ) -> List[Tuple[int, Dict[str, Any], Dict[str, Dict[str, Any]]]]:
        """
        Run backtests for all inference configurations.

        Args:
            model: Trained model
            model_config: Training configuration
            ohlc_data: OHLC data
            training_run_id: Training run ID
            inference_grid: Optional custom inference grid

        Returns:
            List of (backtest_id, overall_metrics, regime_metrics) tuples
        """
        if inference_grid is None:
            inference_grid = self.generate_inference_grid()

        results = []

        logger.info(f"Running {len(inference_grid)} inference backtests...")

        for i, inference_config in enumerate(inference_grid):
            try:
                result = self.backtest_single_inference(
                    model=model,
                    model_config=model_config,
                    inference_config=inference_config,
                    ohlc_data=ohlc_data,
                    training_run_id=training_run_id
                )
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(inference_grid)} inference backtests")

            except Exception as e:
                logger.error(f"Failed inference backtest {i + 1}: {e}")
                continue

        logger.info(f"Completed all {len(results)} inference backtests")

        return results

    def find_best_inference_config(
        self,
        results: List[Tuple[int, Dict[str, Any], Dict[str, Dict[str, Any]]]],
        primary_metric: str = 'sharpe_ratio'
    ) -> Tuple[int, Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Find best performing inference configuration.

        Args:
            results: List of backtest results
            primary_metric: Metric to optimize

        Returns:
            Best result tuple (backtest_id, overall_metrics, regime_metrics)
        """
        if not results:
            raise ValueError("No results to evaluate")

        # Find result with best primary metric
        best_result = max(
            results,
            key=lambda r: r[1].get(primary_metric, float('-inf'))
        )

        logger.info(
            f"Best inference config: backtest_id={best_result[0]}, "
            f"{primary_metric}={best_result[1].get(primary_metric, 0):.4f}"
        )

        return best_result
