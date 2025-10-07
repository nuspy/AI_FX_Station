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
                confidence_threshold
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
        """Recursive multi-step prediction."""
        predictions = []
        current_features = features.copy()

        for i in range(len(features)):
            pred = model.predict(current_features[i:i+1])[0]
            predictions.append(pred)

            # Update features with prediction (simplified)
            if i < len(features) - 1:
                current_features[i+1, 0] = pred  # Update return feature

        return np.array(predictions)

    def _direct_multi_prediction(self, model, features: np.ndarray) -> np.ndarray:
        """Direct multi-step prediction with multiple predictors."""
        # Simplified: just use direct prediction
        # Actual implementation would have separate predictors for different horizons
        return model.predict(features)

    def _apply_ensemble(
        self,
        predictions: np.ndarray,
        ensemble_method: str
    ) -> np.ndarray:
        """Apply ensemble method to predictions."""
        if ensemble_method == 'mean':
            # Simple average (for multiple model predictions)
            return predictions

        elif ensemble_method == 'weighted':
            # Performance-weighted ensemble
            # Simplified: just return predictions
            return predictions

        elif ensemble_method == 'stacking':
            # Meta-model stacking
            # Simplified: just return predictions
            return predictions

        return predictions

    def _apply_confidence_filter(
        self,
        predictions: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Filter predictions by confidence threshold."""
        # For models that provide uncertainty estimates
        # Simplified: assume confidence is inversely proportional to abs(prediction)

        # Set low-confidence predictions to zero (no trade)
        confidence = np.abs(predictions)
        filtered_predictions = predictions.copy()
        filtered_predictions[confidence < threshold] = 0

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
