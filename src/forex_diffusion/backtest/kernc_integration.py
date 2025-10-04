"""
Integration with kernc/backtesting.py library.

Provides seamless adapter between ForexGPT models and the backtesting.py library.
This is complementary to the existing custom BacktestEngine.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Backtesting.py imports
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    logger.warning("backtesting.py library not installed - install with: pip install backtesting")


def prepare_ohlcv_dataframe(
    symbol: str,
    timeframe: str,
    days_history: int,
    db_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Prepare OHLCV DataFrame in format expected by backtesting.py.

    Args:
        symbol: Trading pair (e.g., "EUR/USD")
        timeframe: Timeframe (e.g., "1h", "15m")
        days_history: Number of days of history
        db_path: Optional database path

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        and DatetimeIndex
    """
    from ..training.train_sklearn import fetch_candles_from_db

    # Fetch data from database
    candles = fetch_candles_from_db(symbol, timeframe, days_history)

    # Convert to backtesting.py format
    df = pd.DataFrame({
        'Open': candles['open'].values,
        'High': candles['high'].values,
        'Low': candles['low'].values,
        'Close': candles['close'].values,
        'Volume': candles.get('volume', pd.Series(0, index=candles.index)).values
    })

    # Convert timestamp to datetime index
    df.index = pd.to_datetime(candles['ts_utc'], unit='ms', utc=True)
    df.index.name = 'Date'

    logger.info(f"Prepared {len(df)} bars for backtesting: {symbol} {timeframe}")

    return df


class ForexDiffusionStrategy(Strategy):
    """
    Base strategy class for ForexGPT models with backtesting.py.

    Subclass this to create custom strategies using model predictions.
    """

    # Strategy parameters (can be optimized)
    entry_threshold = 0.001  # Minimum predicted move to enter (e.g., 0.1%)
    stop_loss_pct = 0.02     # Stop loss as percentage (e.g., 2%)
    take_profit_pct = 0.03   # Take profit as percentage (e.g., 3%)
    max_hold_bars = 48       # Maximum bars to hold position
    confidence_threshold = 0.6  # Minimum confidence to trade

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.model = None
        self.predictions = None
        self.hold_counter = 0

    def set_model(self, model_path: Path, device: str = "cpu"):
        """Load trained model for prediction."""
        from forex_diffusion.train.loop import ForexDiffusionLit

        # Load checkpoint
        self.model = ForexDiffusionLit.load_from_checkpoint(str(model_path))
        self.model = self.model.to(device)
        self.model.eval()

        logger.info(f"Loaded model from {model_path}")

    def set_predictions(self, predictions: pd.Series):
        """
        Set precomputed predictions for the strategy.

        Args:
            predictions: Series with same index as data, containing price predictions
        """
        self.predictions = predictions
        logger.info(f"Set {len(predictions)} predictions for backtesting")

    def init(self):
        """Initialize strategy (called once before backtesting)."""
        self.hold_counter = 0

    def next(self):
        """
        Execute strategy logic for each bar.

        This is called for every bar in the dataset during backtesting.
        """
        # Get current bar index
        current_idx = len(self.data.Close) - 1

        # Increment hold counter if in position
        if self.position:
            self.hold_counter += 1

            # Exit if max hold time reached
            if self.hold_counter >= self.max_hold_bars:
                self.position.close()
                self.hold_counter = 0
                return

        # Skip if we don't have predictions
        if self.predictions is None or current_idx >= len(self.predictions):
            return

        # Get prediction for next bar
        pred_return = self.predictions.iloc[current_idx]

        # Current price
        current_price = self.data.Close[-1]

        # Calculate expected price change
        expected_pct_change = pred_return / current_price if current_price > 0 else 0

        # Entry logic: long if expected move exceeds threshold
        if not self.position and abs(expected_pct_change) > self.entry_threshold:
            if expected_pct_change > 0:
                # Long entry
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)

                self.buy(
                    sl=stop_loss,
                    tp=take_profit
                )
                self.hold_counter = 0

            elif expected_pct_change < 0:
                # Short entry
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.take_profit_pct)

                self.sell(
                    sl=stop_loss,
                    tp=take_profit
                )
                self.hold_counter = 0


def run_backtest(
    strategy_class: type,
    data: pd.DataFrame,
    model_path: Optional[Path] = None,
    predictions: Optional[pd.Series] = None,
    cash: float = 10000,
    commission: float = 0.002,
    margin: float = 1.0,
    **strategy_params
) -> Dict[str, Any]:
    """
    Run backtest using backtesting.py library.

    Args:
        strategy_class: Strategy class (subclass of ForexDiffusionStrategy)
        data: OHLCV DataFrame with DatetimeIndex
        model_path: Optional path to model checkpoint
        predictions: Optional precomputed predictions (alternative to model_path)
        cash: Starting cash
        commission: Commission per trade (fraction, e.g., 0.002 = 0.2%)
        margin: Margin requirement (1.0 = no leverage, 0.01 = 100x leverage)
        **strategy_params: Additional strategy parameters to optimize

    Returns:
        Dictionary with backtest results
    """
    if not BACKTESTING_AVAILABLE:
        raise ImportError("backtesting.py library required - install with: pip install backtesting")

    # Create backtest instance
    bt = Backtest(
        data,
        strategy_class,
        cash=cash,
        commission=commission,
        margin=margin,
        exclusive_orders=True
    )

    # Run backtest
    if strategy_params:
        results = bt.run(**strategy_params)
    else:
        results = bt.run()

    # Set model or predictions on strategy instance
    if model_path is not None:
        bt._strategy.set_model(model_path)
    elif predictions is not None:
        bt._strategy.set_predictions(predictions)

    # Run again with model/predictions loaded
    results = bt.run(**strategy_params)

    # Convert results to dictionary
    results_dict = {
        'return': results.get('Return [%]', 0),
        'sharpe_ratio': results.get('Sharpe Ratio', 0),
        'max_drawdown': results.get('Max. Drawdown [%]', 0),
        'win_rate': results.get('Win Rate [%]', 0),
        'num_trades': results.get('# Trades', 0),
        'avg_trade': results.get('Avg. Trade [%]', 0),
        'max_trade_duration': results.get('Max. Trade Duration', 0),
        'avg_trade_duration': results.get('Avg. Trade Duration', 0),
        'profit_factor': results.get('Profit Factor', 0),
        'expectancy': results.get('Expectancy [%]', 0),
        'equity_final': results.get('Equity Final [$]', cash),
        'equity_peak': results.get('Equity Peak [$]', cash)
    }

    logger.info(f"Backtest complete: Return={results_dict['return']:.2f}%, Sharpe={results_dict['sharpe_ratio']:.2f}")

    return results_dict


def optimize_strategy(
    strategy_class: type,
    data: pd.DataFrame,
    optimize_params: Dict[str, Any],
    model_path: Optional[Path] = None,
    predictions: Optional[pd.Series] = None,
    maximize: str = 'Sharpe Ratio',
    constraint: Optional[callable] = None,
    method: str = 'grid',
    max_tries: int = 100,
    return_heatmap: bool = False
) -> Dict[str, Any]:
    """
    Optimize strategy parameters using backtesting.py optimization.

    Args:
        strategy_class: Strategy class to optimize
        data: OHLCV DataFrame
        optimize_params: Dict of parameter ranges, e.g., {'stop_loss_pct': [0.01, 0.02, 0.03]}
        model_path: Optional model checkpoint path
        predictions: Optional precomputed predictions
        maximize: Metric to maximize (e.g., 'Sharpe Ratio', 'Return [%]')
        constraint: Optional constraint function
        method: Optimization method ('grid' or 'skopt')
        max_tries: Maximum optimization iterations (for skopt)
        return_heatmap: Whether to return parameter heatmap

    Returns:
        Dictionary with best parameters and results
    """
    if not BACKTESTING_AVAILABLE:
        raise ImportError("backtesting.py library required")

    # Create backtest instance
    bt = Backtest(
        data,
        strategy_class,
        cash=10000,
        commission=0.002
    )

    # Set model/predictions if provided
    if model_path is not None:
        bt._strategy.set_model(model_path)
    elif predictions is not None:
        bt._strategy.set_predictions(predictions)

    # Run optimization
    logger.info(f"Starting {method} optimization with {len(optimize_params)} parameters")

    if method == 'grid':
        results = bt.optimize(
            **optimize_params,
            maximize=maximize,
            constraint=constraint,
            return_heatmap=return_heatmap
        )
    elif method == 'skopt':
        results = bt.optimize(
            **optimize_params,
            maximize=maximize,
            constraint=constraint,
            method='skopt',
            max_tries=max_tries,
            return_heatmap=return_heatmap
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    # Extract best parameters
    best_params = {k: v for k, v in results._strategy._params.items() if k in optimize_params}

    results_dict = {
        'best_params': best_params,
        'return': results.get('Return [%]', 0),
        'sharpe_ratio': results.get('Sharpe Ratio', 0),
        'max_drawdown': results.get('Max. Drawdown [%]', 0),
        'win_rate': results.get('Win Rate [%]', 0),
        'num_trades': results.get('# Trades', 0)
    }

    logger.info(f"Optimization complete: Best Sharpe={results_dict['sharpe_ratio']:.2f}")
    logger.info(f"Best parameters: {best_params}")

    return results_dict


def generate_predictions_from_model(
    model_path: Path,
    data: pd.DataFrame,
    horizon_bars: int,
    patch_len: int = 64,
    device: str = "cpu"
) -> pd.Series:
    """
    Generate predictions from trained model for backtesting.

    Args:
        model_path: Path to model checkpoint
        data: OHLCV DataFrame with DatetimeIndex
        horizon_bars: Prediction horizon in bars
        patch_len: Model patch length
        device: Device for inference

    Returns:
        Series of price predictions aligned with data index
    """
    from forex_diffusion.train.loop import ForexDiffusionLit
    from forex_diffusion.training.train import _add_time_features

    # Load model
    model = ForexDiffusionLit.load_from_checkpoint(str(model_path))
    model = model.to(device)
    model.eval()

    # Load metadata
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        mu = np.array(meta['mu'])
        sigma = np.array(meta['sigma'])
        channel_order = meta['channel_order']
    else:
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # Prepare data
    df = data.copy()
    df['ts_utc'] = df.index.astype(np.int64) // 10**6  # Convert to milliseconds
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df = _add_time_features(df)

    # Extract features
    values = df[channel_order].values

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for i in range(patch_len, len(df) - horizon_bars):
            # Extract patch
            patch = values[i - patch_len:i].T  # (C, L)

            # Normalize
            patch_norm = (patch - mu[:, None]) / sigma[:, None]

            # Convert to tensor
            x = torch.from_numpy(patch_norm.astype(np.float32)).unsqueeze(0).to(device)  # (1, C, L)

            # Forward pass (use VAE mean for deterministic prediction)
            mu_z, _ = model.vae.encode(x)
            x_rec = model.vae.decode(mu_z)

            # Extract predicted close (channel 3)
            pred_close = x_rec[0, 3, -1].item()

            # Denormalize
            pred_close = pred_close * sigma[3] + mu[3]

            predictions.append(pred_close)

    # Create series aligned with data
    pred_series = pd.Series(
        predictions,
        index=df.index[patch_len:len(predictions) + patch_len]
    )

    logger.info(f"Generated {len(predictions)} predictions from model")

    return pred_series
