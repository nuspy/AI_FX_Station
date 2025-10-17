#!/usr/bin/env python3
"""
LDM4TS Backtest Example

Demonstrates how to run a backtest with LDM4TS predictions.

Usage:
    python examples/backtest_ldm4ts_example.py

Requirements:
    - Trained LDM4TS model checkpoint
    - Historical OHLCV data
    - Features DataFrame
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forex_diffusion.backtest import LDM4TSBacktester, LDM4TSBacktestConfig
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def load_sample_data():
    """
    Load or generate sample OHLCV data for backtesting.
    
    In production, replace with actual historical data loading.
    """
    logger.info("Loading sample data...")
    
    # Generate synthetic data (replace with real data loader)
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='1min')
    n = len(dates)
    
    # Random walk with drift
    np.random.seed(42)
    returns = np.random.randn(n) * 0.0001 + 0.00001
    close = 1.0500 + np.cumsum(returns)
    
    # OHLC from close
    open_prices = close + np.random.randn(n) * 0.00005
    high = np.maximum(open_prices, close) + abs(np.random.randn(n)) * 0.0001
    low = np.minimum(open_prices, close) - abs(np.random.randn(n)) * 0.0001
    volume = np.random.rand(n) * 1000000
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    logger.info(f"Generated {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    
    return data


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for ML ensemble (if used).
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        Features DataFrame
    """
    logger.info("Creating features...")
    
    features = pd.DataFrame(index=data.index)
    
    # Simple technical indicators
    features['returns'] = data['close'].pct_change()
    features['sma_20'] = data['close'].rolling(20).mean()
    features['sma_50'] = data['close'].rolling(50).mean()
    features['rsi'] = 50  # Placeholder
    features['atr'] = data['high'] - data['low']
    
    # Fill NaN
    features = features.fillna(method='bfill').fillna(0)
    
    logger.info(f"Created {len(features.columns)} features")
    
    return features


def create_labels(data: pd.DataFrame, horizon: int = 60) -> pd.Series:
    """
    Create labels for supervised learning.
    
    Args:
        data: OHLCV DataFrame
        horizon: Future bars to predict
        
    Returns:
        Labels Series (1=up, -1=down, 0=neutral)
    """
    logger.info(f"Creating labels (horizon={horizon})...")
    
    future_returns = data['close'].shift(-horizon) / data['close'] - 1
    
    labels = pd.Series(0, index=data.index)
    labels[future_returns > 0.001] = 1  # +0.1% = buy
    labels[future_returns < -0.001] = -1  # -0.1% = sell
    
    logger.info(f"Created labels: {(labels==1).sum()} buys, {(labels==-1).sum()} sells, {(labels==0).sum()} neutral")
    
    return labels


def run_backtest():
    """Run LDM4TS backtest example."""
    
    logger.info("=" * 80)
    logger.info("LDM4TS BACKTEST EXAMPLE")
    logger.info("=" * 80)
    
    # 1. Load data
    data = load_sample_data()
    features = create_features(data)
    labels = create_labels(data, horizon=60)
    
    # 2. Configure backtest
    config = LDM4TSBacktestConfig(
        # Basic settings
        symbol='EUR/USD',
        start_date=data.index[0],
        end_date=data.index[-1],
        initial_capital=10000.0,
        
        # Walk-forward settings
        train_size_days=30,
        test_size_days=7,
        step_size_days=7,
        
        # Risk management
        max_positions=2,
        base_risk_per_trade_pct=1.0,
        daily_loss_limit_pct=3.0,
        
        # Model settings
        use_multi_timeframe=False,  # Disable for this example
        use_stacked_ensemble=False,  # Disable for this example
        use_regime_detection=False,  # Disable for this example
        use_smart_execution=False,  # Disable for this example
        use_multi_level_stops=False,  # Disable for this example
        
        # LDM4TS settings (NEW)
        use_ldm4ts=True,
        ldm4ts_checkpoint_path=None,  # Set to actual checkpoint path
        ldm4ts_horizons=[15, 60, 240],
        ldm4ts_uncertainty_threshold=0.50,
        ldm4ts_min_strength=0.30,
        ldm4ts_position_scaling=True,
        ldm4ts_num_samples=50,
        ldm4ts_window_size=100,
        
        # Signal settings
        min_signal_confidence=0.60
    )
    
    logger.info("\nBacktest Configuration:")
    logger.info(f"  Symbol: {config.symbol}")
    logger.info(f"  Period: {config.start_date} to {config.end_date}")
    logger.info(f"  Initial capital: ${config.initial_capital:,.2f}")
    logger.info(f"  Use LDM4TS: {config.use_ldm4ts}")
    if config.use_ldm4ts:
        logger.info(f"  LDM4TS horizons: {config.ldm4ts_horizons}")
        logger.info(f"  Uncertainty threshold: {config.ldm4ts_uncertainty_threshold:.2f}")
        logger.info(f"  Min strength: {config.ldm4ts_min_strength:.2f}")
    
    # 3. Initialize backtester
    backtester = LDM4TSBacktester(config)
    
    # 4. Run backtest
    logger.info("\nStarting backtest...\n")
    
    try:
        result = backtester.run(
            data=data,
            features=features,
            labels=labels,
            verbose=True
        )
        
        # 5. Display results
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        logger.info(f"\nTotal trades: {result.total_trades}")
        logger.info(f"Win rate: {result.win_rate:.2%}")
        logger.info(f"Net P&L: ${result.net_pnl:,.2f}")
        logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {result.max_drawdown:.2%}")
        logger.info(f"Final capital: ${result.final_capital:,.2f}")
        
        if hasattr(result, 'metadata') and result.metadata and 'ldm4ts' in result.metadata:
            ldm_metrics = result.metadata['ldm4ts']
            logger.info("\nLDM4TS Metrics:")
            logger.info(f"  Total predictions: {ldm_metrics['total_predictions']}")
            logger.info(f"  Total signals: {ldm_metrics['total_signals']}")
            logger.info(f"  Avg uncertainty: {ldm_metrics['avg_uncertainty']:.3f}%")
            logger.info(f"  Bull signals: {ldm_metrics['signals_by_direction']['bull']}")
            logger.info(f"  Bear signals: {ldm_metrics['signals_by_direction']['bear']}")
        
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    result = run_backtest()
    
    if result:
        logger.info("\n✅ Example completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Train an LDM4TS model on historical data")
        logger.info("2. Update ldm4ts_checkpoint_path in config")
        logger.info("3. Run backtest with real predictions")
        logger.info("4. Compare vs baseline (ML ensemble only)")
    else:
        logger.error("\n❌ Example failed. Check logs above.")
