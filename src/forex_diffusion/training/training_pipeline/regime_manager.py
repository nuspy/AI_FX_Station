"""
Regime classification and management for market regimes.

Handles regime detection from OHLC data, regime performance evaluation,
and tracking of best models per regime.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from sqlalchemy.orm import Session

from .database import (
    RegimeDefinition, get_all_active_regimes, get_best_model_for_regime,
    update_best_model_for_regime
)
from .config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class RegimeClassification:
    """Result of regime classification for a time period."""
    regime_name: str
    confidence: float
    start_index: int
    end_index: int
    metrics: Dict[str, float]


class RegimeManager:
    """
    Manages market regime detection and best model tracking.

    Classifies market data into regimes and tracks which models perform
    best in each regime.
    """

    def __init__(
        self,
        session: Session,
        trend_window: Optional[int] = None,
        volatility_window: Optional[int] = None,
        returns_window: Optional[int] = None,
        min_regime_duration: Optional[int] = None
    ):
        """
        Initialize RegimeManager.

        Args:
            session: SQLAlchemy session
            trend_window: Number of bars for trend calculation (default: from config)
            volatility_window: Number of bars for volatility calculation (default: from config)
            returns_window: Number of bars for returns calculation (default: from config)
            min_regime_duration: Minimum bars to classify as a regime (default: from config)
        """
        self.config = get_config()
        self.session = session
        self.trend_window = trend_window or self.config.trend_window
        self.volatility_window = volatility_window or self.config.volatility_window
        self.returns_window = returns_window or self.config.returns_window
        self.min_regime_duration = min_regime_duration or self.config.min_regime_duration

        # Load regime definitions
        self.regime_definitions = self._load_regime_definitions()

    def _load_regime_definitions(self) -> Dict[str, RegimeDefinition]:
        """Load active regime definitions from database."""
        regimes = get_all_active_regimes(self.session)
        return {r.regime_name: r for r in regimes}

    def calculate_market_features(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market features for regime classification.

        Args:
            ohlc_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with additional regime classification features
        """
        df = ohlc_data.copy()

        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Rolling returns
        df['returns_mean'] = df['returns'].rolling(window=self.returns_window).mean()
        df['returns_std'] = df['returns'].rolling(window=self.returns_window).std()

        # Volatility (rolling standard deviation of returns)
        df['volatility'] = df['log_returns'].rolling(window=self.volatility_window).std()
        df['volatility_percentile'] = df['volatility'].rank(pct=True) * 100

        # Trend strength (using linear regression slope)
        def calculate_trend_strength(close_prices):
            """Calculate trend strength using linear regression."""
            if len(close_prices) < 2:
                return 0.0
            x = np.arange(len(close_prices))
            y = close_prices.values
            # Simple linear regression
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2 + 1e-8)
            # Normalize by price level
            trend_strength = abs(slope) / (np.mean(y) + 1e-8)
            return min(trend_strength * 100, 1.0)  # Cap at 1.0

        df['trend_strength'] = df['close'].rolling(window=self.trend_window).apply(
            calculate_trend_strength, raw=False
        )

        # Price momentum (rate of change)
        df['momentum'] = df['close'].pct_change(periods=self.trend_window)

        # Average True Range (ATR) for volatility measure
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()

        # Volume features (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        else:
            df['volume_ratio'] = 1.0

        return df

    def classify_regime(self, ohlc_data: pd.DataFrame) -> str:
        """
        Classify the current market regime based on OHLC data.

        Uses the most recent complete data to determine regime.

        Args:
            ohlc_data: DataFrame with OHLC data

        Returns:
            Regime name (e.g., 'bull_trending', 'bear_trending', etc.)
        """
        # Calculate features
        df = self.calculate_market_features(ohlc_data)

        # Get most recent complete row (ignore last row if incomplete)
        last_row = df.iloc[-1]

        # Extract key metrics
        trend_strength = last_row.get('trend_strength', 0.0)
        returns_mean = last_row.get('returns_mean', 0.0)
        volatility_percentile = last_row.get('volatility_percentile', 50.0)

        # Classification logic based on default regimes
        # Priority order: trending regimes first, then ranging

        # Bull trending: strong trend + positive returns
        if trend_strength > 0.7 and returns_mean > 0:
            return 'bull_trending'

        # Bear trending: strong trend + negative returns
        if trend_strength > 0.7 and returns_mean < 0:
            return 'bear_trending'

        # Volatile ranging: weak trend + high volatility
        if trend_strength < 0.3 and volatility_percentile > 75:
            return 'volatile_ranging'

        # Calm ranging: weak trend + low volatility
        if trend_strength < 0.3 and volatility_percentile < 50:
            return 'calm_ranging'

        # Default to calm_ranging if no clear match
        return 'calm_ranging'

    def classify_regime_history(
        self,
        ohlc_data: pd.DataFrame,
        window_size: Optional[int] = None
    ) -> List[RegimeClassification]:
        """
        Classify regimes throughout historical data.

        Args:
            ohlc_data: DataFrame with OHLC data
            window_size: Size of rolling window for classification (default: trend_window)

        Returns:
            List of RegimeClassification objects
        """
        if window_size is None:
            window_size = self.trend_window

        # Calculate features for entire history
        df = self.calculate_market_features(ohlc_data)

        # Classify each window
        classifications = []
        current_regime = None
        regime_start = 0

        for i in range(len(df)):
            if i < window_size:
                continue  # Skip until we have enough data

            # Get window of data
            window_df = df.iloc[max(0, i - window_size):i + 1]

            # Classify this window
            regime = self.classify_regime(window_df)

            # Track regime changes
            if regime != current_regime:
                # Save previous regime if it met minimum duration
                if current_regime is not None and (i - regime_start) >= self.min_regime_duration:
                    classifications.append(RegimeClassification(
                        regime_name=current_regime,
                        confidence=1.0,  # Could calculate confidence metric
                        start_index=regime_start,
                        end_index=i - 1,
                        metrics=self._extract_regime_metrics(df.iloc[regime_start:i])
                    ))

                # Start new regime
                current_regime = regime
                regime_start = i

        # Add final regime
        if current_regime is not None and (len(df) - regime_start) >= self.min_regime_duration:
            classifications.append(RegimeClassification(
                regime_name=current_regime,
                confidence=1.0,
                start_index=regime_start,
                end_index=len(df) - 1,
                metrics=self._extract_regime_metrics(df.iloc[regime_start:])
            ))

        logger.info(f"Classified {len(classifications)} regime periods in data")

        return classifications

    def _extract_regime_metrics(self, regime_df: pd.DataFrame) -> Dict[str, float]:
        """Extract summary metrics for a regime period."""
        return {
            'duration': len(regime_df),
            'mean_return': regime_df['returns'].mean() if 'returns' in regime_df else 0.0,
            'volatility': regime_df['volatility'].mean() if 'volatility' in regime_df else 0.0,
            'trend_strength': regime_df['trend_strength'].mean() if 'trend_strength' in regime_df else 0.0,
            'max_drawdown': self._calculate_max_drawdown(regime_df['close']) if 'close' in regime_df else 0.0
        }

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def calculate_regime_metrics(
        self,
        ohlc_data: pd.DataFrame,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics broken down by regime.

        Args:
            ohlc_data: DataFrame with OHLC data (must align with backtest)
            backtest_results: Backtest results with trades/positions

        Returns:
            Dictionary mapping regime names to their performance metrics
        """
        # Classify regimes in historical data
        regime_classifications = self.classify_regime_history(ohlc_data)

        # Initialize metrics per regime
        regime_metrics = {regime: {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'duration_bars': 0
        } for regime in self.regime_definitions.keys()}

        # Extract trades from backtest results
        trades = backtest_results.get('trades', [])

        if not trades:
            logger.warning("No trades found in backtest results")
            return regime_metrics

        # Assign each trade to a regime based on entry time
        for trade in trades:
            entry_index = trade.get('entry_index', 0)

            # Find which regime this trade occurred in
            regime_name = None
            for classification in regime_classifications:
                if classification.start_index <= entry_index <= classification.end_index:
                    regime_name = classification.regime_name
                    break

            if regime_name is None:
                continue  # Skip trades outside classified regimes

            # Update metrics
            metrics = regime_metrics[regime_name]
            metrics['total_trades'] += 1
            metrics['duration_bars'] += classification.end_index - classification.start_index

            pnl = trade.get('pnl', 0.0)
            metrics['total_pnl'] += pnl

            if pnl > 0:
                metrics['winning_trades'] += 1
            elif pnl < 0:
                metrics['losing_trades'] += 1

        # Calculate derived metrics for each regime
        for regime_name, metrics in regime_metrics.items():
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']

                # Calculate average win/loss
                winning_pnls = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
                losing_pnls = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]

                metrics['avg_win'] = np.mean(winning_pnls) if winning_pnls else 0.0
                metrics['avg_loss'] = np.mean(losing_pnls) if losing_pnls else 0.0

                # Profit factor
                total_wins = sum(winning_pnls) if winning_pnls else 0.0
                total_losses = abs(sum(losing_pnls)) if losing_pnls else 0.0
                metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')

                # Calculate Sharpe ratio from trade returns
                if len(trades) > 1:
                    trade_returns = [t.get('pnl', 0.0) for t in trades]
                    metrics['sharpe_ratio'] = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) * np.sqrt(252)

        return regime_metrics

    def evaluate_regime_improvements(
        self,
        inference_backtest_id: int,
        regime_metrics: Dict[str, Dict[str, Any]],
        primary_metric: str = 'sharpe_ratio'
    ) -> Dict[str, bool]:
        """
        Evaluate if this model improves performance for any regimes.

        Args:
            inference_backtest_id: ID of the inference backtest
            regime_metrics: Performance metrics by regime
            primary_metric: Metric to use for comparison (default: sharpe_ratio)

        Returns:
            Dictionary mapping regime names to boolean (True if improvement)
        """
        improvements = {}

        for regime_name, metrics in regime_metrics.items():
            current_score = metrics.get(primary_metric, 0.0)

            # Get current best model for this regime
            best_model = get_best_model_for_regime(self.session, regime_name)

            if best_model is None:
                # No existing best model, this is an improvement
                improvements[regime_name] = True
                logger.info(f"No existing best model for {regime_name}, marking as improvement")
            else:
                # Compare against current best
                best_score = best_model.performance_score

                # Check if improvement (with minimum threshold)
                improvement_threshold = 0.01  # 1% minimum improvement
                if current_score > best_score * (1 + improvement_threshold):
                    improvements[regime_name] = True
                    logger.info(
                        f"New best for {regime_name}: {current_score:.4f} > {best_score:.4f} "
                        f"(+{(current_score/best_score - 1)*100:.2f}%)"
                    )
                else:
                    improvements[regime_name] = False

        return improvements

    def update_regime_bests(
        self,
        training_run_id: int,
        inference_backtest_id: int,
        regime_metrics: Dict[str, Dict[str, Any]],
        primary_metric: str = 'sharpe_ratio'
    ) -> List[str]:
        """
        Update best models for regimes where this model improved performance.

        Args:
            training_run_id: ID of the training run
            inference_backtest_id: ID of the inference backtest
            regime_metrics: Performance metrics by regime
            primary_metric: Metric to use for comparison

        Returns:
            List of regime names that were updated
        """
        # Evaluate improvements
        improvements = self.evaluate_regime_improvements(
            inference_backtest_id,
            regime_metrics,
            primary_metric
        )

        updated_regimes = []

        # Update database for improved regimes
        for regime_name, is_improvement in improvements.items():
            if is_improvement:
                performance_score = regime_metrics[regime_name].get(primary_metric, 0.0)
                secondary_metrics = {
                    k: v for k, v in regime_metrics[regime_name].items()
                    if k != primary_metric
                }

                update_best_model_for_regime(
                    self.session,
                    regime_name=regime_name,
                    training_run_id=training_run_id,
                    inference_backtest_id=inference_backtest_id,
                    performance_score=performance_score,
                    secondary_metrics=secondary_metrics
                )

                updated_regimes.append(regime_name)

        if updated_regimes:
            self.session.commit()
            logger.info(f"Updated best models for regimes: {updated_regimes}")
        else:
            logger.info("No regime improvements found")

        return updated_regimes

    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get summary of all regimes and their best models.

        Returns:
            Dictionary with regime summary information
        """
        summary = {
            'total_regimes': len(self.regime_definitions),
            'regimes': {}
        }

        for regime_name, regime_def in self.regime_definitions.items():
            best_model = get_best_model_for_regime(self.session, regime_name)

            regime_info = {
                'description': regime_def.description,
                'is_active': regime_def.is_active,
                'has_best_model': best_model is not None
            }

            if best_model:
                regime_info['best_model'] = {
                    'training_run_id': best_model.training_run_id,
                    'performance_score': best_model.performance_score,
                    'achieved_at': best_model.achieved_at.isoformat() if best_model.achieved_at else None,
                    'secondary_metrics': best_model.secondary_metrics
                }

            summary['regimes'][regime_name] = regime_info

        return summary
