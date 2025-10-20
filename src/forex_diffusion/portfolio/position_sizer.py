"""
Adaptive Position Sizing integrating Portfolio Optimization with Trading Signals.

Uses Riskfolio-Lib optimization to dynamically adjust position sizes based on
diffusion model predictions and current market conditions.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .optimizer import PortfolioOptimizer


class AdaptivePositionSizer:
    """
    Adaptive position sizing strategy using portfolio optimization.

    Integrates diffusion model predictions with quantitative portfolio
    optimization to determine optimal position sizes for each asset.
    """

    def __init__(
        self,
        optimizer: Optional[PortfolioOptimizer] = None,
        lookback_period: int = 60,  # days
        rebalance_frequency: int = 5,  # days
        max_position_size: float = 0.25,  # 25% max per position
        min_position_size: float = 0.01,  # 1% min per position
    ):
        """
        Initialize adaptive position sizer.

        Args:
            optimizer: PortfolioOptimizer instance (creates default if None)
            lookback_period: Historical period for optimization (days)
            rebalance_frequency: How often to reoptimize (days)
            max_position_size: Maximum weight per asset (0-1)
            min_position_size: Minimum weight per asset (0-1)
        """
        self.optimizer = optimizer or PortfolioOptimizer(
            risk_measure="CVaR",
            objective="Sharpe",
            risk_aversion=1.0,
        )

        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size

        # State tracking
        self.last_rebalance_date: Optional[pd.Timestamp] = None
        self.current_weights: Optional[pd.Series] = None

        logger.info(
            f"AdaptivePositionSizer initialized: "
            f"lookback={lookback_period}d, rebalance={rebalance_frequency}d"
        )

    def calculate_positions(
        self,
        predictions: pd.DataFrame,
        historical_returns: pd.DataFrame,
        current_date: pd.Timestamp,
        total_capital: float,
        force_rebalance: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate optimal position sizes for each asset.

        Args:
            predictions: DataFrame with model predictions (returns forecast)
                         columns = assets, index = timestamps
            historical_returns: Historical returns for risk estimation
            current_date: Current timestamp
            total_capital: Total available capital
            force_rebalance: Force reoptimization regardless of schedule

        Returns:
            Dict mapping asset -> position size in base currency
        """
        # Check if rebalancing is needed
        needs_rebalance = self._needs_rebalance(current_date) or force_rebalance

        if not needs_rebalance and self.current_weights is not None:
            logger.debug("Using existing weights (no rebalance needed)")
            return self._weights_to_positions(self.current_weights, total_capital)

        # Prepare data for optimization
        combined_returns = self._prepare_returns_data(
            predictions, historical_returns, current_date
        )

        if combined_returns.empty:
            logger.warning("Insufficient data for optimization")
            return {}

        # Optimize portfolio
        try:
            constraints = {
                "max_weight": self.max_position_size,
                "min_weight": self.min_position_size,
            }

            optimal_weights = self.optimizer.optimize(
                returns=combined_returns,
                constraints=constraints,
                method="Classic",
            )

            # Update state
            self.current_weights = optimal_weights
            self.last_rebalance_date = current_date

            # Convert weights to positions
            positions = self._weights_to_positions(optimal_weights, total_capital)

            logger.info(
                f"Rebalanced portfolio: {len(positions)} positions, "
                f"total allocation={sum(positions.values()) / total_capital:.1%}"
            )

            return positions

        except Exception as e:
            logger.error(f"Position calculation failed: {e}")
            return {}

    def calculate_risk_parity_positions(
        self,
        historical_returns: pd.DataFrame,
        total_capital: float,
        risk_budgets: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate positions using Risk Parity approach.

        Args:
            historical_returns: Historical returns DataFrame
            total_capital: Total available capital
            risk_budgets: Optional risk budget allocation

        Returns:
            Dict mapping asset -> position size
        """
        try:
            weights = self.optimizer.optimize_risk_parity(
                returns=historical_returns,
                risk_budgets=risk_budgets,
            )

            self.current_weights = weights
            positions = self._weights_to_positions(weights, total_capital)

            logger.info(f"Risk Parity positions calculated: {len(positions)} assets")
            return positions

        except Exception as e:
            logger.error(f"Risk Parity position calculation failed: {e}")
            return {}

    def adjust_for_volatility(
        self,
        base_positions: Dict[str, float],
        volatility_estimates: Dict[str, float],
        target_volatility: float = 0.15,  # 15% annualized
    ) -> Dict[str, float]:
        """
        Adjust position sizes based on volatility targeting.

        Args:
            base_positions: Base position sizes (from optimization)
            volatility_estimates: Estimated volatility per asset
            target_volatility: Target portfolio volatility

        Returns:
            Volatility-adjusted position sizes
        """
        adjusted_positions = {}

        for asset, position in base_positions.items():
            if asset in volatility_estimates:
                vol = volatility_estimates[asset]
                if vol > 0:
                    # Scale position inversely to volatility
                    vol_scalar = target_volatility / vol
                    adjusted_positions[asset] = position * vol_scalar
                else:
                    adjusted_positions[asset] = position
            else:
                adjusted_positions[asset] = position

        logger.debug(f"Volatility-adjusted {len(adjusted_positions)} positions")
        return adjusted_positions

    def get_rebalance_trades(
        self,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float],
        min_trade_size: float = 100.0,  # Minimum trade size
    ) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance to target positions.

        Args:
            target_positions: Target position sizes
            current_positions: Current position sizes
            min_trade_size: Minimum trade size (avoid tiny trades)

        Returns:
            Dict mapping asset -> trade size (positive=buy, negative=sell)
        """
        trades = {}

        # Get all assets (current + target)
        all_assets = set(target_positions.keys()) | set(current_positions.keys())

        for asset in all_assets:
            target = target_positions.get(asset, 0.0)
            current = current_positions.get(asset, 0.0)
            trade_size = target - current

            # Only include trades above minimum size
            if abs(trade_size) >= min_trade_size:
                trades[asset] = trade_size

        logger.info(f"Rebalance trades: {len(trades)} positions to adjust")
        return trades

    def _needs_rebalance(self, current_date: pd.Timestamp) -> bool:
        """Check if portfolio needs rebalancing."""
        if self.last_rebalance_date is None:
            return True

        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance >= self.rebalance_frequency

    def _prepare_returns_data(
        self,
        predictions: pd.DataFrame,
        historical_returns: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Prepare combined returns data for optimization.

        Blends historical returns with forward-looking predictions.
        """
        try:
            # Get recent historical returns
            lookback_start = current_date - pd.Timedelta(days=self.lookback_period)
            recent_returns = historical_returns[
                historical_returns.index >= lookback_start
            ]

            # Combine with predictions (give more weight to predictions)
            # This creates a "forward-looking" returns matrix
            combined = pd.concat(
                [
                    recent_returns * 0.4,  # 40% weight on historical
                    predictions * 0.6,  # 60% weight on predictions
                ],
                axis=0,
            )

            # Align columns (only assets present in both)
            common_assets = list(
                set(recent_returns.columns) & set(predictions.columns)
            )
            combined = combined[common_assets]

            return combined.dropna()

        except Exception as e:
            logger.error(f"Failed to prepare returns data: {e}")
            return pd.DataFrame()

    @staticmethod
    def _weights_to_positions(
        weights: pd.Series, total_capital: float
    ) -> Dict[str, float]:
        """Convert portfolio weights to position sizes."""
        positions = {}
        for asset, weight in weights.items():
            position_size = weight * total_capital
            if position_size > 0:  # Only include non-zero positions
                positions[asset] = float(position_size)
        return positions

    def get_position_summary(self) -> Dict:
        """
        Get summary of current position sizing state.

        Returns:
            Dict with summary statistics
        """
        if self.current_weights is None:
            return {"status": "no_positions"}

        return {
            "status": "active",
            "num_positions": len(self.current_weights),
            "max_weight": self.current_weights.max(),
            "min_weight": self.current_weights.min(),
            "concentration": (self.current_weights ** 2).sum(),
            "last_rebalance": self.last_rebalance_date,
            "next_rebalance": (
                self.last_rebalance_date + pd.Timedelta(days=self.rebalance_frequency)
                if self.last_rebalance_date
                else None
            ),
        }
