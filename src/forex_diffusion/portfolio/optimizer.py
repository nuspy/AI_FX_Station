"""
Portfolio Optimizer using Riskfolio-Lib.

Integrates quantitative portfolio optimization with diffusion model predictions
for optimal asset allocation and risk management.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

try:
    import riskfolio as rp
    _HAS_RISKFOLIO = True
except ImportError:
    _HAS_RISKFOLIO = False
    logger.warning("riskfolio-lib not installed. Portfolio optimization disabled.")


class PortfolioOptimizer:
    """
    Portfolio optimizer integrating Riskfolio-Lib with trading signals.

    Supports multiple optimization objectives:
    - Maximum Sharpe Ratio
    - Minimum Risk (Volatility, CVaR, CDaR)
    - Risk Parity
    - Maximum Diversification
    """

    def __init__(
        self,
        risk_measure: str = "CVaR",
        objective: str = "Sharpe",
        risk_free_rate: float = 0.0,
        risk_aversion: float = 1.0,
    ):
        """
        Initialize portfolio optimizer.

        Args:
            risk_measure: Risk measure ('MV', 'CVaR', 'CDaR', 'EVaR', 'WR', 'MDD')
            objective: Optimization objective ('Sharpe', 'MinRisk', 'Utility', 'MaxRet')
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            risk_aversion: Risk aversion parameter (0-inf, higher = more conservative)
        """
        if not _HAS_RISKFOLIO:
            raise ImportError("riskfolio-lib required for portfolio optimization")

        self.risk_measure = risk_measure
        self.objective = objective
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion

        # Portfolio object (created per optimization)
        self.portfolio: Optional[rp.Portfolio] = None

        # Last optimization results
        self.last_weights: Optional[pd.Series] = None
        self.last_frontier: Optional[pd.DataFrame] = None

        logger.info(
            f"Portfolio Optimizer initialized: "
            f"rm={risk_measure}, obj={objective}, lambda={risk_aversion}"
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None,
        method: str = "Classic",
    ) -> pd.Series:
        """
        Optimize portfolio weights based on historical/predicted returns.

        Args:
            returns: DataFrame with returns (columns=assets, index=timestamps)
            constraints: Dict with optimization constraints
                - 'max_weight': Maximum weight per asset (0-1)
                - 'min_weight': Minimum weight per asset (0-1)
                - 'max_leverage': Maximum leverage (>= 1.0)
                - 'target_return': Minimum required return
                - 'target_risk': Maximum allowed risk
            method: Optimization method ('Classic', 'BL', 'FM')

        Returns:
            Series with optimal weights (index=asset names, values=weights)
        """
        if returns.empty:
            logger.warning("Empty returns DataFrame provided")
            return pd.Series()

        try:
            # Create portfolio object
            self.portfolio = rp.Portfolio(returns=returns)

            # Apply constraints
            if constraints:
                self._apply_constraints(constraints)

            # Calculate optimal portfolio
            weights = self.portfolio.optimization(
                model=method,
                rm=self.risk_measure,
                obj=self.objective,
                rf=self.risk_free_rate,
                l=self.risk_aversion,
                hist=True,  # Use historical scenarios
            )

            # Store results
            self.last_weights = weights

            logger.info(
                f"Portfolio optimized: {len(weights)} assets, "
                f"concentration={self._calculate_concentration(weights):.2%}"
            )

            return weights

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Return equal-weight fallback
            n_assets = len(returns.columns)
            return pd.Series(1.0 / n_assets, index=returns.columns)

    def optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        risk_budgets: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """
        Optimize portfolio using Risk Parity approach.

        Args:
            returns: DataFrame with returns
            risk_budgets: Optional risk budget allocation (sums to 1.0)
                         If None, uses equal risk contribution

        Returns:
            Series with risk parity weights
        """
        try:
            self.portfolio = rp.Portfolio(returns=returns)

            weights = self.portfolio.rp_optimization(
                model="Classic",
                rm=self.risk_measure,
                rf=self.risk_free_rate,
                b=risk_budgets,  # Risk budgets (None = equal)
                hist=True,
            )

            self.last_weights = weights

            logger.info(f"Risk Parity portfolio optimized: {len(weights)} assets")
            return weights

        except Exception as e:
            logger.error(f"Risk Parity optimization failed: {e}")
            n_assets = len(returns.columns)
            return pd.Series(1.0 / n_assets, index=returns.columns)

    def calculate_efficient_frontier(
        self,
        returns: pd.DataFrame,
        points: int = 20,
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier for visualization.

        Args:
            returns: DataFrame with returns
            points: Number of points on the frontier

        Returns:
            DataFrame with columns: [return, risk, sharpe] for each point
        """
        try:
            self.portfolio = rp.Portfolio(returns=returns)

            # Calculate frontier
            frontier = self.portfolio.efficient_frontier(
                model="Classic",
                rm=self.risk_measure,
                points=points,
                rf=self.risk_free_rate,
                hist=True,
            )

            self.last_frontier = frontier

            logger.info(f"Efficient frontier calculated: {points} points")
            return frontier

        except Exception as e:
            logger.error(f"Efficient frontier calculation failed: {e}")
            return pd.DataFrame()

    def backtest_strategy(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Backtest portfolio strategy and calculate performance metrics.

        Args:
            weights: Portfolio weights
            returns: Historical returns DataFrame

        Returns:
            Dict with metrics: sharpe, sortino, max_drawdown, cvar, etc.
        """
        try:
            # Calculate portfolio returns
            port_returns = (returns * weights).sum(axis=1)

            # Calculate metrics
            metrics = {
                "total_return": (1 + port_returns).prod() - 1,
                "ann_return": port_returns.mean() * 252,  # Assuming daily data
                "ann_volatility": port_returns.std() * np.sqrt(252),
                "sharpe_ratio": self._calculate_sharpe(port_returns),
                "sortino_ratio": self._calculate_sortino(port_returns),
                "max_drawdown": self._calculate_max_drawdown(port_returns),
                "cvar_95": self._calculate_cvar(port_returns, alpha=0.05),
                "calmar_ratio": self._calculate_calmar(port_returns),
            }

            logger.info(
                f"Backtest complete: Sharpe={metrics['sharpe_ratio']:.2f}, "
                f"MDD={metrics['max_drawdown']:.2%}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {}

    def _apply_constraints(self, constraints: Dict):
        """Apply portfolio constraints."""
        if "max_weight" in constraints:
            self.portfolio.upperlng = constraints["max_weight"]

        if "min_weight" in constraints:
            self.portfolio.lowerret = constraints["min_weight"]

        if "max_leverage" in constraints:
            self.portfolio.sht = constraints["max_leverage"] > 1.0

        if "target_return" in constraints:
            self.portfolio.lowerret = constraints["target_return"]

        if "target_risk" in constraints:
            self.portfolio.upperrisk = constraints["target_risk"]

    @staticmethod
    def _calculate_concentration(weights: pd.Series) -> float:
        """Calculate portfolio concentration (Herfindahl index)."""
        return (weights ** 2).sum()

    @staticmethod
    def _calculate_sharpe(returns: pd.Series, rf: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - rf / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0.0

    @staticmethod
    def _calculate_sortino(returns: pd.Series, rf: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns - rf / 252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0.0

    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def _calculate_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = returns.quantile(alpha)
        return returns[returns <= var].mean()

    @staticmethod
    def _calculate_calmar(returns: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        ann_return = returns.mean() * 252
        max_dd = PortfolioOptimizer._calculate_max_drawdown(returns)
        return ann_return / abs(max_dd) if max_dd != 0 else 0.0

    def get_portfolio_stats(self, weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Get comprehensive portfolio statistics.

        Args:
            weights: Portfolio weights
            returns: Historical returns

        Returns:
            Dict with portfolio statistics
        """
        try:
            port_returns = (returns * weights).sum(axis=1)

            stats = {
                "expected_return": port_returns.mean() * 252,
                "volatility": port_returns.std() * np.sqrt(252),
                "sharpe_ratio": self._calculate_sharpe(port_returns),
                "sortino_ratio": self._calculate_sortino(port_returns),
                "max_drawdown": self._calculate_max_drawdown(port_returns),
                "cvar_95": self._calculate_cvar(port_returns, 0.05),
                "cvar_99": self._calculate_cvar(port_returns, 0.01),
                "skewness": port_returns.skew(),
                "kurtosis": port_returns.kurtosis(),
                "concentration": self._calculate_concentration(weights),
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate portfolio stats: {e}")
            return {}
