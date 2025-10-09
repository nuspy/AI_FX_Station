"""Risk metrics calculator for portfolio analysis."""

from typing import Dict
import pandas as pd
import numpy as np
from loguru import logger


class RiskMetricsCalculator:
    """Calculate comprehensive risk metrics for portfolio analysis."""

    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return returns.quantile(1 - confidence)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = RiskMetricsCalculator.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def calculate_all_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate all risk metrics."""
        return {
            "var_95": RiskMetricsCalculator.calculate_var(returns, 0.95),
            "var_99": RiskMetricsCalculator.calculate_var(returns, 0.99),
            "cvar_95": RiskMetricsCalculator.calculate_cvar(returns, 0.95),
            "cvar_99": RiskMetricsCalculator.calculate_cvar(returns, 0.99),
            "volatility": returns.std() * np.sqrt(252),
            "downside_deviation": returns[returns < 0].std() * np.sqrt(252),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }
