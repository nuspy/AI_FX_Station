"""Riskfolio-Lib Integrator - Portfolio Optimization"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from loguru import logger

try:
    import riskfolio as rp
    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False
    logger.warning("riskfolio-lib not installed")

class RiskfolioIntegrator:
    """Integrates Riskfolio-Lib for portfolio optimization"""
    
    def __init__(self, config: Dict = None):
        if not RISKFOLIO_AVAILABLE:
            logger.warning("Riskfolio-Lib not available, using equal weights")
        
        self.config = config or {}
        self.portfolio = None
    
    def optimize_portfolio(self, returns: pd.DataFrame) -> pd.Series:
        """Optimize portfolio weights"""
        if not RISKFOLIO_AVAILABLE or returns.empty:
            # Equal weights fallback
            n_assets = len(returns.columns) if isinstance(returns, pd.DataFrame) else 1
            return pd.Series(1.0 / n_assets, index=returns.columns if isinstance(returns, pd.DataFrame) else [0])
        
        try:
            # Create portfolio object
            self.portfolio = rp.Portfolio(returns=returns)
            
            # Get config
            risk_measure = self.config.get('riskfolio_risk_measure', 'CVaR')
            objective = self.config.get('riskfolio_objective', 'Sharpe')
            risk_aversion = self.config.get('riskfolio_risk_aversion', 1.0)
            risk_free_rate = self.config.get('riskfolio_risk_free_rate', 0.0)
            
            # Apply constraints
            max_weight = self.config.get('riskfolio_max_weight', 0.3)
            min_weight = self.config.get('riskfolio_min_weight', 0.0)
            
            self.portfolio.upperlng = max_weight
            self.portfolio.lowerret = min_weight
            
            # Optimize
            if self.config.get('riskfolio_use_risk_parity', False):
                weights = self.portfolio.rp_optimization(
                    model='Classic',
                    rm=risk_measure,
                    rf=risk_free_rate,
                    hist=True
                )
            else:
                weights = self.portfolio.optimization(
                    model='Classic',
                    rm=risk_measure,
                    obj=objective,
                    rf=risk_free_rate,
                    l=risk_aversion,
                    hist=True
                )
            
            logger.info(f"Portfolio optimized: {len(weights)} assets, method={objective}")
            return weights
            
        except Exception as e:
            logger.error(f"Riskfolio optimization failed: {e}")
            n_assets = len(returns.columns)
            return pd.Series(1.0 / n_assets, index=returns.columns)
    
    def calculate_portfolio_metrics(self, weights: pd.Series, returns: pd.DataFrame) -> Dict:
        """Calculate portfolio statistics"""
        port_returns = (returns * weights).sum(axis=1)
        
        return {
            'expected_return': port_returns.mean() * 252,
            'volatility': port_returns.std() * np.sqrt(252),
            'sharpe_ratio': (port_returns.mean() / port_returns.std() * np.sqrt(252)) if port_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(port_returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
