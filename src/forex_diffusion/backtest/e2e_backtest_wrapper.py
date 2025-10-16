"""E2E Backtest Wrapper - Integrates all components"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

from ..integrations import (
    SssdIntegrator,
    RiskfolioIntegrator,
    PatternIntegrator,
    RLIntegrator,
    VixFilter,
    SentimentFilter,
    VolumeFilter
)

class E2EBacktestWrapper:
    """
    Wrapper for integrated backtest with E2E parameters.
    
    Connects all component integrators and runs backtest with optimized parameters.
    """
    
    def __init__(self, db_session=None):
        self.db_session = db_session
        
        # Component integrators
        self.sssd_integrator: Optional[SssdIntegrator] = None
        self.riskfolio_integrator: Optional[RiskfolioIntegrator] = None
        self.pattern_integrator: Optional[PatternIntegrator] = None
        self.rl_integrator: Optional[RLIntegrator] = None
        self.vix_filter: Optional[VixFilter] = None
        self.sentiment_filter: Optional[SentimentFilter] = None
        self.volume_filter: Optional[VolumeFilter] = None
    
    def run_backtest(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Run backtest with E2E parameters.
        
        Args:
            data: OHLCV data
            params: Dictionary with all 90+ parameters
            
        Returns:
            Dictionary with backtest results (sharpe, dd, wr, pf, etc.)
        """
        logger.info(f"Running E2E backtest with {len(params)} parameters")
        
        # Initialize integrators with parameters
        self._initialize_integrators(params)
        
        # Run simplified backtest
        result = self._run_simplified_backtest(data, params)
        
        logger.info(f"Backtest complete: Sharpe={result.get('sharpe_ratio', 0):.3f}, DD={result.get('max_drawdown_pct', 0):.2f}%")
        
        return result
    
    def _initialize_integrators(self, params: Dict):
        """Initialize all component integrators with parameters"""
        
        # SSSD
        if params.get('sssd_enabled', False):
            self.sssd_integrator = SssdIntegrator(config=params)
        
        # Riskfolio
        if params.get('riskfolio_enabled', True):
            self.riskfolio_integrator = RiskfolioIntegrator(config=params)
        
        # Patterns
        if params.get('patterns_enabled', True):
            self.pattern_integrator = PatternIntegrator(db_session=self.db_session)
        
        # RL
        if params.get('rl_enabled', False):
            self.rl_integrator = RLIntegrator(config=params)
        
        # Filters
        self.vix_filter = VixFilter(config=params)
        self.sentiment_filter = SentimentFilter(config=params)
        self.volume_filter = VolumeFilter(config=params)
    
    def _run_simplified_backtest(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Simplified backtest implementation.
        
        In production, this would integrate with existing IntegratedBacktester.
        For now, returns mock results based on parameters.
        """
        
        # Extract key parameters
        risk_per_trade = params.get('sizing_base_risk_pct', 1.0) / 100
        stop_loss_pct = params.get('risk_stop_loss_pct', 2.0) / 100
        take_profit_pct = params.get('risk_take_profit_pct', 4.0) / 100
        
        # Simple buy-and-hold with risk management
        returns = data['close'].pct_change().dropna()
        
        # Apply position sizing
        position_sizes = np.ones(len(returns)) * risk_per_trade
        
        # Apply VIX filter (reduce size in high volatility)
        vix_level = 25.0  # Mock VIX
        vix_adjustment = self.vix_filter.get_adjustment_factor(vix_level) if self.vix_filter else 1.0
        position_sizes *= vix_adjustment
        
        # Calculate portfolio returns
        strategy_returns = returns * position_sizes
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        ann_return = strategy_returns.mean() * 252
        ann_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Calculate drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        # Calculate win rate (simple)
        wins = (strategy_returns > 0).sum()
        losses = (strategy_returns < 0).sum()
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.5
        
        # Calculate profit factor
        total_wins = strategy_returns[strategy_returns > 0].sum()
        total_losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 1.0
        
        # Calmar ratio
        calmar = ann_return / max_dd if max_dd > 0 else 0
        
        # Costs (simplified)
        commission_rate = 0.0002  # 2 pips
        total_costs = total_trades * commission_rate
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sharpe * 1.2,  # Approximation
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': int(total_trades),
            'winning_trades': int(wins),
            'losing_trades': int(losses),
            'avg_win': total_wins / wins if wins > 0 else 0,
            'avg_loss': total_losses / losses if losses > 0 else 0,
            'expectancy': (win_rate * (total_wins / wins if wins > 0 else 0) + 
                          (1 - win_rate) * (-total_losses / losses if losses > 0 else 0)),
            'total_costs': total_costs,
            'avg_cost_per_trade': total_costs / total_trades if total_trades > 0 else 0,
            'avg_holding_time_hrs': 24.0,  # Mock
        }
