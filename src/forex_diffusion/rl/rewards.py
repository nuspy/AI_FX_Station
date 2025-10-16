"""
Reward Functions for Portfolio RL

Multi-objective reward functions combining:
- Sharpe ratio improvement
- Transaction costs
- Risk violations (VaR, CVaR)
- Correlation constraints
- Diversification bonus
"""

import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for multi-objective reward function."""
    
    # Reward component weights
    sharpe_weight: float = 5.0
    transaction_cost_weight: float = -10.0
    var_violation_weight: float = -20.0
    cvar_violation_weight: float = -15.0
    correlation_violation_weight: float = -5.0
    diversification_weight: float = 1.0
    drawdown_weight: float = -10.0
    turnover_weight: float = -3.0
    sortino_weight: float = 3.0
    
    # Enable/disable components
    enable_sharpe: bool = True
    enable_transaction_cost: bool = True
    enable_var_violation: bool = True
    enable_cvar_violation: bool = True
    enable_correlation_violation: bool = True
    enable_diversification: bool = True
    enable_drawdown: bool = False
    enable_turnover: bool = False
    enable_sortino: bool = False
    
    # Constraints (from portfolio config)
    max_var: float = 0.10  # 10%
    max_cvar: float = 0.15  # 15%
    max_correlated_exposure: float = 0.50  # 50%
    transaction_cost_bps: float = 5.0  # 5 basis points
    
    # Normalization
    normalize_rewards: bool = True
    reward_clip_min: float = -100.0
    reward_clip_max: float = 100.0


class RewardFunction:
    """
    Base class for reward functions.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        
        # Running statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
    
    def calculate(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_metrics: Dict,
        market_data: Dict
    ) -> float:
        """
        Calculate reward for transition.
        
        Args:
            old_weights: Previous portfolio weights
            new_weights: New portfolio weights
            portfolio_metrics: Current portfolio metrics (Sharpe, VaR, etc.)
            market_data: Market data and correlations
            
        Returns:
            Total reward
        """
        raise NotImplementedError
    
    def normalize(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        if not self.config.normalize_rewards:
            return reward
        
        # Update running mean and std
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std ** 2 + delta * (reward - self.reward_mean)) / 
            self.reward_count
        )
        
        # Z-score normalization
        if self.reward_std > 1e-6:
            normalized = (reward - self.reward_mean) / self.reward_std
        else:
            normalized = reward
        
        # Clip
        normalized = np.clip(
            normalized, 
            self.config.reward_clip_min, 
            self.config.reward_clip_max
        )
        
        return normalized


class MultiObjectiveReward(RewardFunction):
    """
    Multi-objective reward function for portfolio management.
    
    Combines multiple reward components with configurable weights.
    """
    
    def calculate(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_metrics: Dict,
        market_data: Dict
    ) -> float:
        """Calculate multi-objective reward."""
        
        reward = 0.0
        
        # 1. Sharpe Ratio Improvement
        if self.config.enable_sharpe:
            sharpe_delta = self._calculate_sharpe_delta(
                old_weights, new_weights, portfolio_metrics
            )
            reward += self.config.sharpe_weight * sharpe_delta
        
        # 2. Transaction Cost Penalty
        if self.config.enable_transaction_cost:
            cost_penalty = self._calculate_transaction_cost(
                old_weights, new_weights
            )
            reward += self.config.transaction_cost_weight * cost_penalty
        
        # 3. VaR Violation Penalty
        if self.config.enable_var_violation:
            var_penalty = self._calculate_var_violation(portfolio_metrics)
            reward += self.config.var_violation_weight * var_penalty
        
        # 4. CVaR Violation Penalty
        if self.config.enable_cvar_violation:
            cvar_penalty = self._calculate_cvar_violation(portfolio_metrics)
            reward += self.config.cvar_violation_weight * cvar_penalty
        
        # 5. Correlation Violation Penalty
        if self.config.enable_correlation_violation:
            corr_penalty = self._calculate_correlation_violation(
                new_weights, market_data
            )
            reward += self.config.correlation_violation_weight * corr_penalty
        
        # 6. Diversification Bonus
        if self.config.enable_diversification:
            diversification_bonus = self._calculate_diversification(new_weights)
            reward += self.config.diversification_weight * diversification_bonus
        
        # 7. Drawdown Penalty (optional)
        if self.config.enable_drawdown:
            drawdown_penalty = self._calculate_drawdown_penalty(portfolio_metrics)
            reward += self.config.drawdown_weight * drawdown_penalty
        
        # 8. Turnover Penalty (optional)
        if self.config.enable_turnover:
            turnover_penalty = self._calculate_turnover(old_weights, new_weights)
            reward += self.config.turnover_weight * turnover_penalty
        
        # 9. Sortino Ratio Improvement (optional)
        if self.config.enable_sortino:
            sortino_delta = self._calculate_sortino_delta(
                old_weights, new_weights, portfolio_metrics
            )
            reward += self.config.sortino_weight * sortino_delta
        
        # Normalize
        reward = self.normalize(reward)
        
        return reward
    
    def _calculate_sharpe_delta(
        self, 
        old_weights: np.ndarray, 
        new_weights: np.ndarray,
        portfolio_metrics: Dict
    ) -> float:
        """Calculate change in Sharpe ratio."""
        # Get current and previous Sharpe
        current_sharpe = portfolio_metrics.get('sharpe_ratio', 0.0)
        previous_sharpe = portfolio_metrics.get('previous_sharpe_ratio', 0.0)
        
        sharpe_delta = current_sharpe - previous_sharpe
        return sharpe_delta
    
    def _calculate_transaction_cost(
        self, 
        old_weights: np.ndarray, 
        new_weights: np.ndarray
    ) -> float:
        """Calculate transaction cost as fraction of portfolio."""
        turnover = np.sum(np.abs(new_weights - old_weights))
        cost = turnover * self.config.transaction_cost_bps / 10000.0
        return cost
    
    def _calculate_var_violation(self, portfolio_metrics: Dict) -> float:
        """Penalty if VaR exceeds limit."""
        var = portfolio_metrics.get('var_95', 0.0)
        if var > self.config.max_var:
            # Penalty proportional to violation magnitude
            violation = (var - self.config.max_var) / self.config.max_var
            return violation
        return 0.0
    
    def _calculate_cvar_violation(self, portfolio_metrics: Dict) -> float:
        """Penalty if CVaR exceeds limit."""
        cvar = portfolio_metrics.get('cvar_95', 0.0)
        if cvar > self.config.max_cvar:
            violation = (cvar - self.config.max_cvar) / self.config.max_cvar
            return violation
        return 0.0
    
    def _calculate_correlation_violation(
        self, 
        weights: np.ndarray, 
        market_data: Dict
    ) -> float:
        """Penalty if correlated exposure exceeds limit."""
        # Get correlation matrix
        corr_matrix = market_data.get('correlation_matrix')
        if corr_matrix is None:
            return 0.0
        
        # Find highly correlated assets (correlation > 0.7)
        threshold = 0.7
        n_assets = len(weights)
        
        correlated_exposure = 0.0
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                if abs(corr_matrix[i, j]) > threshold:
                    # Both assets are correlated
                    correlated_exposure += weights[i] + weights[j]
        
        if correlated_exposure > self.config.max_correlated_exposure:
            violation = (correlated_exposure - self.config.max_correlated_exposure) / \
                       self.config.max_correlated_exposure
            return violation
        
        return 0.0
    
    def _calculate_diversification(self, weights: np.ndarray) -> float:
        """Bonus for well-diversified portfolio (inverse HHI)."""
        # Herfindahl-Hirschman Index (HHI)
        # HHI = sum(w_i^2), ranges from 1/N (perfect diversification) to 1 (concentrated)
        hhi = np.sum(weights ** 2)
        
        # Diversification score: 1 - HHI
        # Ranges from 0 (fully concentrated) to 1-1/N (perfectly diversified)
        diversification = 1.0 - hhi
        
        return diversification
    
    def _calculate_drawdown_penalty(self, portfolio_metrics: Dict) -> float:
        """Penalty for current drawdown."""
        drawdown = portfolio_metrics.get('current_drawdown', 0.0)
        # Drawdown is negative, so penalty is positive
        return abs(drawdown)
    
    def _calculate_turnover(
        self, 
        old_weights: np.ndarray, 
        new_weights: np.ndarray
    ) -> float:
        """Penalty for excessive portfolio turnover."""
        turnover = np.sum(np.abs(new_weights - old_weights))
        return turnover
    
    def _calculate_sortino_delta(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_metrics: Dict
    ) -> float:
        """Calculate change in Sortino ratio."""
        current_sortino = portfolio_metrics.get('sortino_ratio', 0.0)
        previous_sortino = portfolio_metrics.get('previous_sortino_ratio', 0.0)
        
        sortino_delta = current_sortino - previous_sortino
        return sortino_delta


class CustomReward(RewardFunction):
    """
    Custom reward function with user-defined logic.
    
    Allows users to provide custom Python function via UI.
    """
    
    def __init__(self, config: RewardConfig, custom_fn: Optional[Callable] = None):
        super().__init__(config)
        self.custom_fn = custom_fn
    
    def calculate(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_metrics: Dict,
        market_data: Dict
    ) -> float:
        """Calculate reward using custom function."""
        if self.custom_fn is None:
            raise ValueError("Custom reward function not provided")
        
        try:
            reward = self.custom_fn(
                old_weights, new_weights, portfolio_metrics, market_data
            )
            return self.normalize(reward)
        except Exception as e:
            # Fallback to zero reward on error
            print(f"Custom reward function error: {e}")
            return 0.0
