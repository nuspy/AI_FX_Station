"""
Portfolio Management Gym Environment

State: 137-dimensional continuous
- Portfolio: current weights, P&L, days in position
- Market: returns, volatility, correlations, momentum
- Risk: VaR, CVaR, Sharpe, Sortino, drawdown
- Sentiment: VIX, news, orderbook imbalance

Action: Continuous portfolio weights [0, 1] summing to 1.0

Reward: Multi-objective (Sharpe, costs, risk violations)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
from loguru import logger

from ..rewards import MultiObjectiveReward, RewardConfig


@dataclass
class PortfolioEnvConfig:
    """Configuration for portfolio environment."""
    
    # Assets
    symbols: List[str] = None  # List of asset symbols
    
    # Episode settings
    max_steps: int = 252  # Trading days in episode (1 year default)
    initial_capital: float = 10000.0
    
    # State features
    lookback_returns: int = 20  # Days for returns calculation
    lookback_volatility: int = 20  # Days for volatility
    lookback_momentum: int = 10  # Days for momentum
    
    # Action constraints
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 0.25  # Maximum weight per asset (25%)
    force_sum_one: bool = True  # Force weights to sum to 1.0
    long_only: bool = True  # No shorting
    
    # Reward configuration
    reward_config: Optional[RewardConfig] = None
    
    # Transaction costs
    transaction_cost_bps: float = 5.0  # 5 basis points
    
    # Risk limits
    max_var: float = 0.10  # 10% VaR limit
    max_cvar: float = 0.15  # 15% CVaR limit
    max_correlated_exposure: float = 0.50  # 50% max in correlated assets


class PortfolioEnvironment(gym.Env):
    """
    OpenAI Gym environment for portfolio management.
    
    Agent learns to allocate capital across multiple assets to maximize
    risk-adjusted returns while respecting constraints.
    
    Example:
        >>> config = PortfolioEnvConfig(
        ...     symbols=['EUR/USD', 'GBP/USD', 'USD/JPY'],
        ...     max_steps=252,
        ...     initial_capital=10000.0
        ... )
        >>> env = PortfolioEnvironment(market_data, config)
        >>> state = env.reset()
        >>> action = np.array([0.4, 0.3, 0.3])  # Target weights
        >>> next_state, reward, done, info = env.step(action)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self, 
        market_data: pd.DataFrame,
        config: PortfolioEnvConfig
    ):
        """
        Initialize portfolio environment.
        
        Args:
            market_data: DataFrame with columns ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config
        self.market_data = market_data
        
        # Set default symbols from market data if not provided
        if self.config.symbols is None:
            self.config.symbols = market_data['symbol'].unique().tolist()
        
        self.n_assets = len(self.config.symbols)
        
        # Initialize reward function
        if self.config.reward_config is None:
            self.config.reward_config = RewardConfig()
        self.reward_function = MultiObjectiveReward(self.config.reward_config)
        
        # Define action space: continuous weights for each asset [0, 1]
        self.action_space = spaces.Box(
            low=self.config.min_weight,
            high=self.config.max_weight,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Define observation space: 137-dimensional continuous
        # Structure: [portfolio features (15) + market features (70) + risk features (12) + sentiment features (5)]
        # Actual dimension depends on enabled features, 137 is example with all features
        state_dim = self._calculate_state_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.current_weights = None
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.peak_value = self.config.initial_capital
        
        # Performance tracking
        self.episode_returns = []
        self.episode_weights = []
        self.episode_rewards = []
        self.transaction_costs_total = 0.0
        
        # Market data indexing
        self._prepare_market_data()
        
        logger.info(f"PortfolioEnvironment initialized: {self.n_assets} assets, "
                   f"state_dim={state_dim}, max_steps={config.max_steps}")
    
    def _calculate_state_dimension(self) -> int:
        """Calculate total state dimension based on enabled features."""
        dim = 0
        
        # Portfolio features (always included)
        dim += self.n_assets  # Current weights
        dim += self.n_assets  # Days in position
        dim += self.n_assets  # Unrealized P&L per asset
        dim += 1  # Total portfolio value
        dim += 1  # Cash position
        dim += 1  # Current drawdown
        # Total: 3*n_assets + 3
        
        # Market features
        dim += self.n_assets * self.config.lookback_returns  # Returns history
        dim += self.n_assets  # Current volatility
        dim += (self.n_assets * (self.n_assets - 1)) // 2  # Correlation matrix (upper triangle)
        dim += self.n_assets  # Momentum
        # Total: n_assets * (lookback + 1) + n_assets*(n_assets-1)/2 + n_assets
        
        # Risk features
        dim += 1  # Portfolio VaR
        dim += 1  # Portfolio CVaR
        dim += 1  # Sharpe ratio
        dim += 1  # Sortino ratio
        dim += 1  # Max drawdown
        # Total: 5
        
        # Sentiment features (if available)
        dim += 1  # VIX level
        dim += 1  # VIX percentile
        # Total: 2
        
        return dim
    
    def _prepare_market_data(self):
        """Prepare market data for efficient access during episodes."""
        # Pivot to wide format for faster access
        self.price_data = {}
        
        for symbol in self.config.symbols:
            symbol_data = self.market_data[self.market_data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp')
            symbol_data['returns'] = symbol_data['close'].pct_change()
            self.price_data[symbol] = symbol_data.reset_index(drop=True)
        
        # Find common date range
        min_length = min(len(df) for df in self.price_data.values())
        self.max_episode_start = min_length - self.config.max_steps - self.config.lookback_returns - 1
        
        if self.max_episode_start < 0:
            raise ValueError(f"Not enough data: need at least {self.config.max_steps + self.config.lookback_returns} bars")
        
        logger.debug(f"Market data prepared: {min_length} bars, {self.max_episode_start} possible episode starts")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state observation
        """
        # Random episode start (within valid range)
        self.episode_start_idx = np.random.randint(
            self.config.lookback_returns, 
            self.max_episode_start
        )
        self.current_step = 0
        
        # Reset portfolio
        # Start with equal weights
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.config.initial_capital
        self.cash = 0.0  # All capital invested
        self.peak_value = self.config.initial_capital
        
        # Reset tracking
        self.episode_returns = []
        self.episode_weights = [self.current_weights.copy()]
        self.episode_rewards = []
        self.transaction_costs_total = 0.0
        self.days_in_position = np.zeros(self.n_assets)
        
        # Get initial state
        state = self._get_state()
        
        logger.debug(f"Environment reset: episode start={self.episode_start_idx}, "
                    f"initial_weights={self.current_weights}")
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Target portfolio weights (will be normalized)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Normalize action to valid weights
        new_weights = self._normalize_action(action)
        
        # Calculate portfolio metrics before rebalancing
        old_metrics = self._calculate_portfolio_metrics(self.current_weights)
        
        # Execute rebalancing
        transaction_cost = self._rebalance_portfolio(self.current_weights, new_weights)
        self.transaction_costs_total += transaction_cost
        
        # Update market (simulate one day passing)
        returns = self._get_current_returns()
        
        # Update portfolio value
        portfolio_return = np.sum(new_weights * returns)
        self.portfolio_value *= (1 + portfolio_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        
        # Update weights (market movement changes weights slightly)
        # After returns, each asset weight becomes: w_i * (1 + r_i) / (1 + portfolio_return)
        self.current_weights = new_weights * (1 + returns) / (1 + portfolio_return)
        self.current_weights = self._normalize_weights(self.current_weights)  # Ensure still sums to 1
        
        # Update position tracking
        self.days_in_position[new_weights > 0.01] += 1
        self.days_in_position[new_weights <= 0.01] = 0
        
        # Track performance
        self.episode_returns.append(portfolio_return)
        self.episode_weights.append(self.current_weights.copy())
        
        # Calculate portfolio metrics after rebalancing
        new_metrics = self._calculate_portfolio_metrics(self.current_weights)
        
        # Calculate reward
        market_data = self._get_market_features()
        reward = self.reward_function.calculate(
            old_weights=np.array(self.episode_weights[-2]) if len(self.episode_weights) > 1 else new_weights,
            new_weights=new_weights,
            portfolio_metrics=new_metrics,
            market_data=market_data
        )
        self.episode_rewards.append(reward)
        
        # Advance step
        self.current_step += 1
        done = self.current_step >= self.config.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'sharpe_ratio': new_metrics.get('sharpe_ratio', 0.0),
            'sortino_ratio': new_metrics.get('sortino_ratio', 0.0),
            'var_95': new_metrics.get('var_95', 0.0),
            'cvar_95': new_metrics.get('cvar_95', 0.0),
            'current_drawdown': new_metrics.get('current_drawdown', 0.0),
            'weights': self.current_weights.copy(),
            'step': self.current_step,
        }
        
        # Episode summary (if done)
        if done:
            info['episode'] = {
                'total_return': (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital,
                'sharpe_ratio': self._calculate_episode_sharpe(),
                'sortino_ratio': self._calculate_episode_sortino(),
                'max_drawdown': self._calculate_max_drawdown(),
                'total_transaction_costs': self.transaction_costs_total,
                'total_reward': sum(self.episode_rewards),
            }
            logger.info(f"Episode finished: return={info['episode']['total_return']:.2%}, "
                       f"Sharpe={info['episode']['sharpe_ratio']:.2f}, "
                       f"MaxDD={info['episode']['max_drawdown']:.2%}")
        
        return next_state, reward, done, info
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action to valid portfolio weights.
        
        - Clip to [min_weight, max_weight]
        - Force sum to 1.0
        - Force non-negative if long_only
        """
        weights = np.array(action, dtype=np.float32)
        
        # Clip to bounds
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        
        # Force long-only
        if self.config.long_only:
            weights = np.maximum(weights, 0.0)
        
        # Normalize to sum to 1.0
        if self.config.force_sum_one:
            weight_sum = np.sum(weights)
            if weight_sum > 1e-6:
                weights = weights / weight_sum
            else:
                # If all weights zero, use equal weights
                weights = np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Ensure weights sum to 1.0 (for drift correction)."""
        weight_sum = np.sum(weights)
        if weight_sum > 1e-6:
            return weights / weight_sum
        return np.ones(self.n_assets) / self.n_assets
    
    def _rebalance_portfolio(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """
        Execute portfolio rebalancing and return transaction cost.
        
        Args:
            old_weights: Current portfolio weights
            new_weights: Target portfolio weights
            
        Returns:
            Transaction cost as fraction of portfolio value
        """
        turnover = np.sum(np.abs(new_weights - old_weights))
        cost = turnover * self.config.transaction_cost_bps / 10000.0
        
        # Subtract cost from portfolio value
        self.portfolio_value *= (1 - cost)
        
        return cost
    
    def _get_current_returns(self) -> np.ndarray:
        """Get current period returns for all assets."""
        idx = self.episode_start_idx + self.current_step
        returns = np.zeros(self.n_assets)
        
        for i, symbol in enumerate(self.config.symbols):
            returns[i] = self.price_data[symbol].iloc[idx]['returns']
        
        # Handle NaN
        returns = np.nan_to_num(returns, 0.0)
        
        return returns
    
    def _get_state(self) -> np.ndarray:
        """
        Construct current state observation (137-dim vector).
        
        State components:
        1. Portfolio features (current weights, P&L, days in position)
        2. Market features (returns history, volatility, correlations, momentum)
        3. Risk features (VaR, CVaR, Sharpe, Sortino, drawdown)
        4. Sentiment features (VIX, percentile)
        """
        state_components = []
        
        # 1. Portfolio features
        state_components.append(self.current_weights)  # n_assets
        state_components.append(self.days_in_position / 252.0)  # Normalized by 1 year
        
        # Unrealized P&L per asset
        asset_pnl = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if len(self.episode_weights) > 1:
                weight_change = self.current_weights[i] - self.episode_weights[0][i]
                asset_pnl[i] = weight_change
        state_components.append(asset_pnl)
        
        # Portfolio-level
        state_components.append([self.portfolio_value / self.config.initial_capital])  # Normalized value
        state_components.append([self.cash / self.portfolio_value if self.portfolio_value > 0 else 0.0])
        
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0
        state_components.append([current_drawdown])
        
        # 2. Market features
        market_features = self._get_market_features()
        
        # Returns history (last N days)
        returns_history = market_features['returns_history']  # Shape: (n_assets, lookback)
        state_components.append(returns_history.flatten())
        
        # Current volatility
        state_components.append(market_features['volatility'])  # Shape: (n_assets,)
        
        # Correlation matrix (upper triangle only to avoid redundancy)
        corr_matrix = market_features['correlation_matrix']
        corr_upper = corr_matrix[np.triu_indices(self.n_assets, k=1)]
        state_components.append(corr_upper)
        
        # Momentum
        state_components.append(market_features['momentum'])  # Shape: (n_assets,)
        
        # 3. Risk features
        risk_metrics = self._calculate_portfolio_metrics(self.current_weights)
        state_components.append([
            risk_metrics.get('var_95', 0.0),
            risk_metrics.get('cvar_95', 0.0),
            risk_metrics.get('sharpe_ratio', 0.0),
            risk_metrics.get('sortino_ratio', 0.0),
            risk_metrics.get('max_drawdown', 0.0),
        ])
        
        # 4. Sentiment features (placeholder - can be extended)
        vix = market_features.get('vix', 15.0)  # Default VIX = 15
        vix_percentile = market_features.get('vix_percentile', 0.5)  # Default median
        state_components.append([vix / 50.0, vix_percentile])  # Normalized VIX
        
        # Concatenate all components
        state = np.concatenate([np.array(c).flatten() for c in state_components])
        
        return state.astype(np.float32)
    
    def _get_market_features(self) -> Dict:
        """Extract market features for current step."""
        idx = self.episode_start_idx + self.current_step
        
        features = {}
        
        # Returns history
        returns_history = np.zeros((self.n_assets, self.config.lookback_returns))
        for i, symbol in enumerate(self.config.symbols):
            start_idx = idx - self.config.lookback_returns
            returns_history[i, :] = self.price_data[symbol].iloc[start_idx:idx]['returns'].values
        
        returns_history = np.nan_to_num(returns_history, 0.0)
        features['returns_history'] = returns_history
        
        # Volatility (std of returns)
        volatility = np.std(returns_history, axis=1)
        features['volatility'] = volatility
        
        # Correlation matrix
        if returns_history.shape[1] > 1:
            corr_matrix = np.corrcoef(returns_history)
            corr_matrix = np.nan_to_num(corr_matrix, 0.0)
        else:
            corr_matrix = np.eye(self.n_assets)
        features['correlation_matrix'] = corr_matrix
        
        # Momentum (cumulative return over lookback period)
        momentum = np.sum(returns_history, axis=1)
        features['momentum'] = momentum
        
        # Sentiment (placeholder - can be extended with real data)
        features['vix'] = 15.0  # Default
        features['vix_percentile'] = 0.5
        
        return features
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """Calculate portfolio risk metrics."""
        if len(self.episode_returns) < 2:
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
            }
        
        returns = np.array(self.episode_returns)
        
        # Sharpe ratio (annualized)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 1e-6 else 0.0
        
        # Sortino ratio (annualized, downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
        sortino = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 1e-6 else 0.0
        
        # VaR and CVaR (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0
        
        return {
            'var_95': abs(var_95),
            'cvar_95': abs(cvar_95),
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'previous_sharpe_ratio': sharpe - 0.01 if len(returns) > 1 else 0.0,  # Approx for delta
            'previous_sortino_ratio': sortino - 0.01 if len(returns) > 1 else 0.0,
        }
    
    def _calculate_episode_sharpe(self) -> float:
        """Calculate Sharpe ratio for entire episode."""
        if len(self.episode_returns) < 2:
            return 0.0
        
        returns = np.array(self.episode_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-6:
            return 0.0
        
        # Annualized Sharpe
        sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
        return sharpe
    
    def _calculate_episode_sortino(self) -> float:
        """Calculate Sortino ratio for entire episode."""
        if len(self.episode_returns) < 2:
            return 0.0
        
        returns = np.array(self.episode_returns)
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 100.0  # No downside, infinite Sortino (cap at 100)
        
        downside_std = np.std(downside_returns)
        if downside_std < 1e-6:
            return 0.0
        
        # Annualized Sortino
        sortino = (mean_return * 252) / (downside_std * np.sqrt(252))
        return sortino
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown in episode."""
        if len(self.episode_returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + np.array(self.episode_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        
        max_dd = np.max(drawdown)
        return max_dd
    
    def render(self, mode='human'):
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step}/{self.config.max_steps} ===")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Return: {(self.portfolio_value / self.config.initial_capital - 1) * 100:.2f}%")
            print(f"Current Weights: {dict(zip(self.config.symbols, self.current_weights))}")
            
            if len(self.episode_returns) > 0:
                print(f"Episode Sharpe: {self._calculate_episode_sharpe():.2f}")
                print(f"Max Drawdown: {self._calculate_max_drawdown() * 100:.2f}%")
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        return [seed]
    
    def close(self):
        """Cleanup resources."""
        pass
