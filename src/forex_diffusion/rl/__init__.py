"""
Reinforcement Learning Module for Portfolio Management

Provides Actor-Critic deep RL agents (PPO, SAC, A3C, TD3) for:
- Portfolio weight optimization
- Dynamic rebalancing
- Multi-objective reward maximization

Integration with Riskfolio-Lib optimizer and Trading Engine.
"""

from .rl_portfolio_manager import RLPortfolioManager
from .rewards import RewardFunction, MultiObjectiveReward
from .replay_buffer import ReplayBuffer

__all__ = [
    'RLPortfolioManager',
    'RewardFunction',
    'MultiObjectiveReward',
    'ReplayBuffer',
]
