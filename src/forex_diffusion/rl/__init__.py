"""
Reinforcement Learning Module for Portfolio Management

Provides Actor-Critic deep RL agents (PPO, SAC, A3C, TD3) for:
- Portfolio weight optimization
- Dynamic rebalancing
- Multi-objective reward maximization

Integration with Riskfolio-Lib optimizer and Trading Engine.
"""

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .rewards import RewardFunction, MultiObjectiveReward, RewardConfig
from .environments import PortfolioEnvironment
from .networks import ActorNetwork, CriticNetwork

__all__ = [
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'RewardFunction',
    'MultiObjectiveReward',
    'RewardConfig',
    'PortfolioEnvironment',
    'ActorNetwork',
    'CriticNetwork',
]
