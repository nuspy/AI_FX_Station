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
from .actor_critic import BaseAgent, PPOAgent
from .trainer import RLTrainer, TrainerConfig
from .rl_portfolio_manager import RLPortfolioManager, RLPortfolioConfig

__all__ = [
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'RewardFunction',
    'MultiObjectiveReward',
    'RewardConfig',
    'PortfolioEnvironment',
    'ActorNetwork',
    'CriticNetwork',
    'BaseAgent',
    'PPOAgent',
    'RLTrainer',
    'TrainerConfig',
    'RLPortfolioManager',
    'RLPortfolioConfig',
]
