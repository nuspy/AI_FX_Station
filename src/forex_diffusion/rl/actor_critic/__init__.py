"""
Actor-Critic RL Agents

Implementations of state-of-the-art RL algorithms:
- PPO (Proximal Policy Optimization) - Primary algorithm
- SAC (Soft Actor-Critic) - Continuous control
- A3C (Asynchronous Advantage Actor-Critic) - Parallel training
- TD3 (Twin Delayed DDPG) - Robust Q-learning
"""

from .base_agent import BaseAgent
from .ppo_agent import PPOAgent

__all__ = ['BaseAgent', 'PPOAgent']
