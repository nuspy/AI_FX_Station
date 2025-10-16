"""
Neural Networks for RL Agents

Actor-Critic architectures with LSTM support:
- ActorNetwork: Policy network (state → action distribution)
- CriticNetwork: Value network (state + action → Q-value)
- SharedEncoder: Shared feature extraction layers
"""

from .actor_network import ActorNetwork
from .critic_network import CriticNetwork

__all__ = ['ActorNetwork', 'CriticNetwork']
