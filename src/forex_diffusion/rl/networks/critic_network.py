"""
Critic Network (Value Network)

Maps (state, action) → Q-value

Architecture:
- Input: State (137-dim) + Action (n_assets)
- Hidden: [256, 128] fully connected
- Output: Scalar Q-value

Used in:
- PPO: V(s) for advantage estimation
- SAC/TD3: Q(s,a) for critic loss
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class CriticNetwork(nn.Module):
    """
    Value network for Q-value estimation.
    
    Estimates expected return from state-action pair.
    
    Example:
        >>> critic = CriticNetwork(state_dim=137, action_dim=10, hidden_dims=[256, 128])
        >>> state = torch.randn(32, 137)
        >>> action = torch.randn(32, 10)
        >>> q_value = critic(state, action)
        >>> # q_value shape: (32, 1)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        activation: str = 'relu',
        value_only: bool = False,
    ):
        """
        Initialize Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'elu')
            value_only: If True, estimate V(s) only (ignore action)
        """
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.value_only = value_only
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input dimension: state + action (or just state for value_only)
        input_dim = state_dim if value_only else state_dim + action_dim
        
        # Build fully connected layers
        layers = []
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Output layer: single Q-value or V-value
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim) - optional if value_only=True
            
        Returns:
            value: Q-value or V-value (batch_size, 1)
        """
        if self.value_only:
            # V(s) network
            x = state
        else:
            # Q(s, a) network
            if action is None:
                raise ValueError("Action required for Q-network (value_only=False)")
            x = torch.cat([state, action], dim=-1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        # Value output
        value = self.value_head(x)
        
        return value
    
    def get_value(self, state: np.ndarray, action: Optional[np.ndarray] = None) -> float:
        """
        Get value for single state-action pair.
        
        Args:
            state: State as numpy array
            action: Action as numpy array (optional if value_only=True)
            
        Returns:
            value: Estimated Q-value or V-value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dim
            
            if action is not None:
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
            else:
                action_tensor = None
            
            value = self.forward(state_tensor, action_tensor)
            value_scalar = value.item()
        
        return value_scalar


class TwinCriticNetwork(nn.Module):
    """
    Twin Critic Networks for TD3 and SAC.
    
    Uses two separate Q-networks to reduce overestimation bias.
    Takes minimum of two Q-values for target calculation.
    
    Example:
        >>> twin_critic = TwinCriticNetwork(state_dim=137, action_dim=10)
        >>> state = torch.randn(32, 137)
        >>> action = torch.randn(32, 10)
        >>> q1, q2 = twin_critic(state, action)
        >>> # q1 shape: (32, 1), q2 shape: (32, 1)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        """
        Initialize Twin Critic networks.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer sizes (shared by both critics)
            dropout: Dropout probability
            activation: Activation function
        """
        super(TwinCriticNetwork, self).__init__()
        
        # Two independent critic networks
        self.critic1 = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            value_only=False
        )
        
        self.critic2 = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            value_only=False
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> tuple:
        """
        Forward pass through both critic networks.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            q1: Q-value from first critic
            q2: Q-value from second critic
        """
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first critic only (for actor loss)."""
        return self.critic1(state, action)
    
    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get minimum Q-value (for target calculation)."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
