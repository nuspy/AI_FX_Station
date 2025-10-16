"""
Actor Network (Policy Network)

Maps state → action distribution (portfolio weights)

Architecture:
- Input: State (137-dim)
- Hidden: [256, 128] fully connected + LSTM(64)
- Output: Action logits + Softmax → Portfolio weights

Exploration: Gaussian noise added to logits during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ActorNetwork(nn.Module):
    """
    Policy network for portfolio weight selection.
    
    Outputs continuous action distribution (portfolio weights).
    Uses softmax to ensure weights sum to 1.0.
    
    Example:
        >>> actor = ActorNetwork(state_dim=137, action_dim=10, hidden_dims=[256, 128])
        >>> state = torch.randn(32, 137)  # Batch of 32 states
        >>> action, log_prob = actor(state)
        >>> # action shape: (32, 10), log_prob shape: (32,)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128],
        use_lstm: bool = True,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        """
        Initialize Actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of assets)
            hidden_dims: List of hidden layer sizes
            use_lstm: Whether to use LSTM layer
            lstm_hidden: LSTM hidden size
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm
        self.lstm_hidden = lstm_hidden
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build fully connected layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*layers)
        
        # LSTM layer (optional)
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                dropout=0.0  # No dropout for single-layer LSTM
            )
            lstm_output_dim = lstm_hidden
        else:
            self.lstm = None
            lstm_output_dim = input_dim
        
        # Output layer: action logits
        self.action_head = nn.Linear(lstm_output_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through actor network.
        
        Args:
            state: State tensor (batch_size, state_dim) or (batch_size, seq_len, state_dim) for LSTM
            hidden_state: LSTM hidden state (h, c) if using LSTM
            
        Returns:
            action: Portfolio weights (batch_size, action_dim) summing to 1.0
            hidden_state: Updated LSTM hidden state (if using LSTM)
        """
        # Fully connected layers
        x = self.fc_layers(state)
        
        # LSTM layer (if enabled)
        if self.use_lstm:
            # Reshape to (batch_size, seq_len=1, hidden_dim) if needed
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            if hidden_state is None:
                # Initialize hidden state
                batch_size = x.size(0)
                h0 = torch.zeros(1, batch_size, self.lstm_hidden, device=x.device)
                c0 = torch.zeros(1, batch_size, self.lstm_hidden, device=x.device)
                hidden_state = (h0, c0)
            
            x, hidden_state = self.lstm(x, hidden_state)
            x = x[:, -1, :]  # Take last time step
        
        # Action logits
        logits = self.action_head(x)
        
        # Softmax to get valid portfolio weights (sum to 1.0)
        action = F.softmax(logits, dim=-1)
        
        return action, hidden_state
    
    def sample_action(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        noise_scale: float = 0.1,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Sample action with optional exploration noise.
        
        Args:
            state: State tensor
            hidden_state: LSTM hidden state
            noise_scale: Scale of Gaussian noise for exploration
            deterministic: If True, return mean action without noise
            
        Returns:
            action: Sampled action (portfolio weights)
            log_prob: Log probability of action
            hidden_state: Updated LSTM hidden state
        """
        # Get action distribution
        mean_action, hidden_state = self.forward(state, hidden_state)
        
        if deterministic:
            action = mean_action
            # Log prob not needed for deterministic
            log_prob = torch.zeros(action.size(0), device=action.device)
        else:
            # Add Gaussian noise to logits for exploration
            logits = torch.log(mean_action + 1e-8)  # Convert back to logits
            noise = torch.randn_like(logits) * noise_scale
            noisy_logits = logits + noise
            
            # Re-normalize
            action = F.softmax(noisy_logits, dim=-1)
            
            # Calculate log probability
            # For categorical distribution: log P(a) = sum(a_i * log(p_i))
            log_prob = torch.sum(action * torch.log(mean_action + 1e-8), dim=-1)
        
        return action, log_prob, hidden_state
    
    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given action.
        
        Used during PPO training to compute importance sampling ratios.
        
        Args:
            state: State tensor
            action: Action tensor (portfolio weights)
            hidden_state: LSTM hidden state
            
        Returns:
            log_prob: Log probability of action
            entropy: Entropy of action distribution
        """
        # Get action distribution
        action_probs, _ = self.forward(state, hidden_state)
        
        # Log probability
        log_prob = torch.sum(action * torch.log(action_probs + 1e-8), dim=-1)
        
        # Entropy: H = -sum(p * log(p))
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        
        return log_prob, entropy
    
    def get_action_deterministic(self, state: np.ndarray) -> np.ndarray:
        """
        Get deterministic action for deployment (no exploration).
        
        Args:
            state: State as numpy array
            
        Returns:
            action: Portfolio weights as numpy array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dim
            action, _ = self.forward(state_tensor)
            action_np = action.squeeze(0).cpu().numpy()
        
        return action_np
