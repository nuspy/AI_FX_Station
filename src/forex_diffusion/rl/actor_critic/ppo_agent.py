"""
PPO Agent (Proximal Policy Optimization)

State-of-the-art policy gradient algorithm with:
- Clipped surrogate objective (prevents large policy updates)
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy regularization

Reference: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from loguru import logger

from .base_agent import BaseAgent
from ..networks import ActorNetwork, CriticNetwork


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent.
    
    PPO is a policy gradient method that improves sample efficiency and stability
    by constraining policy updates to a "trust region" using a clipped objective.
    
    Key Features:
    - Clipped surrogate objective: L_CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
    - Generalized Advantage Estimation (GAE) for variance reduction
    - Value function baseline for advantage estimation
    - Multiple epochs on same batch (improves sample efficiency)
    
    Example:
        >>> agent = PPOAgent(
        ...     state_dim=137,
        ...     action_dim=10,
        ...     actor_lr=3e-4,
        ...     critic_lr=1e-3,
        ...     clip_epsilon=0.2,
        ...     gae_lambda=0.95,
        ... )
        >>> action = agent.select_action(state)
        >>> metrics = agent.update(batch)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = 'cpu',
        # Network architecture
        actor_hidden_dims: list = [256, 128],
        critic_hidden_dims: list = [256, 128],
        use_lstm: bool = True,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
        # PPO hyperparameters
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64,
        # Exploration
        noise_scale: float = 0.1,
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for training
            actor_hidden_dims: Hidden layer sizes for actor network
            critic_hidden_dims: Hidden layer sizes for critic network
            use_lstm: Whether to use LSTM in actor
            lstm_hidden: LSTM hidden size
            dropout: Dropout probability
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            clip_epsilon: Clipping parameter for PPO objective (typically 0.1-0.3)
            gae_lambda: GAE lambda parameter (0.9-0.99)
            gamma: Discount factor
            entropy_coef: Entropy bonus coefficient (encourages exploration)
            value_loss_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of epochs to update on same batch
            mini_batch_size: Mini-batch size for SGD updates
            noise_scale: Exploration noise scale
        """
        super().__init__(state_dim, action_dim, device)
        
        # Hyperparameters
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.noise_scale = noise_scale
        
        # Actor network (policy)
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            use_lstm=use_lstm,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
        ).to(self.device)
        
        # Critic network (value function)
        self.critic = CriticNetwork(
            state_dim=state_dim,
            action_dim=0,  # Not used (value_only=True)
            hidden_dims=critic_hidden_dims,
            dropout=dropout,
            value_only=True,  # V(s) network
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Training statistics
        self.train_step = 0
        
        logger.info(f"PPO Agent initialized: state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"Hyperparameters: clip_ε={clip_epsilon}, GAE_λ={gae_lambda}, γ={gamma}")
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: If True, select mean action (no exploration)
            
        Returns:
            action: Selected action (portfolio weights)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                # Deterministic action (mean of policy)
                action, _ = self.actor.forward(state_tensor)
                action = action.squeeze(0).cpu().numpy()
            else:
                # Stochastic action (with exploration noise)
                action, _, _ = self.actor.sample_action(
                    state_tensor,
                    noise_scale=self.noise_scale,
                    deterministic=False
                )
                action = action.squeeze(0).cpu().numpy()
        
        return action
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update PPO agent using batch of experience.
        
        Args:
            batch: Dictionary with keys:
                - 'states': (batch_size, state_dim)
                - 'actions': (batch_size, action_dim)
                - 'rewards': (batch_size,)
                - 'next_states': (batch_size, state_dim)
                - 'dones': (batch_size,)
                - 'old_log_probs': (batch_size,) - log probs from behavior policy
            
        Returns:
            metrics: Training metrics (losses, etc.)
        """
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
        
        # Compute advantages using GAE
        with torch.no_grad():
            advantages, returns = self._compute_gae(
                states, rewards, next_states, dones
            )
        
        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update: multiple epochs on same batch
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        batch_size = states.size(0)
        num_batches = max(1, batch_size // self.mini_batch_size)
        
        for epoch in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)
            
            for i in range(num_batches):
                # Mini-batch indices
                start_idx = i * self.mini_batch_size
                end_idx = min((i + 1) * self.mini_batch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Evaluate actions with current policy
                log_probs, entropy = self.actor.evaluate_action(mb_states, mb_actions)
                
                # Importance sampling ratio: π_new / π_old
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total actor loss
                total_actor_loss_mb = actor_loss + self.entropy_coef * entropy_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                total_actor_loss_mb.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic loss (MSE between V(s) and returns)
                values = self.critic(mb_states).squeeze(-1)
                critic_loss = nn.MSELoss()(values, mb_returns)
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Accumulate metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        # Average metrics
        total_updates = self.ppo_epochs * num_batches
        avg_actor_loss = total_actor_loss / total_updates
        avg_critic_loss = total_critic_loss / total_updates
        avg_entropy = total_entropy / total_updates
        
        self.train_step += 1
        
        metrics = {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy,
            'train_step': self.train_step,
        }
        
        return metrics
    
    def _compute_gae(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE balances bias-variance tradeoff in advantage estimation:
        - λ=0: Low variance, high bias (1-step TD)
        - λ=1: High variance, low bias (Monte Carlo)
        - λ=0.95: Good balance (typical)
        
        Formula: A_t = Σ(γλ)^l * δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        Args:
            states: States
            rewards: Rewards
            next_states: Next states
            dones: Episode done flags
            
        Returns:
            advantages: Advantage estimates
            returns: Target returns for value function
        """
        with torch.no_grad():
            # Get value estimates
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            
            # Compute TD errors: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            
            # Compute GAE advantages
            batch_size = states.size(0)
            advantages = torch.zeros(batch_size, device=self.device)
            
            gae = 0.0
            for t in reversed(range(batch_size)):
                if dones[t]:
                    gae = 0.0  # Reset GAE at episode boundaries
                
                gae = deltas[t] + self.gamma * self.gae_lambda * gae
                advantages[t] = gae
            
            # Returns for value function training: G_t = A_t + V(s_t)
            returns = advantages + values
        
        return advantages, returns
    
    def save(self, path: Path):
        """
        Save PPO agent checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'train_step': self.train_step,
            'hyperparameters': {
                'clip_epsilon': self.clip_epsilon,
                'gae_lambda': self.gae_lambda,
                'gamma': self.gamma,
                'entropy_coef': self.entropy_coef,
                'value_loss_coef': self.value_loss_coef,
                'max_grad_norm': self.max_grad_norm,
                'ppo_epochs': self.ppo_epochs,
                'mini_batch_size': self.mini_batch_size,
                'noise_scale': self.noise_scale,
            },
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"PPO checkpoint saved to {path}")
    
    def load(self, path: Path):
        """
        Load PPO agent checkpoint.
        
        Args:
            path: Path to load checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.train_step = checkpoint['train_step']
        
        # Load hyperparameters
        for key, value in checkpoint['hyperparameters'].items():
            setattr(self, key, value)
        
        logger.info(f"PPO checkpoint loaded from {path}")
    
    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
    
    def train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()
