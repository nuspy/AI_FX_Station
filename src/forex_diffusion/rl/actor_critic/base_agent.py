"""
Base Agent Interface

Abstract base class for all RL agents.
Defines common interface and utilities.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.
    
    All RL agents (PPO, SAC, A3C, TD3) must implement this interface.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for training ('cpu' or 'cuda')
            **kwargs: Algorithm-specific parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        logger.info(f"Initializing {self.__class__.__name__} on device: {self.device}")
    
    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action given current state.
        
        Args:
            state: Current state
            deterministic: If True, select mean action (no exploration)
            
        Returns:
            action: Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent parameters using batch of experience.
        
        Args:
            batch: Dictionary containing 'states', 'actions', 'rewards', 'next_states', 'dones'
            
        Returns:
            metrics: Dictionary of training metrics (losses, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """
        Save agent checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """
        Load agent checkpoint.
        
        Args:
            path: Path to load checkpoint
        """
        pass
    
    def to(self, device: str):
        """Move agent to device."""
        self.device = torch.device(device)
        logger.info(f"Agent moved to device: {self.device}")
    
    def eval_mode(self):
        """Set agent to evaluation mode (disable dropout, etc.)."""
        pass
    
    def train_mode(self):
        """Set agent to training mode (enable dropout, etc.)."""
        pass
