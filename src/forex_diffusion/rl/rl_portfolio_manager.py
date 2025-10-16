"""
RL Portfolio Manager

High-level interface for RL-based portfolio management.

Integrates:
- RL agents (PPO, SAC, TD3, A3C)
- PortfolioEnvironment
- Riskfolio-Lib optimizer (hybrid mode)
- Trading Engine
- Training and deployment workflows

Deployment Modes:
1. RL Only: Pure RL-based portfolio weights
2. RL + Riskfolio Hybrid: RL suggests, Riskfolio constraints
3. RL Advisory: RL provides signals, final decision elsewhere
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Literal
from dataclasses import dataclass
from loguru import logger

from .actor_critic import PPOAgent, BaseAgent
from .environments import PortfolioEnvironment, PortfolioEnvConfig
from .rewards import RewardConfig
from .trainer import RLTrainer, TrainerConfig


DeploymentMode = Literal['rl_only', 'rl_riskfolio_hybrid', 'rl_advisory']


@dataclass
class RLPortfolioConfig:
    """Configuration for RL portfolio manager."""
    
    # Assets
    symbols: List[str] = None
    
    # RL agent settings
    agent_type: str = 'ppo'  # 'ppo', 'sac', 'td3', 'a3c'
    state_dim: int = 137
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    hidden_dims: List[int] = None
    use_lstm: bool = True
    
    # Environment settings
    env_config: Optional[PortfolioEnvConfig] = None
    reward_config: Optional[RewardConfig] = None
    
    # Training settings
    trainer_config: Optional[TrainerConfig] = None
    
    # Deployment settings
    deployment_mode: DeploymentMode = 'rl_riskfolio_hybrid'
    confidence_threshold: float = 0.7  # Min confidence for RL-only mode
    max_deviation_from_riskfolio: float = 0.20  # Max 20% weight deviation
    
    # Safety limits
    max_trades_per_day: int = 10
    emergency_stop_drawdown: float = 0.25  # Stop if 25% drawdown
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]
        if self.env_config is None:
            self.env_config = PortfolioEnvConfig(symbols=self.symbols)
        if self.reward_config is None:
            self.reward_config = RewardConfig()
        if self.trainer_config is None:
            self.trainer_config = TrainerConfig()


class RLPortfolioManager:
    """
    High-level manager for RL-based portfolio optimization.
    
    Handles:
    - Agent initialization (PPO/SAC/TD3/A3C)
    - Training workflow
    - Deployment with safety checks
    - Hybrid RL + Riskfolio integration
    - Live portfolio weight generation
    
    Example:
        >>> config = RLPortfolioConfig(
        ...     symbols=['EUR/USD', 'GBP/USD', 'USD/JPY'],
        ...     deployment_mode='rl_riskfolio_hybrid'
        ... )
        >>> manager = RLPortfolioManager(config)
        >>> manager.train(market_data)
        >>> weights = manager.get_target_weights(current_state)
    """
    
    def __init__(self, config: RLPortfolioConfig):
        """
        Initialize RL portfolio manager.
        
        Args:
            config: RL portfolio configuration
        """
        self.config = config
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Training state
        self.is_trained = False
        self.training_history = None
        self.best_checkpoint_path = None
        
        # Deployment state
        self.deployment_active = False
        self.daily_trades_count = 0
        self.peak_portfolio_value = None
        
        logger.info(f"RLPortfolioManager initialized: agent={config.agent_type}, "
                   f"mode={config.deployment_mode}")
    
    def _create_agent(self) -> BaseAgent:
        """Create RL agent based on config."""
        action_dim = len(self.config.symbols) if self.config.symbols else 10
        
        if self.config.agent_type == 'ppo':
            agent = PPOAgent(
                state_dim=self.config.state_dim,
                action_dim=action_dim,
                actor_lr=self.config.actor_lr,
                critic_lr=self.config.critic_lr,
                actor_hidden_dims=self.config.hidden_dims,
                critic_hidden_dims=self.config.hidden_dims,
                use_lstm=self.config.use_lstm,
            )
        elif self.config.agent_type == 'sac':
            # TODO: Implement SAC agent
            raise NotImplementedError("SAC agent not yet implemented")
        elif self.config.agent_type == 'td3':
            # TODO: Implement TD3 agent
            raise NotImplementedError("TD3 agent not yet implemented")
        elif self.config.agent_type == 'a3c':
            # TODO: Implement A3C agent
            raise NotImplementedError("A3C agent not yet implemented")
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")
        
        return agent
    
    def train(
        self,
        market_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        resume_from_checkpoint: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Train RL agent on market data.
        
        Args:
            market_data: Training market data (OHLCV + symbol)
            validation_data: Validation market data (optional)
            resume_from_checkpoint: Path to checkpoint to resume from
            progress_callback: Callback for live training updates (for UI)
            
        Returns:
            results: Training results and metrics
        """
        logger.info("Starting RL training...")
        
        # Create training environment
        train_env = PortfolioEnvironment(
            market_data=market_data,
            config=self.config.env_config
        )
        
        # Create validation environment (if provided)
        eval_env = None
        if validation_data is not None:
            eval_env = PortfolioEnvironment(
                market_data=validation_data,
                config=self.config.env_config
            )
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            self.agent.load(resume_from_checkpoint)
        
        # Create trainer
        trainer = RLTrainer(
            agent=self.agent,
            env=train_env,
            config=self.config.trainer_config,
            eval_env=eval_env,
            progress_callback=progress_callback,
        )
        
        # Run training
        results = trainer.train()
        
        # Update state
        self.is_trained = True
        self.training_history = results['training_history']
        
        # Find best checkpoint
        best_checkpoints = sorted(
            self.config.trainer_config.checkpoint_dir.glob("best_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if best_checkpoints:
            self.best_checkpoint_path = best_checkpoints[0]
            logger.info(f"Best checkpoint: {self.best_checkpoint_path}")
        
        logger.info("Training complete!")
        
        return results
    
    def get_target_weights(
        self,
        current_state: np.ndarray,
        riskfolio_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Get target portfolio weights from RL agent.
        
        Args:
            current_state: Current state observation (137-dim)
            riskfolio_weights: Riskfolio optimizer weights (for hybrid mode)
            
        Returns:
            result: Dictionary with:
                - 'weights': Target weights
                - 'confidence': Agent confidence (entropy-based)
                - 'mode_used': Which deployment mode was used
                - 'safety_checks': Safety check results
        """
        if not self.is_trained:
            raise RuntimeError("Agent not trained. Call train() first.")
        
        # Get RL weights (deterministic)
        self.agent.eval_mode()
        rl_weights = self.agent.select_action(current_state, deterministic=True)
        
        # Calculate confidence (based on entropy)
        confidence = self._calculate_confidence(current_state, rl_weights)
        
        # Apply deployment mode
        if self.config.deployment_mode == 'rl_only':
            if confidence >= self.config.confidence_threshold:
                final_weights = rl_weights
                mode_used = 'rl_only'
            else:
                # Low confidence, fallback to Riskfolio
                final_weights = riskfolio_weights if riskfolio_weights is not None else rl_weights
                mode_used = 'riskfolio_fallback'
        
        elif self.config.deployment_mode == 'rl_riskfolio_hybrid':
            if riskfolio_weights is None:
                logger.warning("Hybrid mode requires riskfolio_weights, using RL only")
                final_weights = rl_weights
                mode_used = 'rl_only'
            else:
                # Blend RL and Riskfolio with deviation constraint
                final_weights = self._hybrid_blend(rl_weights, riskfolio_weights)
                mode_used = 'rl_riskfolio_hybrid'
        
        elif self.config.deployment_mode == 'rl_advisory':
            # RL provides advisory only, return both
            final_weights = riskfolio_weights if riskfolio_weights is not None else rl_weights
            mode_used = 'rl_advisory'
        
        else:
            raise ValueError(f"Unknown deployment mode: {self.config.deployment_mode}")
        
        # Safety checks
        safety_checks = self._run_safety_checks(final_weights)
        
        result = {
            'weights': final_weights,
            'rl_weights': rl_weights,
            'confidence': confidence,
            'mode_used': mode_used,
            'safety_checks': safety_checks,
        }
        
        return result
    
    def _calculate_confidence(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Calculate agent confidence in action.
        
        Uses entropy of action distribution:
        - High entropy (uncertain) → Low confidence
        - Low entropy (certain) → High confidence
        
        Args:
            state: State observation
            action: Selected action
            
        Returns:
            confidence: Confidence score [0, 1]
        """
        with logger.catch():
            import torch
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.agent.device)
            
            _, entropy = self.agent.actor.evaluate_action(state_tensor, action_tensor)
            entropy = entropy.item()
            
            # Convert entropy to confidence
            # Max entropy for uniform distribution: log(n_assets)
            max_entropy = np.log(len(self.config.symbols))
            confidence = 1.0 - (entropy / max_entropy)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return confidence
        
        # Fallback if error
        return 0.5
    
    def _hybrid_blend(
        self,
        rl_weights: np.ndarray,
        riskfolio_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Blend RL and Riskfolio weights with deviation constraint.
        
        Strategy:
        - Use RL weights as base
        - Constrain deviation from Riskfolio to max_deviation
        - Ensures RL doesn't deviate too far from risk-optimized weights
        
        Args:
            rl_weights: RL agent weights
            riskfolio_weights: Riskfolio optimizer weights
            
        Returns:
            blended_weights: Constrained weights
        """
        # Calculate deviation
        deviation = np.abs(rl_weights - riskfolio_weights)
        
        # If deviation exceeds limit, pull RL weights towards Riskfolio
        if np.max(deviation) > self.config.max_deviation_from_riskfolio:
            # Weighted average (more weight to Riskfolio when deviation high)
            alpha = 0.6  # 60% Riskfolio, 40% RL when constrained
            blended = alpha * riskfolio_weights + (1 - alpha) * rl_weights
            
            # Re-normalize
            blended = blended / np.sum(blended)
        else:
            # Use RL weights directly
            blended = rl_weights
        
        return blended
    
    def _run_safety_checks(self, weights: np.ndarray) -> Dict:
        """
        Run safety checks on proposed weights.
        
        Args:
            weights: Proposed portfolio weights
            
        Returns:
            safety_results: Dictionary of safety check results
        """
        checks = {}
        
        # Check 1: Weights sum to 1.0
        checks['weights_sum_valid'] = np.abs(np.sum(weights) - 1.0) < 0.01
        
        # Check 2: No negative weights (if long-only)
        if self.config.env_config.long_only:
            checks['no_shorts'] = np.all(weights >= 0.0)
        else:
            checks['no_shorts'] = True
        
        # Check 3: Max weight constraint
        checks['max_weight_ok'] = np.all(weights <= self.config.env_config.max_weight + 0.01)
        
        # Check 4: Daily trade limit
        self.daily_trades_count += 1
        checks['daily_limit_ok'] = self.daily_trades_count <= self.config.max_trades_per_day
        
        # Check 5: Emergency stop (drawdown check)
        if self.peak_portfolio_value is not None:
            # This would need current portfolio value from environment
            checks['emergency_stop_ok'] = True  # Placeholder
        else:
            checks['emergency_stop_ok'] = True
        
        # Overall safety
        checks['all_passed'] = all(checks.values())
        
        return checks
    
    def reset_daily_counters(self):
        """Reset daily trade counter (call at start of each trading day)."""
        self.daily_trades_count = 0
        logger.debug("Daily trade counter reset")
    
    def activate_deployment(self):
        """Activate deployment mode."""
        if not self.is_trained:
            raise RuntimeError("Cannot activate deployment without training agent first")
        
        self.deployment_active = True
        self.agent.eval_mode()
        logger.info("RL deployment activated")
    
    def deactivate_deployment(self):
        """Deactivate deployment mode."""
        self.deployment_active = False
        logger.info("RL deployment deactivated")
    
    def save_agent(self, path: Path):
        """Save agent to file."""
        self.agent.save(path)
        logger.info(f"Agent saved to {path}")
    
    def load_agent(self, path: Path):
        """Load agent from file."""
        self.agent.load(path)
        self.is_trained = True
        logger.info(f"Agent loaded from {path}")
    
    def get_training_summary(self) -> Dict:
        """Get summary of training results."""
        if self.training_history is None:
            return {'status': 'not_trained'}
        
        summary = {
            'status': 'trained',
            'total_episodes': len(self.training_history),
            'final_sharpe': self.training_history['sharpe_ratio'].iloc[-1],
            'best_sharpe': self.training_history['sharpe_ratio'].max(),
            'final_return': self.training_history['episode_return'].iloc[-1],
            'best_return': self.training_history['episode_return'].max(),
            'avg_return': self.training_history['episode_return'].mean(),
            'best_checkpoint': str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
        }
        
        return summary
