"""
RL Training Loop

Handles training of RL agents with:
- Episode management
- TensorBoard logging
- Checkpointing (best model + periodic saves)
- Early stopping
- Live metrics tracking
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available - install with: pip install tensorboard")

from .actor_critic import BaseAgent
from .environments import PortfolioEnvironment


@dataclass
class TrainerConfig:
    """Configuration for RL trainer."""
    
    # Training episodes
    num_episodes: int = 1000
    max_steps_per_episode: int = 252  # Trading days
    
    # Evaluation
    eval_frequency: int = 10  # Evaluate every N episodes
    eval_episodes: int = 5  # Number of episodes for evaluation
    
    # Checkpointing
    checkpoint_dir: Path = Path("artifacts/rl_checkpoints")
    save_frequency: int = 50  # Save checkpoint every N episodes
    save_best_only: bool = True  # Save only when performance improves
    
    # Early stopping
    early_stopping_patience: int = 100  # Stop if no improvement for N evaluations
    early_stopping_min_delta: float = 0.01  # Minimum improvement to reset patience
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: Path = Path("artifacts/rl_tensorboard")
    
    # Logging
    log_frequency: int = 1  # Log metrics every N episodes
    verbose: bool = True


class RLTrainer:
    """
    Training loop for RL agents.
    
    Features:
    - Episode-based training with environment interaction
    - Periodic evaluation on validation episodes
    - Best model checkpointing
    - TensorBoard integration
    - Early stopping based on validation performance
    - Live metrics tracking (Sharpe, returns, etc.)
    
    Example:
        >>> env = PortfolioEnvironment(market_data, config)
        >>> agent = PPOAgent(state_dim=137, action_dim=10)
        >>> trainer = RLTrainer(agent, env, config)
        >>> results = trainer.train()
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        env: PortfolioEnvironment,
        config: TrainerConfig,
        eval_env: Optional[PortfolioEnvironment] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize RL trainer.
        
        Args:
            agent: RL agent to train
            env: Training environment
            config: Trainer configuration
            eval_env: Separate environment for evaluation (optional)
            progress_callback: Callback for live progress updates (for UI)
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.eval_env = eval_env if eval_env is not None else env
        self.progress_callback = progress_callback
        
        # Create directories
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.config.tensorboard_dir))
        else:
            self.writer = None
        
        # Training state
        self.episode = 0
        self.best_eval_return = -np.inf
        self.patience_counter = 0
        self.training_history = []
        
        # Replay buffer for batch updates (stores trajectories)
        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'old_log_probs': [],
        }
        
        logger.info(f"RLTrainer initialized: {config.num_episodes} episodes, "
                   f"checkpoint_dir={config.checkpoint_dir}")
        
        if self.writer:
            logger.info(f"TensorBoard logging enabled: {config.tensorboard_dir}")
    
    def train(self) -> Dict:
        """
        Run training loop.
        
        Returns:
            results: Dictionary with training history and final metrics
        """
        logger.info("Starting RL training...")
        
        self.agent.train_mode()
        
        for episode in range(self.config.num_episodes):
            self.episode = episode
            
            # Run training episode
            episode_metrics = self._run_episode(training=True)
            
            # Log metrics
            if episode % self.config.log_frequency == 0:
                self._log_metrics(episode_metrics, prefix="train")
            
            # Update agent (if buffer has enough samples)
            if len(self.trajectory_buffer['states']) >= self.agent.mini_batch_size:
                update_metrics = self._update_agent()
                episode_metrics.update(update_metrics)
            
            # Periodic evaluation
            if episode % self.config.eval_frequency == 0:
                eval_metrics = self._evaluate()
                self._log_metrics(eval_metrics, prefix="eval")
                
                # Checkpointing
                if self._should_save_checkpoint(eval_metrics):
                    self._save_checkpoint(eval_metrics)
                
                # Early stopping check
                if self._check_early_stopping(eval_metrics):
                    logger.info(f"Early stopping triggered at episode {episode}")
                    break
            
            # Periodic checkpoint (regardless of performance)
            if episode % self.config.save_frequency == 0 and episode > 0:
                if not self.config.save_best_only:
                    self._save_checkpoint(episode_metrics, prefix=f"ep{episode}")
            
            # Progress callback for UI
            if self.progress_callback:
                self.progress_callback({
                    'episode': episode,
                    'metrics': episode_metrics,
                    'best_eval_return': self.best_eval_return,
                })
            
            # Store history
            self.training_history.append({
                'episode': episode,
                **episode_metrics
            })
        
        # Final evaluation
        logger.info("Training complete. Running final evaluation...")
        final_eval_metrics = self._evaluate(num_episodes=self.config.eval_episodes * 2)
        
        # Save final checkpoint
        self._save_checkpoint(final_eval_metrics, prefix="final")
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        # Compile results
        results = {
            'training_history': pd.DataFrame(self.training_history),
            'final_eval_metrics': final_eval_metrics,
            'best_eval_return': self.best_eval_return,
            'total_episodes': self.episode + 1,
        }
        
        logger.info(f"Training finished: {self.episode + 1} episodes, "
                   f"best_eval_return={self.best_eval_return:.2%}")
        
        return results
    
    def _run_episode(self, training: bool = True) -> Dict:
        """
        Run a single episode.
        
        Args:
            training: If True, collect experience for training
            
        Returns:
            metrics: Episode metrics
        """
        state = self.env.reset()
        done = False
        step = 0
        
        episode_reward = 0.0
        episode_actions = []
        
        while not done and step < self.config.max_steps_per_episode:
            # Select action
            action = self.agent.select_action(state, deterministic=not training)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store trajectory (if training)
            if training:
                # Get log prob of action (needed for PPO)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.agent.device)
                    log_prob, _ = self.agent.actor.evaluate_action(state_tensor, action_tensor)
                    log_prob = log_prob.item()
                
                self.trajectory_buffer['states'].append(state)
                self.trajectory_buffer['actions'].append(action)
                self.trajectory_buffer['rewards'].append(reward)
                self.trajectory_buffer['next_states'].append(next_state)
                self.trajectory_buffer['dones'].append(float(done))
                self.trajectory_buffer['old_log_probs'].append(log_prob)
            
            episode_reward += reward
            episode_actions.append(action)
            state = next_state
            step += 1
        
        # Episode metrics
        metrics = {
            'episode_return': info.get('episode', {}).get('total_return', 
                (self.env.portfolio_value - self.env.config.initial_capital) / self.env.config.initial_capital),
            'episode_reward': episode_reward,
            'episode_length': step,
            'sharpe_ratio': info.get('episode', {}).get('sharpe_ratio', 0.0),
            'sortino_ratio': info.get('episode', {}).get('sortino_ratio', 0.0),
            'max_drawdown': info.get('episode', {}).get('max_drawdown', 0.0),
            'transaction_costs': info.get('episode', {}).get('total_transaction_costs', 0.0),
            'portfolio_value': self.env.portfolio_value,
        }
        
        return metrics
    
    def _update_agent(self) -> Dict:
        """
        Update agent using collected trajectories.
        
        Returns:
            update_metrics: Training metrics from update
        """
        # Convert buffer to batch
        batch = {
            'states': np.array(self.trajectory_buffer['states']),
            'actions': np.array(self.trajectory_buffer['actions']),
            'rewards': np.array(self.trajectory_buffer['rewards']),
            'next_states': np.array(self.trajectory_buffer['next_states']),
            'dones': np.array(self.trajectory_buffer['dones']),
            'old_log_probs': np.array(self.trajectory_buffer['old_log_probs']),
        }
        
        # Update agent
        update_metrics = self.agent.update(batch)
        
        # Clear buffer
        for key in self.trajectory_buffer:
            self.trajectory_buffer[key] = []
        
        return update_metrics
    
    def _evaluate(self, num_episodes: Optional[int] = None) -> Dict:
        """
        Evaluate agent on validation episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate (default: config value)
            
        Returns:
            eval_metrics: Average metrics across episodes
        """
        if num_episodes is None:
            num_episodes = self.config.eval_episodes
        
        self.agent.eval_mode()
        
        eval_returns = []
        eval_sharpes = []
        eval_sortinos = []
        eval_drawdowns = []
        
        for _ in range(num_episodes):
            metrics = self._run_episode(training=False)
            eval_returns.append(metrics['episode_return'])
            eval_sharpes.append(metrics['sharpe_ratio'])
            eval_sortinos.append(metrics['sortino_ratio'])
            eval_drawdowns.append(metrics['max_drawdown'])
        
        self.agent.train_mode()
        
        eval_metrics = {
            'eval_mean_return': np.mean(eval_returns),
            'eval_std_return': np.std(eval_returns),
            'eval_mean_sharpe': np.mean(eval_sharpes),
            'eval_mean_sortino': np.mean(eval_sortinos),
            'eval_mean_drawdown': np.mean(eval_drawdowns),
        }
        
        return eval_metrics
    
    def _should_save_checkpoint(self, eval_metrics: Dict) -> bool:
        """Check if checkpoint should be saved."""
        if not self.config.save_best_only:
            return True
        
        current_return = eval_metrics['eval_mean_return']
        
        if current_return > self.best_eval_return + self.config.early_stopping_min_delta:
            self.best_eval_return = current_return
            return True
        
        return False
    
    def _save_checkpoint(self, metrics: Dict, prefix: str = "best"):
        """
        Save agent checkpoint.
        
        Args:
            metrics: Current metrics
            prefix: Checkpoint filename prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.config.checkpoint_dir / f"{prefix}_{timestamp}.pt"
        
        self.agent.save(checkpoint_path)
        
        # Save metrics alongside checkpoint
        metrics_path = checkpoint_path.with_suffix('.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _check_early_stopping(self, eval_metrics: Dict) -> bool:
        """
        Check if early stopping criteria met.
        
        Returns:
            should_stop: True if training should stop
        """
        current_return = eval_metrics['eval_mean_return']
        
        if current_return > self.best_eval_return + self.config.early_stopping_min_delta:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.early_stopping_patience:
            return True
        
        return False
    
    def _log_metrics(self, metrics: Dict, prefix: str = ""):
        """
        Log metrics to console and TensorBoard.
        
        Args:
            metrics: Metrics dictionary
            prefix: Prefix for metric names (e.g., "train", "eval")
        """
        # Console logging
        if self.config.verbose:
            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                    for k, v in metrics.items()])
            logger.info(f"Episode {self.episode} [{prefix}]: {metrics_str}")
        
        # TensorBoard logging
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}/{key}", value, self.episode)
    
    def get_training_history(self) -> pd.DataFrame:
        """Get training history as DataFrame."""
        return pd.DataFrame(self.training_history)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load agent from checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.agent.load(checkpoint_path)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        # Try to load associated metrics
        metrics_path = checkpoint_path.with_suffix('.json')
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            if 'eval_mean_return' in metrics:
                self.best_eval_return = metrics['eval_mean_return']
                logger.info(f"Resuming with best_eval_return={self.best_eval_return:.2%}")
