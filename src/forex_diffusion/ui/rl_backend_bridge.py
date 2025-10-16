"""
RL Backend Bridge

Connects RLConfigTab UI to RL backend (RLPortfolioManager, RLTrainer).

Responsibilities:
- Convert UI config to backend config objects
- Start/stop training in background thread
- Forward live progress updates to UI
- Handle deployment activation/deactivation
- Load market data for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from loguru import logger
from PySide6.QtCore import QObject, Signal, QThread, Slot
from datetime import datetime

from ..rl import (
    RLPortfolioManager, RLPortfolioConfig,
    RLTrainer, TrainerConfig,
    PortfolioEnvironment, PortfolioEnvConfig,
    RewardConfig
)
from ..services.db_service import DBService


class TrainingWorker(QThread):
    """
    Background worker for RL training.
    
    Runs training in separate thread to avoid blocking UI.
    """
    
    # Signals
    progress_update = Signal(dict)  # Episode progress
    training_finished = Signal(dict)  # Final results
    training_error = Signal(str)  # Error message
    
    def __init__(
        self,
        manager: RLPortfolioManager,
        market_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.manager = manager
        self.market_data = market_data
        self.validation_data = validation_data
        self._is_running = True
    
    def run(self):
        """Run training in background thread."""
        try:
            logger.info("Training worker started")
            
            # Progress callback for live updates
            def progress_callback(metrics: Dict):
                if self._is_running:
                    self.progress_update.emit(metrics)
            
            # Run training
            results = self.manager.train(
                market_data=self.market_data,
                validation_data=self.validation_data,
                progress_callback=progress_callback
            )
            
            # Emit completion
            if self._is_running:
                self.training_finished.emit(results)
                logger.info("Training completed successfully")
        
        except Exception as e:
            logger.exception(f"Training error: {e}")
            self.training_error.emit(str(e))
    
    def stop(self):
        """Stop training."""
        self._is_running = False
        logger.info("Training worker stop requested")


class RLBackendBridge(QObject):
    """
    Bridge between RLConfigTab UI and RL backend.
    
    Handles:
    - UI config → Backend config conversion
    - Training workflow (start/stop/progress)
    - Deployment activation
    - Market data loading
    - Live updates
    
    Example:
        >>> bridge = RLBackendBridge(ui_tab, db_service)
        >>> bridge.connect_signals()
        >>> # User clicks "Start Training"
        >>> # → UI emits training_started signal
        >>> # → Bridge starts training in background
        >>> # → Live updates sent to UI
    """
    
    def __init__(self, ui_tab, db_service: DBService):
        """
        Initialize backend bridge.
        
        Args:
            ui_tab: RLConfigTab UI widget
            db_service: Database service for market data
        """
        super().__init__()
        
        self.ui_tab = ui_tab
        self.db_service = db_service
        
        # Backend components
        self.manager: Optional[RLPortfolioManager] = None
        self.training_worker: Optional[TrainingWorker] = None
        
        # State
        self.is_training = False
        self.is_deployed = False
        
        logger.info("RLBackendBridge initialized")
    
    def connect_signals(self):
        """Connect UI signals to backend handlers."""
        # Training signals
        self.ui_tab.training_started.connect(self._on_training_started)
        self.ui_tab.training_stopped.connect(self._on_training_stopped)
        
        # Deployment signals
        self.ui_tab.deployment_activated.connect(self._on_deployment_activated)
        self.ui_tab.deployment_deactivated.connect(self._on_deployment_deactivated)
        
        logger.info("Signals connected: UI ↔ Backend")
    
    @Slot(dict)
    def _on_training_started(self, ui_config: Dict):
        """
        Handle training start request from UI.
        
        Args:
            ui_config: Configuration from UI widgets
        """
        if self.is_training:
            logger.warning("Training already in progress")
            return
        
        logger.info("Training start requested")
        
        try:
            # Convert UI config to backend config
            rl_config = self._ui_config_to_rl_config(ui_config)
            
            # Create RL portfolio manager
            self.manager = RLPortfolioManager(rl_config)
            
            # Load market data
            market_data, validation_data = self._load_market_data(ui_config)
            
            if market_data is None or len(market_data) == 0:
                raise ValueError("No market data available for training")
            
            logger.info(f"Market data loaded: {len(market_data)} rows")
            
            # Create training worker
            self.training_worker = TrainingWorker(
                manager=self.manager,
                market_data=market_data,
                validation_data=validation_data
            )
            
            # Connect worker signals
            self.training_worker.progress_update.connect(self._on_training_progress)
            self.training_worker.training_finished.connect(self._on_training_finished)
            self.training_worker.training_error.connect(self._on_training_error)
            
            # Start training
            self.training_worker.start()
            self.is_training = True
            
            logger.info("Training worker started in background thread")
        
        except Exception as e:
            logger.exception(f"Failed to start training: {e}")
            self._on_training_error(str(e))
    
    @Slot()
    def _on_training_stopped(self):
        """Handle training stop request."""
        if not self.is_training or self.training_worker is None:
            return
        
        logger.info("Stopping training...")
        
        # Stop worker
        self.training_worker.stop()
        self.training_worker.wait()  # Wait for thread to finish
        
        self.is_training = False
        logger.info("Training stopped")
    
    @Slot(dict)
    def _on_training_progress(self, metrics: Dict):
        """
        Handle training progress update from worker.
        
        Args:
            metrics: Episode metrics
        """
        # Forward to UI
        episode = metrics.get('episode', 0)
        total = self.manager.config.trainer_config.num_episodes
        
        self.ui_tab.update_training_progress(episode, total, metrics)
    
    @Slot(dict)
    def _on_training_finished(self, results: Dict):
        """
        Handle training completion.
        
        Args:
            results: Training results
        """
        self.is_training = False
        
        logger.info("Training finished successfully")
        logger.info(f"Final results: {self.manager.get_training_summary()}")
        
        # Update UI status
        self.ui_tab.training_status_label.setText("✓ Training Complete")
        self.ui_tab.start_training_btn.setEnabled(True)
        self.ui_tab.pause_training_btn.setEnabled(False)
        self.ui_tab.stop_training_btn.setEnabled(False)
        self.ui_tab.save_checkpoint_btn.setEnabled(True)
        
        # Enable deployment
        self.ui_tab.activate_deployment_btn.setEnabled(True)
    
    @Slot(str)
    def _on_training_error(self, error_msg: str):
        """
        Handle training error.
        
        Args:
            error_msg: Error message
        """
        self.is_training = False
        
        logger.error(f"Training error: {error_msg}")
        
        # Update UI
        self.ui_tab.training_status_label.setText(f"✗ Error: {error_msg}")
        self.ui_tab.start_training_btn.setEnabled(True)
        self.ui_tab.pause_training_btn.setEnabled(False)
        self.ui_tab.stop_training_btn.setEnabled(False)
        
        # Show error dialog
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(
            self.ui_tab,
            "Training Error",
            f"Training failed:\n\n{error_msg}"
        )
    
    @Slot(str)
    def _on_deployment_activated(self, mode: str):
        """
        Handle deployment activation.
        
        Args:
            mode: Deployment mode ('rl_only', 'rl_riskfolio_hybrid', 'rl_advisory')
        """
        if self.manager is None or not self.manager.is_trained:
            logger.warning("Cannot activate deployment: agent not trained")
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self.ui_tab,
                "Not Trained",
                "Agent must be trained before deployment can be activated."
            )
            return
        
        try:
            # Update deployment mode in config
            self.manager.config.deployment_mode = mode
            
            # Activate deployment
            self.manager.activate_deployment()
            
            self.is_deployed = True
            logger.info(f"Deployment activated: mode={mode}")
            
            # Show confirmation
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self.ui_tab,
                "Deployment Active",
                f"RL portfolio management is now ACTIVE\n\nMode: {mode}\n\n"
                "The agent will now provide portfolio weight recommendations."
            )
        
        except Exception as e:
            logger.exception(f"Deployment activation failed: {e}")
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self.ui_tab,
                "Deployment Error",
                f"Failed to activate deployment:\n\n{e}"
            )
    
    @Slot()
    def _on_deployment_deactivated(self):
        """Handle deployment deactivation."""
        if self.manager is None:
            return
        
        try:
            self.manager.deactivate_deployment()
            self.is_deployed = False
            
            logger.info("Deployment deactivated")
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self.ui_tab,
                "Deployment Inactive",
                "RL portfolio management has been deactivated."
            )
        
        except Exception as e:
            logger.exception(f"Deployment deactivation failed: {e}")
    
    def _ui_config_to_rl_config(self, ui_config: Dict) -> RLPortfolioConfig:
        """
        Convert UI configuration to RLPortfolioConfig.
        
        Args:
            ui_config: Configuration from UI widgets
            
        Returns:
            RLPortfolioConfig object
        """
        # Extract symbols (TODO: get from portfolio tab or settings)
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
        
        # Create reward config
        reward_config = RewardConfig(
            # Weights
            sharpe_weight=ui_config['reward_weights'].get('Sharpe Ratio Improvement', 5.0),
            transaction_cost_weight=ui_config['reward_weights'].get('Transaction Costs', -10.0),
            var_violation_weight=ui_config['reward_weights'].get('VaR Violation', -20.0),
            cvar_violation_weight=ui_config['reward_weights'].get('CVaR Violation', -15.0),
            correlation_violation_weight=ui_config['reward_weights'].get('Correlation Violation', -5.0),
            diversification_weight=ui_config['reward_weights'].get('Diversification Bonus', 1.0),
            drawdown_weight=ui_config['reward_weights'].get('Drawdown Penalty', -10.0),
            turnover_weight=ui_config['reward_weights'].get('Turnover Penalty', -3.0),
            sortino_weight=ui_config['reward_weights'].get('Sortino Ratio Improvement', 3.0),
            
            # Constraints
            max_var=ui_config['max_var'],
            max_cvar=ui_config['max_cvar'],
            max_correlated_exposure=ui_config['max_correlated_exposure'],
            transaction_cost_bps=ui_config['transaction_cost_bps'],
            
            # Normalization
            normalize_rewards=ui_config['normalize_rewards'],
            reward_clip_min=ui_config['reward_clip_min'],
            reward_clip_max=ui_config['reward_clip_max'],
        )
        
        # Create environment config
        env_config = PortfolioEnvConfig(
            symbols=symbols,
            max_steps=ui_config['max_steps_per_episode'],
            min_weight=ui_config['min_weight'],
            max_weight=ui_config['max_weight'],
            long_only=ui_config['long_only'],
            force_sum_one=ui_config['force_sum_one'],
            transaction_cost_bps=ui_config['transaction_cost_bps'],
            max_var=ui_config['max_var'],
            max_cvar=ui_config['max_cvar'],
            max_correlated_exposure=ui_config['max_correlated_exposure'],
            reward_config=reward_config,
        )
        
        # Create trainer config
        trainer_config = TrainerConfig(
            num_episodes=ui_config['num_episodes'],
            max_steps_per_episode=ui_config['max_steps_per_episode'],
            eval_frequency=ui_config['eval_frequency'],
            eval_episodes=ui_config['eval_episodes'],
            checkpoint_dir=Path(ui_config['checkpoint_dir']),
            save_frequency=ui_config['save_frequency'],
            save_best_only=ui_config['save_best_only'],
            early_stopping_patience=ui_config['patience'] if ui_config['early_stopping'] else 999999,
            early_stopping_min_delta=ui_config['min_delta'],
            use_tensorboard=ui_config['use_tensorboard'],
            tensorboard_dir=Path(ui_config['tensorboard_dir']),
        )
        
        # Create RL portfolio config
        rl_config = RLPortfolioConfig(
            symbols=symbols,
            agent_type=ui_config['algorithm'],
            state_dim=137,  # Fixed for now
            actor_lr=ui_config['actor_lr'],
            critic_lr=ui_config['critic_lr'],
            hidden_dims=ui_config['actor_hidden_dims'],
            use_lstm=ui_config['use_lstm'],
            env_config=env_config,
            reward_config=reward_config,
            trainer_config=trainer_config,
            deployment_mode=ui_config.get('deployment_mode', 'rl_riskfolio_hybrid'),
            confidence_threshold=ui_config.get('confidence_threshold', 0.7),
            max_deviation_from_riskfolio=ui_config.get('max_deviation', 0.20),
            max_trades_per_day=ui_config.get('max_trades_per_day', 10),
            emergency_stop_drawdown=ui_config.get('emergency_stop_drawdown', 0.25),
        )
        
        return rl_config
    
    def _load_market_data(self, ui_config: Dict) -> tuple:
        """
        Load market data for training.
        
        Args:
            ui_config: UI configuration
            
        Returns:
            (train_data, validation_data): DataFrames
        """
        try:
            # Get symbols
            symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
            
            # Load from database
            # Use last 2 years for training, last 6 months for validation
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            train_start = end_date - timedelta(days=730)  # 2 years
            val_start = end_date - timedelta(days=180)  # 6 months
            
            # Load data for all symbols
            all_data = []
            
            for symbol in symbols:
                # Query database
                query = f"""
                    SELECT timestamp, symbol, open, high, low, close, volume
                    FROM ohlcv
                    WHERE symbol = '{symbol}'
                    AND timeframe = '1h'
                    AND timestamp >= '{train_start.strftime('%Y-%m-%d')}'
                    ORDER BY timestamp
                """
                
                try:
                    df = pd.read_sql(query, self.db_service.engine)
                    if len(df) > 0:
                        all_data.append(df)
                        logger.info(f"Loaded {len(df)} bars for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to load data for {symbol}: {e}")
            
            if not all_data:
                logger.error("No market data found in database")
                return None, None
            
            # Combine all symbols
            market_data = pd.concat(all_data, ignore_index=True)
            market_data = market_data.sort_values('timestamp').reset_index(drop=True)
            
            # Split train/validation
            val_mask = market_data['timestamp'] >= val_start.strftime('%Y-%m-%d')
            validation_data = market_data[val_mask].copy()
            train_data = market_data[~val_mask].copy()
            
            logger.info(f"Train data: {len(train_data)} rows, Validation: {len(validation_data)} rows")
            
            return train_data, validation_data
        
        except Exception as e:
            logger.exception(f"Failed to load market data: {e}")
            return None, None
    
    def get_target_weights(
        self,
        current_state: np.ndarray,
        riskfolio_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Get target portfolio weights from deployed RL agent.
        
        Args:
            current_state: Current state observation (137-dim)
            riskfolio_weights: Riskfolio optimizer weights (for hybrid mode)
            
        Returns:
            result: Dictionary with weights, confidence, mode, safety checks
        """
        if self.manager is None or not self.is_deployed:
            logger.warning("RL agent not deployed - cannot get weights")
            return None
        
        try:
            result = self.manager.get_target_weights(
                current_state=current_state,
                riskfolio_weights=riskfolio_weights
            )
            
            return result
        
        except Exception as e:
            logger.exception(f"Failed to get target weights: {e}")
            return None
