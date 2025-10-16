"""
RL Agent Configuration Tab

6 Level-3 Sub-tabs:
1. Agent Configuration - Algorithm selection, hyperparameters, network architecture
2. Training Settings - Episodes, batch size, checkpointing, TensorBoard
3. State & Action Space - Feature selection, action constraints
4. Reward Function - 9 component weights and configuration
5. Training Progress - Live training metrics, charts, best models
6. Deployment & Testing - Backtest, production deployment, safety limits
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QLineEdit, QTextEdit, QTableWidget,
    QTableWidgetItem, QProgressBar, QFormLayout, QGridLayout,
    QSlider, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from loguru import logger
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available for charts")


class RLConfigTab(QWidget):
    """
    Main RL configuration tab with 6 sub-tabs.
    
    Integrates:
    - RL Agent (PPO/SAC/TD3/A3C)
    - Riskfolio-Lib optimizer
    - Trading Engine
    
    Provides complete workflow:
    - Configure → Train → Evaluate → Deploy
    """
    
    # Signals for backend integration
    training_started = Signal(dict)  # config dict
    training_stopped = Signal()
    deployment_activated = Signal(str)  # mode: 'rl_only', 'hybrid', 'advisory'
    deployment_deactivated = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("RL Agent Configuration")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        self.layout.addWidget(title)
        
        # Main tab widget (6 sub-tabs)
        self.tabs = QTabWidget()
        
        # Create sub-tabs
        self.agent_config_tab = self._create_agent_config_tab()
        self.training_settings_tab = self._create_training_settings_tab()
        self.state_action_tab = self._create_state_action_tab()
        self.reward_function_tab = self._create_reward_function_tab()
        self.training_progress_tab = self._create_training_progress_tab()
        self.deployment_tab = self._create_deployment_tab()
        
        # Add tabs
        self.tabs.addTab(self.agent_config_tab, "1. Agent Config")
        self.tabs.addTab(self.training_settings_tab, "2. Training")
        self.tabs.addTab(self.state_action_tab, "3. State/Action")
        self.tabs.addTab(self.reward_function_tab, "4. Reward")
        self.tabs.addTab(self.training_progress_tab, "5. Progress")
        self.tabs.addTab(self.deployment_tab, "6. Deployment")
        
        self.layout.addWidget(self.tabs)
        
        # Training state
        self.is_training = False
        self.training_history = []
        
        # Timer for live updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_live_metrics)
        
        # Apply i18n tooltips
        self._apply_i18n_tooltips()
        
        logger.info("RLConfigTab initialized with 6 sub-tabs")
    
    def _create_agent_config_tab(self) -> QWidget:
        """
        Tab 1: Agent Configuration
        
        Widgets: 25
        - Algorithm selection (PPO/SAC/TD3/A3C)
        - Hyperparameters (learning rates, clip epsilon, etc.)
        - Network architecture (hidden layers, LSTM, dropout)
        - Optimizer settings
        - Device selection (CPU/GPU)
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Algorithm Selection
        algo_group = QGroupBox("Algorithm")
        algo_layout = QFormLayout(algo_group)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["PPO", "SAC", "TD3", "A3C"])
        self.algorithm_combo.setCurrentText("PPO")
        algo_layout.addRow("Algorithm:", self.algorithm_combo)
        
        layout.addWidget(algo_group)
        
        # PPO Hyperparameters
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QGridLayout(hyper_group)
        
        # Actor LR
        hyper_layout.addWidget(QLabel("Actor LR:"), 0, 0)
        self.actor_lr_spin = QDoubleSpinBox()
        self.actor_lr_spin.setRange(1e-6, 1e-2)
        self.actor_lr_spin.setDecimals(6)
        self.actor_lr_spin.setSingleStep(1e-5)
        self.actor_lr_spin.setValue(3e-4)
        hyper_layout.addWidget(self.actor_lr_spin, 0, 1)
        
        # Critic LR
        hyper_layout.addWidget(QLabel("Critic LR:"), 0, 2)
        self.critic_lr_spin = QDoubleSpinBox()
        self.critic_lr_spin.setRange(1e-6, 1e-2)
        self.critic_lr_spin.setDecimals(6)
        self.critic_lr_spin.setSingleStep(1e-5)
        self.critic_lr_spin.setValue(1e-3)
        hyper_layout.addWidget(self.critic_lr_spin, 0, 3)
        
        # Clip Epsilon (PPO)
        hyper_layout.addWidget(QLabel("Clip ε:"), 1, 0)
        self.clip_epsilon_spin = QDoubleSpinBox()
        self.clip_epsilon_spin.setRange(0.05, 0.5)
        self.clip_epsilon_spin.setDecimals(2)
        self.clip_epsilon_spin.setSingleStep(0.05)
        self.clip_epsilon_spin.setValue(0.2)
        hyper_layout.addWidget(self.clip_epsilon_spin, 1, 1)
        
        # GAE Lambda
        hyper_layout.addWidget(QLabel("GAE λ:"), 1, 2)
        self.gae_lambda_spin = QDoubleSpinBox()
        self.gae_lambda_spin.setRange(0.8, 0.99)
        self.gae_lambda_spin.setDecimals(2)
        self.gae_lambda_spin.setSingleStep(0.05)
        self.gae_lambda_spin.setValue(0.95)
        hyper_layout.addWidget(self.gae_lambda_spin, 1, 3)
        
        # Discount Gamma
        hyper_layout.addWidget(QLabel("Discount γ:"), 2, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.9, 0.999)
        self.gamma_spin.setDecimals(3)
        self.gamma_spin.setSingleStep(0.01)
        self.gamma_spin.setValue(0.99)
        hyper_layout.addWidget(self.gamma_spin, 2, 1)
        
        # Entropy Coefficient
        hyper_layout.addWidget(QLabel("Entropy Coef:"), 2, 2)
        self.entropy_coef_spin = QDoubleSpinBox()
        self.entropy_coef_spin.setRange(0.0, 0.1)
        self.entropy_coef_spin.setDecimals(3)
        self.entropy_coef_spin.setSingleStep(0.005)
        self.entropy_coef_spin.setValue(0.01)
        hyper_layout.addWidget(self.entropy_coef_spin, 2, 3)
        
        # PPO Epochs
        hyper_layout.addWidget(QLabel("PPO Epochs:"), 3, 0)
        self.ppo_epochs_spin = QSpinBox()
        self.ppo_epochs_spin.setRange(1, 50)
        self.ppo_epochs_spin.setValue(10)
        hyper_layout.addWidget(self.ppo_epochs_spin, 3, 1)
        
        # Mini Batch Size
        hyper_layout.addWidget(QLabel("Mini-Batch Size:"), 3, 2)
        self.mini_batch_spin = QSpinBox()
        self.mini_batch_spin.setRange(16, 512)
        self.mini_batch_spin.setSingleStep(16)
        self.mini_batch_spin.setValue(64)
        hyper_layout.addWidget(self.mini_batch_spin, 3, 3)
        
        layout.addWidget(hyper_group)
        
        # Network Architecture
        arch_group = QGroupBox("Network Architecture")
        arch_layout = QGridLayout(arch_group)
        
        # Actor Hidden Layers
        arch_layout.addWidget(QLabel("Actor Hidden:"), 0, 0)
        self.actor_hidden_edit = QLineEdit("256,128")
        arch_layout.addWidget(self.actor_hidden_edit, 0, 1)
        
        # Critic Hidden Layers
        arch_layout.addWidget(QLabel("Critic Hidden:"), 0, 2)
        self.critic_hidden_edit = QLineEdit("256,128")
        arch_layout.addWidget(self.critic_hidden_edit, 0, 3)
        
        # Use LSTM
        self.use_lstm_check = QCheckBox("Use LSTM")
        self.use_lstm_check.setChecked(True)
        arch_layout.addWidget(self.use_lstm_check, 1, 0)
        
        # LSTM Hidden Size
        arch_layout.addWidget(QLabel("LSTM Hidden:"), 1, 1)
        self.lstm_hidden_spin = QSpinBox()
        self.lstm_hidden_spin.setRange(32, 256)
        self.lstm_hidden_spin.setValue(64)
        arch_layout.addWidget(self.lstm_hidden_spin, 1, 2)
        
        # Dropout
        arch_layout.addWidget(QLabel("Dropout:"), 2, 0)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.1)
        arch_layout.addWidget(self.dropout_spin, 2, 1)
        
        # Activation Function
        arch_layout.addWidget(QLabel("Activation:"), 2, 2)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "tanh", "elu"])
        arch_layout.addWidget(self.activation_combo, 2, 3)
        
        layout.addWidget(arch_group)
        
        # Device Selection
        device_group = QGroupBox("Compute Device")
        device_layout = QHBoxLayout(device_group)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        device_layout.addWidget(QLabel("Device:"))
        device_layout.addWidget(self.device_combo)
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                self.device_combo.setCurrentText("cuda")
                device_layout.addWidget(QLabel(f"✓ GPU: {torch.cuda.get_device_name(0)}"))
            else:
                self.device_combo.setCurrentIndex(0)
                self.device_combo.setEnabled(False)
                device_layout.addWidget(QLabel("⚠ CUDA not available"))
        except:
            self.device_combo.setCurrentIndex(0)
            self.device_combo.setEnabled(False)
        
        layout.addWidget(device_group)
        
        layout.addStretch()
        
        return tab
    
    def _create_training_settings_tab(self) -> QWidget:
        """
        Tab 2: Training Settings
        
        Widgets: 20
        - Training mode (offline/online/hybrid)
        - Episodes, max steps per episode
        - Evaluation frequency
        - Checkpointing settings
        - Early stopping
        - TensorBoard logging
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Training Mode
        mode_group = QGroupBox("Training Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        self.training_mode_combo = QComboBox()
        self.training_mode_combo.addItems(["Offline (Historical)", "Online (Live)", "Hybrid"])
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.training_mode_combo)
        
        layout.addWidget(mode_group)
        
        # Episode Settings
        episode_group = QGroupBox("Episode Configuration")
        episode_layout = QGridLayout(episode_group)
        
        # Number of Episodes
        episode_layout.addWidget(QLabel("Total Episodes:"), 0, 0)
        self.num_episodes_spin = QSpinBox()
        self.num_episodes_spin.setRange(10, 10000)
        self.num_episodes_spin.setSingleStep(10)
        self.num_episodes_spin.setValue(1000)
        episode_layout.addWidget(self.num_episodes_spin, 0, 1)
        
        # Max Steps per Episode
        episode_layout.addWidget(QLabel("Max Steps/Episode:"), 0, 2)
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(50, 1000)
        self.max_steps_spin.setValue(252)  # ~1 trading year
        episode_layout.addWidget(self.max_steps_spin, 0, 3)
        
        # Eval Frequency
        episode_layout.addWidget(QLabel("Eval Every N Episodes:"), 1, 0)
        self.eval_freq_spin = QSpinBox()
        self.eval_freq_spin.setRange(1, 100)
        self.eval_freq_spin.setValue(10)
        episode_layout.addWidget(self.eval_freq_spin, 1, 1)
        
        # Eval Episodes
        episode_layout.addWidget(QLabel("Eval Episodes:"), 1, 2)
        self.eval_episodes_spin = QSpinBox()
        self.eval_episodes_spin.setRange(1, 20)
        self.eval_episodes_spin.setValue(5)
        episode_layout.addWidget(self.eval_episodes_spin, 1, 3)
        
        layout.addWidget(episode_group)
        
        # Checkpointing
        checkpoint_group = QGroupBox("Checkpointing")
        checkpoint_layout = QGridLayout(checkpoint_group)
        
        # Checkpoint Directory
        checkpoint_layout.addWidget(QLabel("Save Directory:"), 0, 0)
        self.checkpoint_dir_edit = QLineEdit("artifacts/rl_checkpoints")
        checkpoint_layout.addWidget(self.checkpoint_dir_edit, 0, 1, 1, 2)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_checkpoint_dir)
        checkpoint_layout.addWidget(browse_btn, 0, 3)
        
        # Save Frequency
        checkpoint_layout.addWidget(QLabel("Save Every N Episodes:"), 1, 0)
        self.save_freq_spin = QSpinBox()
        self.save_freq_spin.setRange(1, 500)
        self.save_freq_spin.setValue(50)
        checkpoint_layout.addWidget(self.save_freq_spin, 1, 1)
        
        # Save Best Only
        self.save_best_only_check = QCheckBox("Save Best Only")
        self.save_best_only_check.setChecked(True)
        checkpoint_layout.addWidget(self.save_best_only_check, 1, 2, 1, 2)
        
        layout.addWidget(checkpoint_group)
        
        # Early Stopping
        early_stop_group = QGroupBox("Early Stopping")
        early_stop_layout = QGridLayout(early_stop_group)
        
        # Enable Early Stopping
        self.early_stopping_check = QCheckBox("Enable Early Stopping")
        self.early_stopping_check.setChecked(True)
        early_stop_layout.addWidget(self.early_stopping_check, 0, 0, 1, 2)
        
        # Patience
        early_stop_layout.addWidget(QLabel("Patience:"), 1, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(10, 500)
        self.patience_spin.setValue(100)
        early_stop_layout.addWidget(self.patience_spin, 1, 1)
        
        # Min Delta
        early_stop_layout.addWidget(QLabel("Min Improvement:"), 1, 2)
        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setRange(0.001, 0.1)
        self.min_delta_spin.setDecimals(3)
        self.min_delta_spin.setValue(0.01)
        early_stop_layout.addWidget(self.min_delta_spin, 1, 3)
        
        layout.addWidget(early_stop_group)
        
        # TensorBoard
        tb_group = QGroupBox("TensorBoard Logging")
        tb_layout = QGridLayout(tb_group)
        
        # Enable TensorBoard
        self.use_tensorboard_check = QCheckBox("Enable TensorBoard")
        self.use_tensorboard_check.setChecked(True)
        tb_layout.addWidget(self.use_tensorboard_check, 0, 0, 1, 2)
        
        # TensorBoard Directory
        tb_layout.addWidget(QLabel("Log Directory:"), 1, 0)
        self.tensorboard_dir_edit = QLineEdit("artifacts/rl_tensorboard")
        tb_layout.addWidget(self.tensorboard_dir_edit, 1, 1, 1, 2)
        
        browse_tb_btn = QPushButton("Browse...")
        browse_tb_btn.clicked.connect(self._browse_tensorboard_dir)
        tb_layout.addWidget(browse_tb_btn, 1, 3)
        
        # Open TensorBoard button
        open_tb_btn = QPushButton("🔍 Open TensorBoard")
        open_tb_btn.clicked.connect(self._open_tensorboard)
        tb_layout.addWidget(open_tb_btn, 2, 0, 1, 4)
        
        layout.addWidget(tb_group)
        
        layout.addStretch()
        
        return tab
    
    def _create_state_action_tab(self) -> QWidget:
        """
        Tab 3: State & Action Space
        
        Widgets: 35
        - Portfolio features (always included)
        - Market features (returns, volatility, RSI, MACD, etc.)
        - Risk features (VaR, CVaR, Sharpe, Sortino)
        - Sentiment features (VIX, news, orderbook)
        - Action constraints (min/max weights, smoothing)
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Info label
        info = QLabel("Select features to include in state observation (137-dim max)")
        layout.addWidget(info)
        
        # Portfolio Features (always included)
        portfolio_group = QGroupBox("Portfolio Features (Always Included)")
        portfolio_layout = QVBoxLayout(portfolio_group)
        
        portfolio_features = [
            "Current Weights", "Days in Position", "Unrealized P&L",
            "Portfolio Value", "Cash Position", "Current Drawdown"
        ]
        for feature in portfolio_features:
            check = QCheckBox(feature)
            check.setChecked(True)
            check.setEnabled(False)  # Can't disable
            portfolio_layout.addWidget(check)
        
        layout.addWidget(portfolio_group)
        
        # Market Features
        market_group = QGroupBox("Market Features")
        market_layout = QGridLayout(market_group)
        
        self.market_features = {}
        market_feature_list = [
            "Returns History", "Volatility", "Correlation Matrix", "Momentum",
            "RSI", "MACD", "Bollinger Bands", "ATR", "Volume", "Spread"
        ]
        
        for i, feature in enumerate(market_feature_list):
            check = QCheckBox(feature)
            check.setChecked(i < 4)  # First 4 checked by default
            self.market_features[feature] = check
            market_layout.addWidget(check, i // 2, i % 2)
        
        layout.addWidget(market_group)
        
        # Risk Features
        risk_group = QGroupBox("Risk Features")
        risk_layout = QGridLayout(risk_group)
        
        self.risk_features = {}
        risk_feature_list = [
            "VaR (95%)", "CVaR (95%)", "Sharpe Ratio", "Sortino Ratio", 
            "Max Drawdown", "Volatility Ratio"
        ]
        
        for i, feature in enumerate(risk_feature_list):
            check = QCheckBox(feature)
            check.setChecked(True)  # All checked by default
            self.risk_features[feature] = check
            risk_layout.addWidget(check, i // 3, i % 3)
        
        layout.addWidget(risk_group)
        
        # Sentiment Features
        sentiment_group = QGroupBox("Sentiment Features")
        sentiment_layout = QGridLayout(sentiment_group)
        
        self.sentiment_features = {}
        sentiment_feature_list = [
            "VIX Level", "VIX Percentile", "News Sentiment", "Orderbook Imbalance"
        ]
        
        for i, feature in enumerate(sentiment_feature_list):
            check = QCheckBox(feature)
            check.setChecked(i < 2)  # VIX features checked
            self.sentiment_features[feature] = check
            sentiment_layout.addWidget(check, i // 2, i % 2)
        
        layout.addWidget(sentiment_group)
        
        # Action Constraints
        action_group = QGroupBox("Action Constraints")
        action_layout = QGridLayout(action_group)
        
        # Min Weight
        action_layout.addWidget(QLabel("Min Weight per Asset:"), 0, 0)
        self.min_weight_spin = QDoubleSpinBox()
        self.min_weight_spin.setRange(0.0, 0.5)
        self.min_weight_spin.setDecimals(2)
        self.min_weight_spin.setValue(0.0)
        action_layout.addWidget(self.min_weight_spin, 0, 1)
        
        # Max Weight
        action_layout.addWidget(QLabel("Max Weight per Asset:"), 0, 2)
        self.max_weight_spin = QDoubleSpinBox()
        self.max_weight_spin.setRange(0.0, 1.0)
        self.max_weight_spin.setDecimals(2)
        self.max_weight_spin.setValue(0.25)
        action_layout.addWidget(self.max_weight_spin, 0, 3)
        
        # Long Only
        self.long_only_check = QCheckBox("Long Only (No Shorting)")
        self.long_only_check.setChecked(True)
        action_layout.addWidget(self.long_only_check, 1, 0, 1, 2)
        
        # Force Sum to 1.0
        self.force_sum_check = QCheckBox("Force Weights Sum to 1.0")
        self.force_sum_check.setChecked(True)
        action_layout.addWidget(self.force_sum_check, 1, 2, 1, 2)
        
        # Action Smoothing (EMA)
        self.action_smoothing_check = QCheckBox("Action Smoothing (EMA)")
        action_layout.addWidget(self.action_smoothing_check, 2, 0, 1, 2)
        
        action_layout.addWidget(QLabel("Smoothing Factor:"), 2, 2)
        self.smoothing_alpha_spin = QDoubleSpinBox()
        self.smoothing_alpha_spin.setRange(0.0, 1.0)
        self.smoothing_alpha_spin.setDecimals(2)
        self.smoothing_alpha_spin.setValue(0.5)
        action_layout.addWidget(self.smoothing_alpha_spin, 2, 3)
        
        layout.addWidget(action_group)
        
        layout.addStretch()
        
        return tab
    
    def _create_reward_function_tab(self) -> QWidget:
        """
        Tab 4: Reward Function
        
        Widgets: 20
        - 9 reward components with weights
        - Reward normalization settings
        - Reward clipping
        - Preview chart (optional)
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Info
        info = QLabel("Configure multi-objective reward function (9 components)")
        layout.addWidget(info)
        
        # Reward Components
        reward_group = QGroupBox("Reward Components")
        reward_layout = QGridLayout(reward_group)
        
        # Headers
        reward_layout.addWidget(QLabel("<b>Component</b>"), 0, 0)
        reward_layout.addWidget(QLabel("<b>Enable</b>"), 0, 1)
        reward_layout.addWidget(QLabel("<b>Weight</b>"), 0, 2)
        
        # Component list with default weights
        self.reward_components = {}
        components = [
            ("Sharpe Ratio Improvement", True, 5.0),
            ("Transaction Costs", True, -10.0),
            ("VaR Violation", True, -20.0),
            ("CVaR Violation", True, -15.0),
            ("Correlation Violation", True, -5.0),
            ("Diversification Bonus", True, 1.0),
            ("Drawdown Penalty", False, -10.0),
            ("Turnover Penalty", False, -3.0),
            ("Sortino Ratio Improvement", False, 3.0),
        ]
        
        for i, (name, enabled, weight) in enumerate(components, start=1):
            # Name
            reward_layout.addWidget(QLabel(name), i, 0)
            
            # Enable checkbox
            enable_check = QCheckBox()
            enable_check.setChecked(enabled)
            reward_layout.addWidget(enable_check, i, 1)
            
            # Weight spinbox
            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(-100.0, 100.0)
            weight_spin.setDecimals(1)
            weight_spin.setValue(weight)
            reward_layout.addWidget(weight_spin, i, 2)
            
            self.reward_components[name] = {
                'enable': enable_check,
                'weight': weight_spin
            }
        
        layout.addWidget(reward_group)
        
        # Risk Constraints
        constraint_group = QGroupBox("Risk Constraints")
        constraint_layout = QGridLayout(constraint_group)
        
        # Max VaR
        constraint_layout.addWidget(QLabel("Max VaR (95%):"), 0, 0)
        self.max_var_spin = QDoubleSpinBox()
        self.max_var_spin.setRange(0.01, 0.5)
        self.max_var_spin.setDecimals(2)
        self.max_var_spin.setValue(0.10)
        constraint_layout.addWidget(self.max_var_spin, 0, 1)
        
        # Max CVaR
        constraint_layout.addWidget(QLabel("Max CVaR (95%):"), 0, 2)
        self.max_cvar_spin = QDoubleSpinBox()
        self.max_cvar_spin.setRange(0.01, 0.5)
        self.max_cvar_spin.setDecimals(2)
        self.max_cvar_spin.setValue(0.15)
        constraint_layout.addWidget(self.max_cvar_spin, 0, 3)
        
        # Max Correlated Exposure
        constraint_layout.addWidget(QLabel("Max Correlated Exposure:"), 1, 0)
        self.max_corr_exposure_spin = QDoubleSpinBox()
        self.max_corr_exposure_spin.setRange(0.0, 1.0)
        self.max_corr_exposure_spin.setDecimals(2)
        self.max_corr_exposure_spin.setValue(0.50)
        constraint_layout.addWidget(self.max_corr_exposure_spin, 1, 1)
        
        # Transaction Cost (bps)
        constraint_layout.addWidget(QLabel("Transaction Cost (bps):"), 1, 2)
        self.transaction_cost_spin = QDoubleSpinBox()
        self.transaction_cost_spin.setRange(0.0, 50.0)
        self.transaction_cost_spin.setDecimals(1)
        self.transaction_cost_spin.setValue(5.0)
        constraint_layout.addWidget(self.transaction_cost_spin, 1, 3)
        
        layout.addWidget(constraint_group)
        
        # Normalization
        norm_group = QGroupBox("Reward Normalization")
        norm_layout = QGridLayout(norm_group)
        
        # Enable Normalization
        self.normalize_rewards_check = QCheckBox("Normalize Rewards (Z-score)")
        self.normalize_rewards_check.setChecked(True)
        norm_layout.addWidget(self.normalize_rewards_check, 0, 0, 1, 2)
        
        # Clip Min
        norm_layout.addWidget(QLabel("Clip Min:"), 1, 0)
        self.reward_clip_min_spin = QDoubleSpinBox()
        self.reward_clip_min_spin.setRange(-1000.0, 0.0)
        self.reward_clip_min_spin.setValue(-100.0)
        norm_layout.addWidget(self.reward_clip_min_spin, 1, 1)
        
        # Clip Max
        norm_layout.addWidget(QLabel("Clip Max:"), 1, 2)
        self.reward_clip_max_spin = QDoubleSpinBox()
        self.reward_clip_max_spin.setRange(0.0, 1000.0)
        self.reward_clip_max_spin.setValue(100.0)
        norm_layout.addWidget(self.reward_clip_max_spin, 1, 3)
        
        layout.addWidget(norm_group)
        
        layout.addStretch()
        
        return tab
    
    def _create_training_progress_tab(self) -> QWidget:
        """
        Tab 5: Training Progress
        
        Live monitoring during training:
        - Current episode, progress bar, ETA
        - Real-time metrics (reward, Sharpe, costs)
        - Training curves (4 subplots)
        - Best models leaderboard
        - Pause/Stop/Save buttons
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Status Section
        status_group = QGroupBox("Training Status")
        status_layout = QGridLayout(status_group)
        
        # Episode Counter
        status_layout.addWidget(QLabel("Episode:"), 0, 0)
        self.current_episode_label = QLabel("0 / 0")
        self.current_episode_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.current_episode_label, 0, 1)
        
        # Progress Bar
        status_layout.addWidget(QLabel("Progress:"), 0, 2)
        self.training_progress_bar = QProgressBar()
        status_layout.addWidget(self.training_progress_bar, 0, 3)
        
        # ETA
        status_layout.addWidget(QLabel("ETA:"), 1, 0)
        self.eta_label = QLabel("--:--:--")
        status_layout.addWidget(self.eta_label, 1, 1)
        
        # Status
        status_layout.addWidget(QLabel("Status:"), 1, 2)
        self.training_status_label = QLabel("Not Started")
        status_layout.addWidget(self.training_status_label, 1, 3)
        
        layout.addWidget(status_group)
        
        # Current Episode Metrics
        metrics_group = QGroupBox("Current Episode Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        metrics_labels = [
            ("Reward:", "reward"),
            ("Return:", "return"),
            ("Sharpe:", "sharpe"),
            ("Sortino:", "sortino"),
            ("Max DD:", "drawdown"),
            ("Trans Cost:", "cost"),
        ]
        
        self.metric_labels = {}
        for i, (label, key) in enumerate(metrics_labels):
            metrics_layout.addWidget(QLabel(label), i // 3, (i % 3) * 2)
            value_label = QLabel("--")
            value_label.setStyleSheet("font-weight: bold;")
            metrics_layout.addWidget(value_label, i // 3, (i % 3) * 2 + 1)
            self.metric_labels[key] = value_label
        
        layout.addWidget(metrics_group)
        
        # Training Curves (Matplotlib)
        if MATPLOTLIB_AVAILABLE:
            self.training_figure = Figure(figsize=(10, 6))
            self.training_canvas = FigureCanvas(self.training_figure)
            
            # Create 4 subplots
            self.axes = self.training_figure.subplots(2, 2)
            self.training_figure.tight_layout()
            
            layout.addWidget(self.training_canvas)
        else:
            layout.addWidget(QLabel("Matplotlib not available - install for charts"))
        
        # Best Models Leaderboard
        leaderboard_group = QGroupBox("Best Models (Top 5)")
        leaderboard_layout = QVBoxLayout(leaderboard_group)
        
        self.leaderboard_table = QTableWidget()
        self.leaderboard_table.setColumnCount(5)
        self.leaderboard_table.setHorizontalHeaderLabels([
            "Episode", "Sharpe", "Return", "Max DD", "Checkpoint"
        ])
        self.leaderboard_table.setRowCount(5)
        leaderboard_layout.addWidget(self.leaderboard_table)
        
        layout.addWidget(leaderboard_group)
        
        # Control Buttons
        control_layout = QHBoxLayout()
        
        self.start_training_btn = QPushButton("▶ Start Training")
        self.start_training_btn.clicked.connect(self._on_start_training)
        control_layout.addWidget(self.start_training_btn)
        
        self.pause_training_btn = QPushButton("⏸ Pause")
        self.pause_training_btn.setEnabled(False)
        control_layout.addWidget(self.pause_training_btn)
        
        self.stop_training_btn = QPushButton("⏹ Stop")
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.clicked.connect(self._on_stop_training)
        control_layout.addWidget(self.stop_training_btn)
        
        self.save_checkpoint_btn = QPushButton("💾 Save Checkpoint")
        self.save_checkpoint_btn.setEnabled(False)
        control_layout.addWidget(self.save_checkpoint_btn)
        
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        return tab
    
    def _create_deployment_tab(self) -> QWidget:
        """
        Tab 6: Deployment & Testing
        
        Widgets: 20
        - Backtest configuration
        - Comparison with baselines
        - Deploy to production
        - Deployment mode selection
        - Safety limits
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Backtest Section
        backtest_group = QGroupBox("Backtest Configuration")
        backtest_layout = QGridLayout(backtest_group)
        
        # Load Model
        backtest_layout.addWidget(QLabel("Load Model:"), 0, 0)
        self.model_path_edit = QLineEdit()
        backtest_layout.addWidget(self.model_path_edit, 0, 1, 1, 2)
        
        browse_model_btn = QPushButton("Browse...")
        browse_model_btn.clicked.connect(self._browse_model)
        backtest_layout.addWidget(browse_model_btn, 0, 3)
        
        # Date Range
        backtest_layout.addWidget(QLabel("Start Date:"), 1, 0)
        self.backtest_start_edit = QLineEdit("2023-01-01")
        backtest_layout.addWidget(self.backtest_start_edit, 1, 1)
        
        backtest_layout.addWidget(QLabel("End Date:"), 1, 2)
        self.backtest_end_edit = QLineEdit("2023-12-31")
        backtest_layout.addWidget(self.backtest_end_edit, 1, 3)
        
        # Initial Capital
        backtest_layout.addWidget(QLabel("Initial Capital:"), 2, 0)
        self.backtest_capital_spin = QDoubleSpinBox()
        self.backtest_capital_spin.setRange(1000, 1000000)
        self.backtest_capital_spin.setValue(10000)
        backtest_layout.addWidget(self.backtest_capital_spin, 2, 1)
        
        # Run Backtest Button
        run_backtest_btn = QPushButton("🚀 Run Backtest")
        run_backtest_btn.clicked.connect(self._on_run_backtest)
        backtest_layout.addWidget(run_backtest_btn, 2, 2, 1, 2)
        
        layout.addWidget(backtest_group)
        
        # Comparison Baselines
        comparison_group = QGroupBox("Compare With Baselines")
        comparison_layout = QVBoxLayout(comparison_group)
        
        self.baseline_checks = {}
        baselines = [
            "Equal Weight Portfolio",
            "Riskfolio Mean-Variance",
            "Riskfolio Max Sharpe",
            "Riskfolio Risk Parity",
            "Buy & Hold"
        ]
        
        for baseline in baselines:
            check = QCheckBox(baseline)
            check.setChecked(baseline.startswith("Riskfolio"))
            self.baseline_checks[baseline] = check
            comparison_layout.addWidget(check)
        
        layout.addWidget(comparison_group)
        
        # Backtest Results
        results_group = QGroupBox("Backtest Results")
        results_layout = QVBoxLayout(results_group)
        
        self.backtest_results_table = QTableWidget()
        self.backtest_results_table.setColumnCount(6)
        self.backtest_results_table.setHorizontalHeaderLabels([
            "Strategy", "Total Return", "Sharpe", "Sortino", "Max DD", "Trans Costs"
        ])
        results_layout.addWidget(self.backtest_results_table)
        
        layout.addWidget(results_group)
        
        # Production Deployment
        deploy_group = QGroupBox("Production Deployment")
        deploy_layout = QGridLayout(deploy_group)
        
        # Deployment Mode
        deploy_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.deployment_mode_combo = QComboBox()
        self.deployment_mode_combo.addItems([
            "RL Only",
            "RL + Riskfolio Hybrid",
            "RL Advisory"
        ])
        self.deployment_mode_combo.setCurrentIndex(1)  # Hybrid default
        deploy_layout.addWidget(self.deployment_mode_combo, 0, 1)
        
        # Confidence Threshold
        deploy_layout.addWidget(QLabel("Confidence Threshold:"), 0, 2)
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.0, 1.0)
        self.confidence_threshold_spin.setDecimals(2)
        self.confidence_threshold_spin.setValue(0.7)
        deploy_layout.addWidget(self.confidence_threshold_spin, 0, 3)
        
        # Max Deviation from Riskfolio
        deploy_layout.addWidget(QLabel("Max Deviation (Hybrid):"), 1, 0)
        self.max_deviation_spin = QDoubleSpinBox()
        self.max_deviation_spin.setRange(0.0, 0.5)
        self.max_deviation_spin.setDecimals(2)
        self.max_deviation_spin.setValue(0.20)
        deploy_layout.addWidget(self.max_deviation_spin, 1, 1)
        
        layout.addWidget(deploy_group)
        
        # Safety Limits
        safety_group = QGroupBox("Safety Limits")
        safety_layout = QGridLayout(safety_group)
        
        # Max Trades per Day
        safety_layout.addWidget(QLabel("Max Trades/Day:"), 0, 0)
        self.max_trades_spin = QSpinBox()
        self.max_trades_spin.setRange(1, 100)
        self.max_trades_spin.setValue(10)
        safety_layout.addWidget(self.max_trades_spin, 0, 1)
        
        # Emergency Stop Drawdown
        safety_layout.addWidget(QLabel("Emergency Stop DD:"), 0, 2)
        self.emergency_stop_spin = QDoubleSpinBox()
        self.emergency_stop_spin.setRange(0.0, 0.5)
        self.emergency_stop_spin.setDecimals(2)
        self.emergency_stop_spin.setValue(0.25)
        safety_layout.addWidget(self.emergency_stop_spin, 0, 3)
        
        layout.addWidget(safety_group)
        
        # Deployment Controls
        deploy_control_layout = QHBoxLayout()
        
        self.activate_deployment_btn = QPushButton("✓ Activate Deployment")
        self.activate_deployment_btn.clicked.connect(self._on_activate_deployment)
        deploy_control_layout.addWidget(self.activate_deployment_btn)
        
        self.deactivate_deployment_btn = QPushButton("✗ Deactivate Deployment")
        self.deactivate_deployment_btn.setEnabled(False)
        self.deactivate_deployment_btn.clicked.connect(self._on_deactivate_deployment)
        deploy_control_layout.addWidget(self.deactivate_deployment_btn)
        
        deploy_control_layout.addStretch()
        
        layout.addLayout(deploy_control_layout)
        
        layout.addStretch()
        
        return tab
    
    # Event Handlers
    
    def _browse_checkpoint_dir(self):
        """Browse for checkpoint directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Checkpoint Directory")
        if dir_path:
            self.checkpoint_dir_edit.setText(dir_path)
    
    def _browse_tensorboard_dir(self):
        """Browse for TensorBoard directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select TensorBoard Directory")
        if dir_path:
            self.tensorboard_dir_edit.setText(dir_path)
    
    def _open_tensorboard(self):
        """Open TensorBoard in browser."""
        import subprocess
        log_dir = self.tensorboard_dir_edit.text()
        
        try:
            subprocess.Popen(f"tensorboard --logdir={log_dir}", shell=True)
            QMessageBox.information(
                self, 
                "TensorBoard Started",
                f"TensorBoard started at http://localhost:6006\n\nLog directory: {log_dir}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start TensorBoard: {e}")
    
    def _browse_model(self):
        """Browse for model checkpoint."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "", "PyTorch Models (*.pt *.pth)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def _on_start_training(self):
        """Start RL training."""
        # Collect configuration
        config = self._get_training_config()
        
        # Emit signal
        self.training_started.emit(config)
        
        # Update UI state
        self.is_training = True
        self.start_training_btn.setEnabled(False)
        self.pause_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(True)
        self.training_status_label.setText("Training...")
        
        # Start update timer
        self.update_timer.start(1000)  # Update every second
        
        logger.info("Training started with config: {}", config)
    
    def _on_stop_training(self):
        """Stop RL training."""
        self.training_stopped.emit()
        
        self.is_training = False
        self.start_training_btn.setEnabled(True)
        self.pause_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(False)
        self.training_status_label.setText("Stopped")
        
        self.update_timer.stop()
        
        logger.info("Training stopped")
    
    def _on_run_backtest(self):
        """Run backtest on trained model."""
        logger.info("Backtest requested")
        # TODO: Implement backtest logic
    
    def _on_activate_deployment(self):
        """Activate production deployment."""
        mode_map = {
            "RL Only": "rl_only",
            "RL + Riskfolio Hybrid": "rl_riskfolio_hybrid",
            "RL Advisory": "rl_advisory"
        }
        mode = mode_map[self.deployment_mode_combo.currentText()]
        
        self.deployment_activated.emit(mode)
        
        self.activate_deployment_btn.setEnabled(False)
        self.deactivate_deployment_btn.setEnabled(True)
        
        logger.info(f"Deployment activated: mode={mode}")
    
    def _on_deactivate_deployment(self):
        """Deactivate production deployment."""
        self.deployment_deactivated.emit()
        
        self.activate_deployment_btn.setEnabled(True)
        self.deactivate_deployment_btn.setEnabled(False)
        
        logger.info("Deployment deactivated")
    
    def _update_live_metrics(self):
        """Update live training metrics (called every second)."""
        # TODO: Get metrics from training backend
        pass
    
    def _get_training_config(self) -> Dict:
        """
        Collect all configuration from UI widgets.
        
        Returns:
            config: Complete training configuration
        """
        # Parse hidden dims
        actor_hidden = [int(x.strip()) for x in self.actor_hidden_edit.text().split(',')]
        critic_hidden = [int(x.strip()) for x in self.critic_hidden_edit.text().split(',')]
        
        # Collect reward component weights
        reward_weights = {}
        for name, widgets in self.reward_components.items():
            if widgets['enable'].isChecked():
                reward_weights[name] = widgets['weight'].value()
        
        config = {
            # Agent
            'algorithm': self.algorithm_combo.currentText().lower(),
            'actor_lr': self.actor_lr_spin.value(),
            'critic_lr': self.critic_lr_spin.value(),
            'clip_epsilon': self.clip_epsilon_spin.value(),
            'gae_lambda': self.gae_lambda_spin.value(),
            'gamma': self.gamma_spin.value(),
            'entropy_coef': self.entropy_coef_spin.value(),
            'ppo_epochs': self.ppo_epochs_spin.value(),
            'mini_batch_size': self.mini_batch_spin.value(),
            'actor_hidden_dims': actor_hidden,
            'critic_hidden_dims': critic_hidden,
            'use_lstm': self.use_lstm_check.isChecked(),
            'lstm_hidden': self.lstm_hidden_spin.value(),
            'dropout': self.dropout_spin.value(),
            'activation': self.activation_combo.currentText(),
            'device': self.device_combo.currentText(),
            
            # Training
            'num_episodes': self.num_episodes_spin.value(),
            'max_steps_per_episode': self.max_steps_spin.value(),
            'eval_frequency': self.eval_freq_spin.value(),
            'eval_episodes': self.eval_episodes_spin.value(),
            
            # Checkpointing
            'checkpoint_dir': self.checkpoint_dir_edit.text(),
            'save_frequency': self.save_freq_spin.value(),
            'save_best_only': self.save_best_only_check.isChecked(),
            
            # Early Stopping
            'early_stopping': self.early_stopping_check.isChecked(),
            'patience': self.patience_spin.value(),
            'min_delta': self.min_delta_spin.value(),
            
            # TensorBoard
            'use_tensorboard': self.use_tensorboard_check.isChecked(),
            'tensorboard_dir': self.tensorboard_dir_edit.text(),
            
            # State/Action
            'min_weight': self.min_weight_spin.value(),
            'max_weight': self.max_weight_spin.value(),
            'long_only': self.long_only_check.isChecked(),
            'force_sum_one': self.force_sum_check.isChecked(),
            
            # Reward
            'reward_weights': reward_weights,
            'max_var': self.max_var_spin.value(),
            'max_cvar': self.max_cvar_spin.value(),
            'max_correlated_exposure': self.max_corr_exposure_spin.value(),
            'transaction_cost_bps': self.transaction_cost_spin.value(),
            'normalize_rewards': self.normalize_rewards_check.isChecked(),
            'reward_clip_min': self.reward_clip_min_spin.value(),
            'reward_clip_max': self.reward_clip_max_spin.value(),
            
            # Deployment
            'deployment_mode': self.deployment_mode_combo.currentText(),
            'confidence_threshold': self.confidence_threshold_spin.value(),
            'max_deviation': self.max_deviation_spin.value(),
            'max_trades_per_day': self.max_trades_spin.value(),
            'emergency_stop_drawdown': self.emergency_stop_spin.value(),
        }
        
        return config
    
    def update_training_progress(self, episode: int, total_episodes: int, metrics: Dict):
        """
        Update training progress display.
        
        Called from backend during training.
        
        Args:
            episode: Current episode number
            total_episodes: Total episodes
            metrics: Episode metrics
        """
        # Update labels
        self.current_episode_label.setText(f"{episode} / {total_episodes}")
        self.training_progress_bar.setMaximum(total_episodes)
        self.training_progress_bar.setValue(episode)
        
        # Update metrics
        self.metric_labels['reward'].setText(f"{metrics.get('episode_reward', 0):.2f}")
        self.metric_labels['return'].setText(f"{metrics.get('episode_return', 0):.2%}")
        self.metric_labels['sharpe'].setText(f"{metrics.get('sharpe_ratio', 0):.2f}")
        self.metric_labels['sortino'].setText(f"{metrics.get('sortino_ratio', 0):.2f}")
        self.metric_labels['drawdown'].setText(f"{metrics.get('max_drawdown', 0):.2%}")
        self.metric_labels['cost'].setText(f"{metrics.get('transaction_costs', 0):.4f}")
        
        # Store history
        self.training_history.append(metrics)
        
        # Update charts
        if MATPLOTLIB_AVAILABLE and len(self.training_history) > 1:
            self._update_training_charts()
    
    def _update_training_charts(self):
        """Update matplotlib training charts."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        df = pd.DataFrame(self.training_history)
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Rewards
        self.axes[0, 0].plot(df['episode_reward'], label='Episode Reward')
        self.axes[0, 0].set_title('Episode Reward')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].legend()
        
        # Plot 2: Returns
        self.axes[0, 1].plot(df['episode_return'], label='Episode Return', color='green')
        self.axes[0, 1].set_title('Portfolio Return')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].legend()
        
        # Plot 3: Sharpe Ratio
        self.axes[1, 0].plot(df['sharpe_ratio'], label='Sharpe Ratio', color='orange')
        self.axes[1, 0].set_title('Sharpe Ratio')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].legend()
        
        # Plot 4: Max Drawdown
        self.axes[1, 1].plot(df['max_drawdown'], label='Max Drawdown', color='red')
        self.axes[1, 1].set_title('Max Drawdown')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].legend()
        
        self.training_figure.tight_layout()
        self.training_canvas.draw()
    
    def _apply_i18n_tooltips(self):
        """Apply i18n tooltips to all widgets (51 tooltips for 4 tabs)."""
        from ..i18n.widget_helper import apply_tooltip
        from ..i18n import tr
        
        logger.info("Applying i18n tooltips to RL Agent tab widgets...")
        
        # Tab 1: Agent Configuration (16 widgets)
        apply_tooltip(self.algorithm_combo, "algorithm", "rl_agent")
        apply_tooltip(self.actor_lr_spin, "actor_lr", "rl_agent")
        apply_tooltip(self.critic_lr_spin, "critic_lr", "rl_agent")
        apply_tooltip(self.clip_epsilon_spin, "clip_epsilon", "rl_agent")
        apply_tooltip(self.gae_lambda_spin, "gae_lambda", "rl_agent")
        apply_tooltip(self.gamma_spin, "gamma", "rl_agent")
        apply_tooltip(self.entropy_coef_spin, "entropy_coef", "rl_agent")
        apply_tooltip(self.ppo_epochs_spin, "ppo_epochs", "rl_agent")
        apply_tooltip(self.mini_batch_spin, "mini_batch_size", "rl_agent")
        apply_tooltip(self.actor_hidden_edit, "actor_hidden", "rl_agent")
        apply_tooltip(self.critic_hidden_edit, "critic_hidden", "rl_agent")
        apply_tooltip(self.use_lstm_check, "use_lstm", "rl_agent")
        apply_tooltip(self.lstm_hidden_spin, "lstm_hidden", "rl_agent")
        apply_tooltip(self.dropout_spin, "dropout", "rl_agent")
        apply_tooltip(self.activation_combo, "activation", "rl_agent")
        apply_tooltip(self.device_combo, "device", "rl_agent")
        
        # Tab 2: Training Settings (13 widgets)
        apply_tooltip(self.training_mode_combo, "training_mode", "rl_agent")
        apply_tooltip(self.num_episodes_spin, "num_episodes", "rl_agent")
        apply_tooltip(self.max_steps_spin, "max_steps_per_episode", "rl_agent")
        apply_tooltip(self.eval_freq_spin, "eval_frequency", "rl_agent")
        apply_tooltip(self.eval_episodes_spin, "eval_episodes", "rl_agent")
        apply_tooltip(self.checkpoint_dir_edit, "checkpoint_dir", "rl_agent")
        apply_tooltip(self.save_freq_spin, "save_frequency", "rl_agent")
        apply_tooltip(self.save_best_only_check, "save_best_only", "rl_agent")
        apply_tooltip(self.early_stopping_check, "early_stopping", "rl_agent")
        apply_tooltip(self.patience_spin, "patience", "rl_agent")
        apply_tooltip(self.min_delta_spin, "min_delta", "rl_agent")
        apply_tooltip(self.use_tensorboard_check, "use_tensorboard", "rl_agent")
        apply_tooltip(self.tensorboard_dir_edit, "tensorboard_dir", "rl_agent")
        
        # Tab 3: State & Action Space (13 widgets - market/risk features)
        # Market features
        if hasattr(self, 'market_features'):
            if 'Returns History' in self.market_features:
                apply_tooltip(self.market_features['Returns History'], "market_returns_history", "rl_agent")
            if 'Volatility' in self.market_features:
                apply_tooltip(self.market_features['Volatility'], "market_volatility", "rl_agent")
            if 'Correlation Matrix' in self.market_features:
                apply_tooltip(self.market_features['Correlation Matrix'], "market_correlation", "rl_agent")
            if 'Momentum' in self.market_features:
                apply_tooltip(self.market_features['Momentum'], "market_momentum", "rl_agent")
            if 'RSI' in self.market_features:
                apply_tooltip(self.market_features['RSI'], "market_rsi", "rl_agent")
            if 'MACD' in self.market_features:
                apply_tooltip(self.market_features['MACD'], "market_macd", "rl_agent")
        
        # Risk features
        if hasattr(self, 'risk_features'):
            if 'VaR (95%)' in self.risk_features:
                apply_tooltip(self.risk_features['VaR (95%)'], "risk_var", "rl_agent")
            if 'CVaR (95%)' in self.risk_features:
                apply_tooltip(self.risk_features['CVaR (95%)'], "risk_cvar", "rl_agent")
            if 'Sharpe Ratio' in self.risk_features:
                apply_tooltip(self.risk_features['Sharpe Ratio'], "risk_sharpe", "rl_agent")
            if 'Sortino Ratio' in self.risk_features:
                apply_tooltip(self.risk_features['Sortino Ratio'], "risk_sortino", "rl_agent")
        
        # Sentiment features
        if hasattr(self, 'sentiment_features'):
            if 'VIX Level' in self.sentiment_features:
                apply_tooltip(self.sentiment_features['VIX Level'], "sentiment_vix", "rl_agent")
            if 'VIX Percentile' in self.sentiment_features:
                apply_tooltip(self.sentiment_features['VIX Percentile'], "sentiment_vix_percentile", "rl_agent")
        
        # Action constraints
        apply_tooltip(self.min_weight_spin, "min_weight", "rl_agent")
        apply_tooltip(self.max_weight_spin, "max_weight", "rl_agent")
        apply_tooltip(self.long_only_check, "long_only", "rl_agent")
        apply_tooltip(self.force_sum_check, "force_sum_one", "rl_agent")
        apply_tooltip(self.action_smoothing_check, "action_smoothing", "rl_agent")
        apply_tooltip(self.smoothing_alpha_spin, "smoothing_alpha", "rl_agent")
        
        # Tab 4: Reward Function (17 widgets)
        if hasattr(self, 'reward_components'):
            if 'Sharpe Ratio Improvement' in self.reward_components:
                apply_tooltip(self.reward_components['Sharpe Ratio Improvement']['weight'], "reward_sharpe", "rl_agent")
            if 'Transaction Costs' in self.reward_components:
                apply_tooltip(self.reward_components['Transaction Costs']['weight'], "reward_transaction_cost", "rl_agent")
            if 'VaR Violation' in self.reward_components:
                apply_tooltip(self.reward_components['VaR Violation']['weight'], "reward_var_violation", "rl_agent")
            if 'CVaR Violation' in self.reward_components:
                apply_tooltip(self.reward_components['CVaR Violation']['weight'], "reward_cvar_violation", "rl_agent")
            if 'Correlation Violation' in self.reward_components:
                apply_tooltip(self.reward_components['Correlation Violation']['weight'], "reward_correlation_violation", "rl_agent")
            if 'Diversification Bonus' in self.reward_components:
                apply_tooltip(self.reward_components['Diversification Bonus']['weight'], "reward_diversification", "rl_agent")
            if 'Drawdown Penalty' in self.reward_components:
                apply_tooltip(self.reward_components['Drawdown Penalty']['weight'], "reward_drawdown", "rl_agent")
            if 'Turnover Penalty' in self.reward_components:
                apply_tooltip(self.reward_components['Turnover Penalty']['weight'], "reward_turnover", "rl_agent")
            if 'Sortino Ratio Improvement' in self.reward_components:
                apply_tooltip(self.reward_components['Sortino Ratio Improvement']['weight'], "reward_sortino", "rl_agent")
        
        apply_tooltip(self.max_var_spin, "max_var", "rl_agent")
        apply_tooltip(self.max_cvar_spin, "max_cvar", "rl_agent")
        apply_tooltip(self.max_corr_exposure_spin, "max_corr_exposure", "rl_agent")
        apply_tooltip(self.transaction_cost_spin, "transaction_cost_bps", "rl_agent")
        apply_tooltip(self.normalize_rewards_check, "normalize_rewards", "rl_agent")
        apply_tooltip(self.reward_clip_min_spin, "reward_clip_min", "rl_agent")
        apply_tooltip(self.reward_clip_max_spin, "reward_clip_max", "rl_agent")
        
        logger.info("RL Agent i18n tooltips applied: 51 widgets")
