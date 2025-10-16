# 🤖 Actor-Critic Deep RL Integration - Complete Specification

## 🎯 Overview

Integrazione di un sistema **Actor-Critic Deep Reinforcement Learning** per Portfolio Management e Trading, basato su:
- **PPO (Proximal Policy Optimization)** - state-of-the-art continuous action RL
- **SAC (Soft Actor-Critic)** - maximum entropy RL per exploration
- **A3C (Asynchronous Advantage Actor-Critic)** - parallel training
- **TD3 (Twin Delayed DDPG)** - stable continuous control

---

## 📚 Teoria: Actor-Critic per Portfolio Management

### **Problema di RL**
```
State (s_t):
  - Portfolio weights: [w_EUR, w_GBP, w_JPY, w_CASH]
  - Market features: [returns, volatility, Sharpe, correlations]
  - Risk metrics: [VaR, CVaR, drawdown]
  - Sentiment: [VIX, orderbook imbalance, news sentiment]
  
Action (a_t):
  - Target portfolio weights: [w'_EUR, w'_GBP, w'_JPY, w'_CASH]
  - Continuous actions in [-1, 1], mapped to [0%, 100%]
  
Reward (r_t):
  - Primary: Sharpe ratio improvement
  - Penalties: Transaction costs, VaR violations, correlation breaches
  
Transition: s_t → a_t → s_(t+1), r_t
```

### **Actor-Critic Architecture**

```
┌────────────────────────────────────────────────────────────┐
│                        ACTOR NETWORK                        │
│  Input: State (portfolio, market, risk)                   │
│  ├─ Dense(256) + ReLU                                      │
│  ├─ Dense(128) + ReLU                                      │
│  ├─ LSTM(64) - temporal patterns                           │
│  └─ Dense(n_assets) + Softmax → Portfolio Weights         │
└────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                       CRITIC NETWORK                        │
│  Input: State + Action                                     │
│  ├─ Dense(256) + ReLU                                      │
│  ├─ Dense(128) + ReLU                                      │
│  └─ Dense(1) → Q-value (expected return)                  │
└────────────────────────────────────────────────────────────┘
```

### **Reward Function** (Multi-Objective)
```python
reward = (
    alpha * sharpe_improvement          # +1.0 for +1 Sharpe
    - beta * transaction_costs          # -0.1 for 1% costs
    - gamma * var_penalty               # -2.0 if VaR > limit
    - delta * correlation_penalty       # -1.0 if corr > limit
    + epsilon * diversification_bonus   # +0.5 if well-diversified
)

# Typical values:
alpha = 5.0    # Sharpe is primary objective
beta = 10.0    # Transaction costs heavily penalized
gamma = 20.0   # VaR violations severely penalized
delta = 5.0    # Correlation breaches penalized
epsilon = 1.0  # Diversification mildly rewarded
```

---

## 📂 File Structure

```
src/forex_diffusion/
├── rl/                                    # NEW MODULE
│   ├── __init__.py
│   ├── actor_critic/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py                  # PPO implementation
│   │   ├── sac_agent.py                  # SAC implementation
│   │   ├── a3c_agent.py                  # A3C implementation
│   │   ├── td3_agent.py                  # TD3 implementation
│   │   └── base_agent.py                 # Base RL Agent
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── portfolio_env.py              # Gym environment for portfolio
│   │   └── trading_env.py                # Gym environment for trading
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── actor_network.py              # Actor architecture
│   │   ├── critic_network.py             # Critic architecture
│   │   └── shared_networks.py            # Shared layers (LSTM, etc)
│   ├── replay_buffer.py                  # Experience replay
│   ├── rewards.py                        # Reward function builder
│   └── trainer.py                        # RL training loop
│
├── ui/
│   └── rl_config_tab.py                  # NEW: RL Configuration Tab
│
└── trading/
    └── rl_portfolio_manager.py           # NEW: RL-driven portfolio manager
```

---

## 🎨 UI: RL Configuration Tab (Level 2 under Trading Intelligence)

### **NEW Tab Structure**

```
Trading Intelligence (Level 1)
├── Portfolio (Level 2) - [6 sub-tabs Level 3]
├── Signals (Level 2)
└── RL Agent (Level 2) ← NEW
    ├── Agent Configuration (Level 3)      # Algorithm, hyperparameters
    ├── Training Settings (Level 3)         # Episodes, batch size, etc.
    ├── State & Action Space (Level 3)      # Input features, output actions
    ├── Reward Function (Level 3)           # Multi-objective weights
    ├── Training Progress (Level 3)         # Live training metrics
    └── Deployment & Testing (Level 3)      # Backtest RL agent
```

---

## 🔧 Level 3 - Tab 1: Agent Configuration

```python
┌─────────────────────────────────────────────────────────────┐
│ Reinforcement Learning Agent                                │
│                                                              │
│ ☑ Enable RL-Based Portfolio Management                     │
│   When enabled, RL agent co-pilots with Riskfolio optimizer│
│                                                              │
│ RL Algorithm:                                                │
│ ⚫ PPO - Proximal Policy Optimization (Recommended)         │
│   └─ Stable, sample-efficient, good for continuous actions │
│                                                              │
│ ○ SAC - Soft Actor-Critic                                   │
│   └─ Maximum entropy, better exploration                   │
│                                                              │
│ ○ A3C - Asynchronous Advantage Actor-Critic                 │
│   └─ Parallel training, faster convergence                 │
│                                                              │
│ ○ TD3 - Twin Delayed DDPG                                   │
│   └─ Very stable, low variance                             │
│                                                              │
│ ○ Ensemble - Combine Multiple Agents                        │
│   └─ Agents: ☑ PPO  ☑ SAC  ☐ A3C  ☐ TD3                  │
│   └─ Aggregation: ⚫ Weighted Average  ○ Voting            │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `enable_rl_check`: QCheckBox (default False)
- `rl_algorithm_combo`: QComboBox ["PPO", "SAC", "A3C", "TD3", "Ensemble"]
- `ensemble_agents_group`: QGroupBox with checkboxes (enabled only if Ensemble)
- `ensemble_aggregation_combo`: QComboBox ["Weighted Average", "Voting", "Meta-Learner"]

### **PPO Hyperparameters**
```python
┌─────────────────────────────────────────────────────────────┐
│ PPO Hyperparameters                                          │
│                                                              │
│ Clip Epsilon (ε):        [0.20] (0.1-0.3)  🛈              │
│   → Policy update clipping, higher = larger updates        │
│                                                              │
│ Value Function Coef (c1): [0.50] (0.1-1.0) 🛈              │
│   → Weight of value loss in total loss                     │
│                                                              │
│ Entropy Coef (c2):       [0.01] (0.001-0.1) 🛈             │
│   → Exploration bonus, higher = more exploration           │
│                                                              │
│ Learning Rate (α):       [3e-4] (1e-5 to 1e-3) 🛈          │
│   Adaptive: ☑ Linear Decay  Start: [3e-4] → End: [1e-5]  │
│                                                              │
│ Discount Factor (γ):     [0.99] (0.9-0.999) 🛈             │
│   → Future reward discounting                              │
│                                                              │
│ GAE Lambda (λ):          [0.95] (0.9-0.99) 🛈              │
│   → Generalized Advantage Estimation smoothing            │
│                                                              │
│ Mini-Batch Size:         [64  ] (32-256) 🛈                │
│ Optimization Epochs:     [10  ] (5-20) 🛈                  │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `ppo_clip_epsilon_spin`: QDoubleSpinBox (0.1-0.3, default 0.2)
- `ppo_value_coef_spin`: QDoubleSpinBox (0.1-1.0, default 0.5)
- `ppo_entropy_coef_spin`: QDoubleSpinBox (0.001-0.1, default 0.01)
- `ppo_learning_rate_spin`: QDoubleSpinBox (1e-5 to 1e-3, default 3e-4, scientific notation)
- `ppo_lr_decay_check`: QCheckBox (default True)
- `ppo_lr_start_spin`: QDoubleSpinBox (enabled if decay)
- `ppo_lr_end_spin`: QDoubleSpinBox (enabled if decay)
- `ppo_gamma_spin`: QDoubleSpinBox (0.9-0.999, default 0.99)
- `ppo_gae_lambda_spin`: QDoubleSpinBox (0.9-0.99, default 0.95)
- `ppo_batch_size_spin`: QSpinBox (32-256, default 64)
- `ppo_epochs_spin`: QSpinBox (5-20, default 10)

### **Actor-Critic Networks Architecture**
```python
┌─────────────────────────────────────────────────────────────┐
│ Neural Network Architecture                                  │
│                                                              │
│ Actor Network:                                               │
│   Hidden Layers: [256, 128, 64]  (comma-separated)         │
│   Activation:    ⚫ ReLU  ○ Tanh  ○ LeakyReLU              │
│   ☑ Use LSTM Layer (64 units) for temporal patterns        │
│   ☑ Batch Normalization                                     │
│   Dropout:       [0.2 ] (0.0-0.5)                          │
│                                                              │
│ Critic Network:                                              │
│   Hidden Layers: [256, 128]  (comma-separated)             │
│   Activation:    ⚫ ReLU  ○ Tanh  ○ LeakyReLU              │
│   ☐ Use LSTM Layer                                          │
│   ☑ Batch Normalization                                     │
│   Dropout:       [0.2 ] (0.0-0.5)                          │
│                                                              │
│ Optimizer:                                                   │
│   ⚫ Adam    ○ AdamW   ○ RMSprop   ○ SGD                   │
│   Beta1: [0.9]  Beta2: [0.999]  Epsilon: [1e-8]           │
│                                                              │
│ Device:                                                      │
│   ⚫ GPU (cuda:0) ✓ Available                              │
│   ○ CPU (fallback)                                          │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `actor_hidden_layers_edit`: QLineEdit (default "256,128,64")
- `actor_activation_combo`: QComboBox ["ReLU", "Tanh", "LeakyReLU", "ELU"]
- `actor_use_lstm_check`: QCheckBox (default True)
- `actor_lstm_units_spin`: QSpinBox (32-256, default 64)
- `actor_batch_norm_check`: QCheckBox (default True)
- `actor_dropout_spin`: QDoubleSpinBox (0-0.5, default 0.2)
- `critic_hidden_layers_edit`: QLineEdit (default "256,128")
- `critic_activation_combo`: QComboBox
- `critic_use_lstm_check`: QCheckBox (default False)
- `critic_batch_norm_check`: QCheckBox
- `critic_dropout_spin`: QDoubleSpinBox
- `optimizer_combo`: QComboBox ["Adam", "AdamW", "RMSprop", "SGD"]
- `optimizer_beta1_spin`: QDoubleSpinBox (0.8-0.99, default 0.9)
- `optimizer_beta2_spin`: QDoubleSpinBox (0.9-0.999, default 0.999)
- `optimizer_epsilon_spin`: QDoubleSpinBox (1e-10 to 1e-6, default 1e-8)
- `device_combo`: QComboBox ["GPU (cuda:0)", "CPU", "Auto"]

---

## 🔧 Level 3 - Tab 2: Training Settings

```python
┌─────────────────────────────────────────────────────────────┐
│ Training Configuration                                       │
│                                                              │
│ Training Mode:                                               │
│ ⚫ Offline (Historical Data)                                 │
│   └─ Train on: Last [365] days                             │
│   └─ Validation Split: [20]%                               │
│                                                              │
│ ○ Online (Live Trading + Learning)                          │
│   └─ Update Frequency: [Daily] (Hourly/Daily/Weekly)      │
│   └─ ⚠ Warning: Can be unstable in production              │
│                                                              │
│ ○ Hybrid (Pretrain Offline, Fine-tune Online)               │
│   └─ Offline Episodes: [1000]                              │
│   └─ Online Learning Rate: [1e-5] (10× lower)             │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `training_mode_combo`: QComboBox ["Offline", "Online", "Hybrid"]
- `offline_days_spin`: QSpinBox (30-1000, default 365, enabled if Offline/Hybrid)
- `validation_split_spin`: QSpinBox (10-30%, default 20%)
- `online_update_frequency_combo`: QComboBox ["Hourly", "Daily", "Weekly"]
- `hybrid_offline_episodes_spin`: QSpinBox (100-10000, default 1000)
- `hybrid_online_lr_spin`: QDoubleSpinBox

### **Episode Configuration**
```python
┌─────────────────────────────────────────────────────────────┐
│ Training Episodes                                            │
│                                                              │
│ Total Episodes:          [5000 ] (100-50000)  🛈           │
│   Each episode = 1 portfolio rebalancing trajectory        │
│                                                              │
│ Steps per Episode:       [252  ] (trading days)  🛈        │
│   1 year = 252 trading days (default)                      │
│                                                              │
│ Parallel Environments:   [4    ] (1-16)  🛈                │
│   Run N environments in parallel (A3C-style)               │
│                                                              │
│ Replay Buffer Size:      [100000] (10k-1M)  🛈             │
│   Store last N transitions for experience replay           │
│                                                              │
│ Warmup Steps:            [1000 ] (0-10000)  🛈             │
│   Random actions before training starts                    │
│                                                              │
│ Target Update Frequency: [1000 ] steps  🛈                 │
│   Copy online network → target network every N steps      │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `total_episodes_spin`: QSpinBox (100-50000, default 5000)
- `steps_per_episode_spin`: QSpinBox (50-500, default 252)
- `parallel_envs_spin`: QSpinBox (1-16, default 4)
- `replay_buffer_size_spin`: QSpinBox (10000-1000000, default 100000)
- `warmup_steps_spin`: QSpinBox (0-10000, default 1000)
- `target_update_frequency_spin`: QSpinBox (100-10000, default 1000)

### **Early Stopping & Checkpointing**
```python
┌─────────────────────────────────────────────────────────────┐
│ Training Control                                             │
│                                                              │
│ ☑ Early Stopping                                            │
│   Metric:    ⚫ Mean Episode Reward  ○ Validation Sharpe   │
│   Patience:  [50  ] episodes without improvement           │
│   Min Delta: [0.01] minimum improvement to count           │
│                                                              │
│ ☑ Save Best Model                                           │
│   Checkpoint every: [100] episodes                          │
│   Keep best:        [5  ] checkpoints                       │
│   Save path:        [artifacts/rl_checkpoints/]            │
│                                                              │
│ ☑ TensorBoard Logging                                       │
│   Log directory:    [runs/rl_training/]                    │
│   Update every:     [10 ] episodes                          │
│                                                              │
│ ☑ Evaluation During Training                                │
│   Eval frequency:   [100] episodes                          │
│   Eval episodes:    [10 ] episodes                          │
│   Eval on:          ⚫ Validation Set  ○ Test Set          │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `early_stopping_check`: QCheckBox (default True)
- `early_stopping_metric_combo`: QComboBox ["Mean Reward", "Validation Sharpe", "Validation Sortino"]
- `early_stopping_patience_spin`: QSpinBox (10-200, default 50)
- `early_stopping_delta_spin`: QDoubleSpinBox (0.001-0.1, default 0.01)
- `save_best_model_check`: QCheckBox (default True)
- `checkpoint_frequency_spin`: QSpinBox (10-1000, default 100)
- `keep_best_checkpoints_spin`: QSpinBox (1-10, default 5)
- `checkpoint_path_edit`: QLineEdit (with Browse button)
- `tensorboard_logging_check`: QCheckBox (default True)
- `tensorboard_log_dir_edit`: QLineEdit
- `tensorboard_update_frequency_spin`: QSpinBox (1-100, default 10)
- `evaluation_during_training_check`: QCheckBox (default True)
- `eval_frequency_spin`: QSpinBox (10-1000, default 100)
- `eval_episodes_spin`: QSpinBox (5-50, default 10)
- `eval_dataset_combo`: QComboBox ["Validation Set", "Test Set"]

---

## 🔧 Level 3 - Tab 3: State & Action Space

### **State Space Configuration**
```python
┌─────────────────────────────────────────────────────────────┐
│ State Representation (Input to Actor-Critic)                │
│                                                              │
│ Portfolio Features (Always Included):                        │
│   ☑ Current Weights (n_assets)                             │
│   ☑ Current P&L (1)                                         │
│   ☑ Days in Position (n_assets)                            │
│   ☑ Unrealized P&L per Asset (n_assets)                    │
│                                                              │
│ Market Features:                                             │
│   ☑ Returns (lookback: [20] days)                          │
│   ☑ Volatility (lookback: [20] days)                       │
│   ☑ Correlation Matrix (flattened, n×n)                    │
│   ☑ Momentum (lookback: [10] days)                         │
│   ☑ RSI (14-period)                                         │
│   ☐ MACD                                                    │
│   ☐ Bollinger Bands (% position)                           │
│   ☐ ATR (normalized)                                        │
│                                                              │
│ Risk Features:                                               │
│   ☑ Portfolio VaR (95%)                                     │
│   ☑ Portfolio CVaR (95%)                                    │
│   ☑ Current Drawdown                                        │
│   ☑ Sharpe Ratio (rolling 30d)                             │
│   ☑ Sortino Ratio (rolling 30d)                            │
│   ☐ Max Drawdown Duration                                  │
│                                                              │
│ Sentiment Features (if available):                          │
│   ☑ VIX Level                                               │
│   ☑ VIX Percentile (historical)                            │
│   ☐ News Sentiment Score                                   │
│   ☐ Orderbook Imbalance                                    │
│   ☐ Twitter Sentiment                                       │
│                                                              │
│ Total State Dimension: [137] (calculated)                   │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- **Portfolio Features** (always enabled, grayed out):
  - Labels only (no widgets)
- **Market Features**:
  - `state_returns_check`: QCheckBox (default True)
  - `state_returns_lookback_spin`: QSpinBox (5-60, default 20)
  - `state_volatility_check`: QCheckBox (default True)
  - `state_volatility_lookback_spin`: QSpinBox
  - `state_correlation_check`: QCheckBox (default True)
  - `state_momentum_check`: QCheckBox (default True)
  - `state_momentum_lookback_spin`: QSpinBox (5-30, default 10)
  - `state_rsi_check`: QCheckBox (default True)
  - `state_macd_check`: QCheckBox (default False)
  - `state_bollinger_check`: QCheckBox (default False)
  - `state_atr_check`: QCheckBox (default False)
- **Risk Features**:
  - `state_var_check`: QCheckBox (default True)
  - `state_cvar_check`: QCheckBox (default True)
  - `state_drawdown_check`: QCheckBox (default True)
  - `state_sharpe_check`: QCheckBox (default True)
  - `state_sortino_check`: QCheckBox (default True)
  - `state_max_dd_duration_check`: QCheckBox (default False)
- **Sentiment Features**:
  - `state_vix_check`: QCheckBox (default True)
  - `state_vix_percentile_check`: QCheckBox (default True)
  - `state_news_sentiment_check`: QCheckBox (default False)
  - `state_orderbook_imbalance_check`: QCheckBox (default False)
  - `state_twitter_sentiment_check`: QCheckBox (default False)
- `total_state_dimension_label`: QLabel (auto-calculated)

### **Action Space Configuration**
```python
┌─────────────────────────────────────────────────────────────┐
│ Action Representation (Output from Actor)                   │
│                                                              │
│ Action Type:                                                 │
│ ⚫ Continuous - Target Portfolio Weights [0, 1]             │
│   Each asset gets weight in [0%, 100%], sum = 100%        │
│   Actor outputs: Softmax(logits) → normalized weights     │
│                                                              │
│ ○ Discrete - Rebalancing Decisions {-1, 0, +1}             │
│   -1 = Reduce weight, 0 = Hold, +1 = Increase weight      │
│   Step size: [5]% per action                               │
│                                                              │
│ Action Constraints:                                          │
│   Min Weight per Asset: [0.0 ] % (from Portfolio Tab)     │
│   Max Weight per Asset: [25.0] % (from Portfolio Tab)     │
│   ☑ Force Sum to 100% (renormalize if needed)             │
│   ☑ Long-Only (no shorting, weights ≥ 0)                  │
│                                                              │
│ Action Smoothing (reduce volatility):                       │
│   ☑ Exponential Moving Average                             │
│     α (smoothing): [0.3] (0.1=smooth, 0.9=reactive)       │
│                                                              │
│ Total Action Dimension: [4] (n_assets)                      │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `action_type_combo`: QComboBox ["Continuous (Weights)", "Discrete (Steps)"]
- `discrete_step_size_spin`: QDoubleSpinBox (1-20%, default 5%, enabled only if Discrete)
- `min_weight_per_asset_label`: QLabel (read from Portfolio Tab, grayed)
- `max_weight_per_asset_label`: QLabel (read from Portfolio Tab, grayed)
- `force_sum_100_check`: QCheckBox (default True)
- `long_only_check`: QCheckBox (default True)
- `action_smoothing_check`: QCheckBox (default True)
- `action_smoothing_alpha_spin`: QDoubleSpinBox (0.1-0.9, default 0.3)
- `total_action_dimension_label`: QLabel (auto-calculated)

---

## 🔧 Level 3 - Tab 4: Reward Function

```python
┌─────────────────────────────────────────────────────────────┐
│ Multi-Objective Reward Function                             │
│                                                              │
│ Total Reward = Σ(weight_i × component_i)                   │
│                                                              │
│ ☑ Sharpe Ratio Improvement                                  │
│   Weight: [5.0 ] (-10 to +10)  🛈                          │
│   Reward: +weight × Δ Sharpe                               │
│   └─ Example: +5.0 if Sharpe improves by 1.0              │
│                                                              │
│ ☑ Transaction Cost Penalty                                  │
│   Weight: [-10.0] (always negative!)  🛈                   │
│   Reward: weight × (cost / portfolio_value)                │
│   └─ Example: -1.0 if costs are 0.1% of portfolio         │
│                                                              │
│ ☑ VaR Violation Penalty                                     │
│   Weight: [-20.0] (always negative!)  🛈                   │
│   Reward: weight if VaR > limit, else 0                    │
│   └─ Example: -20.0 if VaR exceeds 10% limit              │
│                                                              │
│ ☑ CVaR Violation Penalty                                    │
│   Weight: [-15.0] (always negative!)  🛈                   │
│   Reward: weight if CVaR > limit, else 0                   │
│                                                              │
│ ☑ Correlation Violation Penalty                             │
│   Weight: [-5.0 ] (always negative!)  🛈                   │
│   Reward: weight if correlated_exposure > limit            │
│                                                              │
│ ☑ Diversification Bonus                                     │
│   Weight: [1.0  ] (always positive!)  🛈                   │
│   Reward: weight × (1 - HHI)  where HHI = Σ(w²)           │
│   └─ Example: +0.8 if portfolio well-diversified          │
│                                                              │
│ ☐ Drawdown Penalty                                          │
│   Weight: [-10.0] (always negative!)  🛈                   │
│   Reward: weight × drawdown_pct                            │
│                                                              │
│ ☐ Turnover Penalty                                          │
│   Weight: [-3.0 ] (always negative!)  🛈                   │
│   Reward: weight × (turnover_pct)                          │
│   └─ Penalize excessive rebalancing                       │
│                                                              │
│ ☐ Sortino Ratio Improvement                                 │
│   Weight: [3.0  ] (always positive!)  🛈                   │
│   Reward: +weight × Δ Sortino                              │
│                                                              │
│ Custom Reward Components:                                    │
│   [ + Add Custom Component ]                                │
│                                                              │
│ Preview Reward Function:                                     │
│   [ Show Reward Breakdown Chart ]                           │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `reward_sharpe_check`: QCheckBox (default True)
- `reward_sharpe_weight_spin`: QDoubleSpinBox (-10 to +10, default 5.0)
- `reward_transaction_cost_check`: QCheckBox (default True)
- `reward_transaction_cost_weight_spin`: QDoubleSpinBox (-20 to 0, default -10.0)
- `reward_var_violation_check`: QCheckBox (default True)
- `reward_var_violation_weight_spin`: QDoubleSpinBox (-50 to 0, default -20.0)
- `reward_cvar_violation_check`: QCheckBox (default True)
- `reward_cvar_violation_weight_spin`: QDoubleSpinBox (-30 to 0, default -15.0)
- `reward_correlation_violation_check`: QCheckBox (default True)
- `reward_correlation_violation_weight_spin`: QDoubleSpinBox (-10 to 0, default -5.0)
- `reward_diversification_check`: QCheckBox (default True)
- `reward_diversification_weight_spin`: QDoubleSpinBox (0 to +10, default 1.0)
- `reward_drawdown_check`: QCheckBox (default False)
- `reward_drawdown_weight_spin`: QDoubleSpinBox (-20 to 0, default -10.0)
- `reward_turnover_check`: QCheckBox (default False)
- `reward_turnover_weight_spin`: QDoubleSpinBox (-10 to 0, default -3.0)
- `reward_sortino_check`: QCheckBox (default False)
- `reward_sortino_weight_spin`: QDoubleSpinBox (0 to +10, default 3.0)
- `add_custom_reward_btn`: QPushButton (opens dialog for custom Python function)
- `preview_reward_btn`: QPushButton (shows matplotlib chart)

### **Reward Shaping & Normalization**
```python
┌─────────────────────────────────────────────────────────────┐
│ Reward Processing                                            │
│                                                              │
│ Reward Normalization:                                        │
│ ⚫ Running Z-Score (mean=0, std=1)                          │
│ ○ Min-Max Scaling [0, 1]                                    │
│ ○ Clipping [-10, +10]                                       │
│ ○ None (use raw rewards)                                    │
│                                                              │
│ Reward Clipping:                                             │
│   Min Reward: [-100.0] (prevent extreme penalties)         │
│   Max Reward: [+100.0] (prevent extreme bonuses)           │
│                                                              │
│ ☑ Reward Shaping (Potential-Based)                          │
│   Add intermediate rewards to guide learning               │
│   Example: +0.1 per step if Sharpe > 1.0                  │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `reward_normalization_combo`: QComboBox ["Z-Score", "Min-Max", "Clipping", "None"]
- `reward_min_clip_spin`: QDoubleSpinBox (-1000 to 0, default -100)
- `reward_max_clip_spin`: QDoubleSpinBox (0 to +1000, default +100)
- `reward_shaping_check`: QCheckBox (default True)

---

## 🔧 Level 3 - Tab 5: Training Progress

### **Live Training Metrics** (Real-time updates during training)
```python
┌─────────────────────────────────────────────────────────────┐
│ Training Status                                              │
│                                                              │
│ Status: ⚫ Training (Episode 1243 / 5000)                   │
│ Progress: ████████████████████░░░░░░░░  24.86%             │
│ Elapsed Time:  02:45:33                                     │
│ Est. Remaining: 08:23:15                                     │
│ ETA: 2025-01-21 18:45:00                                    │
│                                                              │
│ Current Episode Metrics:                                     │
│   Episode Reward:    +35.42                                 │
│   Episode Length:    252 steps (1 year)                    │
│   Avg Sharpe:        1.85                                   │
│   Transaction Costs: $45.20 (0.45%)                        │
│   VaR Violations:    0                                      │
│                                                              │
│ [ Pause Training ]  [ Stop Training ]  [ Save Checkpoint ] │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `training_status_label`: QLabel (auto-updated)
- `training_progress_bar`: QProgressBar (0-100%)
- `elapsed_time_label`: QLabel (HH:MM:SS)
- `remaining_time_label`: QLabel (HH:MM:SS)
- `eta_label`: QLabel (datetime)
- `current_episode_reward_label`: QLabel (live)
- `current_episode_length_label`: QLabel
- `current_avg_sharpe_label`: QLabel
- `current_transaction_costs_label`: QLabel
- `current_var_violations_label`: QLabel
- `pause_training_btn`: QPushButton
- `stop_training_btn`: QPushButton
- `save_checkpoint_btn`: QPushButton

### **Training Curves** (Matplotlib live plots)
```python
┌─────────────────────────────────────────────────────────────┐
│ Training Performance                                         │
│                                                              │
│ [Matplotlib Figure with 4 subplots]                         │
│                                                              │
│ 1) Mean Episode Reward (rolling 100 episodes)               │
│    Y-axis: Reward, X-axis: Episode                         │
│    Line: Mean ± Std                                         │
│                                                              │
│ 2) Actor Loss + Critic Loss                                 │
│    Y-axis: Loss, X-axis: Training Step                     │
│    Two lines: Actor (blue), Critic (orange)                │
│                                                              │
│ 3) Portfolio Sharpe Ratio (eval episodes)                   │
│    Y-axis: Sharpe, X-axis: Episode                         │
│    Scatter points + trendline                               │
│                                                              │
│ 4) Transaction Costs (cumulative)                           │
│    Y-axis: Total Costs ($), X-axis: Episode                │
│    Bar chart per evaluation                                 │
│                                                              │
│ Update Frequency: ⚫ Real-time  ○ Every 10 episodes         │
│ [ Export Plots PNG ]  [ Open in TensorBoard ]              │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `training_curves_canvas`: Custom QWidget with embedded matplotlib FigureCanvas (4 subplots)
- `update_frequency_combo`: QComboBox ["Real-time", "Every 10 episodes", "Every 50 episodes"]
- `export_plots_btn`: QPushButton (saves PNG)
- `open_tensorboard_btn`: QPushButton (launches TensorBoard server)

### **Best Models Leaderboard**
```python
┌─────────────────────────────────────────────────────────────┐
│ Top 5 Checkpoints (by Validation Sharpe)                    │
│                                                              │
│ Rank  Episode  Val Sharpe  Val Sortino  Max DD  Costs      │
│ ────────────────────────────────────────────────────────────│
│  1    1156     2.34        3.12         -6.2%   $320       │
│  2    1089     2.28        3.05         -7.1%   $340       │
│  3    1201     2.25        2.98         -5.8%   $380       │
│  4    982      2.20        2.89         -8.3%   $290       │
│  5    1134     2.18        2.85         -6.9%   $350       │
│                                                              │
│ Selected Model: Rank 1 (Episode 1156)                      │
│ [ Load for Deployment ]  [ Compare Models ]  [ Delete ]    │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `best_models_table`: QTableWidget (6 columns, top 5 rows)
- `selected_model_label`: QLabel (shows current selection)
- `load_for_deployment_btn`: QPushButton (loads selected model)
- `compare_models_btn`: QPushButton (opens comparison dialog)
- `delete_model_btn`: QPushButton (deletes checkpoint file)

---

## 🔧 Level 3 - Tab 6: Deployment & Testing

### **Backtest RL Agent**
```python
┌─────────────────────────────────────────────────────────────┐
│ Backtest Configuration                                       │
│                                                              │
│ Model to Test:                                               │
│   ⚫ Best Checkpoint (Episode 1156, Sharpe 2.34)            │
│   ○ Latest Checkpoint (Episode 1243)                        │
│   ○ Load from File: [Browse...]                            │
│                                                              │
│ Backtest Period:                                             │
│   Start Date:  [2024-01-01]  (date picker)                 │
│   End Date:    [2024-12-31]  (date picker)                 │
│   └─ Duration: 365 days (calculated)                       │
│                                                              │
│ Initial Capital: [$10,000.00]                               │
│ Rebalancing:     ⚫ RL Agent Decisions  ○ Daily  ○ Weekly  │
│                                                              │
│ Comparison Baselines:                                        │
│   ☑ Buy & Hold Equal Weight                                │
│   ☑ Riskfolio Optimizer (no RL)                            │
│   ☑ Random Agent                                            │
│   ☐ Custom Strategy: [___________]                         │
│                                                              │
│ [ Run Backtest ]                                             │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `backtest_model_combo`: QComboBox ["Best Checkpoint", "Latest Checkpoint", "Load from File"]
- `backtest_model_path_edit`: QLineEdit (with Browse button, enabled if "Load from File")
- `backtest_start_date`: QDateEdit (calendar popup)
- `backtest_end_date`: QDateEdit
- `backtest_duration_label`: QLabel (auto-calculated)
- `backtest_initial_capital_spin`: QDoubleSpinBox (1000-1000000, default 10000)
- `backtest_rebalancing_combo`: QComboBox ["RL Agent", "Daily", "Weekly", "Monthly"]
- `compare_equal_weight_check`: QCheckBox (default True)
- `compare_riskfolio_check`: QCheckBox (default True)
- `compare_random_check`: QCheckBox (default True)
- `compare_custom_check`: QCheckBox (default False)
- `compare_custom_edit`: QLineEdit (Python class name)
- `run_backtest_btn`: QPushButton (starts backtest)

### **Backtest Results** (After running)
```python
┌─────────────────────────────────────────────────────────────┐
│ Backtest Results (2024-01-01 to 2024-12-31)                │
│                                                              │
│ Strategy         Final  Return  Sharpe  Sortino MaxDD Costs│
│ ────────────────────────────────────────────────────────────│
│ RL Agent (PPO)   $13,456 +34.6%  2.28    3.05   -6.2% $380│
│ Riskfolio        $12,890 +28.9%  1.85    2.45   -8.1% $420│
│ Equal Weight     $11,234 +12.3%  0.98    1.23  -12.5% $150│
│ Random Agent     $ 9,567  -4.3% -0.23   -0.45  -18.3% $650│
│                                                              │
│ Winner: 🏆 RL Agent (PPO) - Best Sharpe Ratio              │
│                                                              │
│ [Equity Curve Chart]  [Drawdown Chart]  [Export CSV]       │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `backtest_results_table`: QTableWidget (7 columns, N strategies)
- `winner_label`: QLabel (highlights best strategy)
- `show_equity_curve_btn`: QPushButton (opens matplotlib plot)
- `show_drawdown_chart_btn`: QPushButton
- `export_results_csv_btn`: QPushButton

### **Deploy to Live Trading**
```python
┌─────────────────────────────────────────────────────────────┐
│ Production Deployment                                        │
│                                                              │
│ ⚠ WARNING: Deploy to live trading ONLY after validation!   │
│                                                              │
│ Deployment Mode:                                             │
│ ⚫ RL Agent Only (RL decisions override Riskfolio)          │
│ ○ RL + Riskfolio Hybrid (average both agents)              │
│ ○ RL Advisory (RL suggests, Riskfolio decides)             │
│                                                              │
│ Safety Limits:                                               │
│   ☑ Max Daily Trades: [10  ] (prevent overtrading)        │
│   ☑ Max Position Deviation from Riskfolio: [10]%          │
│   ☑ Disable RL if Sharpe < [0.5] for [7] days             │
│   ☑ Emergency Stop if Portfolio Loss > [15]%               │
│                                                              │
│ Monitoring:                                                  │
│   ☑ Log all RL decisions to database                       │
│   ☑ Email alerts on anomalous actions                      │
│   ☑ Daily performance report                                │
│                                                              │
│ Current Status: ○ Not Deployed                              │
│                                                              │
│ [ Deploy to Production ]  (requires admin password)         │
│ [ Disable RL Agent ]                                         │
└─────────────────────────────────────────────────────────────┘
```

**Widgets**:
- `deployment_mode_combo`: QComboBox ["RL Only", "RL + Riskfolio Hybrid", "RL Advisory"]
- `max_daily_trades_check`: QCheckBox
- `max_daily_trades_spin`: QSpinBox (1-100, default 10)
- `max_deviation_check`: QCheckBox
- `max_deviation_spin`: QSpinBox (5-50%, default 10%)
- `disable_on_low_sharpe_check`: QCheckBox
- `disable_sharpe_threshold_spin`: QDoubleSpinBox (0-2.0, default 0.5)
- `disable_sharpe_days_spin`: QSpinBox (1-30, default 7)
- `emergency_stop_check`: QCheckBox
- `emergency_stop_loss_spin`: QSpinBox (5-50%, default 15%)
- `log_decisions_check`: QCheckBox (default True)
- `email_alerts_check`: QCheckBox (default True)
- `daily_report_check`: QCheckBox (default True)
- `deployment_status_label`: QLabel ("Not Deployed" / "Active" with color)
- `deploy_to_production_btn`: QPushButton (protected, requires password dialog)
- `disable_rl_agent_btn`: QPushButton (immediate disable)

---

## 🔗 Backend Implementation

### **Key Classes**

#### 1. **RLPortfolioManager** (`rl_portfolio_manager.py`)
```python
class RLPortfolioManager:
    """
    Manages RL agent for portfolio optimization
    
    Integrates with:
    - PortfolioOptimizer (Riskfolio)
    - AutomatedTradingEngine
    - RL Agent (PPO/SAC/A3C/TD3)
    """
    
    def __init__(self, config: RLConfig, portfolio_optimizer, trading_engine):
        self.config = config
        self.portfolio_optimizer = portfolio_optimizer
        self.trading_engine = trading_engine
        
        # Create RL agent based on config
        self.agent = self._create_agent()
        
        # Gym environment
        self.env = PortfolioEnvironment(...)
    
    def train(self, episodes: int):
        """Train RL agent"""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                state = next_state
            
            # Update agent
            self.agent.update()
            
            # Log metrics
            self._log_episode_metrics(episode, info)
    
    def get_target_weights(self, state: np.ndarray) -> np.ndarray:
        """Get portfolio weights from RL agent"""
        action = self.agent.select_action(state, deterministic=True)
        weights = self._action_to_weights(action)
        return weights
    
    def _create_agent(self):
        """Factory for RL agents"""
        if self.config.algorithm == 'PPO':
            return PPOAgent(...)
        elif self.config.algorithm == 'SAC':
            return SACAgent(...)
        elif self.config.algorithm == 'A3C':
            return A3CAgent(...)
        elif self.config.algorithm == 'TD3':
            return TD3Agent(...)
```

#### 2. **PortfolioEnvironment** (`environments/portfolio_env.py`)
```python
class PortfolioEnvironment(gym.Env):
    """
    OpenAI Gym environment for portfolio management
    
    State: [weights, returns, volatility, correlations, VaR, ...]
    Action: Target portfolio weights [w1, w2, ..., wn]
    Reward: Multi-objective (Sharpe, costs, risk violations, ...)
    """
    
    def __init__(self, data, config):
        self.data = data
        self.config = config
        
        # State space: 137-dim continuous
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(137,)
        )
        
        # Action space: n_assets-dim continuous [0, 1]
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_assets,)
        )
    
    def reset(self):
        """Reset to initial state"""
        self.current_step = 0
        self.portfolio_weights = np.array([1/n_assets] * n_assets)
        return self._get_state()
    
    def step(self, action):
        """Execute action, return next state and reward"""
        # Action = target weights
        new_weights = self._normalize_weights(action)
        
        # Calculate reward
        reward = self._calculate_reward(self.portfolio_weights, new_weights)
        
        # Update state
        self.portfolio_weights = new_weights
        self.current_step += 1
        
        # Check termination
        done = self.current_step >= self.max_steps
        
        state = self._get_state()
        info = self._get_info()
        
        return state, reward, done, info
    
    def _calculate_reward(self, old_weights, new_weights):
        """Multi-objective reward function"""
        reward = 0.0
        
        # Sharpe improvement
        if self.config.reward_sharpe:
            sharpe_delta = self._calculate_sharpe_delta(old_weights, new_weights)
            reward += self.config.reward_sharpe_weight * sharpe_delta
        
        # Transaction costs
        if self.config.reward_transaction_cost:
            turnover = np.sum(np.abs(new_weights - old_weights))
            cost = turnover * self.config.transaction_cost_bps / 10000
            reward += self.config.reward_transaction_cost_weight * cost
        
        # VaR violation
        if self.config.reward_var_violation:
            var = self._calculate_var(new_weights)
            if var > self.config.max_var:
                reward += self.config.reward_var_violation_weight
        
        # ... other reward components
        
        return reward
```

#### 3. **PPOAgent** (`actor_critic/ppo_agent.py`)
```python
class PPOAgent:
    """
    Proximal Policy Optimization agent
    
    Based on: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    """
    
    def __init__(self, state_dim, action_dim, config):
        self.actor = ActorNetwork(state_dim, action_dim, config)
        self.critic = CriticNetwork(state_dim, config)
        self.old_policy = copy.deepcopy(self.actor)
        
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.config = config
    
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.actor.sample(state_tensor, deterministic)
        
        return action.cpu().numpy()[0]
    
    def update(self):
        """PPO update step"""
        # Sample minibatch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # Calculate advantages using GAE
        advantages = self._calculate_gae(states, rewards, next_states, dones)
        
        # PPO policy update
        for _ in range(self.config.ppo_epochs):
            # Get current policy probabilities
            _, log_probs_new = self.actor.evaluate(states, actions)
            
            # Get old policy probabilities
            with torch.no_grad():
                _, log_probs_old = self.old_policy.evaluate(states, actions)
            
            # Calculate ratio and clipped surrogate
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 
                1 - self.config.ppo_clip_epsilon, 
                1 + self.config.ppo_clip_epsilon
            ) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            values = self.critic(states)
            critic_loss = F.mse_loss(values, rewards + self.config.gamma * next_values)
            
            # Entropy bonus
            entropy = self.actor.entropy(states).mean()
            
            # Total loss
            total_loss = (
                actor_loss + 
                self.config.ppo_value_coef * critic_loss -
                self.config.ppo_entropy_coef * entropy
            )
            
            # Update
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            total_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        
        # Update old policy
        self.old_policy.load_state_dict(self.actor.state_dict())
```

---

## 🔗 Integration with Trading Intelligence

### **Modified Portfolio Tab Structure**
```
Trading Intelligence (Level 1)
├── Portfolio (Level 2)
│   ├── Portfolio Optimization (Level 3)
│   ├── Transaction Costs (Level 3)
│   ├── Risk Management (Level 3)
│   ├── Correlation (Level 3)
│   ├── Rebalancing (Level 3)
│   └── Performance (Level 3)
├── Signals (Level 2)
└── RL Agent (Level 2) ← NEW
    ├── Agent Configuration (Level 3)
    ├── Training Settings (Level 3)
    ├── State & Action Space (Level 3)
    ├── Reward Function (Level 3)
    ├── Training Progress (Level 3)
    └── Deployment & Testing (Level 3)
```

### **Bridge: RL ↔ Riskfolio ↔ Trading Engine**
```python
class IntelligentPortfolioManager:
    """
    Master controller integrating:
    1. Riskfolio Optimizer (analytical)
    2. RL Agent (learned)
    3. Trading Engine (execution)
    """
    
    def __init__(self, config):
        self.riskfolio = PortfolioOptimizer(...)
        self.rl_manager = RLPortfolioManager(...) if config.enable_rl else None
        self.trading_engine = AutomatedTradingEngine(...)
    
    def get_optimal_weights(self, market_data):
        """Get portfolio weights from both systems"""
        
        # 1. Riskfolio analytical optimization
        riskfolio_weights = self.riskfolio.optimize(market_data)
        
        # 2. RL agent learned policy
        if self.rl_manager and self.rl_manager.enabled:
            state = self._prepare_state(market_data)
            rl_weights = self.rl_manager.get_target_weights(state)
            
            # Combine based on deployment mode
            if self.config.deployment_mode == 'RL Only':
                final_weights = rl_weights
            elif self.config.deployment_mode == 'RL + Riskfolio Hybrid':
                # Weighted average
                final_weights = (
                    0.6 * rl_weights + 
                    0.4 * riskfolio_weights
                )
            elif self.config.deployment_mode == 'RL Advisory':
                # Riskfolio decides, RL suggests
                final_weights = riskfolio_weights
                # Log RL suggestion for analysis
                self._log_rl_suggestion(rl_weights)
        else:
            final_weights = riskfolio_weights
        
        return final_weights
```

---

## 📊 Summary

**Total Implementation**:
- **New Module**: `src/forex_diffusion/rl/` (~5,000 lines)
- **New UI Tab**: `rl_config_tab.py` (~2,000 lines)
- **6 Level-3 Sub-Tabs**: Agent Config, Training, State/Action, Reward, Progress, Deployment
- **Total Widgets**: ~120 widgets (settings + live monitoring)
- **i18n Tooltips**: ~120 professional tooltips
- **Backend Classes**: 10 new classes (Agent, Environment, Networks, Trainer, ...)

**Key Features**:
✅ **4 RL Algorithms**: PPO, SAC, A3C, TD3
✅ **Customizable State Space**: 137-dim default, extendable
✅ **Multi-Objective Rewards**: 9 components, fully configurable
✅ **Live Training Monitoring**: Real-time plots, TensorBoard
✅ **Comprehensive Backtesting**: Compare RL vs Riskfolio vs baselines
✅ **Production Deployment**: Safety limits, monitoring, emergency stops
✅ **Hybrid Modes**: RL-only, RL+Riskfolio, RL-advisory
✅ **Fully Integrated**: Works with existing Portfolio Tab and Trading Engine

---

**Pronto per implementazione?**

**A)** Fase 1: RL Backend (PPO agent + Environment) - 2 settimane
**B)** Fase 2: RL UI Tab (120 widgets + tooltips) - 1 settimana
**C)** Fase 3: Training Infrastructure (TensorBoard, checkpoints) - 1 settimana
**D)** Fase 4: Integration with Portfolio Tab - 1 settimana
**E)** Fase 5: Testing & Validation - 1 settimana

**TOTALE: 6 settimane per sistema RL completo**

Quale fase vuoi iniziare?
