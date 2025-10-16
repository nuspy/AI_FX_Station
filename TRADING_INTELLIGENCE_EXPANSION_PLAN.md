# 🎯 Trading Intelligence Tab - Struttura Completa a 3 Livelli

## 📊 Architettura Proposta

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING INTELLIGENCE (Level 1)                │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐│
│  │   PORTFOLIO TAB      │  │       SIGNALS TAB                ││
│  │   (Level 2 - Nested) │  │   (Level 2 - Display only)       ││
│  │                      │  │                                  ││
│  │  [6 Sub-Tabs L3]     │  │   • Refresh button              ││
│  └──────────────────────┘  │   • Limit spinbox               ││
│                            │   • Signals table                ││
│                            └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 PORTFOLIO TAB - Level 2 (Container con 6 Sub-Tabs Level 3)

### **LEVEL 3 - Tab 1: Portfolio Optimization**
*Settings per Riskfolio-Lib core optimization*

#### **Section A: Risk Measure & Objective**
```python
┌─────────────────────────────────────────────────────────┐
│ Risk Measure                                            │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ⚫ CVaR - Conditional Value at Risk (Expected Short) │ │
│ │ ○ MV - Mean-Variance (Standard Deviation)          │ │
│ │ ○ CDaR - Conditional Drawdown at Risk              │ │
│ │ ○ EVaR - Entropic Value at Risk                    │ │
│ │ ○ WR - Worst Realization (Worst Case)              │ │
│ │ ○ MDD - Maximum Drawdown                           │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                           │
│ Optimization Objective                                   │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ⚫ Sharpe - Maximum Sharpe Ratio                    │ │
│ │ ○ MinRisk - Minimum Risk                           │ │
│ │ ○ Utility - Maximum Utility Function               │ │
│ │ ○ MaxRet - Maximum Return                          │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `risk_measure_combo`: QComboBox con 6 opzioni
- `objective_combo`: QComboBox con 4 opzioni
- `risk_free_rate_spin`: QDoubleSpinBox (0-10%, default 2%)
- `risk_aversion_spin`: QDoubleSpinBox (0.1-10, default 1.0)

#### **Section B: Portfolio Constraints**
```python
┌─────────────────────────────────────────────────────────┐
│ Position Size Constraints                               │
│                                                           │
│ Max Weight per Asset:    [25.0] %  🛈                   │
│ Min Weight per Asset:    [1.0 ] %  🛈                   │
│ Max Leverage:            [1.0 ] ×  🛈                   │
│ Max Total Exposure:      [100 ] %  🛈                   │
│                                                           │
│ ☑ Long-Only Portfolio (No Shorting)                     │
│ ☐ Allow Short Positions                                 │
│ ☐ Market Neutral (Beta=0)                               │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `max_weight_spin`: QDoubleSpinBox (1-100%, default 25%)
- `min_weight_spin`: QDoubleSpinBox (0-50%, default 1%)
- `max_leverage_spin`: QDoubleSpinBox (1-5×, default 1×)
- `max_exposure_spin`: QDoubleSpinBox (0-500%, default 100%)
- `long_only_check`: QCheckBox (default True)
- `allow_short_check`: QCheckBox (default False)
- `market_neutral_check`: QCheckBox (default False)

#### **Section C: Rolling Window & Update Frequency**
```python
┌─────────────────────────────────────────────────────────┐
│ Data Window Settings                                     │
│                                                           │
│ Rolling Window (days):   [60  ] days  🛈                │
│ Min Data Required:       [252 ] days  🛈                │
│                                                           │
│ Reoptimization Frequency                                 │
│ ⚫ Weekly (every 7 days)                                 │
│ ○ Daily (every day)                                      │
│ ○ Custom: [_7_] days                                     │
│                                                           │
│ ☑ Use Exponential Weighting (recent data weighted more) │
│ └─ Decay Factor: [0.94] (0.5-0.99)                      │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `rolling_window_spin`: QSpinBox (20-500 days, default 60)
- `min_data_required_spin`: QSpinBox (60-1000 days, default 252)
- `reoptimize_frequency_combo`: QComboBox ["Daily", "Weekly", "Monthly", "Custom"]
- `reoptimize_custom_spin`: QSpinBox (1-90 days, enabled only if Custom)
- `exponential_weighting_check`: QCheckBox (default True)
- `decay_factor_spin`: QDoubleSpinBox (0.5-0.99, default 0.94)

---

### **LEVEL 3 - Tab 2: Transaction Costs & Execution**
*Settings per ottimizzazione costi e smart execution*

#### **Section A: Transaction Cost Model**
```python
┌─────────────────────────────────────────────────────────┐
│ Cost Structure (per round-trip trade)                   │
│                                                           │
│ Spread (bps):            [5.0 ] basis points  🛈        │
│ Commission ($ per lot):  [0.00] USD           🛈        │
│ Slippage (bps):          [1.0 ] basis points  🛈        │
│                                                           │
│ Total Estimated Cost:    [6.0 bps] = [0.06%]           │
│                                                           │
│ Spread Model                                             │
│ ⚫ Fixed (use value above)                               │
│ ○ Variable (from live DOM data)                         │
│ ○ Historical Average (60-day mean)                      │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `spread_bps_spin`: QDoubleSpinBox (0.5-50 bps, default 5.0)
- `commission_spin`: QDoubleSpinBox (0-10 USD, default 0)
- `slippage_bps_spin`: QDoubleSpinBox (0.1-10 bps, default 1.0)
- `total_cost_label`: QLabel (calculated, read-only)
- `spread_model_combo`: QComboBox ["Fixed", "Variable (DOM)", "Historical"]

#### **Section B: No-Trade Zones (Cost Optimization)**
```python
┌─────────────────────────────────────────────────────────┐
│ Rebalancing Thresholds                                   │
│                                                           │
│ Min Weight Change:       [2.0 ] %  🛈                   │
│   → Skip rebalance if change < threshold                │
│                                                           │
│ Min Trade Size:          [100 ] USD  🛈                 │
│   → Skip trades smaller than this                       │
│                                                           │
│ Max Trades per Day:      [10  ] trades  🛈              │
│   → Limit total daily turnover                          │
│                                                           │
│ ☑ Cost-Aware Optimization                               │
│   Penalize frequent rebalancing:                        │
│   Cost Penalty Lambda: [1.0] (0.1=aggressive, 5.0=safe)│
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `min_weight_change_spin`: QDoubleSpinBox (0.5-10%, default 2%)
- `min_trade_size_spin`: QDoubleSpinBox (10-1000 USD, default 100)
- `max_trades_per_day_spin`: QSpinBox (1-100, default 10)
- `cost_aware_optimization_check`: QCheckBox (default True)
- `cost_penalty_lambda_spin`: QDoubleSpinBox (0.1-5.0, default 1.0)

#### **Section C: Smart Execution (DOM Integration)**
```python
┌─────────────────────────────────────────────────────────┐
│ Order Execution Strategy                                 │
│                                                           │
│ ⚫ TWAP - Time-Weighted Average Price                    │
│   └─ Duration: [300] seconds (1-3600)                   │
│                                                           │
│ ○ VWAP - Volume-Weighted Average Price                  │
│   └─ Target % of Volume: [10%] (5-50%)                  │
│                                                           │
│ ○ Immediate - Market Order (no slicing)                 │
│                                                           │
│ ☑ Use DOM Liquidity Data                                │
│   └─ Skip execution if spread > [20] bps                │
│                                                           │
│ ☑ Avoid High Volatility Periods                         │
│   └─ Skip if ATR percentile > [95]% (50-99%)           │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `execution_strategy_combo`: QComboBox ["TWAP", "VWAP", "Immediate"]
- `twap_duration_spin`: QSpinBox (30-3600 sec, default 300)
- `vwap_target_volume_spin`: QDoubleSpinBox (5-50%, default 10%)
- `use_dom_liquidity_check`: QCheckBox (default True)
- `max_spread_threshold_spin`: QDoubleSpinBox (5-100 bps, default 20)
- `avoid_high_volatility_check`: QCheckBox (default True)
- `atr_percentile_threshold_spin`: QSpinBox (50-99%, default 95%)

---

### **LEVEL 3 - Tab 3: Risk Management & Stop Loss**
*Portfolio-level risk controls*

#### **Section A: Portfolio Risk Limits**
```python
┌─────────────────────────────────────────────────────────┐
│ Portfolio-Level Risk Constraints                         │
│                                                           │
│ Max Portfolio VaR (95%):  [10.0] %  🛈                  │
│ Max Portfolio CVaR (95%): [15.0] %  🛈                  │
│ Max Drawdown Allowed:     [20.0] %  🛈                  │
│                                                           │
│ Risk Calculation Method                                  │
│ ⚫ Historical Simulation (500 scenarios)                 │
│ ○ Monte Carlo (10,000 simulations)                      │
│ ○ Parametric (Gaussian assumption)                      │
│                                                           │
│ Confidence Level:         [95  ] % (90-99%)             │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `max_portfolio_var_spin`: QDoubleSpinBox (5-50%, default 10%)
- `max_portfolio_cvar_spin`: QDoubleSpinBox (5-50%, default 15%)
- `max_drawdown_spin`: QDoubleSpinBox (5-50%, default 20%)
- `risk_calc_method_combo`: QComboBox ["Historical", "Monte Carlo", "Parametric"]
- `confidence_level_spin`: QSpinBox (90-99%, default 95%)

#### **Section B: Position-Level Stop Loss (Risk Budgeting)**
```python
┌─────────────────────────────────────────────────────────┐
│ Stop Loss Calculation Method                             │
│                                                           │
│ ⚫ Risk Budgeting (from Marginal VaR)                    │
│   Each position gets stop loss based on its contribution│
│   to portfolio risk. Correlated assets have tighter SL. │
│                                                           │
│ ○ Fixed Percentage (traditional)                        │
│   └─ Stop Loss: [2.0]% below entry                     │
│                                                           │
│ ○ ATR-Based (volatility-adaptive)                       │
│   └─ ATR Multiplier: [2.0]× (1.0-5.0×)                 │
│                                                           │
│ Risk Budget per Position                                 │
│ Max Risk Contribution:    [3.0 ] %  🛈                  │
│   → Position SL ensures it contributes max 3% to VaR   │
│                                                           │
│ ☑ Trailing Stop Loss                                    │
│   └─ Trailing Activation: [50]% of profit (0-100%)     │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `stop_loss_method_combo`: QComboBox ["Risk Budgeting", "Fixed %", "ATR-Based"]
- `fixed_stop_loss_spin`: QDoubleSpinBox (0.5-10%, default 2%, enabled only if Fixed)
- `atr_multiplier_spin`: QDoubleSpinBox (1-5×, default 2×, enabled only if ATR)
- `max_risk_contribution_spin`: QDoubleSpinBox (1-10%, default 3%)
- `trailing_stop_check`: QCheckBox (default True)
- `trailing_activation_spin`: QSpinBox (10-100%, default 50%)

#### **Section C: Adaptive Risk Adjustment**
```python
┌─────────────────────────────────────────────────────────┐
│ Dynamic Risk Scaling                                     │
│                                                           │
│ ☑ Reduce Exposure in High Volatility                    │
│   VIX Threshold:          [25  ] (15-50)               │
│   Risk Reduction Factor:  [0.5 ] (0.1-1.0)             │
│   → If VIX > 25, multiply all positions by 0.5×        │
│                                                           │
│ ☑ Reduce Exposure After Losses                          │
│   Drawdown Threshold:     [10  ] %                      │
│   Risk Reduction Factor:  [0.7 ] (0.1-1.0)             │
│   → If portfolio down 10%, reduce sizing to 70%        │
│                                                           │
│ ☑ Increase Exposure in Low Volatility                   │
│   VIX Threshold:          [12  ] (5-20)                │
│   Risk Increase Factor:   [1.3 ] (1.0-2.0)             │
│   → If VIX < 12, multiply positions by 1.3×            │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `reduce_on_high_vol_check`: QCheckBox (default True)
- `vix_high_threshold_spin`: QSpinBox (15-50, default 25)
- `high_vol_reduction_spin`: QDoubleSpinBox (0.1-1.0, default 0.5)
- `reduce_on_drawdown_check`: QCheckBox (default True)
- `drawdown_threshold_spin`: QDoubleSpinBox (5-30%, default 10%)
- `drawdown_reduction_spin`: QDoubleSpinBox (0.1-1.0, default 0.7)
- `increase_on_low_vol_check`: QCheckBox (default False)
- `vix_low_threshold_spin`: QSpinBox (5-20, default 12)
- `low_vol_increase_spin`: QDoubleSpinBox (1.0-2.0, default 1.3)

---

### **LEVEL 3 - Tab 4: Correlation & Diversification**
*Correlation-aware order filtering and hedging*

#### **Section A: Correlation Monitoring**
```python
┌─────────────────────────────────────────────────────────┐
│ Correlation Heatmap (Live Updates)                      │
│                                                           │
│           EUR/USD  GBP/USD  USD/JPY  AUD/USD            │
│ EUR/USD    1.00    0.87    -0.45     0.62               │
│ GBP/USD    0.87    1.00    -0.52     0.58               │
│ USD/JPY   -0.45   -0.52     1.00    -0.38               │
│ AUD/USD    0.62    0.58    -0.38     1.00               │
│                                                           │
│ Rolling Window: [30] days (7-90 days)                   │
│                                                           │
│ [⚠] High Correlation Detected:                          │
│     EUR/USD ↔ GBP/USD: 0.87 (>0.70 threshold)          │
│     Current Combined Exposure: 45% (limit: 50%)         │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `correlation_heatmap`: Custom QWidget con matplotlib canvas
- `correlation_window_spin`: QSpinBox (7-90 days, default 30)
- `correlation_warnings_table`: QTableWidget (read-only, shows alerts)

#### **Section B: Correlation Constraints**
```python
┌─────────────────────────────────────────────────────────┐
│ Diversification Rules                                    │
│                                                           │
│ Correlation Threshold:    [0.70] (0.5-0.95)  🛈         │
│   → Assets with correlation >0.70 are "correlated"     │
│                                                           │
│ Max Correlated Exposure:  [50  ] %  🛈                  │
│   → Max total weight in correlated assets              │
│                                                           │
│ ☑ Block Orders Violating Correlation Limit             │
│   Order will be rejected if it pushes correlated       │
│   exposure above threshold                              │
│                                                           │
│ ☐ Auto-Suggest Hedge Trades                            │
│   When correlation limit breached, suggest:            │
│   • Negatively correlated assets (corr < -0.5)         │
│   • Hedge size: [30]% of concentrated exposure         │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `correlation_threshold_spin`: QDoubleSpinBox (0.5-0.95, default 0.70)
- `max_correlated_exposure_spin`: QDoubleSpinBox (10-100%, default 50%)
- `block_correlated_orders_check`: QCheckBox (default True)
- `auto_hedge_check`: QCheckBox (default False)
- `hedge_size_pct_spin`: QSpinBox (10-50%, default 30%)

#### **Section C: Sector/Factor Limits** (Advanced)
```python
┌─────────────────────────────────────────────────────────┐
│ Factor Exposure Limits (Optional - Advanced)            │
│                                                           │
│ ☐ Enable Factor-Based Constraints                       │
│                                                           │
│ Currency Exposure Limits:                                │
│   Max USD Exposure:  [60  ] % (single currency)        │
│   Max EUR Exposure:  [40  ] %                           │
│   Max GBP Exposure:  [30  ] %                           │
│   Max JPY Exposure:  [30  ] %                           │
│                                                           │
│ Current Exposures:                                       │
│   USD:  [35%] ████████░░░  Safe ✓                      │
│   EUR:  [45%] ███████████░  ⚠ Near Limit               │
│   GBP:  [20%] ██████░░░░░  Safe ✓                      │
│   JPY:  [15%] ████░░░░░░░  Safe ✓                      │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `enable_factor_constraints_check`: QCheckBox (default False)
- `max_usd_exposure_spin`: QSpinBox (10-100%, default 60%)
- `max_eur_exposure_spin`: QSpinBox (10-100%, default 40%)
- `max_gbp_exposure_spin`: QSpinBox (10-100%, default 30%)
- `max_jpy_exposure_spin`: QSpinBox (10-100%, default 30%)
- `currency_exposure_bars`: Custom QWidget con progress bars

---

### **LEVEL 3 - Tab 5: Rebalancing & Triggers**
*When and how to rebalance portfolio*

#### **Section A: Rebalancing Strategy**
```python
┌─────────────────────────────────────────────────────────┐
│ Rebalancing Triggers (Multiple can be active)           │
│                                                           │
│ ☑ Time-Based Rebalancing                                │
│   Frequency: ⚫ Weekly  ○ Daily  ○ Monthly              │
│   Next Rebalance: 2025-01-23 14:00 UTC (in 3d 5h)      │
│                                                           │
│ ☑ Threshold-Based Rebalancing                           │
│   Drift Threshold:        [5.0 ] %  🛈                  │
│   Current Max Drift:      [3.2%] (EUR/USD)             │
│   Status: ✓ Within tolerance                           │
│                                                           │
│ ☑ Volatility-Based Rebalancing                          │
│   High Vol Threshold:     [25  ] VIX  🛈                │
│   Low Vol Threshold:      [12  ] VIX  🛈                │
│   Current VIX:            [18.5]                        │
│   → Normal volatility, standard frequency               │
│                                                           │
│ ☐ Event-Based Rebalancing                               │
│   Trigger on: ☐ Major News  ☐ Regime Change            │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `time_rebalancing_check`: QCheckBox (default True)
- `rebalance_frequency_combo`: QComboBox ["Daily", "Weekly", "Monthly"]
- `next_rebalance_label`: QLabel (calculated, countdown timer)
- `threshold_rebalancing_check`: QCheckBox (default True)
- `drift_threshold_spin`: QDoubleSpinBox (1-20%, default 5%)
- `current_drift_label`: QLabel (live updated)
- `volatility_rebalancing_check`: QCheckBox (default True)
- `high_vol_rebalance_spin`: QSpinBox (20-50, default 25)
- `low_vol_rebalance_spin`: QSpinBox (5-20, default 12)
- `current_vix_label`: QLabel (live from VIX service)
- `event_rebalancing_check`: QCheckBox (default False)

#### **Section B: Rebalancing Preview**
```python
┌─────────────────────────────────────────────────────────┐
│ Pending Rebalancing Trades (If Executed Now)            │
│                                                           │
│ Symbol    Current  Target  Change  Trade     Cost       │
│ ────────────────────────────────────────────────────────│
│ EUR/USD   20.0%   25.0%   +5.0%   +$500   $3.00 (6bps)│
│ GBP/USD   18.0%   15.0%   -3.0%   -$300   $1.80 (6bps)│
│ USD/JPY   12.0%   10.0%   -2.0%   -$200   $1.20 (6bps)│
│ AUD/USD    8.0%   12.0%   +4.0%   +$400   $2.40 (6bps)│
│ ────────────────────────────────────────────────────────│
│ TOTAL:                             $800    $8.40        │
│                                                           │
│ Estimated Impact:                                        │
│   Total Transaction Costs:  $8.40 (0.084% of portfolio)│
│   Expected Benefit:         +0.15% Sharpe improvement   │
│   Net Benefit:              +0.066% (benefit - costs)   │
│                                                           │
│ [ Preview Rebalance ]  [ Execute Rebalance ]            │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `rebalancing_preview_table`: QTableWidget (7 columns, live calculated)
- `total_cost_label`: QLabel (sum of costs)
- `estimated_sharpe_improvement_label`: QLabel (calculated from Riskfolio)
- `net_benefit_label`: QLabel (benefit - costs)
- `preview_rebalance_btn`: QPushButton (generates preview)
- `execute_rebalance_btn`: QPushButton (executes trades)

#### **Section C: Rebalancing History**
```python
┌─────────────────────────────────────────────────────────┐
│ Recent Rebalancing Events                                │
│                                                           │
│ Date         Trigger      Trades  Cost   Benefit        │
│ ──────────────────────────────────────────────────────  │
│ 2025-01-20  Time-based    4      $8.40  +0.12% Sharpe │
│ 2025-01-13  Threshold     3      $6.20  +0.08% Sharpe │
│ 2025-01-06  Time-based    5     $10.50  +0.15% Sharpe │
│ 2024-12-30  Volatility    6     $12.30  +0.22% Sharpe │
│                                                           │
│ Total Rebalancing Costs (30d): $37.40 (0.37% of AUM)   │
│ Total Benefit (30d):           +0.57% Sharpe            │
│ Net Impact:                    +0.20% Sharpe            │
│                                                           │
│ [ Export History CSV ]                                   │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `rebalancing_history_table`: QTableWidget (5 columns)
- `total_costs_30d_label`: QLabel (calculated)
- `total_benefit_30d_label`: QLabel (calculated)
- `net_impact_30d_label`: QLabel (calculated)
- `export_history_btn`: QPushButton (exports to CSV)

---

### **LEVEL 3 - Tab 6: Performance & Analytics**
*Real-time portfolio performance monitoring*

#### **Section A: Portfolio Performance Metrics**
```python
┌─────────────────────────────────────────────────────────┐
│ Real-Time Performance (Updated every 5 seconds)         │
│                                                           │
│ Portfolio Value:     $10,342.50  (+3.43%)              │
│ Daily P&L:           +$342.50    (+3.43%)              │
│ Weekly P&L:          +$1,205.30  (+13.22%)             │
│ Monthly P&L:         +$2,450.80  (+31.08%)             │
│ YTD P&L:             +$2,450.80  (+31.08%)             │
│                                                           │
│ Risk-Adjusted Returns:                                   │
│   Sharpe Ratio (30d):     1.85  (Rolling)              │
│   Sortino Ratio (30d):    2.34                         │
│   Calmar Ratio (30d):     3.12                         │
│                                                           │
│ Risk Metrics:                                            │
│   Current VaR (95%):      -$450 (-4.35%)               │
│   Current CVaR (95%):     -$680 (-6.57%)               │
│   Max Drawdown (30d):     -8.2%                        │
│   Current Drawdown:       0.0% (at peak)               │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `portfolio_value_label`: QLabel (live updated)
- `daily_pnl_label`: QLabel (green/red colored)
- `weekly_pnl_label`: QLabel
- `monthly_pnl_label`: QLabel
- `ytd_pnl_label`: QLabel
- `sharpe_ratio_label`: QLabel (rolling 30d)
- `sortino_ratio_label`: QLabel
- `calmar_ratio_label`: QLabel
- `current_var_label`: QLabel
- `current_cvar_label`: QLabel
- `max_drawdown_label`: QLabel
- `current_drawdown_label`: QLabel

#### **Section B: Position Breakdown**
```python
┌─────────────────────────────────────────────────────────┐
│ Current Positions                                        │
│                                                           │
│ Symbol   Weight Target Drift  P&L    Risk Contrib.     │
│ ──────────────────────────────────────────────────────  │
│ EUR/USD  25.0%  25.0%  0.0%  +$85   2.8% (VaR)        │
│ GBP/USD  15.0%  15.0%  0.0%  +$52   2.1%              │
│ USD/JPY  10.0%  10.0%  0.0%  +$38   1.5%              │
│ AUD/USD  12.0%  12.0%  0.0%  +$45   1.8%              │
│ CASH     38.0%  38.0%  0.0%  +$0    0.0%              │
│                                                           │
│ Portfolio Risk Contribution:                             │
│ ████████████████░░░░  8.2% / 10.0% limit               │
│                                                           │
│ Correlation Status:                                      │
│   ✓ All constraints satisfied                           │
│   Max correlated exposure: 40% (limit: 50%)            │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `positions_breakdown_table`: QTableWidget (7 columns, live updated)
- `portfolio_risk_progressbar`: QProgressBar (shows risk / limit)
- `correlation_status_label`: QLabel (✓ or ⚠)

#### **Section C: Optimization Tracking**
```python
┌─────────────────────────────────────────────────────────┐
│ Portfolio vs Optimal Weights (Tracking Error)           │
│                                                           │
│ Chart: Target Weights (bars) vs Actual Weights (dots)   │
│ [Matplotlib chart with dual bar+scatter plot]           │
│                                                           │
│ Tracking Error:          0.23% (RMS deviation)          │
│ Avg Deviation:           0.15%                          │
│ Max Deviation:           0.35% (EUR/USD)                │
│                                                           │
│ Rebalancing Effectiveness:                               │
│   Avg Time in Drift:     2.3 days                       │
│   Rebalancing Frequency: 5.2 days avg                   │
│   Cost per Rebalance:    $8.50 avg                      │
│   Benefit per Rebalance: +0.12% Sharpe avg              │
│                                                           │
│ [ View Optimization History ]                            │
└─────────────────────────────────────────────────────────┘
```

**Widgets**:
- `tracking_chart`: Custom QWidget con matplotlib (target vs actual)
- `tracking_error_label`: QLabel (RMS)
- `avg_deviation_label`: QLabel
- `max_deviation_label`: QLabel
- `avg_drift_time_label`: QLabel
- `rebalance_freq_label`: QLabel
- `cost_per_rebalance_label`: QLabel
- `benefit_per_rebalance_label`: QLabel
- `view_history_btn`: QPushButton (opens dialog)

---

## 📊 SIGNALS TAB - Level 2 (No sub-tabs, already simple)

**Current Structure** (keep as-is, già completato):
```python
┌─────────────────────────────────────────────────────────┐
│ Signals Display                                          │
│                                                           │
│ [ Refresh Signals ]  Limit: [100 ]                      │
│                                                           │
│ ID  Symbol    TF    Created          Entry   Target  Stop│
│ ──────────────────────────────────────────────────────  │
│ 1   EUR/USD   15m   2025-01-20 14:23  1.0850  1.0920  ...│
│ 2   GBP/USD   1h    2025-01-20 14:15  1.2650  1.2720  ...│
│ ...                                                       │
│                                                           │
│ Auto-refresh: every 60 seconds                          │
└─────────────────────────────────────────────────────────┘
```

**NO CHANGES** - già completo con 2 tooltips i18n.

---

## 🔗 Integration with Trading Engine

### **New Bridge Component**: `PortfolioTradingBridge`

```python
class PortfolioTradingBridge:
    """
    Bridge between Portfolio Tab UI and AutomatedTradingEngine
    
    Responsibilities:
    1. Read settings from Portfolio Tab UI
    2. Create PortfolioTradingConfig
    3. Pass config to AutomatedTradingEngine
    4. Update UI with live portfolio state
    5. Execute rebalancing trades
    """
    
    def __init__(self, portfolio_tab: PortfolioOptimizationTab, 
                 trading_engine: AutomatedTradingEngine):
        self.portfolio_tab = portfolio_tab
        self.trading_engine = trading_engine
        
        # Connect signals
        self.portfolio_tab.settings_changed.connect(self._on_settings_changed)
        self.portfolio_tab.rebalance_requested.connect(self._on_rebalance_requested)
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui_from_engine)
        self.update_timer.start(5000)  # Update every 5 seconds
    
    def _on_settings_changed(self, settings: dict):
        """Apply new settings to trading engine"""
        config = self._create_config_from_ui(settings)
        self.trading_engine.update_config(config)
    
    def _on_rebalance_requested(self):
        """Execute rebalancing trades"""
        trades = self.trading_engine.get_rebalancing_trades()
        self.trading_engine.execute_trades(trades)
    
    def _update_ui_from_engine(self):
        """Update UI with current portfolio state"""
        state = self.trading_engine.get_portfolio_state()
        self.portfolio_tab.update_display(state)
```

---

## 📋 Implementation Checklist

### **Phase 1: Expand Portfolio Tab UI (Week 1-2)**
- [ ] Add Tab 1: Portfolio Optimization (15 widgets)
- [ ] Add Tab 2: Transaction Costs (12 widgets)
- [ ] Add Tab 3: Risk Management (18 widgets)
- [ ] Add Tab 4: Correlation (10 widgets)
- [ ] Add Tab 5: Rebalancing (16 widgets)
- [ ] Add Tab 6: Performance (14 widgets)
- **Total: 85 new widgets across 6 Level-3 tabs**

### **Phase 2: i18n Tooltips (Week 2)**
- [ ] Create 85 professional tooltips in en_US.json
- [ ] Apply 6-section schema (WHAT/HOW/WHY/EFFECTS/RANGE/NOTES)
- [ ] Update _apply_i18n_tooltips() in portfolio_tab.py

### **Phase 3: Backend Integration (Week 3-4)**
- [ ] Implement PortfolioPositionSizer
- [ ] Implement TransactionCostAwareOptimizer
- [ ] Implement PortfolioRiskStopLoss
- [ ] Implement CorrelationFilter
- [ ] Implement RebalancingEngine
- [ ] Create PortfolioTradingBridge

### **Phase 4: Connect UI ↔ Engine (Week 4)**
- [ ] Wire all UI widgets to backend logic
- [ ] Implement live updates (Portfolio state → UI)
- [ ] Implement live controls (UI → Trading Engine commands)
- [ ] Add rebalancing preview/execution

### **Phase 5: Testing & Polish (Week 5)**
- [ ] Test all 85 widgets
- [ ] Verify all tooltips appear
- [ ] Test rebalancing logic
- [ ] Test transaction cost optimization
- [ ] Backtest full system

---

## 📊 Summary

**Total Expansion**:
- **Level 2 Tabs**: 2 (Portfolio, Signals)
- **Level 3 Tabs**: 6 (under Portfolio)
- **Total Settings Widgets**: 85 new + 2 existing = **87 widgets**
- **i18n Tooltips**: 85 professional tooltips (6-section schema)
- **Backend Classes**: 5 new integration classes
- **Lines of Code**: ~3,000 lines UI + 2,000 lines backend = **5,000 lines**

**Expected Outcome**:
✅ **Complete Riskfolio ↔ Trading Engine integration**
✅ **All settings accessible via UI at 3 levels**
✅ **Professional tooltips for every parameter**
✅ **Live portfolio monitoring and control**
✅ **Zero settings left behind**

---

**Pronto a iniziare? Quale fase vuoi che implementi per prima?**

A) Phase 1: Expand UI (85 widgets)
B) Phase 2: i18n (85 tooltips)
C) Phase 3: Backend (5 integration classes)
D) Tutto in sequenza (5 settimane)
