# Portfolio Optimization Integration Guide

## Overview

This guide describes the integration of **Riskfolio-Lib** portfolio optimization into the ForexGPT trading system. The implementation provides quantitative portfolio optimization, adaptive position sizing, and comprehensive risk management.

## Components

### 1. Core Portfolio Modules

Located in `src/forex_diffusion/portfolio/`:

#### `optimizer.py` - PortfolioOptimizer
Main portfolio optimization engine using Riskfolio-Lib.

**Features:**
- Multiple risk measures: MV (Mean-Variance), CVaR, CDaR, EVaR, WR, MDD
- Optimization objectives: Sharpe ratio, MinRisk, Utility, MaxRet
- Risk Parity allocation
- Efficient frontier calculation
- Portfolio backtesting with comprehensive metrics

**Key Methods:**
```python
# Standard optimization
weights = optimizer.optimize(
    returns=historical_returns_df,
    constraints={"max_weight": 0.25, "min_weight": 0.01},
    method="Classic"
)

# Risk Parity optimization
weights = optimizer.optimize_risk_parity(
    returns=historical_returns_df,
    risk_budgets=None  # Equal risk contribution
)

# Calculate efficient frontier
frontier = optimizer.calculate_efficient_frontier(
    returns=historical_returns_df,
    points=20
)

# Backtest strategy
metrics = optimizer.backtest_strategy(
    weights=weights,
    returns=historical_returns_df
)
```

#### `position_sizer.py` - AdaptivePositionSizer
Integrates portfolio optimization with diffusion model predictions for dynamic position sizing.

**Features:**
- Scheduled rebalancing (configurable frequency)
- Volatility targeting
- Combines historical returns with model predictions
- Trade calculation for rebalancing

**Key Methods:**
```python
# Calculate optimal positions
positions = position_sizer.calculate_positions(
    predictions=model_predictions_df,
    historical_returns=historical_returns_df,
    current_date=pd.Timestamp.now(),
    total_capital=100000.0,
    force_rebalance=False
)

# Risk Parity positions
positions = position_sizer.calculate_risk_parity_positions(
    historical_returns=historical_returns_df,
    total_capital=100000.0,
    risk_budgets=None
)

# Volatility-adjusted positions
adjusted = position_sizer.adjust_for_volatility(
    base_positions=positions,
    volatility_estimates=vol_dict,
    target_volatility=0.15
)

# Calculate rebalancing trades
trades = position_sizer.get_rebalance_trades(
    target_positions=new_positions,
    current_positions=current_positions,
    min_trade_size=100.0
)
```

#### `risk_metrics.py` - RiskMetricsCalculator
Calculates comprehensive risk metrics for portfolio analysis.

**Features:**
- Value at Risk (VaR)
- Conditional VaR (CVaR / Expected Shortfall)
- Volatility and downside deviation
- Skewness and kurtosis

**Key Methods:**
```python
# Calculate all metrics
metrics = RiskMetricsCalculator.calculate_all_metrics(portfolio_returns)
# Returns: {var_95, var_99, cvar_95, cvar_99, volatility, downside_deviation, skewness, kurtosis}
```

### 2. GUI Components

Located in `src/forex_diffusion/ui/`:

#### `portfolio_tab.py` - PortfolioOptimizationTab
Main GUI interface for portfolio optimization configuration.

**Features:**
- 5 sub-tabs:
  1. **Optimizer Settings**: Risk measure, objective, parameters, constraints
  2. **Position Sizing**: Adaptive sizing, lookback period, rebalancing
  3. **Portfolio Stats**: Current weights and position sizes
  4. **Risk Metrics**: Comprehensive risk dashboard
  5. **Efficient Frontier**: Interactive visualization

**Signals:**
- `settings_changed`: Emitted when user applies new settings

**Public Methods:**
```python
# Update portfolio display
portfolio_tab.update_portfolio_display(
    weights={"EURUSD": 0.4, "GBPUSD": 0.3, "USDJPY": 0.3},
    stats={
        "expected_return": 0.12,
        "volatility": 0.15,
        "sharpe_ratio": 0.8,
        # ... other metrics
    }
)

# Update efficient frontier visualization
portfolio_tab.update_efficient_frontier(
    frontier_data=frontier_df,
    current_portfolio={"return": 0.12, "risk": 0.15, "sharpe_ratio": 0.8},
    asset_data=asset_stats_df
)
```

#### `portfolio_viz.py` - EfficientFrontierWidget
Interactive matplotlib-based visualization widget.

**Features:**
- Efficient frontier plot
- Current portfolio marker with Sharpe ratio annotation
- Individual asset scatter plot
- Maximum Sharpe ratio point highlighting
- Export to PNG/PDF/SVG

**Signals:**
- `point_clicked`: Emitted when user clicks a point on the frontier

**Public Methods:**
```python
# Plot efficient frontier
frontier_viz.plot_efficient_frontier(
    frontier_data=frontier_df,  # columns: [return, risk, sharpe]
    current_portfolio=portfolio_dict,
    asset_data=asset_df  # columns: [return, risk]
)

# Export plot
frontier_viz.export_plot("efficient_frontier.png")

# Clear plot
frontier_viz.clear_plot()
```

### 3. Main Application Integration

The Portfolio tab is integrated into the main application in `src/forex_diffusion/ui/app.py`:

```python
# Portfolio tab is added as a nested tab under "Chart"
chart_container.addTab(portfolio_tab, "Portfolio")
```

**Navigation path in GUI:**
```
Chart (level_1) → Portfolio (level_2)
```

## Usage Workflows

### Workflow 1: Basic Portfolio Optimization

1. **Configure Optimizer** (Tab: Optimizer Settings):
   - Select risk measure (e.g., CVaR)
   - Select objective (e.g., Sharpe)
   - Set risk-free rate (default: 0%)
   - Set risk aversion (default: 1.0)
   - Set position constraints (min: 1%, max: 25%)

2. **Click "Optimize Portfolio"**:
   - Creates optimizer instance with current settings
   - Ready to receive historical data

3. **View Results** (Tab: Portfolio Stats):
   - Asset weights table
   - Portfolio statistics

4. **Review Risk Metrics** (Tab: Risk Metrics):
   - Expected return, volatility, Sharpe ratio
   - CVaR, maximum drawdown
   - Skewness, kurtosis, concentration

5. **Visualize Frontier** (Tab: Efficient Frontier):
   - Click "Calculate Efficient Frontier"
   - View risk-return tradeoff
   - See current portfolio position

### Workflow 2: Adaptive Position Sizing

1. **Configure Position Sizing** (Tab: Position Sizing):
   - Enable adaptive sizing
   - Set lookback period (default: 60 days)
   - Set rebalance frequency (default: 5 days)
   - Optionally enable Risk Parity
   - Optionally enable Volatility Targeting (target: 15%)

2. **Apply Settings**:
   - Click "Apply Settings"
   - Settings are emitted via `settings_changed` signal

3. **Integration with Trading System**:
   - Position sizer receives diffusion model predictions
   - Combines with historical returns (40% historical, 60% predictions)
   - Optimizes portfolio weights
   - Calculates position sizes based on total capital

### Workflow 3: Risk Parity Allocation

1. **Enable Risk Parity** (Tab: Position Sizing):
   - Check "Use Risk Parity Allocation"
   - This ensures equal risk contribution from each asset

2. **Configure and Optimize**:
   - Set other parameters as needed
   - Click "Optimize Portfolio"

3. **Result**:
   - Portfolio weights are calculated to equalize risk contribution
   - Lower volatility assets get higher weights
   - Higher volatility assets get lower weights

## Integration with Backtesting System

The portfolio optimizer can be integrated with the backtesting framework to test optimized strategies:

```python
# In backtesting loop
for date in backtest_dates:
    # Get historical returns up to current date
    historical_returns = get_historical_returns(end_date=date)

    # Get diffusion model predictions
    predictions = get_model_predictions(date)

    # Calculate optimal positions
    positions = position_sizer.calculate_positions(
        predictions=predictions,
        historical_returns=historical_returns,
        current_date=date,
        total_capital=portfolio_value,
        force_rebalance=False  # Only rebalance on schedule
    )

    # Execute trades
    trades = position_sizer.get_rebalance_trades(
        target_positions=positions,
        current_positions=current_holdings,
        min_trade_size=100.0
    )

    # Apply trades and update portfolio
    execute_trades(trades)
```

## Data Requirements

### Historical Returns Format
```python
# DataFrame with columns = assets, index = timestamps
historical_returns = pd.DataFrame({
    "EURUSD": [0.001, -0.002, 0.003, ...],
    "GBPUSD": [0.002, 0.001, -0.001, ...],
    "USDJPY": [-0.001, 0.002, 0.001, ...],
}, index=pd.date_range("2024-01-01", periods=100, freq="D"))
```

### Model Predictions Format
```python
# DataFrame with columns = assets, index = timestamps (future dates)
predictions = pd.DataFrame({
    "EURUSD": [0.002, 0.003, 0.001, ...],
    "GBPUSD": [0.001, -0.001, 0.002, ...],
    "USDJPY": [0.003, 0.001, -0.001, ...],
}, index=pd.date_range("2024-04-11", periods=10, freq="D"))
```

## Configuration Settings

Settings are stored in a dictionary with the following keys:

```python
settings = {
    "risk_measure": "CVaR",  # MV, CVaR, CDaR, EVaR, WR, MDD
    "objective": "Sharpe",  # Sharpe, MinRisk, Utility, MaxRet
    "risk_free_rate": 0.0,  # Annual risk-free rate (0-10%)
    "risk_aversion": 1.0,  # Lambda parameter (0.1-10.0)
    "max_weight": 0.25,  # Maximum weight per asset (0-1)
    "min_weight": 0.01,  # Minimum weight per asset (0-1)
    "adaptive_sizing_enabled": True,
    "lookback_period": 60,  # Days
    "rebalance_frequency": 5,  # Days
    "risk_parity_enabled": False,
    "vol_targeting_enabled": False,
    "target_volatility": 0.15,  # Annual target volatility (0-1)
}
```

## Dependencies

Required packages (already in `pyproject.toml`):

```toml
riskfolio-lib>=7.0,<8.0
arch>=7.0,<8.0
cvxpy>=1.3.0,<2.0
clarabel>=0.5.0,<1.0
matplotlib>=3.7.0,<4.0
xlsxwriter>=3.0.0  # For Excel export (future feature)
```

## Future Enhancements

1. **Real-time Data Integration**:
   - Connect to cTrader/Tiingo for live historical returns
   - Auto-refresh on new data availability

2. **Portfolio Performance Tracking**:
   - Store optimization results in database
   - Track portfolio evolution over time
   - Compare actual vs. predicted performance

3. **Advanced Visualizations**:
   - Risk contribution pie chart
   - Portfolio evolution over time
   - Correlation heatmap
   - Drawdown plot

4. **Export and Reporting**:
   - Excel export for portfolio reports
   - PDF report generation
   - Email alerts for rebalancing signals

5. **Black-Litterman Model**:
   - Incorporate market views
   - Blend equilibrium returns with predictions

6. **Transaction Costs**:
   - Include transaction costs in optimization
   - Optimize rebalancing frequency considering costs

## Troubleshooting

### Issue: "Portfolio optimization modules not available"
**Solution:** Install riskfolio-lib and dependencies:
```bash
pip install "riskfolio-lib>=7.0" "arch>=7.0" "cvxpy>=1.3.0" "clarabel>=0.5.0"
```

### Issue: "Visualization not available - install matplotlib"
**Solution:** Install matplotlib:
```bash
pip install "matplotlib>=3.7.0"
```

### Issue: Optimization fails with "Solver error"
**Solution:** This usually indicates:
- Insufficient historical data (need at least 20-30 data points)
- Singular covariance matrix (assets are perfectly correlated)
- Infeasible constraints (min_weight too high, max_weight too low)

**Fix:**
- Increase lookback period
- Reduce number of assets
- Relax constraints

### Issue: "Frontier visualization widget not initialized"
**Solution:** Make sure you're viewing the correct tab:
1. Navigate to Chart → Portfolio → Efficient Frontier
2. The widget is only created if matplotlib is installed

## References

- [Riskfolio-Lib Documentation](https://riskfolio-lib.readthedocs.io/)
- [CVXPY Documentation](https://www.cvxpy.org/)
- [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Risk Parity](https://en.wikipedia.org/wiki/Risk_parity)
- [Conditional Value at Risk](https://en.wikipedia.org/wiki/Expected_shortfall)

## Contact

For questions or issues regarding portfolio optimization integration, please check:
- GitHub Issues: https://github.com/anthropics/claude-code/issues
- Project documentation in `docs/`
