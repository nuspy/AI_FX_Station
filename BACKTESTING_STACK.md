# ForexGPT Backtesting Stack - Architecture Overview

**Date:** 2025-10-18  
**Status:** Production Active

---

## 📋 Executive Summary

ForexGPT **NON usa VectorBT Pro** attualmente, nonostante sia presente nel progetto. Utilizza invece un **sistema custom multi-layer** di backtesting engines ottimizzati per forex forecasting.

**VectorBT Pro:** Installato ma **non integrato** (file .whl presente, zero import nel codice)

---

## 🔧 Stack Backtesting Attuale

### **1. Custom Backtest Engines** ⭐ PRINCIPALE

#### **A. BacktestEngine** (`backtest/engine.py`)
**Specializzazione:** Diffusion model forecasting

```python
from forex_diffusion.backtest import BacktestEngine

engine = BacktestEngine()
results = engine.run_backtest(df, quantiles)
```

**Features:**
- ✅ Walk-forward splits con purge/embargo (anti data-leakage)
- ✅ Strategy: median crossing + quantile-based targets
- ✅ First-passage simulation con max_hold timeout
- ✅ Baseline: Random Walk + sigma
- ✅ Metrics: Sharpe annualized, Max Drawdown, Turnover, Net P&L
- ✅ Async DB writer per persistenza segnali

**Perché custom:**
- Ottimizzato per **quantile forecasts** (q05, q50, q95)
- Integrato con **conformal prediction** (uncertainty bands)
- Supporta **multi-timeframe** walk-forward
- Pipeline specifico per **forex** (spread, slippage, swap)

#### **B. AdvancedBacktestEngine** (`backtesting/advanced_backtest_engine.py`)
**Specializzazione:** Professional backtesting suite

```python
from forex_diffusion.backtesting import AdvancedBacktestEngine

engine = AdvancedBacktestEngine(initial_capital=100000)
results = engine.run_backtest(data, strategy)
```

**Features:**
- ✅ Monte Carlo simulation
- ✅ Walk-forward analysis
- ✅ Professional risk metrics (Sortino, Calmar, Omega)
- ✅ Trade-by-trade analysis
- ✅ Equity curve + drawdown series
- ✅ Win rate, profit factor
- ✅ Max drawdown duration

**Comparazione con VectorBT:**
| Feature | AdvancedBacktestEngine | VectorBT Pro |
|---------|------------------------|--------------|
| Monte Carlo | ✅ Custom | ✅ Built-in |
| Walk-forward | ✅ Custom | ✅ Built-in |
| Risk metrics | ✅ Professional | ✅ Extensive |
| Forex-specific | ✅ Spread/Slippage/Swap | ❌ Generic |
| Quantile forecasts | ✅ Native | ❌ Requires adapter |
| Conformal prediction | ✅ Integrated | ❌ Manual |

#### **C. LDM4TSBacktestEngine** (`backtest/ldm4ts_backtest.py`)
**Specializzazione:** Vision-enhanced forecasting

```python
from forex_diffusion.backtest import LDM4TSBacktestEngine

engine = LDM4TSBacktestEngine()
results = engine.run(checkpoint, symbol, timeframe)
```

**Features:**
- ✅ LDM4TS model integration
- ✅ Vision Transformer forecasts
- ✅ Multi-horizon evaluation
- ✅ Chart pattern recognition backtesting

#### **D. UnifiedBacktestEngine** (`backtest/unified_engine.py`)
**Specializzazione:** Multi-model backtesting

```python
from forex_diffusion.backtest import UnifiedBacktestEngine

engine = UnifiedBacktestEngine()
results = engine.backtest_multiple_models([model1, model2])
```

**Features:**
- ✅ Compare multiple models
- ✅ Ensemble strategies
- ✅ Unified metrics across models

### **2. Portfolio Optimization** 🎯

#### **Riskfolio-Lib Integration** (`portfolio/optimizer.py`)

```python
from forex_diffusion.portfolio import PortfolioOptimizer

optimizer = PortfolioOptimizer(
    risk_measure="CVaR",
    objective="Sharpe",
    risk_aversion=1.0
)

weights = optimizer.optimize(returns, constraints)
```

**Features:**
- ✅ Multiple risk measures: CVaR, CDaR, EVaR, WR, MDD
- ✅ Optimization objectives: Sharpe, MinRisk, Utility, MaxRet
- ✅ Constraints: max_weight, min_weight, leverage, target_return/risk
- ✅ Methods: Classic, Black-Litterman, Factor Models
- ✅ Efficient frontier calculation
- ✅ Risk parity portfolios

**Perché Riskfolio-Lib:**
- ✅ State-of-the-art portfolio optimization
- ✅ Academic-grade risk measures
- ✅ Active development
- ✅ Better per forex multi-asset than VectorBT

### **3. Advanced Features** 🚀

#### **Genetic Optimizer** (`backtest/genetic_optimizer.py`)
```python
# NSGA-II multi-objective optimization
- Optimize strategy parameters
- Multiple fitness objectives (Sharpe, Drawdown, Win Rate)
- Pareto frontier discovery
```

#### **Resumable Optimizer** (`backtest/resumable_optimizer.py`)
```python
# Long-running optimizations with checkpointing
- Save/restore optimization state
- Resume after crash/interruption
- Progress tracking
```

#### **Walk-Forward Engine** (`backtest/walk_forward.py`)
```python
# Professional walk-forward testing
- Rolling windows with purge/embargo
- Anti-lookahead bias
- Stability analysis
```

#### **Transaction Costs** (`backtest/transaction_costs.py`)
```python
# Realistic cost modeling
- Spread (bid-ask)
- Slippage (market impact)
- Swap (overnight holding)
- Commission
```

---

## 📊 Comparison: Current Stack vs VectorBT Pro

| Aspect | Current Custom Stack | VectorBT Pro |
|--------|---------------------|--------------|
| **Forex-Specific** | ✅✅✅ Spread/Slippage/Swap native | ❌ Generic (manual) |
| **Quantile Forecasts** | ✅✅✅ Native support | ❌ Requires wrapper |
| **Conformal Prediction** | ✅✅ Integrated | ❌ Manual |
| **Multi-Horizon** | ✅✅✅ Built-in | ⚠️ Possible but complex |
| **Walk-Forward** | ✅✅ Purge/Embargo | ✅✅ Built-in |
| **Monte Carlo** | ✅ Custom | ✅✅ Extensive |
| **Portfolio Optimization** | ✅✅ Riskfolio-Lib | ⚠️ Basic |
| **Performance** | ✅ Optimized for use case | ✅✅ Numba-accelerated |
| **Flexibility** | ✅✅✅ Full control | ⚠️ Framework constraints |
| **Learning Curve** | ⚠️ Custom code | ✅ Well-documented |
| **Community** | ❌ Internal only | ✅✅ Large community |
| **Maintenance** | ⚠️ Self-maintained | ✅ Active development |

**Legend:**
- ✅✅✅ Excellent
- ✅✅ Very Good
- ✅ Good
- ⚠️ Limited/Manual
- ❌ Not Available

---

## 🤔 Should You Integrate VectorBT Pro?

### **Arguments FOR Integration:**

1. **Performance Gains**
   - Numba-accelerated (100-1000x faster than pure Python)
   - Vectorized operations
   - Parallel backtesting

2. **Rich Feature Set**
   - 100+ indicators built-in
   - Advanced portfolio analytics
   - Interactive visualizations (Plotly)
   - Comprehensive metrics library

3. **Proven & Tested**
   - Battle-tested in production
   - Large community
   - Active development
   - Extensive documentation

4. **Reduced Maintenance**
   - Less custom code to maintain
   - Bug fixes from community
   - Regular updates

### **Arguments AGAINST Integration:**

1. **Already Have Optimal Solution**
   - Custom engines optimized for your exact use case
   - Forex-specific features (spread, swap, slippage)
   - Quantile forecast integration
   - Conformal prediction native

2. **Integration Complexity**
   - Need adapters for quantile forecasts
   - Rewrite existing backtests
   - Migration effort significant
   - Training debt (team learning curve)

3. **Framework Lock-in**
   - Less flexibility than custom code
   - Harder to customize for edge cases
   - Updates may break custom logic

4. **Performance Already Good**
   - Current stack is fast enough
   - Numba can be added to custom code if needed
   - Bottleneck is model inference, not backtest

---

## ✅ Recommendations

### **Option 1: KEEP CURRENT STACK** ⭐ RECOMMENDED

**Why:**
- ✅ Already production-ready
- ✅ Optimized for your specific needs
- ✅ Forex-specific features built-in
- ✅ Quantile + conformal prediction native
- ✅ Team knows the codebase
- ✅ No migration risk

**Action Items:**
1. Continue using current engines
2. Add Numba JIT to hot loops if needed
3. Enhance visualization (Plotly/Dash)
4. Document current stack thoroughly

### **Option 2: HYBRID APPROACH** 

**Use VectorBT Pro for:**
- ✅ Quick exploratory backtests
- ✅ Standard technical strategies
- ✅ Portfolio analytics
- ✅ Interactive visualizations

**Keep Custom Engines for:**
- ✅ Diffusion model forecasting
- ✅ Quantile-based strategies
- ✅ Conformal prediction
- ✅ Production backtests

**Implementation:**
```python
# Portfolio analytics with VectorBT
import vectorbtpro as vbt
portfolio = vbt.Portfolio.from_signals(...)
portfolio.stats()

# Diffusion backtesting with custom engine
from forex_diffusion.backtest import BacktestEngine
results = BacktestEngine().run_backtest(df, quantiles)
```

### **Option 3: FULL MIGRATION**

**NOT RECOMMENDED** unless:
- ❌ Current stack has performance issues (it doesn't)
- ❌ Team can't maintain custom code (unlikely)
- ❌ Need features only VectorBT has (what features?)

---

## 🚀 If You Decide to Integrate VectorBT Pro

### **Phase 1: Install & Test**

```bash
# Install from wheel
pip install ./VectorBt_PRO/vectorbtpro-2025.10.15-py3-none-any.whl

# Test import
python -c "import vectorbtpro as vbt; print(vbt.__version__)"
```

### **Phase 2: Create Adapters**

```python
# src/forex_diffusion/integrations/vectorbt_adapter.py

import vectorbtpro as vbt
import pandas as pd
from typing import Dict, Any

class QuantileForecastAdapter:
    """Adapt quantile forecasts to VectorBT signals."""
    
    def __init__(self, quantiles: pd.DataFrame):
        self.quantiles = quantiles
    
    def to_signals(self, strategy: str = "median_cross") -> Dict[str, pd.Series]:
        """Convert quantiles to entry/exit signals."""
        if strategy == "median_cross":
            entries = self.quantiles['q50'] > self.quantiles['q50'].shift(1)
            exits = self.quantiles['q50'] < self.quantiles['q50'].shift(1)
        
        return {
            'entries': entries,
            'exits': exits
        }
    
    def run_backtest(self, price: pd.Series, **kwargs) -> vbt.Portfolio:
        """Run backtest with VectorBT."""
        signals = self.to_signals()
        
        portfolio = vbt.Portfolio.from_signals(
            price,
            entries=signals['entries'],
            exits=signals['exits'],
            **kwargs
        )
        
        return portfolio
```

### **Phase 3: Integrate in UI**

```python
# ui/backtesting_tab.py

def _run_vectorbt_backtest(self):
    """Alternative backtest engine using VectorBT Pro."""
    try:
        import vectorbtpro as vbt
        
        # Get quantiles from model
        quantiles = self._get_quantiles()
        
        # Create adapter
        from ..integrations.vectorbt_adapter import QuantileForecastAdapter
        adapter = QuantileForecastAdapter(quantiles)
        
        # Run backtest
        portfolio = adapter.run_backtest(
            self.price_data,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005
        )
        
        # Display results
        self._show_vectorbt_results(portfolio)
        
    except ImportError:
        QMessageBox.warning(self, "VectorBT Pro", 
            "VectorBT Pro not installed. Using custom engine.")
        self._run_custom_backtest()
```

### **Phase 4: A/B Testing**

```python
# Compare results side-by-side
def compare_engines(df, quantiles):
    # Custom engine
    custom_results = BacktestEngine().run_backtest(df, quantiles)
    
    # VectorBT engine
    vbt_adapter = QuantileForecastAdapter(quantiles)
    vbt_results = vbt_adapter.run_backtest(df['close'])
    
    # Compare
    comparison = pd.DataFrame({
        'Custom Sharpe': [custom_results.sharpe],
        'VBT Sharpe': [vbt_results.sharpe_ratio],
        'Custom MaxDD': [custom_results.max_drawdown],
        'VBT MaxDD': [vbt_results.max_drawdown],
    })
    
    return comparison
```

---

## 📝 Current Architecture Summary

```
BACKTESTING LAYER:
├─ Custom Engines (Primary)
│  ├─ BacktestEngine (diffusion quantiles)
│  ├─ AdvancedBacktestEngine (professional metrics)
│  ├─ LDM4TSBacktestEngine (vision forecasts)
│  └─ UnifiedBacktestEngine (multi-model)
│
├─ Optimization
│  ├─ GeneticOptimizer (NSGA-II)
│  ├─ ResumableOptimizer (long-running)
│  └─ WalkForwardEngine (anti-lookahead)
│
└─ Portfolio Layer
   ├─ Riskfolio-Lib (optimization)
   └─ PortfolioOptimizer (wrapper)

VECTORBT PRO:
└─ Installed but NOT integrated
   └─ ./VectorBt_PRO/vectorbtpro-2025.10.15-py3-none-any.whl
```

---

## 🎯 Final Verdict

### **KEEP CURRENT STACK** ⭐

**Your custom backtesting system is:**
- ✅ **Production-ready**
- ✅ **Forex-optimized**
- ✅ **Quantile-native**
- ✅ **Conformal-integrated**
- ✅ **Feature-complete**
- ✅ **Well-architected**

**VectorBT Pro would add:**
- ⚠️ Performance (not needed currently)
- ⚠️ Visualizations (can add Plotly separately)
- ⚠️ Standard indicators (already have)
- ❌ But require significant migration effort

**Recommendation:**
1. **Keep current stack** as primary
2. **Optionally add VectorBT** for quick explorations
3. **Use Riskfolio-Lib** for portfolio optimization (already done)
4. **Add Numba** to custom engines if performance becomes bottleneck

---

## 📚 References

### Internal Modules
- `backtest/engine.py` - Main diffusion backtest engine
- `backtesting/advanced_backtest_engine.py` - Professional metrics
- `portfolio/optimizer.py` - Riskfolio-Lib integration
- `backtest/genetic_optimizer.py` - Parameter optimization
- `backtest/walk_forward.py` - Anti-lookahead testing

### External Libraries
- **Riskfolio-Lib:** https://riskfolio-lib.readthedocs.io/
- **VectorBT Pro:** https://vectorbt.pro/ (installed, not used)

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-18  
**Status:** ✅ Current stack recommended, VectorBT optional
