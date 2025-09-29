"""
Professional Backtesting and Risk Management Module

This module provides comprehensive backtesting capabilities with:
- Advanced backtest engine with Monte Carlo simulation
- Professional risk management suite
- Portfolio analytics and optimization
- Position sizing algorithms
- Stress testing and VaR calculations

Main Components:
- AdvancedBacktestEngine: Comprehensive backtesting with professional metrics
- RiskManagementSuite: Advanced risk analytics and portfolio monitoring
- PositionSizingEngine: Intelligent position sizing with Kelly criterion
- TradingStrategy: Abstract base class for strategy development

Usage:
    from forex_diffusion.backtesting import AdvancedBacktestEngine, MACrossoverStrategy

    engine = AdvancedBacktestEngine(initial_capital=100000)
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
    results = engine.run_backtest(data, strategy)
"""

from .advanced_backtest_engine import (
    AdvancedBacktestEngine,
    TradingStrategy,
    MACrossoverStrategy,
    Trade,
    BacktestResults
)

from .risk_management import (
    PortfolioRiskAnalyzer,
    PositionSizingEngine,
    RiskMetrics,
    PositionSizingResult
)

__all__ = [
    'AdvancedBacktestEngine',
    'TradingStrategy',
    'MACrossoverStrategy',
    'Trade',
    'BacktestResults',
    'PortfolioRiskAnalyzer',
    'PositionSizingEngine',
    'RiskMetrics',
    'PositionSizingResult'
]