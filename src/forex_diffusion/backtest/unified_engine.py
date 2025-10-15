"""
Unified Backtest Engine Interface

Provides unified API to all backtest engines (STRUCT-001 partial solution).

Three specialized engines exist with different purposes:
1. QuantileBacktest (backtest/engine.py) - quantile-based strategy
2. ForecastBacktest (backtesting/forecast_backtest_engine.py) - probabilistic forecasts
3. IntegratedBacktest (backtest/integrated_backtest.py) - complete end-to-end system

This module provides:
- Unified factory pattern
- Common interface
- Migration path
- Deprecation handling
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .engine import BacktestEngine as QuantileBacktestEngine
    from ..backtesting.forecast_backtest_engine import ForecastBacktestEngine
    from .integrated_backtest import IntegratedBacktest


class BacktestType(Enum):
    """Backtest engine types"""
    QUANTILE = "quantile"  # Original quantile-based strategy
    FORECAST = "forecast"  # Probabilistic forecast evaluation
    INTEGRATED = "integrated"  # Complete end-to-end system
    AUTO = "auto"  # Auto-detect based on configuration


class UnifiedBacktestEngine:
    """
    Unified factory for all backtest engines.
    
    Usage:
        # Create engine
        engine = UnifiedBacktestEngine.create(
            backtest_type=BacktestType.QUANTILE,
            cfg=config
        )
        
        # Run backtest
        results = engine.run(market_data, predictions)
        
        # Get metrics
        metrics = engine.get_metrics(results)
    """
    
    @staticmethod
    def create(
        backtest_type: BacktestType = BacktestType.QUANTILE,
        cfg: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Create backtest engine by type.
        
        Args:
            backtest_type: Type of backtest engine to create
            cfg: Configuration object
            **kwargs: Additional arguments passed to engine constructor
            
        Returns:
            Appropriate backtest engine instance
        """
        logger.info(f"Creating backtest engine: {backtest_type.value}")
        
        if backtest_type == BacktestType.QUANTILE:
            from .engine import BacktestEngine
            return BacktestEngine(cfg=cfg, **kwargs)
        
        elif backtest_type == BacktestType.FORECAST:
            from ..backtesting.forecast_backtest_engine import ForecastBacktestEngine
            return ForecastBacktestEngine(**kwargs)
        
        elif backtest_type == BacktestType.INTEGRATED:
            from .integrated_backtest import IntegratedBacktest
            return IntegratedBacktest(**kwargs)
        
        elif backtest_type == BacktestType.AUTO:
            # Auto-detect based on configuration
            return UnifiedBacktestEngine._auto_create(cfg, **kwargs)
        
        else:
            raise ValueError(f"Unknown backtest type: {backtest_type}")
    
    @staticmethod
    def _auto_create(cfg: Optional[Any] = None, **kwargs) -> Any:
        """Auto-detect appropriate engine from configuration"""
        
        # Check for integrated backtest markers
        if kwargs.get('use_multi_timeframe') or kwargs.get('use_regime_detection'):
            logger.info("Auto-detected: IntegratedBacktest")
            from .integrated_backtest import IntegratedBacktest
            return IntegratedBacktest(**kwargs)
        
        # Check for forecast evaluation markers
        if kwargs.get('models_dir') or kwargs.get('probabilistic_metrics'):
            logger.info("Auto-detected: ForecastBacktest")
            from ..backtesting.forecast_backtest_engine import ForecastBacktestEngine
            return ForecastBacktestEngine(**kwargs)
        
        # Default to quantile backtest
        logger.info("Auto-detected: QuantileBacktest (default)")
        from .engine import BacktestEngine
        return BacktestEngine(cfg=cfg, **kwargs)
    
    @staticmethod
    def get_available_engines() -> Dict[str, str]:
        """Get list of available backtest engines with descriptions"""
        return {
            'quantile': 'Quantile-based strategy backtest (original engine)',
            'forecast': 'Probabilistic forecast evaluation (diffusion models)',
            'integrated': 'Complete end-to-end trading system (production-ready)'
        }
    
    @staticmethod
    def get_recommendation(use_case: str) -> BacktestType:
        """
        Get recommended engine for specific use case.
        
        Args:
            use_case: One of 'strategy', 'forecast', 'production', 'research'
            
        Returns:
            Recommended BacktestType
        """
        recommendations = {
            'strategy': BacktestType.QUANTILE,
            'forecast': BacktestType.FORECAST,
            'production': BacktestType.INTEGRATED,
            'research': BacktestType.FORECAST,
            'trading': BacktestType.INTEGRATED,
            'development': BacktestType.QUANTILE
        }
        
        recommended = recommendations.get(use_case.lower(), BacktestType.QUANTILE)
        logger.info(f"Recommended engine for '{use_case}': {recommended.value}")
        
        return recommended


# Convenience functions
def create_quantile_backtest(cfg: Optional[Any] = None, **kwargs):
    """
    Create quantile-based backtest engine.
    
    Best for: Testing quantile-based trading strategies
    
    Usage:
        engine = create_quantile_backtest()
        results = engine.simulate_trades(market_df, quantiles_df)
    """
    return UnifiedBacktestEngine.create(BacktestType.QUANTILE, cfg=cfg, **kwargs)


def create_forecast_backtest(**kwargs):
    """
    Create forecast evaluation backtest engine.
    
    Best for: Evaluating probabilistic model forecasts (diffusion, VAE)
    
    Usage:
        engine = create_forecast_backtest()
        engine.add_forecast_record(...)
        metrics = engine.evaluate_forecasts()
    """
    return UnifiedBacktestEngine.create(BacktestType.FORECAST, **kwargs)


def create_integrated_backtest(**kwargs):
    """
    Create integrated end-to-end backtest engine.
    
    Best for: Complete system validation (production-ready)
    
    Features:
    - Multi-timeframe ensemble
    - Regime detection
    - Multi-level risk management
    - Transaction costs
    - Smart execution
    
    Usage:
        config = BacktestConfig(...)
        engine = create_integrated_backtest(config=config)
        results = engine.run_walk_forward_backtest()
    """
    return UnifiedBacktestEngine.create(BacktestType.INTEGRATED, **kwargs)


# Legacy compatibility (with deprecation warnings)
def create_backtest_engine(*args, **kwargs):
    """
    ⚠️ DEPRECATED: Use UnifiedBacktestEngine.create() or specific create functions.
    
    Legacy function for backward compatibility.
    """
    logger.warning(
        "create_backtest_engine() is deprecated. "
        "Use UnifiedBacktestEngine.create() or create_quantile_backtest()/create_forecast_backtest()/create_integrated_backtest()"
    )
    return create_quantile_backtest(*args, **kwargs)


# Example usage
if __name__ == "__main__":
    print("=== Unified Backtest Engine ===\n")
    
    print("Available engines:")
    for engine_type, description in UnifiedBacktestEngine.get_available_engines().items():
        print(f"  - {engine_type}: {description}")
    
    print("\nRecommendations:")
    for use_case in ['strategy', 'forecast', 'production']:
        recommended = UnifiedBacktestEngine.get_recommendation(use_case)
        print(f"  - {use_case}: {recommended.value}")
    
    print("\nCreating engines:")
    
    # Quantile backtest
    print("\n1. Quantile Backtest:")
    engine1 = create_quantile_backtest()
    print(f"   Created: {type(engine1).__name__}")
    
    # Forecast backtest
    print("\n2. Forecast Backtest:")
    engine2 = create_forecast_backtest()
    print(f"   Created: {type(engine2).__name__}")
    
    # Integrated backtest
    print("\n3. Integrated Backtest:")
    try:
        engine3 = create_integrated_backtest()
        print(f"   Created: {type(engine3).__name__}")
    except ImportError as e:
        print(f"   Not available (missing dependencies)")
    
    # Auto-detect
    print("\n4. Auto-detect:")
    engine4 = UnifiedBacktestEngine.create(BacktestType.AUTO)
    print(f"   Created: {type(engine4).__name__}")
