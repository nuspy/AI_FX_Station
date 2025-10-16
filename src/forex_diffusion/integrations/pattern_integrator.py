"""Pattern Parameter Integrator - Load from DB"""
from __future__ import annotations
from typing import Dict, Optional
from loguru import logger

class PatternIntegrator:
    """Loads regime-specific pattern parameters from database"""
    
    def __init__(self, db_session=None, cache: bool = True):
        self.db_session = db_session
        self.cache_enabled = cache
        self.param_cache: Dict[str, Dict] = {}
    
    def load_parameters(self, symbol: str, timeframe: str, regime: str) -> Dict:
        """Load optimized pattern parameters for (symbol, timeframe, regime)"""
        cache_key = f"{symbol}_{timeframe}_{regime}"
        
        # Check cache
        if self.cache_enabled and cache_key in self.param_cache:
            logger.debug(f"Pattern params loaded from cache: {cache_key}")
            return self.param_cache[cache_key]
        
        # Load from database
        if self.db_session:
            try:
                from ..database.e2e_optimization_models import E2ERegimeParameter
                
                # Query active parameters
                params = self.db_session.query(E2ERegimeParameter).filter_by(
                    symbol=symbol,
                    timeframe=timeframe,
                    regime=regime,
                    is_active=True
                ).first()
                
                if params:
                    param_dict = params.get_parameters()
                    
                    # Cache
                    if self.cache_enabled:
                        self.param_cache[cache_key] = param_dict
                    
                    logger.info(f"Pattern params loaded from DB: {cache_key}")
                    return param_dict
                else:
                    logger.warning(f"No active params found for {cache_key}, using defaults")
                    
            except Exception as e:
                logger.error(f"Failed to load pattern params from DB: {e}")
        
        # Return defaults
        return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict:
        """Get default pattern parameters"""
        return {
            'pattern_confidence_threshold': 0.6,
            'pattern_lookback_period': 50,
            'pattern_min_pattern_size': 5,
            'pattern_max_pattern_size': 15,
            'pattern_use_volume_confirmation': True,
            'pattern_volume_threshold': 1.5,
            'pattern_use_regime_filter': True,
            'pattern_target_multiplier': 2.0,
            'pattern_stop_loss_multiplier': 1.0,
        }
    
    def clear_cache(self):
        """Clear parameter cache"""
        self.param_cache = {}
        logger.info("Pattern parameter cache cleared")
