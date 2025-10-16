"""SSSD Integrator - Quantile-based Position Sizing"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from loguru import logger

class SssdIntegrator:
    """Integrates SSSD model for uncertainty-aware position sizing"""
    
    def __init__(self, model=None, config: Dict = None):
        self.model = model
        self.config = config or {}
        self.quantiles = [0.05, 0.50, 0.95]
    
    def predict_quantiles(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Predict q05, q50, q95 quantiles"""
        if not self.model:
            # Mock implementation - return last price with uncertainty
            last_price = data['close'].iloc[-1] if len(data) > 0 else 1.0
            volatility = data['close'].pct_change().std() if len(data) > 1 else 0.02
            q50 = last_price
            q05 = q50 * (1 - 2 * volatility)
            q95 = q50 * (1 + 2 * volatility)
            return (q05, q50, q95)
        
        # Real SSSD inference
        try:
            forecast_horizon = self.config.get('sssd_forecast_horizon', 4)
            samples = []
            for _ in range(100):  # Monte Carlo samples
                sample = self.model.sample(data, horizon=forecast_horizon)
                samples.append(sample)
            
            samples = np.array(samples)
            q05 = np.quantile(samples, 0.05)
            q50 = np.quantile(samples, 0.50)
            q95 = np.quantile(samples, 0.95)
            
            return (float(q05), float(q50), float(q95))
        except Exception as e:
            logger.error(f"SSSD prediction failed: {e}")
            return self.predict_quantiles(data)  # Fallback to mock
    
    def calculate_uncertainty(self, q05: float, q50: float, q95: float) -> float:
        """Calculate uncertainty spread"""
        if q50 == 0:
            return 1.0
        spread = (q95 - q05) / q50
        return spread
    
    def calculate_confidence_multiplier(self, uncertainty: float, threshold: float = 0.2) -> float:
        """Adjust position size based on uncertainty"""
        if uncertainty < threshold:
            return 1.5  # High confidence → increase size
        elif uncertainty > threshold * 2:
            return 0.5  # High uncertainty → reduce size
        else:
            return 1.0  # Normal
    
    def get_stop_take_from_quantiles(self, q05: float, q50: float, q95: float, 
                                     direction: str) -> Tuple[float, float]:
        """Set stop loss and take profit based on quantiles"""
        if direction == 'long':
            stop_loss = q05  # Pessimistic scenario
            take_profit = q95  # Optimistic scenario
        else:
            stop_loss = q95
            take_profit = q05
        
        return (stop_loss, take_profit)
