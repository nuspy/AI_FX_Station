"""RL Actor-Critic Integrator"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from loguru import logger

class RLIntegrator:
    """Integrates RL Agent for portfolio weight prediction"""
    
    def __init__(self, rl_manager=None, config: Dict = None):
        self.rl_manager = rl_manager
        self.config = config or {}
        self.hybrid_alpha = self.config.get('rl_hybrid_alpha', 0.5)
    
    def predict_weights(self, state: np.ndarray) -> np.ndarray:
        """Predict portfolio weights from state"""
        if not self.rl_manager:
            # Return equal weights
            n_assets = 1
            return np.array([1.0 / n_assets])
        
        try:
            # Use RL agent
            weights = self.rl_manager.predict_weights_backtest(state, deterministic=True)
            return weights
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            return np.array([1.0])
    
    def blend_weights(self, rl_weights: np.ndarray, riskfolio_weights: np.ndarray) -> np.ndarray:
        """Blend RL and Riskfolio weights (hybrid mode)"""
        alpha = self.hybrid_alpha
        blended = alpha * rl_weights + (1 - alpha) * riskfolio_weights
        
        # Normalize to sum to 1
        blended = blended / blended.sum()
        
        logger.debug(f"Blended weights: alpha={alpha}, RL={rl_weights[0]:.3f}, Riskfolio={riskfolio_weights[0]:.3f}, Final={blended[0]:.3f}")
        
        return blended
    
    def prepare_state(self, data: pd.DataFrame, portfolio_state: Dict) -> np.ndarray:
        """Prepare 137-dimensional state for RL agent"""
        # Simplified state (would be full 137-dim in production)
        features = []
        
        # Portfolio returns (last 20 days)
        returns = data['close'].pct_change().dropna().tail(20)
        features.extend(returns.values.tolist())
        
        # Volatility
        vol = returns.std()
        features.append(vol)
        
        # Current portfolio value
        features.append(portfolio_state.get('value', 1.0))
        
        # Pad to 137 dimensions
        while len(features) < 137:
            features.append(0.0)
        
        return np.array(features[:137], dtype=np.float32)
