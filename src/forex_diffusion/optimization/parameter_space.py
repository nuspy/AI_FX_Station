"""
Parameter Space Definition for E2E Optimization

Defines all 90+ optimizable parameters across 7 component groups:
1. SSSD (10 params)
2. Riskfolio (8 params)
3. Pattern Parameters (20 params)
4. RL Actor-Critic (15 params)
5. Risk Management (12 params)
6. Position Sizing (10 params)
7. Market Filters (15 params: VIX, Sentiment, Volume)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Callable, Optional
import numpy as np


@dataclass
class ParameterDef:
    """Single parameter definition with bounds and constraints"""
    name: str
    group: str  # 'sssd', 'riskfolio', 'patterns', 'rl', 'risk', 'sizing', 'filters'
    type: str  # 'int', 'float', 'categorical', 'bool'
    bounds: Tuple[Any, Any] | List[Any]  # (min, max) for numeric, List of choices for categorical
    default: Any
    description: str
    log_scale: bool = False  # For learning rates, etc.
    constraints: List[Callable] = field(default_factory=list)


class ParameterSpace:
    """
    Complete parameter space for E2E optimization (90+ parameters).
    
    All parameters are organized into 7 groups with proper bounds,
    defaults, and constraints.
    """
    
    def __init__(self):
        self.parameters: Dict[str, ParameterDef] = {}
        self._define_parameters()
    
    def _define_parameters(self):
        """Define all 90+ parameters"""
        
        # ========== GROUP 1: SSSD (10 parameters) ==========
        self._add_param('sssd_diffusion_steps', 'sssd', 'int', (10, 100), 50,
                       'Number of diffusion steps for SSSD sampling')
        self._add_param('sssd_noise_schedule', 'sssd', 'categorical', 
                       ['linear', 'cosine', 'sigmoid'], 'cosine',
                       'Noise schedule type for diffusion process')
        self._add_param('sssd_sampling_method', 'sssd', 'categorical',
                       ['ddpm', 'ddim'], 'ddim',
                       'Sampling method (DDPM slower but more accurate)')
        self._add_param('sssd_guidance_scale', 'sssd', 'float', (0.0, 2.0), 1.0,
                       'Guidance scale for conditional generation')
        self._add_param('sssd_temperature', 'sssd', 'float', (0.5, 1.5), 1.0,
                       'Temperature for sampling randomness')
        self._add_param('sssd_quantile_confidence', 'sssd', 'float', (0.6, 0.95), 0.8,
                       'Confidence level for quantile-based sizing')
        self._add_param('sssd_uncertainty_threshold', 'sssd', 'float', (0.1, 0.5), 0.2,
                       'Threshold for uncertainty-based position reduction')
        self._add_param('sssd_forecast_horizon', 'sssd', 'int', (1, 24), 4,
                       'Forecast horizon in bars')
        self._add_param('sssd_min_samples', 'sssd', 'int', (100, 500), 200,
                       'Minimum samples for SSSD inference')
        self._add_param('sssd_use_ensemble', 'sssd', 'bool', [True, False], True,
                       'Use ensemble of SSSD samples')
        
        # ========== GROUP 2: RISKFOLIO (8 parameters) ==========
        self._add_param('riskfolio_risk_measure', 'riskfolio', 'categorical',
                       ['MV', 'CVaR', 'CDaR', 'EVaR', 'WR'], 'CVaR',
                       'Risk measure for portfolio optimization')
        self._add_param('riskfolio_objective', 'riskfolio', 'categorical',
                       ['Sharpe', 'MinRisk', 'Utility', 'MaxRet'], 'Sharpe',
                       'Optimization objective')
        self._add_param('riskfolio_risk_aversion', 'riskfolio', 'float', (0.1, 5.0), 1.0,
                       'Risk aversion coefficient', log_scale=True)
        self._add_param('riskfolio_max_weight', 'riskfolio', 'float', (0.1, 0.5), 0.3,
                       'Maximum weight per asset')
        self._add_param('riskfolio_min_weight', 'riskfolio', 'float', (0.0, 0.05), 0.0,
                       'Minimum weight per asset')
        self._add_param('riskfolio_risk_free_rate', 'riskfolio', 'float', (0.0, 0.05), 0.0,
                       'Risk-free rate for Sharpe calculation')
        self._add_param('riskfolio_use_risk_parity', 'riskfolio', 'bool', [True, False], False,
                       'Use risk parity instead of mean-variance')
        self._add_param('riskfolio_rebalance_freq', 'riskfolio', 'int', (1, 30), 7,
                       'Rebalancing frequency in days')
        
        # ========== GROUP 3: PATTERN PARAMETERS (20 parameters) ==========
        self._add_param('pattern_confidence_threshold', 'patterns', 'float', (0.5, 0.9), 0.6,
                       'Minimum confidence for pattern signals')
        self._add_param('pattern_lookback_period', 'patterns', 'int', (20, 100), 50,
                       'Lookback period for pattern detection')
        self._add_param('pattern_min_pattern_size', 'patterns', 'int', (3, 10), 5,
                       'Minimum bars in pattern')
        self._add_param('pattern_max_pattern_size', 'patterns', 'int', (10, 30), 15,
                       'Maximum bars in pattern')
        self._add_param('pattern_use_volume_confirmation', 'patterns', 'bool', [True, False], True,
                       'Require volume confirmation')
        self._add_param('pattern_volume_threshold', 'patterns', 'float', (1.2, 3.0), 1.5,
                       'Volume spike threshold vs average')
        self._add_param('pattern_use_regime_filter', 'patterns', 'bool', [True, False], True,
                       'Filter patterns by regime')
        self._add_param('pattern_regime_confirmation_bars', 'patterns', 'int', (3, 10), 5,
                       'Bars to confirm regime')
        self._add_param('pattern_false_breakout_filter', 'patterns', 'bool', [True, False], True,
                       'Filter false breakouts')
        self._add_param('pattern_breakout_retest_bars', 'patterns', 'int', (2, 8), 3,
                       'Bars to allow breakout retest')
        self._add_param('pattern_target_multiplier', 'patterns', 'float', (1.0, 3.0), 2.0,
                       'Take profit as multiple of pattern size')
        self._add_param('pattern_stop_loss_multiplier', 'patterns', 'float', (0.5, 1.5), 1.0,
                       'Stop loss as multiple of pattern size')
        self._add_param('pattern_trailing_activation_pct', 'patterns', 'float', (0.5, 2.0), 1.0,
                       'Profit % to activate trailing stop')
        self._add_param('pattern_trailing_step_pct', 'patterns', 'float', (0.1, 0.5), 0.2,
                       'Trailing stop step size %')
        self._add_param('pattern_max_age_bars', 'patterns', 'int', (5, 50), 20,
                       'Maximum age of pattern before expiry')
        self._add_param('pattern_confluence_weight', 'patterns', 'float', (0.0, 1.0), 0.5,
                       'Weight for pattern confluence')
        self._add_param('pattern_timeframe_alignment', 'patterns', 'bool', [True, False], True,
                       'Require multi-timeframe alignment')
        self._add_param('pattern_support_resistance_weight', 'patterns', 'float', (0.0, 1.0), 0.3,
                       'Weight for S/R levels')
        self._add_param('pattern_trend_alignment_weight', 'patterns', 'float', (0.0, 1.0), 0.4,
                       'Weight for trend alignment')
        self._add_param('pattern_momentum_confirmation', 'patterns', 'bool', [True, False], True,
                       'Require momentum confirmation')
        
        # ========== GROUP 4: RL ACTOR-CRITIC (15 parameters) ==========
        self._add_param('rl_actor_lr', 'rl', 'float', (1e-5, 1e-3), 3e-4,
                       'Actor learning rate', log_scale=True)
        self._add_param('rl_critic_lr', 'rl', 'float', (1e-5, 1e-3), 3e-4,
                       'Critic learning rate', log_scale=True)
        self._add_param('rl_gamma', 'rl', 'float', (0.90, 0.99), 0.95,
                       'Discount factor for future rewards')
        self._add_param('rl_clip_epsilon', 'rl', 'float', (0.1, 0.3), 0.2,
                       'PPO clipping parameter')
        self._add_param('rl_gae_lambda', 'rl', 'float', (0.90, 0.99), 0.95,
                       'GAE lambda for advantage estimation')
        self._add_param('rl_entropy_coef', 'rl', 'float', (0.0, 0.1), 0.01,
                       'Entropy coefficient for exploration')
        self._add_param('rl_value_coef', 'rl', 'float', (0.1, 1.0), 0.5,
                       'Value loss coefficient')
        self._add_param('rl_max_grad_norm', 'rl', 'float', (0.1, 1.0), 0.5,
                       'Max gradient norm for clipping')
        self._add_param('rl_ppo_epochs', 'rl', 'int', (3, 10), 5,
                       'PPO epochs per update')
        self._add_param('rl_mini_batch_size', 'rl', 'int', (32, 256), 64,
                       'Mini-batch size for PPO')
        self._add_param('rl_actor_hidden_layers', 'rl', 'categorical',
                       ['[256,128]', '[512,256]', '[256,256,128]'], '[256,128]',
                       'Actor network hidden layers')
        self._add_param('rl_critic_hidden_layers', 'rl', 'categorical',
                       ['[256,128]', '[512,256]', '[256,256,128]'], '[256,128]',
                       'Critic network hidden layers')
        self._add_param('rl_use_lstm', 'rl', 'bool', [True, False], False,
                       'Use LSTM in networks')
        self._add_param('rl_lstm_hidden_size', 'rl', 'int', (64, 256), 128,
                       'LSTM hidden size')
        self._add_param('rl_hybrid_alpha', 'rl', 'float', (0.0, 1.0), 0.5,
                       'Blend factor: RL * alpha + Riskfolio * (1-alpha)')
        
        # ========== GROUP 5: RISK MANAGEMENT (12 parameters) ==========
        self._add_param('risk_stop_loss_pct', 'risk', 'float', (0.5, 5.0), 2.0,
                       'Initial stop loss %')
        self._add_param('risk_take_profit_pct', 'risk', 'float', (1.0, 10.0), 4.0,
                       'Take profit %')
        self._add_param('risk_trailing_stop_pct', 'risk', 'float', (0.5, 3.0), 1.0,
                       'Trailing stop %')
        self._add_param('risk_trailing_activation_pct', 'risk', 'float', (0.5, 2.0), 1.0,
                       'Profit % to activate trailing')
        self._add_param('risk_atr_multiplier', 'risk', 'float', (1.0, 3.0), 2.0,
                       'ATR multiplier for stop loss')
        self._add_param('risk_atr_period', 'risk', 'int', (10, 30), 14,
                       'ATR calculation period')
        self._add_param('risk_use_trailing_stop', 'risk', 'bool', [True, False], True,
                       'Enable trailing stops')
        self._add_param('risk_use_time_based_exit', 'risk', 'bool', [True, False], True,
                       'Enable time-based exits')
        self._add_param('risk_max_holding_hours', 'risk', 'int', (6, 48), 24,
                       'Maximum holding period in hours')
        self._add_param('risk_daily_loss_limit_pct', 'risk', 'float', (2.0, 5.0), 3.0,
                       'Daily loss limit %')
        self._add_param('risk_use_breakeven_stop', 'risk', 'bool', [True, False], True,
                       'Move stop to breakeven after profit')
        self._add_param('risk_breakeven_trigger_pct', 'risk', 'float', (0.5, 1.5), 1.0,
                       'Profit % to trigger breakeven')
        
        # ========== GROUP 6: POSITION SIZING (10 parameters) ==========
        self._add_param('sizing_method', 'sizing', 'categorical',
                       ['fixed_fraction', 'kelly', 'optimal_f', 'volatility_adjusted'], 'kelly',
                       'Position sizing method')
        self._add_param('sizing_base_risk_pct', 'sizing', 'float', (0.5, 2.0), 1.0,
                       'Base risk per trade %')
        self._add_param('sizing_kelly_fraction', 'sizing', 'float', (0.1, 0.5), 0.25,
                       'Kelly fraction (0.25 = quarter Kelly)')
        self._add_param('sizing_max_position_size_pct', 'sizing', 'float', (10.0, 50.0), 20.0,
                       'Maximum position size % of capital')
        self._add_param('sizing_regime_trending_multiplier', 'sizing', 'float', (1.0, 2.0), 1.5,
                       'Size multiplier in trending regimes')
        self._add_param('sizing_regime_ranging_multiplier', 'sizing', 'float', (0.5, 1.0), 0.7,
                       'Size multiplier in ranging regimes')
        self._add_param('sizing_regime_volatile_multiplier', 'sizing', 'float', (0.3, 0.8), 0.5,
                       'Size multiplier in volatile regimes')
        self._add_param('sizing_confidence_scaling', 'sizing', 'bool', [True, False], True,
                       'Scale size by signal confidence')
        self._add_param('sizing_confidence_min_multiplier', 'sizing', 'float', (0.3, 0.7), 0.5,
                       'Min size multiplier at low confidence')
        self._add_param('sizing_confidence_max_multiplier', 'sizing', 'float', (1.0, 2.0), 1.5,
                       'Max size multiplier at high confidence')
        
        # ========== GROUP 7: MARKET FILTERS (15 parameters) ==========
        # VIX Filter (5 params)
        self._add_param('filter_vix_enabled', 'filters', 'bool', [True, False], True,
                       'Enable VIX filter')
        self._add_param('filter_vix_high_threshold', 'filters', 'float', (20.0, 40.0), 30.0,
                       'VIX level considered high')
        self._add_param('filter_vix_extreme_threshold', 'filters', 'float', (35.0, 60.0), 50.0,
                       'VIX level considered extreme')
        self._add_param('filter_vix_high_reduction_pct', 'filters', 'float', (0.3, 0.7), 0.5,
                       'Position size reduction when VIX high')
        self._add_param('filter_vix_extreme_reduction_pct', 'filters', 'float', (0.5, 0.9), 0.7,
                       'Position size reduction when VIX extreme')
        
        # Sentiment Filter (5 params)
        self._add_param('filter_sentiment_enabled', 'filters', 'bool', [True, False], True,
                       'Enable sentiment filter')
        self._add_param('filter_sentiment_contrarian_threshold', 'filters', 'float', (0.6, 0.9), 0.75,
                       'Sentiment level to trigger contrarian strategy')
        self._add_param('filter_sentiment_confidence_threshold', 'filters', 'float', (0.5, 0.8), 0.6,
                       'Minimum sentiment confidence')
        self._add_param('filter_sentiment_fade_strength', 'filters', 'float', (0.5, 1.5), 1.0,
                       'Strength of contrarian signal')
        self._add_param('filter_sentiment_use_news', 'filters', 'bool', [True, False], False,
                       'Use news sentiment (requires news feed)')
        
        # Volume Filter (5 params)
        self._add_param('filter_volume_enabled', 'filters', 'bool', [True, False], True,
                       'Enable volume filter')
        self._add_param('filter_volume_obv_period', 'filters', 'int', (10, 50), 20,
                       'On-Balance Volume period')
        self._add_param('filter_volume_vwap_period', 'filters', 'int', (20, 100), 50,
                       'VWAP period')
        self._add_param('filter_volume_spike_threshold', 'filters', 'float', (1.5, 3.0), 2.0,
                       'Volume spike threshold vs average')
        self._add_param('filter_volume_min_liquidity_pct', 'filters', 'float', (0.5, 0.9), 0.7,
                       'Minimum liquidity for trade execution')
    
    def _add_param(self, name: str, group: str, type: str, bounds: Tuple | List,
                   default: Any, description: str, log_scale: bool = False):
        """Add parameter to space"""
        self.parameters[name] = ParameterDef(
            name=name,
            group=group,
            type=type,
            bounds=bounds,
            default=default,
            description=description,
            log_scale=log_scale
        )
    
    def get_parameter_definitions(self) -> Dict[str, ParameterDef]:
        """Get all parameter definitions"""
        return self.parameters
    
    def get_parameter_bounds(self) -> Dict[str, Tuple]:
        """Get parameter bounds (for optimization)"""
        return {name: param.bounds for name, param in self.parameters.items()}
    
    def get_parameter_groups(self) -> Dict[str, List[str]]:
        """Get parameters grouped by component"""
        groups = {}
        for name, param in self.parameters.items():
            if param.group not in groups:
                groups[param.group] = []
            groups[param.group].append(name)
        return groups
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters"""
        return {name: param.default for name, param in self.parameters.items()}
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter values"""
        for name, value in params.items():
            if name not in self.parameters:
                return False
            
            param_def = self.parameters[name]
            
            # Type validation
            if param_def.type == 'int' and not isinstance(value, (int, np.integer)):
                return False
            elif param_def.type == 'float' and not isinstance(value, (float, int, np.floating)):
                return False
            elif param_def.type == 'bool' and not isinstance(value, bool):
                return False
            elif param_def.type == 'categorical' and value not in param_def.bounds:
                return False
            
            # Bounds validation (numeric)
            if param_def.type in ('int', 'float'):
                min_val, max_val = param_def.bounds
                if not (min_val <= value <= max_val):
                    return False
        
        return True
    
    def apply_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter constraints (e.g., stop_loss < take_profit)"""
        # Constraint 1: stop_loss < take_profit
        if params.get('risk_stop_loss_pct') and params.get('risk_take_profit_pct'):
            if params['risk_stop_loss_pct'] >= params['risk_take_profit_pct']:
                params['risk_take_profit_pct'] = params['risk_stop_loss_pct'] * 2.0
        
        # Constraint 2: min_pattern_size < max_pattern_size
        if params.get('pattern_min_pattern_size') and params.get('pattern_max_pattern_size'):
            if params['pattern_min_pattern_size'] >= params['pattern_max_pattern_size']:
                params['pattern_max_pattern_size'] = params['pattern_min_pattern_size'] + 5
        
        # Constraint 3: VIX thresholds ordered
        if params.get('filter_vix_high_threshold') and params.get('filter_vix_extreme_threshold'):
            if params['filter_vix_high_threshold'] >= params['filter_vix_extreme_threshold']:
                params['filter_vix_extreme_threshold'] = params['filter_vix_high_threshold'] + 10.0
        
        return params
    
    def count_parameters(self) -> int:
        """Count total parameters"""
        return len(self.parameters)
    
    def count_by_group(self) -> Dict[str, int]:
        """Count parameters by group"""
        groups = self.get_parameter_groups()
        return {group: len(params) for group, params in groups.items()}
