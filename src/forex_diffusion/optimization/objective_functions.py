"""Objective Functions for Multi-Objective Optimization"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class MultiObjectiveResult:
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    cost_ratio: float
    combined_score: float
    
    # LDM4TS specific metrics (optional)
    ldm4ts_avg_uncertainty: float = 0.0
    ldm4ts_signal_acceptance_rate: float = 0.0
    ldm4ts_directional_accuracy: float = 0.0

class ObjectiveCalculator:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'sharpe': 0.4, 
            'max_dd': 0.25, 
            'pf': 0.15, 
            'cost': 0.05,
            'ldm4ts_uncertainty': 0.10,  # NEW: Reward low uncertainty
            'ldm4ts_accuracy': 0.05  # NEW: Reward high directional accuracy
        }
    
    def calculate(self, backtest_result: Dict) -> MultiObjectiveResult:
        sharpe = backtest_result.get('sharpe_ratio', 0.0)
        max_dd = abs(backtest_result.get('max_drawdown_pct', 0.0))
        pf = backtest_result.get('profit_factor', 1.0)
        total_return = backtest_result.get('total_return', 0.0)
        total_costs = backtest_result.get('total_costs', 0.0)
        cost_ratio = total_costs / abs(total_return) if total_return != 0 else 1.0
        
        # LDM4TS metrics (if available)
        ldm4ts_metrics = backtest_result.get('metadata', {}).get('ldm4ts', {})
        ldm4ts_avg_uncertainty = ldm4ts_metrics.get('avg_uncertainty', 0.0)
        ldm4ts_total_signals = ldm4ts_metrics.get('total_signals', 0)
        ldm4ts_total_predictions = ldm4ts_metrics.get('total_predictions', 1)
        ldm4ts_signal_acceptance = ldm4ts_total_signals / ldm4ts_total_predictions if ldm4ts_total_predictions > 0 else 0.0
        ldm4ts_directional_accuracy = ldm4ts_metrics.get('directional_accuracy', 0.0)
        
        # Combined score (higher is better)
        combined = (
            self.weights['sharpe'] * sharpe - 
            self.weights['max_dd'] * max_dd / 100.0 +
            self.weights['pf'] * (pf - 1.0) -
            self.weights['cost'] * cost_ratio
        )
        
        # Add LDM4TS contribution (if enabled)
        if ldm4ts_total_predictions > 0:
            # Reward low uncertainty (0-1 scale, lower is better → negate for maximization)
            uncertainty_score = -ldm4ts_avg_uncertainty / 100.0  # Normalize to 0-1
            # Reward high directional accuracy (0-1 scale, higher is better)
            accuracy_score = ldm4ts_directional_accuracy
            
            combined += (
                self.weights.get('ldm4ts_uncertainty', 0.0) * uncertainty_score +
                self.weights.get('ldm4ts_accuracy', 0.0) * accuracy_score
            )
        
        return MultiObjectiveResult(
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            profit_factor=pf,
            cost_ratio=cost_ratio,
            combined_score=combined,
            ldm4ts_avg_uncertainty=ldm4ts_avg_uncertainty,
            ldm4ts_signal_acceptance_rate=ldm4ts_signal_acceptance,
            ldm4ts_directional_accuracy=ldm4ts_directional_accuracy
        )
    
    def get_multi_objective_tuple(self, result: MultiObjectiveResult) -> Tuple[float, float, float, float]:
        """Return tuple for multi-objective optimization (maximize all)"""
        return (result.sharpe_ratio, -result.max_drawdown_pct, result.profit_factor, -result.cost_ratio)
