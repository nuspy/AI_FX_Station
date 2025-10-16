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

class ObjectiveCalculator:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'sharpe': 0.5, 'max_dd': 0.3, 'pf': 0.15, 'cost': 0.05}
    
    def calculate(self, backtest_result: Dict) -> MultiObjectiveResult:
        sharpe = backtest_result.get('sharpe_ratio', 0.0)
        max_dd = abs(backtest_result.get('max_drawdown_pct', 0.0))
        pf = backtest_result.get('profit_factor', 1.0)
        total_return = backtest_result.get('total_return', 0.0)
        total_costs = backtest_result.get('total_costs', 0.0)
        cost_ratio = total_costs / abs(total_return) if total_return != 0 else 1.0
        
        # Combined score (higher is better)
        combined = (self.weights['sharpe'] * sharpe - 
                   self.weights['max_dd'] * max_dd / 100.0 +
                   self.weights['pf'] * (pf - 1.0) -
                   self.weights['cost'] * cost_ratio)
        
        return MultiObjectiveResult(
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            profit_factor=pf,
            cost_ratio=cost_ratio,
            combined_score=combined
        )
    
    def get_multi_objective_tuple(self, result: MultiObjectiveResult) -> Tuple[float, float, float, float]:
        """Return tuple for multi-objective optimization (maximize all)"""
        return (result.sharpe_ratio, -result.max_drawdown_pct, result.profit_factor, -result.cost_ratio)
