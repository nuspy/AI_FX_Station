"""Convergence Detection for Optimization"""
from __future__ import annotations
import numpy as np
from typing import List

class ConvergenceDetector:
    def __init__(self, patience: int = 20, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = -np.inf
        self.trials_without_improvement = 0
        self.history: List[float] = []
    
    def update(self, current_value: float) -> bool:
        """Update with new trial. Returns True if converged."""
        self.history.append(current_value)
        
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1
        
        # Check convergence
        converged = self.trials_without_improvement >= self.patience
        
        # Also check plateau
        if len(self.history) >= self.patience:
            recent_std = np.std(self.history[-self.patience:])
            if recent_std < self.min_delta:
                return True
        
        return converged
    
    def reset(self):
        """Reset detector"""
        self.best_value = -np.inf
        self.trials_without_improvement = 0
        self.history = []
