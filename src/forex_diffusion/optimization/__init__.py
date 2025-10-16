"""
E2E Optimization Module

Complete end-to-end parameter optimization system integrating:
- SSSD, Riskfolio, Pattern Parameters, RL Actor-Critic
- VIX, Sentiment, Volume filters
- Bayesian Optimization (Optuna) and Genetic Algorithms (NSGA-II)
"""

from .parameter_space import ParameterSpace, ParameterDef
from .e2e_optimizer import E2EOptimizer, E2EOptimizerConfig
from .bayesian_optimizer import BayesianOptimizer, BayesianConfig
from .objective_functions import ObjectiveCalculator, MultiObjectiveResult
from .convergence_detector import ConvergenceDetector

__all__ = [
    'ParameterSpace',
    'ParameterDef',
    'E2EOptimizer',
    'E2EOptimizerConfig',
    'BayesianOptimizer',
    'BayesianConfig',
    'ObjectiveCalculator',
    'MultiObjectiveResult',
    'ConvergenceDetector',
]
