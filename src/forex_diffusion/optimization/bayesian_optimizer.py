"""Bayesian Optimization using Optuna"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, List, Any
import json
from loguru import logger

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")

from .parameter_space import ParameterSpace

@dataclass
class BayesianConfig:
    sampler: str = 'tpe'  # 'tpe', 'random', 'cmaes'
    n_startup_trials: int = 10
    pruner: str = 'median'  # 'median', 'hyperband', 'none'
    pruner_warmup_steps: int = 5
    direction: List[str] = None  # ['maximize', 'minimize'] for multi-obj
    n_jobs: int = 1
    timeout: int = None

class BayesianOptimizer:
    def __init__(self, parameter_space: ParameterSpace, config: BayesianConfig = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required. Install: pip install optuna")
        
        self.parameter_space = parameter_space
        self.config = config or BayesianConfig()
        self.study: optuna.Study = None
    
    def optimize(self, objective_func: Callable, n_trials: int, study_name: str = None) -> Dict:
        """Run Bayesian optimization"""
        logger.info(f"Starting Bayesian optimization: {n_trials} trials")
        
        # Create study
        directions = self.config.direction or ['maximize']
        
        self.study = optuna.create_study(
            study_name=study_name or "e2e_optimization",
            directions=directions if len(directions) > 1 else None,
            direction=directions[0] if len(directions) == 1 else None,
            sampler=self._create_sampler(),
            pruner=self._create_pruner()
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self._objective_wrapper(trial, objective_func),
            n_trials=n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            show_progress_bar=True
        )
        
        # Get best results
        if len(directions) == 1:
            best_params = self.study.best_params
            best_value = self.study.best_value
        else:
            # Multi-objective: return Pareto front
            best_params = self._get_pareto_front()
            best_value = None
        
        logger.info(f"Optimization complete. Best Sharpe: {best_value if best_value else 'Multiple'}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def _objective_wrapper(self, trial: optuna.Trial, objective_func: Callable):
        """Wrap objective function with parameter suggestions"""
        params = {}
        
        for param_name, param_def in self.parameter_space.parameters.items():
            if param_def.type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_def.bounds[0], param_def.bounds[1])
            elif param_def.type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_def.bounds[0], param_def.bounds[1],
                    log=param_def.log_scale
                )
            elif param_def.type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_def.bounds)
            elif param_def.type == 'bool':
                params[param_name] = trial.suggest_categorical(param_name, [True, False])
        
        # Apply constraints
        params = self.parameter_space.apply_constraints(params)
        
        # Call objective function
        return objective_func(trial.number, params)
    
    def _create_sampler(self):
        """Create Optuna sampler"""
        if self.config.sampler == 'tpe':
            return TPESampler(n_startup_trials=self.config.n_startup_trials)
        elif self.config.sampler == 'random':
            return optuna.samplers.RandomSampler()
        elif self.config.sampler == 'cmaes':
            return optuna.samplers.CmaEsSampler()
        else:
            return TPESampler()
    
    def _create_pruner(self):
        """Create Optuna pruner"""
        if self.config.pruner == 'median':
            return MedianPruner(n_warmup_steps=self.config.pruner_warmup_steps)
        elif self.config.pruner == 'hyperband':
            return optuna.pruners.HyperbandPruner()
        else:
            return optuna.pruners.NopPruner()
    
    def _get_pareto_front(self) -> List[Dict]:
        """Get Pareto optimal solutions"""
        pareto_trials = [t for t in self.study.best_trials if t.values is not None]
        return [t.params for t in pareto_trials]
    
    def get_best_params(self) -> Dict:
        """Get best parameters"""
        if not self.study:
            return {}
        return self.study.best_params
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history"""
        if not self.study:
            return []
        return [{'trial': t.number, 'value': t.value, 'params': t.params} 
                for t in self.study.trials]
