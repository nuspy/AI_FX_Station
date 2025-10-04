"""
Genetic Algorithm optimizer for strategy parameters using NSGA-II.

Multi-objective optimization to maximize return while minimizing drawdown
and maximizing Sharpe ratio.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

# pymoo imports for NSGA-II
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logger.warning("pymoo not installed - install with: pip install pymoo")

# Import backtesting components
from .kernc_integration import run_backtest, BACKTESTING_AVAILABLE


@dataclass
class ParameterSpace:
    """
    Parameter space definition for optimization.

    Attributes:
        name: Parameter name
        min_value: Minimum value
        max_value: Maximum value
        step: Optional step size (for discrete parameters)
        log_scale: Whether to use log scale
    """
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    log_scale: bool = False

    def sample(self) -> float:
        """Sample a random value from this parameter space."""
        if self.log_scale:
            log_min = np.log10(self.min_value)
            log_max = np.log10(self.max_value)
            value = 10 ** np.random.uniform(log_min, log_max)
        else:
            value = np.random.uniform(self.min_value, self.max_value)

        if self.step is not None:
            value = round(value / self.step) * self.step

        return float(np.clip(value, self.min_value, self.max_value))


class StrategyOptimizationProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for strategy parameters.

    Objectives:
    1. Maximize return
    2. Minimize max drawdown
    3. Maximize Sharpe ratio
    """

    def __init__(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_spaces: List[ParameterSpace],
        predictions: Optional[pd.Series] = None,
        model_path: Optional[str] = None,
        constraints: Optional[List[Callable]] = None
    ):
        """
        Initialize optimization problem.

        Args:
            strategy_class: Strategy class to optimize
            data: OHLCV data for backtesting
            param_spaces: List of parameter spaces to optimize
            predictions: Optional precomputed predictions
            model_path: Optional model checkpoint path
            constraints: Optional constraint functions
        """
        self.strategy_class = strategy_class
        self.data = data
        self.param_spaces = param_spaces
        self.predictions = predictions
        self.model_path = model_path
        self.constraints = constraints or []

        # Define bounds for parameters
        xl = np.array([p.min_value for p in param_spaces])
        xu = np.array([p.max_value for p in param_spaces])

        # Initialize problem (3 objectives, 0 equality constraints, N inequality constraints)
        super().__init__(
            n_var=len(param_spaces),
            n_obj=3,  # return, -drawdown, sharpe
            n_constr=len(self.constraints),
            xl=xl,
            xu=xu
        )

        self.eval_count = 0

    def _evaluate(self, x: np.ndarray, out: Dict[str, np.ndarray], *args, **kwargs):
        """
        Evaluate strategy with given parameters.

        Args:
            x: Parameter vector
            out: Output dictionary (modified in-place)
        """
        # Convert parameter vector to dict
        params = {}
        for i, param_space in enumerate(self.param_spaces):
            value = x[i]

            # Apply step if defined
            if param_space.step is not None:
                value = round(value / param_space.step) * param_space.step

            params[param_space.name] = float(value)

        # Run backtest
        try:
            results = run_backtest(
                strategy_class=self.strategy_class,
                data=self.data,
                predictions=self.predictions,
                model_path=self.model_path,
                **params
            )

            # Extract objectives
            ret = results.get('return', 0)
            max_dd = results.get('max_drawdown', 100)
            sharpe = results.get('sharpe_ratio', 0)

            # Objectives (to be minimized, so negate return and sharpe)
            out["F"] = [
                -ret,           # Maximize return -> minimize -return
                max_dd,         # Minimize max drawdown
                -sharpe         # Maximize Sharpe -> minimize -Sharpe
            ]

            # Constraints
            constraint_values = []
            for constraint_fn in self.constraints:
                constraint_values.append(constraint_fn(results, params))

            if constraint_values:
                out["G"] = constraint_values

            self.eval_count += 1

            if self.eval_count % 10 == 0:
                logger.info(
                    f"Eval {self.eval_count}: Return={ret:.2f}%, DD={max_dd:.2f}%, Sharpe={sharpe:.2f}"
                )

        except Exception as e:
            logger.error(f"Backtest failed with params {params}: {e}")
            # Return worst possible objectives
            out["F"] = [0, 100, 0]
            if self.constraints:
                out["G"] = [1e6] * len(self.constraints)


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer using NSGA-II for multi-objective optimization.

    Optimizes strategy parameters to maximize return, minimize drawdown,
    and maximize Sharpe ratio simultaneously.
    """

    def __init__(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_spaces: List[ParameterSpace],
        predictions: Optional[pd.Series] = None,
        model_path: Optional[str] = None,
        population_size: int = 100,
        n_generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        constraints: Optional[List[Callable]] = None,
        seed: int = 0
    ):
        """
        Initialize Genetic Optimizer.

        Args:
            strategy_class: Strategy class to optimize
            data: OHLCV DataFrame for backtesting
            param_spaces: List of parameter spaces
            predictions: Optional precomputed predictions
            model_path: Optional model checkpoint path
            population_size: GA population size
            n_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            constraints: Optional constraint functions
            seed: Random seed
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo required - install with: pip install pymoo")

        if not BACKTESTING_AVAILABLE:
            raise ImportError("backtesting.py required - install with: pip install backtesting")

        self.strategy_class = strategy_class
        self.data = data
        self.param_spaces = param_spaces
        self.predictions = predictions
        self.model_path = model_path
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.constraints = constraints
        self.seed = seed

        # Create problem
        self.problem = StrategyOptimizationProblem(
            strategy_class=strategy_class,
            data=data,
            param_spaces=param_spaces,
            predictions=predictions,
            model_path=model_path,
            constraints=constraints
        )

    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.

        Returns:
            Dictionary with Pareto front results and best solutions
        """
        logger.info(f"Starting GA optimization: pop={self.population_size}, gen={self.n_generations}")
        logger.info(f"Optimizing {len(self.param_spaces)} parameters: {[p.name for p in self.param_spaces]}")

        # Configure NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.crossover_prob, eta=15),
            mutation=PM(prob=self.mutation_prob, eta=20),
            eliminate_duplicates=True
        )

        # Termination criterion
        termination = get_termination("n_gen", self.n_generations)

        # Run optimization
        res = minimize(
            self.problem,
            algorithm,
            termination,
            seed=self.seed,
            verbose=verbose
        )

        # Extract Pareto front
        pareto_x = res.X  # Parameter vectors
        pareto_f = res.F  # Objective values

        # Convert to results
        pareto_solutions = []
        for i in range(len(pareto_x)):
            params = {}
            for j, param_space in enumerate(self.param_spaces):
                params[param_space.name] = float(pareto_x[i, j])

            pareto_solutions.append({
                'params': params,
                'return': -pareto_f[i, 0],  # Negate back to positive
                'max_drawdown': pareto_f[i, 1],
                'sharpe_ratio': -pareto_f[i, 2]  # Negate back to positive
            })

        # Sort by Sharpe ratio (best first)
        pareto_solutions.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

        logger.info(f"Optimization complete: Found {len(pareto_solutions)} Pareto-optimal solutions")
        logger.info(f"Best Sharpe solution: {pareto_solutions[0]['sharpe_ratio']:.2f}")

        return {
            'pareto_front': pareto_solutions,
            'best_sharpe': pareto_solutions[0],
            'best_return': max(pareto_solutions, key=lambda x: x['return']),
            'best_drawdown': min(pareto_solutions, key=lambda x: x['max_drawdown']),
            'n_evaluations': self.problem.eval_count
        }

    def optimize_and_select(
        self,
        selection_criterion: str = 'sharpe',
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_return: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run optimization and select best solution based on criteria.

        Args:
            selection_criterion: Criterion for selection ('sharpe', 'return', 'balanced')
            min_sharpe: Minimum Sharpe ratio filter
            max_drawdown: Maximum drawdown filter
            min_return: Minimum return filter

        Returns:
            Selected solution with parameters and metrics
        """
        results = self.optimize(verbose=True)
        pareto_front = results['pareto_front']

        # Apply filters
        filtered = pareto_front
        if min_sharpe is not None:
            filtered = [s for s in filtered if s['sharpe_ratio'] >= min_sharpe]
        if max_drawdown is not None:
            filtered = [s for s in filtered if s['max_drawdown'] <= max_drawdown]
        if min_return is not None:
            filtered = [s for s in filtered if s['return'] >= min_return]

        if not filtered:
            logger.warning("No solutions match criteria, using best Sharpe")
            filtered = pareto_front

        # Select based on criterion
        if selection_criterion == 'sharpe':
            selected = max(filtered, key=lambda x: x['sharpe_ratio'])
        elif selection_criterion == 'return':
            selected = max(filtered, key=lambda x: x['return'])
        elif selection_criterion == 'balanced':
            # Balanced: maximize (sharpe * return) / drawdown
            selected = max(filtered, key=lambda x: (x['sharpe_ratio'] * x['return']) / (x['max_drawdown'] + 0.1))
        else:
            raise ValueError(f"Unknown criterion: {selection_criterion}")

        logger.info(f"Selected solution ({selection_criterion}): {selected}")

        return selected


def create_parameter_space_from_ranges(param_ranges: Dict[str, Tuple[float, float]]) -> List[ParameterSpace]:
    """
    Create ParameterSpace list from simple ranges dict.

    Args:
        param_ranges: Dict mapping parameter name to (min, max) tuple

    Returns:
        List of ParameterSpace objects
    """
    spaces = []
    for name, (min_val, max_val) in param_ranges.items():
        spaces.append(ParameterSpace(
            name=name,
            min_value=min_val,
            max_value=max_val
        ))
    return spaces
