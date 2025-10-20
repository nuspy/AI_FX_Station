"""
Hybrid optimization combining grid search and genetic algorithms.

Strategy:
1. Coarse grid search to identify promising regions
2. Fine-grained genetic algorithm in best regions
3. Final local refinement
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

from .genetic_optimizer import ParameterSpace, GeneticOptimizer
from .kernc_integration import run_backtest, BACKTESTING_AVAILABLE


@dataclass
class GridPoint:
    """Single point in grid search."""
    params: Dict[str, float]
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int


class HybridOptimizer:
    """
    Hybrid optimizer combining grid search and genetic algorithms.

    Phase 1: Coarse grid search across entire parameter space
    Phase 2: Genetic algorithm focused on top K regions
    Phase 3: Local refinement of best solution
    """

    def __init__(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_spaces: List[ParameterSpace],
        predictions: Optional[pd.Series] = None,
        model_path: Optional[str] = None,
        grid_divisions: int = 5,
        top_k_regions: int = 3,
        ga_population: int = 50,
        ga_generations: int = 30,
        n_workers: int = 4,
        seed: int = 0
    ):
        """
        Initialize Hybrid Optimizer.

        Args:
            strategy_class: Strategy class to optimize
            data: OHLCV DataFrame
            param_spaces: List of parameter spaces
            predictions: Optional precomputed predictions
            model_path: Optional model checkpoint path
            grid_divisions: Number of divisions per parameter in grid search
            top_k_regions: Number of top regions to refine with GA
            ga_population: GA population size
            ga_generations: GA generations
            n_workers: Number of parallel workers
            seed: Random seed
        """
        if not BACKTESTING_AVAILABLE:
            raise ImportError("backtesting.py required")

        self.strategy_class = strategy_class
        self.data = data
        self.param_spaces = param_spaces
        self.predictions = predictions
        self.model_path = model_path
        self.grid_divisions = grid_divisions
        self.top_k_regions = top_k_regions
        self.ga_population = ga_population
        self.ga_generations = ga_generations
        self.n_workers = n_workers
        self.seed = seed

        self.grid_results: List[GridPoint] = []
        self.ga_results: List[Dict[str, Any]] = []
        self.best_solution: Optional[Dict[str, Any]] = None

    def _create_grid_points(self) -> List[Dict[str, float]]:
        """Create grid points for coarse search."""
        param_values = []
        for param_space in self.param_spaces:
            if param_space.log_scale:
                values = np.logspace(
                    np.log10(param_space.min_value),
                    np.log10(param_space.max_value),
                    self.grid_divisions
                )
            else:
                values = np.linspace(
                    param_space.min_value,
                    param_space.max_value,
                    self.grid_divisions
                )

            # Apply step if defined
            if param_space.step is not None:
                values = np.round(values / param_space.step) * param_space.step

            param_values.append(values)

        # Create all combinations
        grid_points = []
        for combination in product(*param_values):
            point = {}
            for i, param_space in enumerate(self.param_spaces):
                point[param_space.name] = float(combination[i])
            grid_points.append(point)

        logger.info(f"Created {len(grid_points)} grid points for evaluation")
        return grid_points

    def _evaluate_grid_point(self, params: Dict[str, float]) -> GridPoint:
        """Evaluate single grid point."""
        try:
            results = run_backtest(
                strategy_class=self.strategy_class,
                data=self.data,
                predictions=self.predictions,
                model_path=self.model_path,
                **params
            )

            return GridPoint(
                params=params,
                return_pct=results.get('return', 0),
                sharpe_ratio=results.get('sharpe_ratio', 0),
                max_drawdown=results.get('max_drawdown', 100),
                num_trades=results.get('num_trades', 0)
            )

        except Exception as e:
            logger.error(f"Grid evaluation failed for {params}: {e}")
            return GridPoint(
                params=params,
                return_pct=0,
                sharpe_ratio=0,
                max_drawdown=100,
                num_trades=0
            )

    def phase1_grid_search(self) -> List[GridPoint]:
        """
        Phase 1: Coarse grid search.

        Returns:
            List of GridPoint results sorted by Sharpe ratio
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Coarse Grid Search")
        logger.info("=" * 60)

        grid_points = self._create_grid_points()

        # Evaluate grid points in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._evaluate_grid_point, params): params for params in grid_points}

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)

                if i % 10 == 0:
                    logger.info(f"Evaluated {i}/{len(grid_points)} grid points")

        # Sort by Sharpe ratio
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        self.grid_results = results

        logger.info(f"Grid search complete: Best Sharpe={results[0].sharpe_ratio:.2f}")
        logger.info(f"Best parameters: {results[0].params}")

        return results

    def _create_focused_param_spaces(self, center: Dict[str, float], region_size: float = 0.2) -> List[ParameterSpace]:
        """
        Create focused parameter spaces around a center point.

        Args:
            center: Center parameter values
            region_size: Region size as fraction of original range (e.g., 0.2 = 20%)

        Returns:
            List of focused ParameterSpace objects
        """
        focused_spaces = []
        for param_space in self.param_spaces:
            center_value = center[param_space.name]
            original_range = param_space.max_value - param_space.min_value
            new_range = original_range * region_size

            min_val = max(param_space.min_value, center_value - new_range / 2)
            max_val = min(param_space.max_value, center_value + new_range / 2)

            focused_spaces.append(ParameterSpace(
                name=param_space.name,
                min_value=min_val,
                max_value=max_val,
                step=param_space.step,
                log_scale=param_space.log_scale
            ))

        return focused_spaces

    def phase2_genetic_refinement(self) -> List[Dict[str, Any]]:
        """
        Phase 2: Genetic algorithm refinement in top K regions.

        Returns:
            List of best solutions from each GA run
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Genetic Algorithm Refinement")
        logger.info("=" * 60)

        if not self.grid_results:
            raise RuntimeError("Must run phase1_grid_search first")

        # Get top K regions
        top_regions = self.grid_results[:self.top_k_regions]

        ga_solutions = []
        for i, region in enumerate(top_regions, 1):
            logger.info(f"Refining region {i}/{self.top_k_regions} centered at: {region.params}")

            # Create focused parameter spaces
            focused_spaces = self._create_focused_param_spaces(region.params, region_size=0.3)

            # Run GA
            ga_optimizer = GeneticOptimizer(
                strategy_class=self.strategy_class,
                data=self.data,
                param_spaces=focused_spaces,
                predictions=self.predictions,
                model_path=self.model_path,
                population_size=self.ga_population,
                n_generations=self.ga_generations,
                seed=self.seed + i
            )

            ga_result = ga_optimizer.optimize_and_select(selection_criterion='sharpe')
            ga_solutions.append(ga_result)

            logger.info(f"Region {i} best: Sharpe={ga_result['sharpe_ratio']:.2f}, Return={ga_result['return']:.2f}%")

        self.ga_results = ga_solutions
        return ga_solutions

    def phase3_local_refinement(self, best_solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 3: Local refinement of best solution.

        Uses fine grid search around best solution.

        Args:
            best_solution: Best solution from phase 2

        Returns:
            Refined solution
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Local Refinement")
        logger.info("=" * 60)

        # Create very focused parameter spaces (10% range)
        focused_spaces = self._create_focused_param_spaces(best_solution['params'], region_size=0.1)

        # Fine grid search (more divisions)
        fine_grid_points = []
        param_values = []

        for param_space in focused_spaces:
            if param_space.log_scale:
                values = np.logspace(
                    np.log10(param_space.min_value),
                    np.log10(param_space.max_value),
                    7  # Finer divisions
                )
            else:
                values = np.linspace(
                    param_space.min_value,
                    param_space.max_value,
                    7
                )

            if param_space.step is not None:
                values = np.round(values / param_space.step) * param_space.step

            param_values.append(values)

        # Create grid
        for combination in product(*param_values):
            point = {}
            for i, param_space in enumerate(focused_spaces):
                point[param_space.name] = float(combination[i])
            fine_grid_points.append(point)

        logger.info(f"Refining with {len(fine_grid_points)} fine grid points")

        # Evaluate
        refined_results = []
        for params in fine_grid_points:
            result = self._evaluate_grid_point(params)
            refined_results.append(result)

        # Sort by Sharpe
        refined_results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        best = refined_results[0]

        refined_solution = {
            'params': best.params,
            'return': best.return_pct,
            'sharpe_ratio': best.sharpe_ratio,
            'max_drawdown': best.max_drawdown,
            'num_trades': best.num_trades
        }

        logger.info(f"Refinement complete: Sharpe={refined_solution['sharpe_ratio']:.2f}")
        logger.info(f"Final parameters: {refined_solution['params']}")

        return refined_solution

    def optimize(self, skip_phase3: bool = False) -> Dict[str, Any]:
        """
        Run full hybrid optimization.

        Args:
            skip_phase3: Whether to skip phase 3 local refinement

        Returns:
            Best solution with parameters and metrics
        """
        logger.info("=" * 60)
        logger.info("HYBRID OPTIMIZATION")
        logger.info(f"Parameters: {[p.name for p in self.param_spaces]}")
        logger.info(f"Grid divisions: {self.grid_divisions}")
        logger.info(f"GA population: {self.ga_population}, generations: {self.ga_generations}")
        logger.info("=" * 60)

        # Phase 1: Grid search
        self.phase1_grid_search()

        # Phase 2: GA refinement
        ga_solutions = self.phase2_genetic_refinement()

        # Find best from GA
        best_ga = max(ga_solutions, key=lambda x: x['sharpe_ratio'])

        # Phase 3: Local refinement (optional)
        if not skip_phase3:
            final_solution = self.phase3_local_refinement(best_ga)
        else:
            final_solution = best_ga

        self.best_solution = final_solution

        # Summary
        logger.info("=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"Final Sharpe Ratio: {final_solution['sharpe_ratio']:.2f}")
        logger.info(f"Final Return: {final_solution['return']:.2f}%")
        logger.info(f"Final Max Drawdown: {final_solution['max_drawdown']:.2f}%")
        logger.info(f"Final Parameters: {final_solution['params']}")
        logger.info("=" * 60)

        return final_solution

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization process.

        Returns:
            Dictionary with results from all phases
        """
        if not self.best_solution:
            raise RuntimeError("Must run optimize() first")

        return {
            'best_solution': self.best_solution,
            'grid_search_top_10': [
                {
                    'params': gp.params,
                    'sharpe': gp.sharpe_ratio,
                    'return': gp.return_pct,
                    'drawdown': gp.max_drawdown
                }
                for gp in self.grid_results[:10]
            ],
            'ga_refinements': self.ga_results,
            'total_grid_evaluations': len(self.grid_results),
            'n_parameters': len(self.param_spaces),
            'parameter_names': [p.name for p in self.param_spaces]
        }
