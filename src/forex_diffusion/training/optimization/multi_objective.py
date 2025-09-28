"""
Multi-objective optimization for pattern parameters using Pareto frontiers.

This module provides NSGA-II inspired multi-objective optimization with support for
dual-dataset optimization, Pareto front calculation, and compromise solution selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from loguru import logger

@dataclass
class ObjectiveResult:
    """Result for a single objective evaluation"""
    value: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_feasible: bool = True
    constraint_violations: List[str] = field(default_factory=list)

@dataclass
class MultiObjectiveResult:
    """Multi-objective evaluation result"""
    objectives: Dict[str, ObjectiveResult] = field(default_factory=dict)
    combined_score: Optional[float] = None
    pareto_rank: Optional[int] = None
    crowding_distance: Optional[float] = None
    dominates: List[int] = field(default_factory=list)
    dominated_by: List[int] = field(default_factory=list)

class MultiObjectiveEvaluator:
    """
    Evaluates pattern performance across multiple objectives and datasets.

    Supports:
    - Success rate optimization on D1 and D2
    - Expectancy maximization
    - Risk-adjusted returns (Sharpe, Sortino)
    - Robustness metrics
    - Constraint validation
    """

    def __init__(self):
        self.objective_functions = {
            "success_rate_d1": self._eval_success_rate_d1,
            "success_rate_d2": self._eval_success_rate_d2,
            "expectancy_d1": self._eval_expectancy_d1,
            "expectancy_d2": self._eval_expectancy_d2,
            "profit_factor_d1": self._eval_profit_factor_d1,
            "profit_factor_d2": self._eval_profit_factor_d2,
            "max_drawdown_d1": self._eval_max_drawdown_d1,
            "max_drawdown_d2": self._eval_max_drawdown_d2,
            "robustness": self._eval_robustness,
            "consistency": self._eval_consistency
        }

    def evaluate_multi_objective(self, metrics: Dict[str, Dict[str, Any]],
                                objectives: List[str] = None) -> MultiObjectiveResult:
        """
        Evaluate multiple objectives for multi-objective optimization.

        Args:
            metrics: Performance metrics by dataset {"D1": {...}, "D2": {...}}
            objectives: List of objectives to evaluate (default: success_rate + expectancy)

        Returns:
            Multi-objective evaluation result
        """
        if objectives is None:
            objectives = ["success_rate_d1", "expectancy_d1"]
            if "D2" in metrics:
                objectives.extend(["success_rate_d2", "expectancy_d2"])

        result = MultiObjectiveResult()

        # Evaluate each objective
        for obj_name in objectives:
            if obj_name in self.objective_functions:
                try:
                    obj_result = self.objective_functions[obj_name](metrics)
                    result.objectives[obj_name] = obj_result
                except Exception as e:
                    logger.warning(f"Failed to evaluate objective {obj_name}: {e}")
                    result.objectives[obj_name] = ObjectiveResult(
                        value=0.0, is_feasible=False,
                        constraint_violations=[f"Evaluation error: {e}"]
                    )

        # Check overall feasibility
        is_feasible = all(obj.is_feasible for obj in result.objectives.values())

        if is_feasible:
            # Calculate combined score for single-objective fallback
            result.combined_score = self._calculate_combined_score(result.objectives)
        else:
            result.combined_score = -1e6  # Heavy penalty for infeasible solutions

        return result

    def evaluate_single_objective(self, metrics: Dict[str, Dict[str, Any]],
                                 weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evaluate single weighted objective combining multiple metrics.

        Args:
            metrics: Performance metrics by dataset
            weights: Weights for combining D1/D2 metrics

        Returns:
            Single objective scores
        """
        if weights is None:
            weights = {"D1": 0.7, "D2": 0.3}

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        combined_score = 0.0
        component_scores = {}

        for dataset_id, dataset_metrics in metrics.items():
            weight = weights.get(dataset_id, 0.0)
            if weight > 0:
                # Calculate dataset score combining success rate and expectancy
                success_rate = dataset_metrics.get("success_rate", 0.0)
                expectancy = dataset_metrics.get("expectancy", 0.0)

                # Normalize expectancy to 0-1 range (assuming reasonable bounds)
                normalized_expectancy = max(0, min(1, (expectancy + 0.5) / 1.0))

                dataset_score = 0.6 * success_rate + 0.4 * normalized_expectancy
                component_scores[f"{dataset_id}_score"] = dataset_score
                combined_score += weight * dataset_score

        return {
            "combined_score": combined_score,
            **component_scores
        }

    def _eval_success_rate_d1(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate success rate on dataset D1"""
        d1_metrics = metrics.get("D1", {})
        success_rate = d1_metrics.get("success_rate", 0.0)
        total_signals = d1_metrics.get("total_signals", 0)

        # Constraint: minimum signals required
        is_feasible = total_signals >= 10
        violations = [] if is_feasible else ["Insufficient signals on D1"]

        return ObjectiveResult(
            value=success_rate,
            metrics={"total_signals": total_signals},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_success_rate_d2(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate success rate on dataset D2"""
        d2_metrics = metrics.get("D2", {})
        if not d2_metrics:
            return ObjectiveResult(value=0.0, is_feasible=False,
                                 constraint_violations=["D2 metrics not available"])

        success_rate = d2_metrics.get("success_rate", 0.0)
        total_signals = d2_metrics.get("total_signals", 0)

        is_feasible = total_signals >= 5  # Lower threshold for D2
        violations = [] if is_feasible else ["Insufficient signals on D2"]

        return ObjectiveResult(
            value=success_rate,
            metrics={"total_signals": total_signals},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_expectancy_d1(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate expectancy on dataset D1"""
        d1_metrics = metrics.get("D1", {})
        expectancy = d1_metrics.get("expectancy", 0.0)

        # Constraint: positive expectancy preferred
        is_feasible = expectancy >= -0.1  # Allow slight negative
        violations = [] if is_feasible else ["Highly negative expectancy on D1"]

        return ObjectiveResult(
            value=expectancy,
            metrics={"raw_expectancy": expectancy},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_expectancy_d2(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate expectancy on dataset D2"""
        d2_metrics = metrics.get("D2", {})
        if not d2_metrics:
            return ObjectiveResult(value=0.0, is_feasible=False,
                                 constraint_violations=["D2 metrics not available"])

        expectancy = d2_metrics.get("expectancy", 0.0)

        is_feasible = expectancy >= -0.2  # More lenient for D2
        violations = [] if is_feasible else ["Highly negative expectancy on D2"]

        return ObjectiveResult(
            value=expectancy,
            metrics={"raw_expectancy": expectancy},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_profit_factor_d1(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate profit factor on dataset D1"""
        d1_metrics = metrics.get("D1", {})
        profit_factor = d1_metrics.get("profit_factor", 0.0)

        is_feasible = profit_factor >= 0.8
        violations = [] if is_feasible else ["Poor profit factor on D1"]

        return ObjectiveResult(
            value=profit_factor,
            metrics={"profit_factor": profit_factor},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_profit_factor_d2(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate profit factor on dataset D2"""
        d2_metrics = metrics.get("D2", {})
        if not d2_metrics:
            return ObjectiveResult(value=0.0, is_feasible=False,
                                 constraint_violations=["D2 metrics not available"])

        profit_factor = d2_metrics.get("profit_factor", 0.0)

        is_feasible = profit_factor >= 0.7  # More lenient for D2
        violations = [] if is_feasible else ["Poor profit factor on D2"]

        return ObjectiveResult(
            value=profit_factor,
            metrics={"profit_factor": profit_factor},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_max_drawdown_d1(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate maximum drawdown on dataset D1 (minimize)"""
        d1_metrics = metrics.get("D1", {})
        max_dd = d1_metrics.get("max_drawdown", 1.0)

        # Convert to maximization problem (lower drawdown is better)
        value = 1.0 - abs(max_dd)

        is_feasible = abs(max_dd) <= 0.3  # Max 30% drawdown
        violations = [] if is_feasible else ["Excessive drawdown on D1"]

        return ObjectiveResult(
            value=value,
            metrics={"max_drawdown": max_dd},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_max_drawdown_d2(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate maximum drawdown on dataset D2 (minimize)"""
        d2_metrics = metrics.get("D2", {})
        if not d2_metrics:
            return ObjectiveResult(value=0.0, is_feasible=False,
                                 constraint_violations=["D2 metrics not available"])

        max_dd = d2_metrics.get("max_drawdown", 1.0)
        value = 1.0 - abs(max_dd)

        is_feasible = abs(max_dd) <= 0.4  # More lenient for D2
        violations = [] if is_feasible else ["Excessive drawdown on D2"]

        return ObjectiveResult(
            value=value,
            metrics={"max_drawdown": max_dd},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_robustness(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate robustness across datasets"""
        if len(metrics) < 2:
            return ObjectiveResult(value=0.0, is_feasible=False,
                                 constraint_violations=["Need multiple datasets for robustness"])

        # Calculate consistency of performance across datasets
        success_rates = [m.get("success_rate", 0.0) for m in metrics.values()]
        expectancies = [m.get("expectancy", 0.0) for m in metrics.values()]

        # Robustness = 1 - variance of performance metrics
        sr_var = np.var(success_rates) if len(success_rates) > 1 else 0
        exp_var = np.var(expectancies) if len(expectancies) > 1 else 0

        robustness = 1.0 - min(1.0, sr_var + exp_var * 0.5)

        is_feasible = robustness >= 0.3
        violations = [] if is_feasible else ["Poor robustness across datasets"]

        return ObjectiveResult(
            value=robustness,
            metrics={"success_rate_variance": sr_var, "expectancy_variance": exp_var},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _eval_consistency(self, metrics: Dict[str, Dict[str, Any]]) -> ObjectiveResult:
        """Evaluate temporal consistency"""
        # Average consistency score across datasets
        consistency_scores = []

        for dataset_metrics in metrics.values():
            consistency = dataset_metrics.get("consistency_score", 0.5)
            consistency_scores.append(consistency)

        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

        is_feasible = avg_consistency >= 0.3
        violations = [] if is_feasible else ["Poor temporal consistency"]

        return ObjectiveResult(
            value=avg_consistency,
            metrics={"individual_consistencies": consistency_scores},
            is_feasible=is_feasible,
            constraint_violations=violations
        )

    def _calculate_combined_score(self, objectives: Dict[str, ObjectiveResult]) -> float:
        """Calculate combined score from multiple objectives"""
        if not objectives:
            return 0.0

        # Weight objectives by importance
        weights = {
            "success_rate_d1": 0.3,
            "success_rate_d2": 0.25,
            "expectancy_d1": 0.25,
            "expectancy_d2": 0.15,
            "robustness": 0.05
        }

        total_score = 0.0
        total_weight = 0.0

        for obj_name, obj_result in objectives.items():
            weight = weights.get(obj_name, 0.1)  # Default weight for unknown objectives
            if obj_result.is_feasible:
                total_score += weight * obj_result.value
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

class ParetoOptimizer:
    """
    Pareto frontier optimization for multi-objective pattern optimization.

    Implements NSGA-II inspired algorithms for:
    - Pareto dominance ranking
    - Crowding distance calculation
    - Best compromise solution selection
    """

    def calculate_pareto_front(self, trial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate Pareto frontier from trial results.

        Args:
            trial_results: List of completed trial results with objectives

        Returns:
            List of non-dominated solutions (Pareto front)
        """
        if not trial_results:
            return []

        # Extract objective values for each trial
        objectives_data = []
        for i, trial in enumerate(trial_results):
            if trial["status"] != "completed" or "metrics" not in trial:
                continue

            objectives = self._extract_objectives(trial)
            if objectives:
                objectives_data.append({
                    "trial_index": i,
                    "trial": trial,
                    "objectives": objectives
                })

        if not objectives_data:
            return []

        # Calculate Pareto ranks
        pareto_ranks = self._calculate_pareto_ranks(objectives_data)

        # Get first front (rank 0)
        pareto_front = []
        for i, data in enumerate(objectives_data):
            if pareto_ranks[i] == 0:
                trial_copy = dict(data["trial"])
                trial_copy["pareto_rank"] = 0
                trial_copy["objectives"] = data["objectives"]
                pareto_front.append(trial_copy)

        # Calculate crowding distances for the front
        if len(pareto_front) > 2:
            crowding_distances = self._calculate_crowding_distances(pareto_front)
            for i, trial in enumerate(pareto_front):
                trial["crowding_distance"] = crowding_distances[i]

        logger.info(f"Calculated Pareto front with {len(pareto_front)} solutions")
        return pareto_front

    def select_best_compromise(self, pareto_front: List[Dict[str, Any]],
                              preferences: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """
        Select best compromise solution from Pareto front.

        Args:
            pareto_front: Non-dominated solutions
            preferences: User preferences for different objectives

        Returns:
            Best compromise solution
        """
        if not pareto_front:
            return None

        if len(pareto_front) == 1:
            return pareto_front[0]

        # If no preferences specified, use crowding distance
        if not preferences:
            return max(pareto_front,
                      key=lambda x: x.get("crowding_distance", 0))

        # Calculate weighted distance from ideal point
        best_solution = None
        best_score = -float('inf')

        for solution in pareto_front:
            objectives = solution.get("objectives", {})
            score = self._calculate_preference_score(objectives, preferences)

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def _extract_objectives(self, trial: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract objective values from trial result"""
        metrics = trial.get("metrics", {})
        if not metrics:
            return None

        objectives = {}

        # Extract success rates
        for dataset_id in ["D1", "D2"]:
            if dataset_id in metrics:
                dataset_metrics = metrics[dataset_id]
                objectives[f"success_rate_{dataset_id.lower()}"] = dataset_metrics.get("success_rate", 0.0)
                objectives[f"expectancy_{dataset_id.lower()}"] = dataset_metrics.get("expectancy", 0.0)

        return objectives if objectives else None

    def _calculate_pareto_ranks(self, objectives_data: List[Dict[str, Any]]) -> List[int]:
        """Calculate Pareto dominance ranks using fast non-dominated sorting"""
        n = len(objectives_data)
        ranks = [0] * n

        # Count domination relationships
        domination_count = [0] * n  # How many solutions dominate this one
        dominated_solutions = [[] for _ in range(n)]  # Which solutions this one dominates

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives_data[i]["objectives"],
                                     objectives_data[j]["objectives"]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives_data[j]["objectives"],
                                       objectives_data[i]["objectives"]):
                        domination_count[i] += 1

        # Find first front (rank 0)
        current_front = []
        for i in range(n):
            if domination_count[i] == 0:
                ranks[i] = 0
                current_front.append(i)

        # Find subsequent fronts
        current_rank = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = current_rank + 1
                        next_front.append(j)

            current_front = next_front
            current_rank += 1

        return ranks

    def _dominates(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)"""
        if not obj1 or not obj2:
            return False

        all_better_or_equal = True
        at_least_one_better = False

        for key in set(obj1.keys()) | set(obj2.keys()):
            val1 = obj1.get(key, 0.0)
            val2 = obj2.get(key, 0.0)

            if val1 < val2:
                all_better_or_equal = False
                break
            elif val1 > val2:
                at_least_one_better = True

        return all_better_or_equal and at_least_one_better

    def _calculate_crowding_distances(self, solutions: List[Dict[str, Any]]) -> List[float]:
        """Calculate crowding distances for diversity preservation"""
        n = len(solutions)
        distances = [0.0] * n

        if n <= 2:
            return [float('inf')] * n

        # Get all objective names
        all_objectives = set()
        for solution in solutions:
            objectives = solution.get("objectives", {})
            all_objectives.update(objectives.keys())

        # Calculate crowding distance for each objective
        for obj_name in all_objectives:
            # Sort by this objective
            sorted_indices = sorted(range(n),
                                  key=lambda i: solutions[i].get("objectives", {}).get(obj_name, 0))

            # Get objective values
            obj_values = [solutions[i].get("objectives", {}).get(obj_name, 0)
                         for i in sorted_indices]

            # Set boundary points to infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Calculate range
            obj_range = obj_values[-1] - obj_values[0]
            if obj_range == 0:
                continue

            # Calculate crowding distance for intermediate points
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                if distances[idx] != float('inf'):
                    distances[idx] += (obj_values[i + 1] - obj_values[i - 1]) / obj_range

        return distances

    def _calculate_preference_score(self, objectives: Dict[str, float],
                                  preferences: Dict[str, float]) -> float:
        """Calculate preference-weighted score"""
        score = 0.0
        total_weight = 0.0

        for obj_name, obj_value in objectives.items():
            weight = preferences.get(obj_name, 1.0)
            score += weight * obj_value
            total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0