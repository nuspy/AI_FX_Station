"""
Genetic Algorithm for focused parameter optimization.

Implements a specialized genetic algorithm that:
1. Starts with wide exploration of parameter space
2. Identifies high-performance regions
3. Focuses search on promising areas
4. Converges to optimal parameters for both D1 and D2 objectives
"""

from __future__ import annotations

import numpy as np
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population"""
    parameters: Dict[str, Any]
    fitness_d1: float = 0.0  # Profit maximization score
    fitness_d2: float = 0.0  # Risk minimization score
    combined_fitness: float = 0.0
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    generation: int = 0


@dataclass
class GAConfig:
    """Configuration for genetic algorithm"""
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_percentage: float = 0.2
    convergence_threshold: float = 0.01
    exploration_phases: int = 3


class ParameterBounds:
    """Manages parameter bounds and sampling"""

    def __init__(self, parameter_ranges: Dict[str, Tuple]):
        self.ranges = parameter_ranges
        self.current_bounds = parameter_ranges.copy()
        self.exploration_history = []

    def get_bounds(self, parameter: str) -> Tuple:
        """Get current bounds for a parameter"""
        return self.current_bounds.get(parameter, self.ranges[parameter])

    def narrow_bounds(self, parameter: str, center: float, reduction_factor: float = 0.3):
        """Narrow bounds around a center point"""
        min_val, max_val = self.ranges[parameter]
        current_range = max_val - min_val
        new_range = current_range * reduction_factor

        new_min = max(min_val, center - new_range / 2)
        new_max = min(max_val, center + new_range / 2)

        self.current_bounds[parameter] = (new_min, new_max)
        logger.debug(f"Narrowed bounds for {parameter}: {self.current_bounds[parameter]}")

    def reset_bounds(self):
        """Reset bounds to original ranges"""
        self.current_bounds = self.ranges.copy()


class GeneticAlgorithm:
    """
    Genetic Algorithm for multi-objective parameter optimization.

    Implements NSGA-II inspired approach with adaptive exploration phases.
    """

    def __init__(self, config: GAConfig, parameter_ranges: Dict[str, Tuple]):
        self.config = config
        self.parameter_bounds = ParameterBounds(parameter_ranges)
        self.population: List[Individual] = []
        self.best_individuals_d1: List[Individual] = []
        self.best_individuals_d2: List[Individual] = []
        self.generation = 0
        self.exploration_phase = 0

    def initialize_population(self) -> List[Individual]:
        """Initialize random population"""
        population = []

        for _ in range(self.config.population_size):
            parameters = {}
            for param_name, bounds in self.parameter_bounds.ranges.items():
                if isinstance(bounds[0], int):
                    # Integer parameter
                    parameters[param_name] = random.randint(bounds[0], bounds[1])
                else:
                    # Float parameter
                    parameters[param_name] = random.uniform(bounds[0], bounds[1])

            individual = Individual(
                parameters=parameters,
                generation=self.generation
            )
            population.append(individual)

        self.population = population
        logger.info(f"Initialized GA population with {len(population)} individuals")
        return population

    def evaluate_population(self, evaluation_function):
        """Evaluate fitness for all individuals in population"""
        for individual in self.population:
            if individual.fitness_d1 == 0.0 and individual.fitness_d2 == 0.0:
                # Evaluate individual
                d1_score, d2_score = evaluation_function(individual.parameters)
                individual.fitness_d1 = d1_score
                individual.fitness_d2 = d2_score
                individual.combined_fitness = self._calculate_combined_fitness(d1_score, d2_score)

        # Calculate Pareto ranks and crowding distances
        self._calculate_pareto_ranking()

    def _calculate_combined_fitness(self, d1_score: float, d2_score: float) -> float:
        """Calculate combined fitness score"""
        # Weighted combination (can be adjusted based on strategy preference)
        return 0.6 * d1_score + 0.4 * d2_score

    def _calculate_pareto_ranking(self):
        """Calculate Pareto ranking and crowding distances"""
        # Simple Pareto ranking implementation
        population_size = len(self.population)
        domination_count = [0] * population_size
        dominated_individuals = [[] for _ in range(population_size)]
        fronts = [[]]

        # Calculate domination relationships
        for i in range(population_size):
            for j in range(population_size):
                if i != j:
                    if self._dominates(self.population[i], self.population[j]):
                        dominated_individuals[i].append(j)
                    elif self._dominates(self.population[j], self.population[i]):
                        domination_count[i] += 1

            if domination_count[i] == 0:
                self.population[i].pareto_rank = 0
                fronts[0].append(i)

        # Build subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            for i in fronts[front_index]:
                for j in dominated_individuals[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        self.population[j].pareto_rank = front_index + 1
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)
            front_index += 1

        # Calculate crowding distances
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    self.population[i].crowding_distance = float('inf')
            else:
                # Sort by each objective and calculate distances
                for i in front:
                    self.population[i].crowding_distance = 0

                # D1 objective
                front.sort(key=lambda x: self.population[x].fitness_d1)
                self.population[front[0]].crowding_distance = float('inf')
                self.population[front[-1]].crowding_distance = float('inf')

                max_d1 = self.population[front[-1]].fitness_d1
                min_d1 = self.population[front[0]].fitness_d1
                d1_range = max_d1 - min_d1 if max_d1 != min_d1 else 1

                for i in range(1, len(front) - 1):
                    distance = (self.population[front[i + 1]].fitness_d1 -
                              self.population[front[i - 1]].fitness_d1) / d1_range
                    self.population[front[i]].crowding_distance += distance

                # D2 objective
                front.sort(key=lambda x: self.population[x].fitness_d2)
                max_d2 = self.population[front[-1]].fitness_d2
                min_d2 = self.population[front[0]].fitness_d2
                d2_range = max_d2 - min_d2 if max_d2 != min_d2 else 1

                for i in range(1, len(front) - 1):
                    distance = (self.population[front[i + 1]].fitness_d2 -
                              self.population[front[i - 1]].fitness_d2) / d2_range
                    self.population[front[i]].crowding_distance += distance

    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if individual 1 dominates individual 2"""
        return (ind1.fitness_d1 >= ind2.fitness_d1 and ind1.fitness_d2 >= ind2.fitness_d2 and
                (ind1.fitness_d1 > ind2.fitness_d1 or ind1.fitness_d2 > ind2.fitness_d2))

    def select_parents(self) -> List[Individual]:
        """Select parents for reproduction using tournament selection"""
        parents = []
        tournament_size = 3

        for _ in range(self.config.population_size):
            # Tournament selection
            tournament = random.sample(self.population, tournament_size)
            winner = min(tournament, key=lambda x: (x.pareto_rank, -x.crowding_distance))
            parents.append(winner)

        return parents

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Create offspring through crossover"""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2

        child1_params = {}
        child2_params = {}

        for param_name in parent1.parameters:
            if random.random() < 0.5:
                child1_params[param_name] = parent1.parameters[param_name]
                child2_params[param_name] = parent2.parameters[param_name]
            else:
                child1_params[param_name] = parent2.parameters[param_name]
                child2_params[param_name] = parent1.parameters[param_name]

        child1 = Individual(parameters=child1_params, generation=self.generation + 1)
        child2 = Individual(parameters=child2_params, generation=self.generation + 1)

        return child1, child2

    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual"""
        if random.random() > self.config.mutation_rate:
            return individual

        mutated_params = individual.parameters.copy()

        # Select random parameter to mutate
        param_to_mutate = random.choice(list(mutated_params.keys()))
        bounds = self.parameter_bounds.get_bounds(param_to_mutate)

        if isinstance(bounds[0], int):
            # Integer parameter
            mutated_params[param_to_mutate] = random.randint(bounds[0], bounds[1])
        else:
            # Float parameter - add gaussian noise
            current_value = mutated_params[param_to_mutate]
            noise_std = (bounds[1] - bounds[0]) * 0.1  # 10% of range
            new_value = current_value + random.gauss(0, noise_std)
            new_value = max(bounds[0], min(bounds[1], new_value))
            mutated_params[param_to_mutate] = new_value

        return Individual(parameters=mutated_params, generation=self.generation + 1)

    def evolve_generation(self, evaluation_function) -> List[Individual]:
        """Evolve one generation"""
        # Select parents
        parents = self.select_parents()

        # Generate offspring
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = self.crossover(parents[i], parents[i + 1])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            offspring.extend([child1, child2])

        # Evaluate offspring
        for individual in offspring:
            d1_score, d2_score = evaluation_function(individual.parameters)
            individual.fitness_d1 = d1_score
            individual.fitness_d2 = d2_score
            individual.combined_fitness = self._calculate_combined_fitness(d1_score, d2_score)

        # Combine population and offspring
        combined_population = self.population + offspring

        # Environmental selection (keep best individuals)
        combined_population.sort(key=lambda x: (x.pareto_rank, -x.crowding_distance))
        self.population = combined_population[:self.config.population_size]

        self.generation += 1

        # Update best individuals
        self._update_best_individuals()

        return self.population

    def _update_best_individuals(self):
        """Update lists of best individuals for each objective"""
        # Best for D1 (profit maximization)
        d1_sorted = sorted(self.population, key=lambda x: x.fitness_d1, reverse=True)
        self.best_individuals_d1 = d1_sorted[:5]

        # Best for D2 (risk minimization)
        d2_sorted = sorted(self.population, key=lambda x: x.fitness_d2, reverse=True)
        self.best_individuals_d2 = d2_sorted[:5]

    def should_advance_exploration_phase(self) -> bool:
        """Determine if we should advance to next exploration phase"""
        if self.generation < 20:  # Minimum generations per phase
            return False

        # Check convergence of top individuals
        if len(self.best_individuals_d1) < 3:
            return False

        # Calculate diversity in top solutions
        d1_scores = [ind.fitness_d1 for ind in self.best_individuals_d1[:3]]
        d2_scores = [ind.fitness_d2 for ind in self.best_individuals_d2[:3]]

        d1_diversity = max(d1_scores) - min(d1_scores) if d1_scores else 0
        d2_diversity = max(d2_scores) - min(d2_scores) if d2_scores else 0

        # If diversity is low, advance to next phase
        return d1_diversity < self.config.convergence_threshold and d2_diversity < self.config.convergence_threshold

    def advance_exploration_phase(self):
        """Advance to next exploration phase by narrowing parameter space"""
        if self.exploration_phase >= self.config.exploration_phases:
            return

        logger.info(f"Advancing to exploration phase {self.exploration_phase + 1}")

        # Find centers of high-performance regions
        for param_name in self.parameter_bounds.ranges:
            # Get parameter values from best individuals
            d1_values = [ind.parameters[param_name] for ind in self.best_individuals_d1]
            d2_values = [ind.parameters[param_name] for ind in self.best_individuals_d2]

            # Calculate centers
            d1_center = np.mean(d1_values) if d1_values else None
            d2_center = np.mean(d2_values) if d2_values else None

            # Use the center that shows more promise
            if d1_center is not None and d2_center is not None:
                # Choose center based on which objective shows better convergence
                center = d1_center if len(set(d1_values)) <= len(set(d2_values)) else d2_center
            else:
                center = d1_center or d2_center

            if center is not None:
                # Narrow bounds around center
                reduction_factor = 0.5 - (self.exploration_phase * 0.15)  # Gradually narrow more
                self.parameter_bounds.narrow_bounds(param_name, center, reduction_factor)

        self.exploration_phase += 1

        # Re-initialize some population for exploration of new bounds
        new_individuals_count = self.config.population_size // 3
        for i in range(new_individuals_count):
            parameters = {}
            for param_name, bounds in self.parameter_bounds.current_bounds.items():
                if isinstance(bounds[0], int):
                    parameters[param_name] = random.randint(bounds[0], bounds[1])
                else:
                    parameters[param_name] = random.uniform(bounds[0], bounds[1])

            self.population[i] = Individual(parameters=parameters, generation=self.generation)

    def get_best_parameters(self, strategy: str = "balanced") -> Dict[str, Any]:
        """Get best parameters based on strategy"""
        if strategy == "high_return":
            return self.best_individuals_d1[0].parameters if self.best_individuals_d1 else {}
        elif strategy == "low_risk":
            return self.best_individuals_d2[0].parameters if self.best_individuals_d2 else {}
        else:  # balanced
            # Return parameters from best combined fitness
            best_combined = max(self.population, key=lambda x: x.combined_fitness)
            return best_combined.parameters

    def get_pareto_front(self) -> List[Individual]:
        """Get current Pareto front"""
        return [ind for ind in self.population if ind.pareto_rank == 0]

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics"""
        pareto_front = self.get_pareto_front()

        return {
            "generation": self.generation,
            "exploration_phase": self.exploration_phase,
            "population_size": len(self.population),
            "pareto_front_size": len(pareto_front),
            "best_d1_score": max(ind.fitness_d1 for ind in self.population),
            "best_d2_score": max(ind.fitness_d2 for ind in self.population),
            "best_combined_score": max(ind.combined_fitness for ind in self.population),
            "avg_d1_score": np.mean([ind.fitness_d1 for ind in self.population]),
            "avg_d2_score": np.mean([ind.fitness_d2 for ind in self.population]),
            "parameter_bounds": dict(self.parameter_bounds.current_bounds)
        }