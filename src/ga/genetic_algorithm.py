"""
Genetic Algorithm Core Implementation
Supports multiple representations, selection, crossover, and mutation methods
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from ..data.solomon_parser import VRPInstance


@dataclass
class GAConfig:
    """GA Configuration"""
    population_size: int = 100
    max_generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1
    tournament_size: int = 3
    selection_method: str = "tournament"
    crossover_method: str = "pmx"
    mutation_method: str = "swap"
    representation: str = "permutation"
    verbose: bool = True


@dataclass
class GAResult:
    """GA Execution Result"""
    best_fitness: float
    best_chromosome: np.ndarray
    best_routes: List[List[int]]
    fitness_history: List[float]
    diversity_history: List[float]
    runtime: float
    convergence_generation: int
    final_population: List[np.ndarray]
    final_fitness: np.ndarray


class GeneticAlgorithm:
    """Genetic Algorithm with configurable operators"""
    
    def __init__(self, instance: VRPInstance, config: GAConfig):
        self.instance = instance
        self.config = config
        self.dist_matrix = instance.get_distance_matrix()
        
        # Initialize representation
        self.representation = self._get_representation()
        
        # Initialize operators
        self.selection = self._get_selection()
        self.crossover = self._get_crossover()
        self.mutation = self._get_mutation()
        
        # History tracking
        self.fitness_history = []
        self.diversity_history = []
    
    def _get_representation(self):
        """Get representation object"""
        if self.config.representation == "permutation":
            from ..representation.permutation import PermutationRepresentation
            return PermutationRepresentation(self.instance)
        elif self.config.representation == "binary":
            from ..representation.binary import BinaryRepresentation
            return BinaryRepresentation(self.instance)
        elif self.config.representation == "real_valued":
            from ..representation.real_valued import RealValuedRepresentation
            return RealValuedRepresentation(self.instance)
        else:
            raise ValueError(f"Unknown representation: {self.config.representation}")
    
    def _get_selection(self):
        """Get selection method"""
        from ..selection.selection_methods import get_selection_method
        return get_selection_method(self.config.selection_method)
    
    def _get_crossover(self):
        """Get crossover operator"""
        if self.config.representation == "permutation":
            from ..crossover.permutation_crossover import get_permutation_crossover
            return get_permutation_crossover(self.config.crossover_method)
        elif self.config.representation == "binary":
            from ..crossover.binary_crossover import get_binary_crossover
            return get_binary_crossover(self.config.crossover_method)
        elif self.config.representation == "real_valued":
            from ..crossover.real_crossover import get_real_crossover
            return get_real_crossover(self.config.crossover_method)
        else:
            raise ValueError(f"Unknown representation: {self.config.representation}")
    
    def _get_mutation(self):
        """Get mutation operator"""
        if self.config.representation == "permutation":
            from ..mutation.permutation_mutation import get_permutation_mutation
            return get_permutation_mutation(self.config.mutation_method)
        elif self.config.representation == "binary":
            from ..mutation.binary_mutation import get_binary_mutation
            return get_binary_mutation(self.config.mutation_method)
        elif self.config.representation == "real_valued":
            from ..mutation.real_mutation import get_real_mutation
            return get_real_mutation(self.config.mutation_method)
        else:
            raise ValueError(f"Unknown representation: {self.config.representation}")
    
    def initialize_population(self) -> List[np.ndarray]:
        """Initialize random population"""
        population = []
        for _ in range(self.config.population_size):
            chromosome = self.representation.create_chromosome()
            population.append(chromosome)
        return population
    
    def evaluate_population(self, population: List[np.ndarray]) -> np.ndarray:
        """Evaluate fitness of entire population"""
        fitness = np.zeros(len(population))
        for i, chromosome in enumerate(population):
            fitness[i] = self.representation.calculate_fitness(
                chromosome, self.dist_matrix
            )
        return fitness
    
    def calculate_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 0.0
        
        # For permutation: use hamming distance
        if self.config.representation == "permutation":
            total_diff = 0
            comparisons = 0
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    diff = np.sum(population[i] != population[j])
                    total_diff += diff
                    comparisons += 1
            return total_diff / comparisons if comparisons > 0 else 0.0
        else:
            # For binary/real: use mean pairwise distance
            total_diff = 0
            comparisons = 0
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    diff = np.linalg.norm(population[i] - population[j])
                    total_diff += diff
                    comparisons += 1
            return total_diff / comparisons if comparisons > 0 else 0.0
    
    def apply_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply crossover operator"""
        if self.config.representation == "permutation" and self.config.crossover_method == "scx":
            # SCX needs distance matrix
            return self.crossover.crossover(
                parent1, parent2, 
                self.config.crossover_rate,
                dist_matrix=self.dist_matrix
            )
        elif self.config.representation == "real_valued" and self.config.crossover_method == "sbx":
            return self.crossover.crossover(
                parent1, parent2,
                self.config.crossover_rate,
                eta=20.0
            )
        elif self.config.representation == "real_valued" and self.config.crossover_method == "blx_alpha":
            return self.crossover.crossover(
                parent1, parent2,
                self.config.crossover_rate,
                alpha=0.5
            )
        else:
            return self.crossover.crossover(
                parent1, parent2,
                self.config.crossover_rate
            )
    
    def apply_mutation(self, chromosome: np.ndarray, generation: int = 0) -> np.ndarray:
        """Apply mutation operator"""
        if self.config.representation == "real_valued" and self.config.mutation_method == "non_uniform":
            return self.mutation.mutate(
                chromosome,
                self.config.mutation_rate,
                current_gen=generation,
                max_gen=self.config.max_generations
            )
        elif self.config.representation == "real_valued" and self.config.mutation_method == "gaussian":
            return self.mutation.mutate(
                chromosome,
                self.config.mutation_rate,
                sigma=0.1
            )
        elif self.config.representation == "real_valued" and self.config.mutation_method == "polynomial":
            return self.mutation.mutate(
                chromosome,
                self.config.mutation_rate,
                eta=20.0
            )
        else:
            return self.mutation.mutate(chromosome, self.config.mutation_rate)
    
    def run(self) -> GAResult:
        """Run genetic algorithm"""
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        # Track best
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_chromosome = population[best_idx].copy()
        convergence_gen = 0
        no_improvement = 0
        
        # Main loop
        for generation in range(self.config.max_generations):
            # Elitism: preserve best individuals
            elite_size = int(self.config.population_size * self.config.elitism_rate)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite = [population[i].copy() for i in elite_indices]
            
            # Create new population
            new_population = elite.copy()
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Selection
                parent_indices = self.selection.select(
                    population, fitness, num_parents=2,
                    tournament_size=self.config.tournament_size
                )
                parent1 = population[parent_indices[0]]
                parent2 = population[parent_indices[1]]
                
                # Crossover
                child1, child2 = self.apply_crossover(parent1, parent2)
                
                # Mutation
                child1 = self.apply_mutation(child1, generation)
                child2 = self.apply_mutation(child2, generation)
                
                # Repair if needed (for permutation)
                if self.config.representation == "permutation":
                    from ..representation.permutation import PermutationRepresentation
                    perm_rep = PermutationRepresentation(self.instance)
                    child1 = perm_rep.repair(child1)
                    child2 = perm_rep.repair(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            population = new_population[:self.config.population_size]
            fitness = self.evaluate_population(population)
            
            # Update best
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[current_best_idx].copy()
                convergence_gen = generation
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Track history
            self.fitness_history.append(best_fitness)
            diversity = self.calculate_diversity(population)
            self.diversity_history.append(diversity)
            
            # Early stopping disabled - always run to max_generations
            # (Removed to ensure consistent comparison across all methods)
            
            # Progress output
            if self.config.verbose and (generation + 1) % 50 == 0:
                print(f"Generation {generation + 1}/{self.config.max_generations}: "
                      f"Best Fitness = {best_fitness:.2f}, Diversity = {diversity:.2f}")
        
        # Decode best solution to routes
        if self.config.representation == "permutation":
            best_routes = self.representation.decode_to_routes(best_chromosome)
        else:
            # For binary/real, decode to permutation first
            if self.config.representation == "binary":
                perm_chromosome = self.representation.decode_to_permutation(best_chromosome)
            else:
                perm_chromosome = self.representation.decode_to_permutation(best_chromosome)
            
            from ..representation.permutation import PermutationRepresentation
            perm_rep = PermutationRepresentation(self.instance)
            best_routes = perm_rep.decode_to_routes(perm_chromosome)
        
        runtime = time.time() - start_time
        
        return GAResult(
            best_fitness=best_fitness,
            best_chromosome=best_chromosome,
            best_routes=best_routes,
            fitness_history=self.fitness_history.copy(),
            diversity_history=self.diversity_history.copy(),
            runtime=runtime,
            convergence_generation=convergence_gen,
            final_population=population,
            final_fitness=fitness
        )

