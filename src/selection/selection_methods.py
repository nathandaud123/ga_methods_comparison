"""
Selection Methods for Genetic Algorithm
Implements various selection strategies
"""

import numpy as np
from typing import List, Tuple
import math


class SelectionMethod:
    """Base class for selection methods"""
    
    def select(self, population: List, fitness_values: np.ndarray, 
               num_parents: int, **kwargs) -> List:
        """
        Select parents from population
        
        Args:
            population: List of chromosomes
            fitness_values: Array of fitness values (lower is better for minimization)
            num_parents: Number of parents to select
            **kwargs: Additional parameters
            
        Returns:
            List of selected parent indices
        """
        raise NotImplementedError


class RouletteWheelSelection(SelectionMethod):
    """
    Roulette Wheel Selection (Fitness Proportional Selection)
    Goldberg (1989)
    """
    
    def select(self, population: List, fitness_values: np.ndarray, 
               num_parents: int, **kwargs) -> List:
        # Convert to maximization problem (invert fitness)
        # For minimization: higher fitness = lower cost
        max_fitness = np.max(fitness_values)
        if max_fitness > 0:
            # Use inverse for minimization
            probabilities = (max_fitness - fitness_values + 1e-10) / \
                          (np.sum(max_fitness - fitness_values) + 1e-10)
        else:
            # Fallback: use negative fitness
            probabilities = (-fitness_values + 1e-10) / \
                          (np.sum(-fitness_values) + 1e-10)
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Select parents
        selected_indices = np.random.choice(
            len(population),
            size=num_parents,
            p=probabilities,
            replace=True
        )
        
        return selected_indices.tolist()


class TournamentSelection(SelectionMethod):
    """
    Tournament Selection
    Brindle (1981); Fang & Li (2010)
    """
    
    def select(self, population: List, fitness_values: np.ndarray, 
               num_parents: int, tournament_size: int = 3, **kwargs) -> List:
        selected_indices = []
        
        for _ in range(num_parents):
            # Randomly select tournament participants
            tournament_indices = np.random.choice(
                len(population),
                size=tournament_size,
                replace=False
            )
            
            # Select best (lowest fitness for minimization)
            tournament_fitness = fitness_values[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(winner_idx)
        
        return selected_indices


class RankSelection(SelectionMethod):
    """
    Rank Selection
    Baker (1985)
    """
    
    def select(self, population: List, fitness_values: np.ndarray, 
               num_parents: int, **kwargs) -> List:
        # Rank individuals (1 = best, N = worst)
        sorted_indices = np.argsort(fitness_values)
        ranks = np.zeros(len(population))
        
        for rank, idx in enumerate(sorted_indices, start=1):
            ranks[idx] = rank
        
        # Assign probabilities based on rank (linear ranking)
        # Better rank = higher probability
        max_rank = len(population)
        probabilities = (max_rank - ranks + 1) / np.sum(max_rank - ranks + 1)
        
        # Select parents
        selected_indices = np.random.choice(
            len(population),
            size=num_parents,
            p=probabilities,
            replace=True
        )
        
        return selected_indices.tolist()


class StochasticUniversalSampling(SelectionMethod):
    """
    Stochastic Universal Sampling (SUS)
    Baker (1987)
    """
    
    def select(self, population: List, fitness_values: np.ndarray, 
               num_parents: int, **kwargs) -> List:
        # Convert to maximization problem
        max_fitness = np.max(fitness_values)
        if max_fitness > 0:
            scaled_fitness = max_fitness - fitness_values + 1e-10
        else:
            scaled_fitness = -fitness_values + 1e-10
        
        # Calculate cumulative probabilities
        total_fitness = np.sum(scaled_fitness)
        probabilities = scaled_fitness / total_fitness
        cumulative = np.cumsum(probabilities)
        
        # SUS selection
        selected_indices = []
        pointer_distance = 1.0 / num_parents
        start = np.random.uniform(0, pointer_distance)
        
        for i in range(num_parents):
            pointer = start + i * pointer_distance
            # Find which individual this pointer points to
            idx = np.searchsorted(cumulative, pointer)
            selected_indices.append(min(idx, len(population) - 1))
        
        return selected_indices


class ElitismSelection(SelectionMethod):
    """
    Elitism Selection
    De Jong (1975)
    """
    
    def select(self, population: List, fitness_values: np.ndarray, 
               num_parents: int, elite_size: int = None, **kwargs) -> List:
        if elite_size is None:
            elite_size = max(1, num_parents // 10)
        
        # Select elite individuals (best fitness)
        elite_indices = np.argsort(fitness_values)[:elite_size]
        
        # Fill remaining with random selection
        remaining = num_parents - len(elite_indices)
        if remaining > 0:
            # Use roulette wheel for remaining
            rw = RouletteWheelSelection()
            additional = rw.select(population, fitness_values, remaining)
            selected_indices = list(elite_indices) + additional
        else:
            selected_indices = list(elite_indices[:num_parents])
        
        return selected_indices


class BoltzmannSelection(SelectionMethod):
    """
    Boltzmann Selection (Simulated Annealing-like)
    Goldberg (1990)
    """
    
    def select(self, population: List, fitness_values: np.ndarray, 
               num_parents: int, temperature: float = 1.0, **kwargs) -> List:
        # Convert fitness to probabilities using Boltzmann distribution
        # For minimization: exp(-fitness / temperature)
        scaled_fitness = -fitness_values / (temperature + 1e-10)
        exp_fitness = np.exp(scaled_fitness - np.max(scaled_fitness))  # Numerical stability
        
        probabilities = exp_fitness / np.sum(exp_fitness)
        
        # Select parents
        selected_indices = np.random.choice(
            len(population),
            size=num_parents,
            p=probabilities,
            replace=True
        )
        
        return selected_indices.tolist()


class StairwiseSelection(SelectionMethod):
    """
    Stairwise Selection (SWS)
    Umair et al. (2019)
    """
    
    def select(self, population: List, fitness_values: np.ndarray, 
               num_parents: int, **kwargs) -> List:
        # Sort by fitness
        sorted_indices = np.argsort(fitness_values)
        
        # Assign stairwise probabilities
        n = len(population)
        probabilities = np.zeros(n)
        
        for i, idx in enumerate(sorted_indices):
            # Stairwise: better individuals get higher probability
            # Probability decreases in steps
            step = (n - i) / n
            probabilities[idx] = step
        
        probabilities = probabilities / np.sum(probabilities)
        
        # Select parents
        selected_indices = np.random.choice(
            len(population),
            size=num_parents,
            p=probabilities,
            replace=True
        )
        
        return selected_indices.tolist()


# Selection method registry
SELECTION_METHODS = {
    'roulette_wheel': RouletteWheelSelection,
    'tournament': TournamentSelection,
    'rank': RankSelection,
    'stochastic_universal': StochasticUniversalSampling,
    'elitism': ElitismSelection,
    'boltzmann': BoltzmannSelection,
    'stairwise': StairwiseSelection,
}


def get_selection_method(name: str) -> SelectionMethod:
    """Get selection method by name"""
    if name not in SELECTION_METHODS:
        raise ValueError(f"Unknown selection method: {name}")
    return SELECTION_METHODS[name]()

