"""
Real-valued Mutation Operators
"""

import numpy as np
import math


class RealMutation:
    """Base class for real-valued mutation"""
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float, **kwargs) -> np.ndarray:
        raise NotImplementedError


class GaussianMutation(RealMutation):
    """
    Gaussian Mutation
    Schwefel (1981)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float, 
               sigma: float = 0.1, **kwargs) -> np.ndarray:
        mutated = chromosome.copy()
        mask = np.random.random(len(chromosome)) < mutation_rate
        noise = np.random.normal(0, sigma, size=len(chromosome))
        mutated[mask] = np.clip(mutated[mask] + noise[mask], 0, 1)
        return mutated


class PolynomialMutation(RealMutation):
    """
    Polynomial Mutation
    Deb & Goyal (1996)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float, 
               eta: float = 20.0, **kwargs) -> np.ndarray:
        mutated = chromosome.copy()
        mask = np.random.random(len(chromosome)) < mutation_rate
        
        for i in range(len(chromosome)):
            if mask[i]:
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                
                mutated[i] = np.clip(mutated[i] + delta, 0, 1)
        
        return mutated


class UniformMutation(RealMutation):
    """
    Uniform Mutation
    Michalewicz (1996)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float, **kwargs) -> np.ndarray:
        mutated = chromosome.copy()
        mask = np.random.random(len(chromosome)) < mutation_rate
        mutated[mask] = np.random.uniform(0, 1, size=np.sum(mask))
        return mutated


class NonUniformMutation(RealMutation):
    """
    Non-uniform Mutation
    Michalewicz (1996)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float, 
               current_gen: int = 0, max_gen: int = 100, b: float = 5.0, **kwargs) -> np.ndarray:
        mutated = chromosome.copy()
        mask = np.random.random(len(chromosome)) < mutation_rate
        
        for i in range(len(chromosome)):
            if mask[i]:
                tau = (1 - current_gen / max_gen) ** b
                u = np.random.random()
                
                if u < 0.5:
                    delta = (1 - mutated[i]) * ((1 - u) ** tau - 1)
                else:
                    delta = mutated[i] * ((1 - u) ** tau - 1)
                
                mutated[i] = np.clip(mutated[i] + delta, 0, 1)
        
        return mutated


# Registry
REAL_MUTATION_METHODS = {
    'gaussian': GaussianMutation,
    'polynomial': PolynomialMutation,
    'uniform': UniformMutation,
    'non_uniform': NonUniformMutation,
}


def get_real_mutation(name: str):
    """Get real-valued mutation method by name"""
    if name not in REAL_MUTATION_METHODS:
        raise ValueError(f"Unknown real mutation method: {name}")
    return REAL_MUTATION_METHODS[name]

