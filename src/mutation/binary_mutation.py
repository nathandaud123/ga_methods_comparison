"""
Binary Mutation Operators
"""

import numpy as np


class BinaryMutation:
    """Base class for binary mutation"""
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        raise NotImplementedError


class BitFlipMutation(BinaryMutation):
    """
    Bit Flip Mutation
    Holland (1975)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        mutated = chromosome.copy()
        mask = np.random.random(len(chromosome)) < mutation_rate
        mutated[mask] = 1 - mutated[mask]
        return mutated


class UniformMutation(BinaryMutation):
    """
    Uniform Mutation
    Mühlenbein (1992)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        mutated = chromosome.copy()
        mask = np.random.random(len(chromosome)) < mutation_rate
        mutated[mask] = np.random.randint(0, 2, size=np.sum(mask))
        return mutated


class InterchangingMutation(BinaryMutation):
    """
    Interchanging Mutation
    Back (1993)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        indices = np.random.choice(len(chromosome), size=2, replace=False)
        mutated[indices[0]], mutated[indices[1]] = mutated[indices[1]], mutated[indices[0]]
        return mutated


class ReversingMutation(BinaryMutation):
    """
    Reversing Mutation
    Banzhaf (1990)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        start, end = sorted(np.random.choice(len(chromosome), size=2, replace=False))
        mutated[start:end+1] = mutated[start:end+1][::-1]
        return mutated


# Registry
BINARY_MUTATION_METHODS = {
    'bit_flip': BitFlipMutation,
    'uniform': UniformMutation,
    'interchanging': InterchangingMutation,
    'reversing': ReversingMutation,
}


def get_binary_mutation(name: str):
    """Get binary mutation method by name"""
    if name not in BINARY_MUTATION_METHODS:
        raise ValueError(f"Unknown binary mutation method: {name}")
    return BINARY_MUTATION_METHODS[name]

