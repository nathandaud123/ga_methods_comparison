"""
Binary Crossover Operators
"""

import numpy as np
from typing import Tuple


class BinaryCrossover:
    """Base class for binary crossover"""
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SinglePointCrossover(BinaryCrossover):
    """
    Single-point Crossover
    Holland (1975)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return child1, child2


class TwoPointCrossover(BinaryCrossover):
    """
    Two-point Crossover
    Goldberg (1989)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        points = sorted(np.random.choice(len(parent1), size=2, replace=False))
        child1 = np.concatenate([
            parent1[:points[0]],
            parent2[points[0]:points[1]],
            parent1[points[1]:]
        ])
        child2 = np.concatenate([
            parent2[:points[0]],
            parent1[points[0]:points[1]],
            parent2[points[1]:]
        ])
        
        return child1, child2


class MultiPointCrossover(BinaryCrossover):
    """
    Multi-point Crossover
    De Jong (1975)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8, num_points: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        points = sorted(np.random.choice(len(parent1), size=num_points, replace=False))
        points = [0] + list(points) + [len(parent1)]
        
        child1 = np.array([], dtype=parent1.dtype)
        child2 = np.array([], dtype=parent2.dtype)
        
        for i in range(len(points) - 1):
            start, end = points[i], points[i+1]
            if i % 2 == 0:
                child1 = np.concatenate([child1, parent1[start:end]])
                child2 = np.concatenate([child2, parent2[start:end]])
            else:
                child1 = np.concatenate([child1, parent2[start:end]])
                child2 = np.concatenate([child2, parent1[start:end]])
        
        return child1, child2


class UniformCrossover(BinaryCrossover):
    """
    Uniform Crossover
    Syswerda (1989)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2


class ShuffleCrossover(BinaryCrossover):
    """
    Shuffle Crossover
    Eshelman et al. (1989)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Shuffle indices
        indices = np.arange(len(parent1))
        np.random.shuffle(indices)
        
        # Apply single-point crossover on shuffled
        point = np.random.randint(1, len(parent1))
        shuffled1 = parent1[indices]
        shuffled2 = parent2[indices]
        
        child1_shuffled = np.concatenate([shuffled1[:point], shuffled2[point:]])
        child2_shuffled = np.concatenate([shuffled2[:point], shuffled1[point:]])
        
        # Unshuffle
        reverse_indices = np.argsort(indices)
        child1 = child1_shuffled[reverse_indices]
        child2 = child2_shuffled[reverse_indices]
        
        return child1, child2


class ArithmeticCrossover(BinaryCrossover):
    """
    Arithmetic Crossover
    Michalewicz (1996)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Convert to float, perform arithmetic, convert back to binary
        p1_float = parent1.astype(float)
        p2_float = parent2.astype(float)
        
        child1_float = alpha * p1_float + (1 - alpha) * p2_float
        child2_float = alpha * p2_float + (1 - alpha) * p1_float
        
        # Convert back to binary (threshold at 0.5)
        child1 = (child1_float >= 0.5).astype(int)
        child2 = (child2_float >= 0.5).astype(int)
        
        return child1, child2


# Registry
BINARY_CROSSOVER_METHODS = {
    'single_point': SinglePointCrossover,
    'two_point': TwoPointCrossover,
    'multi_point': MultiPointCrossover,
    'uniform': UniformCrossover,
    'shuffle': ShuffleCrossover,
    'arithmetic': ArithmeticCrossover,
}


def get_binary_crossover(name: str):
    """Get binary crossover method by name"""
    if name not in BINARY_CROSSOVER_METHODS:
        raise ValueError(f"Unknown binary crossover method: {name}")
    return BINARY_CROSSOVER_METHODS[name]

