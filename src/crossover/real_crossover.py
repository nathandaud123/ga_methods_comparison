"""
Real-valued Crossover Operators
"""

import numpy as np
from typing import Tuple
import math


class RealCrossover:
    """Base class for real-valued crossover"""
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SimulatedBinaryCrossover(RealCrossover):
    """
    Simulated Binary Crossover (SBX)
    Deb & Agrawal (1995)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8, eta: float = 20.0, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    # Calculate beta
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    # Generate children
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    # Clip to [0, 1] range
                    c1 = np.clip(c1, 0, 1)
                    c2 = np.clip(c2, 0, 1)
                    
                    child1[i] = c1
                    child2[i] = c2
        
        return child1, child2


class BlendCrossover(RealCrossover):
    """
    Blend Crossover (BLX-α)
    Eshelman & Schaffer (1993)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8, alpha: float = 0.5, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if parent1[i] < parent2[i]:
                low, high = parent1[i], parent2[i]
            else:
                low, high = parent2[i], parent1[i]
            
            range_val = high - low
            lower_bound = low - alpha * range_val
            upper_bound = high + alpha * range_val
            
            # Clip to [0, 1]
            lower_bound = max(0, lower_bound)
            upper_bound = min(1, upper_bound)
            
            child1[i] = np.random.uniform(lower_bound, upper_bound)
            child2[i] = np.random.uniform(lower_bound, upper_bound)
        
        return child1, child2


class FlatCrossover(RealCrossover):
    """
    Flat Crossover
    Wright (1991)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            low = min(parent1[i], parent2[i])
            high = max(parent1[i], parent2[i])
            
            child1[i] = np.random.uniform(low, high)
            child2[i] = np.random.uniform(low, high)
        
        return child1, child2


# Registry
REAL_CROSSOVER_METHODS = {
    'sbx': SimulatedBinaryCrossover,
    'blx_alpha': BlendCrossover,
    'flat': FlatCrossover,
}


def get_real_crossover(name: str):
    """Get real-valued crossover method by name"""
    if name not in REAL_CROSSOVER_METHODS:
        raise ValueError(f"Unknown real crossover method: {name}")
    return REAL_CROSSOVER_METHODS[name]

