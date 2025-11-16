"""
Real-valued Representation
Encodes solution as continuous values
"""

import numpy as np
from typing import List
from ..data.solomon_parser import VRPInstance


class RealValuedRepresentation:
    """Real-valued chromosome representation"""
    
    def __init__(self, instance: VRPInstance):
        self.instance = instance
        self.num_customers = instance.num_customers
    
    def create_chromosome(self) -> np.ndarray:
        """
        Create random real-valued chromosome
        Values represent priorities/weights for customer ordering
        """
        return np.random.uniform(0, 1, size=self.num_customers)
    
    def decode_to_permutation(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Decode real-valued chromosome to permutation
        Uses real values as priorities for customer ordering
        
        Args:
            chromosome: Real-valued chromosome
            
        Returns:
            Permutation of customer IDs
        """
        priorities = list(enumerate(chromosome, start=1))
        priorities.sort(key=lambda x: x[1])
        permutation = np.array([customer_id for customer_id, _ in priorities])
        
        return permutation
    
    def calculate_fitness(self, chromosome: np.ndarray, 
                         dist_matrix: np.ndarray) -> float:
        """
        Calculate fitness of real-valued chromosome
        """
        permutation = self.decode_to_permutation(chromosome)
        # Use permutation representation's decode method
        from .permutation import PermutationRepresentation
        perm_rep = PermutationRepresentation(self.instance)
        routes = perm_rep.decode_to_routes(permutation)
        
        total_distance = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total_distance += dist_matrix[route[i]][route[i+1]]
        
        return total_distance

