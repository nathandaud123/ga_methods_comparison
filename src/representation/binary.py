"""
Binary Representation
Encodes solution as binary string
"""

import numpy as np
from typing import List
from ..data.solomon_parser import VRPInstance


class BinaryRepresentation:
    """Binary chromosome representation"""
    
    def __init__(self, instance: VRPInstance, bits_per_customer: int = 8):
        self.instance = instance
        self.num_customers = instance.num_customers
        self.bits_per_customer = bits_per_customer
        self.chromosome_length = instance.num_customers * bits_per_customer
    
    def create_chromosome(self) -> np.ndarray:
        """
        Create random binary chromosome
        """
        return np.random.randint(0, 2, size=self.chromosome_length)
    
    def decode_to_permutation(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Decode binary chromosome to permutation
        Uses binary values as priorities for customer ordering
        
        Args:
            chromosome: Binary chromosome
            
        Returns:
            Permutation of customer IDs
        """
        priorities = []
        for i in range(self.num_customers):
            start_idx = i * self.bits_per_customer
            end_idx = start_idx + self.bits_per_customer
            binary_segment = chromosome[start_idx:end_idx]
            
            # Convert binary to integer priority
            priority = int(''.join(map(str, binary_segment)), 2)
            priorities.append((priority, i + 1))
        
        # Sort by priority to get permutation
        priorities.sort()
        permutation = np.array([customer_id for _, customer_id in priorities])
        
        return permutation
    
    def calculate_fitness(self, chromosome: np.ndarray, 
                         dist_matrix: np.ndarray) -> float:
        """
        Calculate fitness of binary chromosome
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

