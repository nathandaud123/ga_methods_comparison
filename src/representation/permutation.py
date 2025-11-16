"""
Permutation Representation for VRP/TSP
Most suitable for order-based problems
"""

import numpy as np
from typing import List, Tuple
from ..data.solomon_parser import VRPInstance


class PermutationRepresentation:
    """Permutation-based chromosome representation"""
    
    def __init__(self, instance: VRPInstance):
        self.instance = instance
        self.num_customers = instance.num_customers
    
    def create_chromosome(self) -> np.ndarray:
        """
        Create random permutation chromosome
        Returns array of customer IDs in random order
        """
        customers = list(range(1, self.num_customers + 1))
        np.random.shuffle(customers)
        return np.array(customers)
    
    def decode_to_routes(self, chromosome: np.ndarray) -> List[List[int]]:
        """
        Decode permutation to VRP routes using greedy approach
        
        Args:
            chromosome: Permutation of customer IDs
            
        Returns:
            List of routes (each route includes depot at start/end)
        """
        routes = []
        current_route = [0]  # Start at depot
        current_demand = 0.0
        
        for customer_id in chromosome:
            customer = self.instance.customers[customer_id - 1]
            
            # Check if adding customer exceeds capacity
            if current_demand + customer.demand > self.instance.vehicle_capacity:
                # Close current route and start new one
                current_route.append(0)  # Return to depot
                routes.append(current_route)
                current_route = [0]
                current_demand = 0.0
            
            # Add customer to current route
            current_route.append(customer_id)
            current_demand += customer.demand
        
        # Close last route
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        
        return routes
    
    def calculate_fitness(self, chromosome: np.ndarray, 
                         dist_matrix: np.ndarray) -> float:
        """
        Calculate fitness (total distance) of chromosome
        
        Args:
            chromosome: Permutation chromosome
            dist_matrix: Distance matrix
            
        Returns:
            Total distance (lower is better)
        """
        routes = self.decode_to_routes(chromosome)
        total_distance = 0.0
        
        for route in routes:
            for i in range(len(route) - 1):
                total_distance += dist_matrix[route[i]][route[i+1]]
        
        return total_distance
    
    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Repair chromosome to ensure valid permutation
        (Remove duplicates, add missing customers)
        
        Args:
            chromosome: Potentially invalid chromosome
            
        Returns:
            Valid permutation
        """
        # Remove duplicates while preserving order
        seen = set()
        repaired = []
        for gene in chromosome:
            if gene not in seen and 1 <= gene <= self.num_customers:
                seen.add(gene)
                repaired.append(gene)
        
        # Add missing customers
        missing = set(range(1, self.num_customers + 1)) - seen
        repaired.extend(sorted(missing))
        
        return np.array(repaired[:self.num_customers])

