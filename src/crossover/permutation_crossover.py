"""
Permutation Crossover Operators for VRP/TSP
"""

import numpy as np
from typing import Tuple, List, Set


class PermutationCrossover:
    """Base class for permutation crossover"""
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class PartiallyMappedCrossover(PermutationCrossover):
    """
    Partially Mapped Crossover (PMX)
    Goldberg & Lingle (1985)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n = len(parent1)
        point1, point2 = sorted(np.random.choice(n, size=2, replace=False))
        
        def pmx_helper(p1, p2):
            child = np.zeros(n, dtype=int)
            child[point1:point2] = p1[point1:point2]
            
            # Create mapping
            mapping = {}
            for i in range(point1, point2):
                if p2[i] not in p1[point1:point2]:
                    val = p2[i]
                    while val in p1[point1:point2]:
                        idx = np.where(p1 == val)[0][0]
                        val = p2[idx]
                    mapping[p2[i]] = val
            
            # Fill remaining positions
            for i in range(n):
                if i < point1 or i >= point2:
                    if p2[i] not in child:
                        child[i] = p2[i]
                    else:
                        # Use mapping
                        val = p2[i]
                        while val in child:
                            val = mapping.get(val, val)
                            if val in child:
                                val = p2[np.where(p1 == val)[0][0]]
                        child[i] = val
            
            return child
        
        child1 = pmx_helper(parent1, parent2)
        child2 = pmx_helper(parent2, parent1)
        
        return child1, child2


class OrderCrossover(PermutationCrossover):
    """
    Order Crossover (OX)
    Davis (1985)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n = len(parent1)
        point1, point2 = sorted(np.random.choice(n, size=2, replace=False))
        
        def ox_helper(p1, p2):
            child = np.zeros(n, dtype=int)
            child[point1:point2] = p1[point1:point2]
            
            # Get remaining elements from p2 in order
            remaining = [x for x in p2 if x not in child]
            idx = 0
            for i in range(n):
                if i < point1 or i >= point2:
                    child[i] = remaining[idx]
                    idx += 1
            
            return child
        
        child1 = ox_helper(parent1, parent2)
        child2 = ox_helper(parent2, parent1)
        
        return child1, child2


class CycleCrossover(PermutationCrossover):
    """
    Cycle Crossover (CX)
    Oliver et al. (1987)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n = len(parent1)
        child1 = np.zeros(n, dtype=int)
        child2 = np.zeros(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        
        cycle = 0
        while not visited.all():
            # Find starting position
            start = np.where(~visited)[0][0]
            current = start
            
            # Build cycle
            cycle_indices = []
            while not visited[current]:
                visited[current] = True
                cycle_indices.append(current)
                value = parent1[current]
                current = np.where(parent2 == value)[0][0]
            
            # Assign cycle to children
            if cycle % 2 == 0:
                for idx in cycle_indices:
                    child1[idx] = parent1[idx]
                    child2[idx] = parent2[idx]
            else:
                for idx in cycle_indices:
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]
            
            cycle += 1
        
        return child1, child2


class OrderBasedCrossover(PermutationCrossover):
    """
    Order-Based Crossover (OBX)
    Syswerda (1991)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n = len(parent1)
        num_positions = np.random.randint(1, n // 2)
        positions = sorted(np.random.choice(n, size=num_positions, replace=False))
        
        def obx_helper(p1, p2):
            child = np.zeros(n, dtype=int)
            selected_values = p1[positions]
            
            # Fill positions with selected values
            for i, pos in enumerate(positions):
                child[pos] = selected_values[i]
            
            # Fill remaining with p2 in order
            remaining = [x for x in p2 if x not in selected_values]
            idx = 0
            for i in range(n):
                if child[i] == 0:
                    child[i] = remaining[idx]
                    idx += 1
            
            return child
        
        child1 = obx_helper(parent1, parent2)
        child2 = obx_helper(parent2, parent1)
        
        return child1, child2


class PositionBasedCrossover(PermutationCrossover):
    """
    Position-Based Crossover (POS)
    Syswerda (1991)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        n = len(parent1)
        num_positions = np.random.randint(1, n // 2)
        positions = sorted(np.random.choice(n, size=num_positions, replace=False))
        
        def pos_helper(p1, p2):
            child = np.zeros(n, dtype=int)
            selected_values = set(p1[positions])
            
            # Fill positions with selected values from p1
            for i, pos in enumerate(positions):
                child[pos] = p1[positions[i]]
            
            # Fill remaining with p2 in order
            remaining = [x for x in p2 if x not in selected_values]
            idx = 0
            for i in range(n):
                if child[i] == 0:
                    child[i] = remaining[idx]
                    idx += 1
            
            return child
        
        child1 = pos_helper(parent1, parent2)
        child2 = pos_helper(parent2, parent1)
        
        return child1, child2


class EdgeRecombinationCrossover(PermutationCrossover):
    """
    Edge Recombination Crossover (ERX)
    Whitley et al. (1989)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        def erx_helper(p1, p2):
            n = len(p1)
            child = np.zeros(n, dtype=int)
            
            # Build edge table
            edge_table = {}
            for i in range(n):
                neighbors = set()
                # From parent1
                neighbors.add(p1[(i - 1) % n])
                neighbors.add(p1[(i + 1) % n])
                # From parent2
                neighbors.add(p2[(i - 1) % n])
                neighbors.add(p2[(i + 1) % n])
                edge_table[p1[i]] = neighbors
            
            # Start with random element
            current = np.random.choice(p1)
            child[0] = current
            
            # Build child using edge table
            for i in range(1, n):
                # Remove current from all edge tables
                for key in edge_table:
                    edge_table[key].discard(current)
                
                # Choose next element
                if len(edge_table[current]) > 0:
                    # Choose neighbor with fewest remaining neighbors
                    neighbors = list(edge_table[current])
                    neighbor_counts = [len(edge_table[n]) for n in neighbors]
                    current = neighbors[np.argmin(neighbor_counts)]
                else:
                    # No neighbors left, choose random unvisited
                    remaining = [x for x in p1 if x not in child[:i]]
                    if remaining:
                        current = np.random.choice(remaining)
                    else:
                        break
                
                child[i] = current
            
            return child
        
        child1 = erx_helper(parent1, parent2)
        child2 = erx_helper(parent2, parent1)
        
        return child1, child2


class SequentialConstructiveCrossover(PermutationCrossover):
    """
    Sequential Constructive Crossover (SCX)
    Ahmed (2010)
    """
    
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float = 0.8, dist_matrix: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if dist_matrix is None:
            # Fallback to OX if no distance matrix
            return OrderCrossover.crossover(parent1, parent2, crossover_rate)
        
        def scx_helper(p1, p2):
            n = len(p1)
            child = [p1[0]]  # Start with first element of p1
            current = p1[0]
            
            p1_dict = {val: idx for idx, val in enumerate(p1)}
            p2_dict = {val: idx for idx, val in enumerate(p2)}
            
            while len(child) < n:
                # Find next candidates
                p1_idx = p1_dict[current]
                p2_idx = p2_dict[current]
                
                next_p1 = p1[(p1_idx + 1) % n] if (p1_idx + 1) % n != 0 else None
                next_p2 = p2[(p2_idx + 1) % n] if (p2_idx + 1) % n != 0 else None
                
                candidates = []
                if next_p1 and next_p1 not in child:
                    candidates.append((next_p1, dist_matrix[current][next_p1]))
                if next_p2 and next_p2 not in child:
                    candidates.append((next_p2, dist_matrix[current][next_p2]))
                
                if candidates:
                    # Choose candidate with minimum distance
                    candidates.sort(key=lambda x: x[1])
                    current = candidates[0][0]
                else:
                    # No valid candidates, choose random unvisited
                    remaining = [x for x in p1 if x not in child]
                    if remaining:
                        current = np.random.choice(remaining)
                    else:
                        break
                
                child.append(current)
            
            return np.array(child)
        
        child1 = scx_helper(parent1, parent2)
        child2 = scx_helper(parent2, parent1)
        
        return child1, child2


# Registry
PERMUTATION_CROSSOVER_METHODS = {
    'pmx': PartiallyMappedCrossover,
    'ox': OrderCrossover,
    'cx': CycleCrossover,
    'obx': OrderBasedCrossover,
    'pos': PositionBasedCrossover,
    'erx': EdgeRecombinationCrossover,
    'scx': SequentialConstructiveCrossover,
}


def get_permutation_crossover(name: str):
    """Get permutation crossover method by name"""
    if name not in PERMUTATION_CROSSOVER_METHODS:
        raise ValueError(f"Unknown permutation crossover method: {name}")
    return PERMUTATION_CROSSOVER_METHODS[name]

