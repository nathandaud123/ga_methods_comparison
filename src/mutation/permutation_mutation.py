"""
Permutation Mutation Operators
"""

import numpy as np


class PermutationMutation:
    """Base class for permutation mutation"""
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        raise NotImplementedError


class SwapMutation(PermutationMutation):
    """
    Swap Mutation
    Banzhaf (1990)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        indices = np.random.choice(len(chromosome), size=2, replace=False)
        mutated[indices[0]], mutated[indices[1]] = mutated[indices[1]], mutated[indices[0]]
        return mutated


class InsertMutation(PermutationMutation):
    """
    Insert Mutation
    Fogel (1990)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        indices = np.random.choice(len(chromosome), size=2, replace=False)
        from_idx, to_idx = indices[0], indices[1]
        
        value = mutated[from_idx]
        if from_idx < to_idx:
            mutated[from_idx:to_idx] = mutated[from_idx+1:to_idx+1]
            mutated[to_idx] = value
        else:
            mutated[to_idx+1:from_idx+1] = mutated[to_idx:from_idx]
            mutated[to_idx] = value
        
        return mutated


class InversionMutation(PermutationMutation):
    """
    Inversion Mutation (2-opt like)
    Holland (1975)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        start, end = sorted(np.random.choice(len(chromosome), size=2, replace=False))
        mutated[start:end+1] = mutated[start:end+1][::-1]
        return mutated


class ScrambleMutation(PermutationMutation):
    """
    Scramble Mutation
    Syswerda (1991)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        start, end = sorted(np.random.choice(len(chromosome), size=2, replace=False))
        segment = mutated[start:end+1].copy()
        np.random.shuffle(segment)
        mutated[start:end+1] = segment
        return mutated


class DisplacementMutation(PermutationMutation):
    """
    Displacement Mutation
    Michalewicz (1992)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        n = len(chromosome)
        
        # Select segment to move
        start1, end1 = sorted(np.random.choice(n, size=2, replace=False))
        segment_length = end1 - start1 + 1
        
        # Select insertion point
        valid_positions = [i for i in range(n) if i < start1 or i > end1]
        if not valid_positions:
            return mutated
        
        insert_pos = np.random.choice(valid_positions)
        
        # Extract segment
        segment = mutated[start1:end1+1].copy()
        
        # Remove segment
        remaining = np.concatenate([mutated[:start1], mutated[end1+1:]])
        
        # Insert segment at new position
        if insert_pos < start1:
            new_pos = insert_pos
        else:
            new_pos = insert_pos - segment_length
        
        mutated = np.concatenate([
            remaining[:new_pos],
            segment,
            remaining[new_pos:]
        ])
        
        return mutated


class ExchangeMutation(PermutationMutation):
    """
    Exchange Mutation
    Deep & Mebrahtu (2011)
    """
    
    @staticmethod
    def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        # Find worst position (for VRP, this could be based on distance)
        # For simplicity, use random exchange
        indices = np.random.choice(len(chromosome), size=2, replace=False)
        mutated[indices[0]], mutated[indices[1]] = mutated[indices[1]], mutated[indices[0]]
        
        return mutated


# Registry
PERMUTATION_MUTATION_METHODS = {
    'swap': SwapMutation,
    'insert': InsertMutation,
    'inversion': InversionMutation,
    'scramble': ScrambleMutation,
    'displacement': DisplacementMutation,
    'exchange': ExchangeMutation,
}


def get_permutation_mutation(name: str):
    """Get permutation mutation method by name"""
    if name not in PERMUTATION_MUTATION_METHODS:
        raise ValueError(f"Unknown permutation mutation method: {name}")
    return PERMUTATION_MUTATION_METHODS[name]

