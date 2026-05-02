import numpy as np
import random
from typing import List, Tuple, Callable, Optional

class MathematicalAGA:
    """
    Implementation of Adaptive Genetic Algorithm (AGA) based on mathematical functions.
    This implementation follows the Srinivas and Patnaik (1994) approach 
    adapted for minimization problems (e.g., VRP, TSP).
    
    Attributes:
        pc_max, pc_min: Range for adaptive Crossover Probability.
        pm_max, pm_min: Range for adaptive Mutation Probability.
        selection_op: Method used for parent selection (e.g., Tournament, Roulette).
        crossover_op: Method used for crossover (e.g., PMX, OX, SCX).
        mutation_op: Method used for mutation (e.g., Swap, Inversion).
    """
    
    def __init__(self, 
                 pc_max: float = 0.9, 
                 pc_min: float = 0.6, 
                 pm_max: float = 0.3, 
                 pm_min: float = 0.1,
                 selection_op: Optional[Callable] = None,
                 crossover_op: Optional[Callable] = None,
                 mutation_op: Optional[Callable] = None):
        
        # Adaptive Parameter Ranges
        self.pc_max = pc_max
        self.pc_min = pc_min
        self.pm_max = pm_max
        self.pm_min = pm_min
        
        # Genetic Operators (Configurable as Parameters)
        self.selection_op = selection_op
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op

    def calculate_pc(self, f_prime: float, f_avg: float, f_min: float) -> float:
        """Calculates dynamic Crossover Probability (Pc) for minimization."""
        if f_prime < f_avg:
            if f_avg != f_min:
                pc = self.pc_min + (self.pc_max - self.pc_min) * ((f_prime - f_min) / (f_avg - f_min))
            else:
                pc = self.pc_min
        else:
            pc = self.pc_max
        return max(self.pc_min, min(self.pc_max, pc))

    def calculate_pm(self, f: float, f_avg: float, f_min: float) -> float:
        """Calculates dynamic Mutation Probability (Pm) for minimization."""
        if f < f_avg:
            if f_avg != f_min:
                pm = self.pm_min + (self.pm_max - self.pm_min) * ((f - f_min) / (f_avg - f_min))
            else:
                pm = self.pm_min
        else:
            pm = self.pm_max
        return max(self.pm_min, min(self.pm_max, pm))

    def evolve(self, population: List, fitness_values: np.ndarray) -> List:
        """
        Evolution strategy using mathematical adaptation and configurable operators.
        """
        # 1. Population Statistics
        f_avg = np.mean(fitness_values)
        f_min = np.min(fitness_values)
        new_population = []
        
        while len(new_population) < len(population):
            # 2. Selection (as a parameter)
            if self.selection_op:
                indices = self.selection_op(population, fitness_values, num=2)
                p1, p2 = population[indices[0]], population[indices[1]]
                f1, f2 = fitness_values[indices[0]], fitness_values[indices[1]]
            else:
                # Default logic if no operator provided
                p1, p2 = random.sample(population, 2)
                f1, f2 = 0, 0 # Dummy values
            
            # 3. Dynamic Parameter Calculation
            f_prime = min(f1, f2) # Better parent fitness
            pc = self.calculate_pc(f_prime, f_avg, f_min)
            
            # 4. Crossover (as a parameter)
            if self.crossover_op:
                c1, c2 = self.crossover_op(p1, p2, pc)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # 5. Dynamic Mutation Calculation
            pm1 = self.calculate_pm(f1, f_avg, f_min)
            pm2 = self.calculate_pm(f2, f_avg, f_min)
            
            # 6. Mutation (as a parameter)
            if self.mutation_op:
                c1 = self.mutation_op(c1, pm1)
                c2 = self.mutation_op(c2, pm2)
            
            new_population.extend([c1, c2])
            
        return new_population[:len(population)]

# --- Contoh Implementasi Modular untuk Dokumen Tesis ---
def tournament_selection(pop, fits, num=2):
    """Example Selection Operator"""
    return np.random.choice(len(pop), size=num, replace=False).tolist()

def scx_crossover(p1, p2, pc):
    """Example Crossover Operator (SCX)"""
    if random.random() < pc:
        # Implementation of crossover logic...
        return p1.copy(), p2.copy()
    return p1.copy(), p2.copy()

def inversion_mutation(chrom, pm):
    """Example Mutation Operator"""
    if random.random() < pm:
        # Implementation of mutation logic...
        return chrom.copy()
    return chrom

if __name__ == "__main__":
    # Inisialisasi AGA dengan operator sebagai parameter
    aga = MathematicalAGA(
        pc_max=0.9, pc_min=0.6, 
        pm_max=0.3, pm_min=0.1,
        selection_op=tournament_selection,
        crossover_op=scx_crossover,
        mutation_op=inversion_mutation
    )
    
    print("AGA initialized with modular operators as parameters.")
    print(f"Selection: {aga.selection_op.__name__}")
    print(f"Crossover: {aga.crossover_op.__name__}")
    print(f"Mutation:  {aga.mutation_op.__name__}")
