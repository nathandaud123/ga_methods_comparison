"""
Evaluation Metrics for GA Performance
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run"""
    fitness: float
    runtime: float
    convergence_generation: int
    diversity: float
    solution_quality: float  # Gap from best known (if available)
    num_routes: int
    total_distance: float


@dataclass
class ComparisonMetrics:
    """Aggregated metrics for comparison"""
    method_name: str
    mean_fitness: float
    std_fitness: float
    mean_runtime: float
    std_runtime: float
    mean_convergence: float
    std_convergence: float
    mean_diversity: float
    std_diversity: float
    best_fitness: float
    worst_fitness: float
    success_rate: float  # Percentage of runs that found good solutions


class MetricsCalculator:
    """Calculate various performance metrics"""
    
    @staticmethod
    def calculate_convergence_speed(fitness_history: List[float], 
                                   threshold: float = 0.01) -> int:
        """
        Calculate generation at which convergence occurred
        Convergence: when improvement < threshold for last 10% of generations
        """
        if len(fitness_history) < 10:
            return len(fitness_history)
        
        window_size = max(10, len(fitness_history) // 10)
        recent_improvements = []
        
        for i in range(len(fitness_history) - window_size, len(fitness_history) - 1):
            improvement = abs(fitness_history[i] - fitness_history[i+1])
            recent_improvements.append(improvement)
        
        if np.mean(recent_improvements) < threshold:
            return len(fitness_history) - window_size
        
        return len(fitness_history)
    
    @staticmethod
    def calculate_diversity_metric(population: List[np.ndarray]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 0.0
        
        total_diff = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if isinstance(population[i], np.ndarray) and isinstance(population[j], np.ndarray):
                    diff = np.sum(population[i] != population[j])
                    total_diff += diff
                    comparisons += 1
        
        return total_diff / comparisons if comparisons > 0 else 0.0
    
    @staticmethod
    def aggregate_metrics(experiment_results: List[ExperimentMetrics]) -> ComparisonMetrics:
        """Aggregate metrics from multiple runs"""
        if not experiment_results:
            raise ValueError("No experiment results provided")
        
        fitnesses = [r.fitness for r in experiment_results]
        runtimes = [r.runtime for r in experiment_results]
        convergences = [r.convergence_generation for r in experiment_results]
        diversities = [r.diversity for r in experiment_results]
        
        return ComparisonMetrics(
            method_name="",
            mean_fitness=np.mean(fitnesses),
            std_fitness=np.std(fitnesses),
            mean_runtime=np.mean(runtimes),
            std_runtime=np.std(runtimes),
            mean_convergence=np.mean(convergences),
            std_convergence=np.std(convergences),
            mean_diversity=np.mean(diversities),
            std_diversity=np.std(diversities),
            best_fitness=np.min(fitnesses),
            worst_fitness=np.max(fitnesses),
            success_rate=1.0  # Can be customized based on solution quality
        )
    
    @staticmethod
    def calculate_solution_quality(obtained_fitness: float, 
                                   best_known: float = None) -> float:
        """
        Calculate solution quality as gap from best known
        Returns percentage gap
        """
        if best_known is None or best_known == 0:
            return 0.0
        
        gap = ((obtained_fitness - best_known) / best_known) * 100
        return gap

