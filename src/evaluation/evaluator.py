"""
Evaluator for running multiple experiments and collecting results
"""

import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional
from ..ga.genetic_algorithm import GeneticAlgorithm, GAConfig, GAResult
from ..data.solomon_parser import VRPInstance
from .metrics import ExperimentMetrics, ComparisonMetrics, MetricsCalculator
import time


class ExperimentEvaluator:
    """Run multiple experiments and evaluate performance"""
    
    def __init__(self, instance: VRPInstance, n_runs: int = 10, save_history: bool = True, 
                 history_dir: Optional[str] = None):
        self.instance = instance
        self.n_runs = n_runs
        self.metrics_calc = MetricsCalculator()
        self.save_history = save_history
        self.history_dir = history_dir or "results/convergence"
        if self.save_history:
            os.makedirs(self.history_dir, exist_ok=True)
    
    def run_experiment(self, config: GAConfig, method_name: str = "") -> List[ExperimentMetrics]:
        """
        Run multiple independent runs of GA with given configuration
        
        Args:
            config: GA configuration
            method_name: Name of the method (for saving history)
            
        Returns:
            List of experiment metrics for each run
        """
        results = []
        all_fitness_history = []
        all_diversity_history = []
        
        for run in range(self.n_runs):
            print(f"Run {run + 1}/{self.n_runs}...")
            
            # Run GA
            ga = GeneticAlgorithm(self.instance, config)
            result = ga.run()
            
            # Store history for CSV
            if self.save_history and method_name:
                all_fitness_history.append(result.fitness_history)
                all_diversity_history.append(result.diversity_history)
            
            # Calculate metrics
            metrics = ExperimentMetrics(
                fitness=result.best_fitness,
                runtime=result.runtime,
                convergence_generation=result.convergence_generation,
                diversity=np.mean(result.diversity_history) if result.diversity_history else 0.0,
                solution_quality=0.0,  # Can be updated with best known solution
                num_routes=len(result.best_routes),
                total_distance=result.best_fitness
            )
            
            results.append(metrics)
        
        # Save convergence history to CSV
        if self.save_history and method_name and all_fitness_history:
            self._save_convergence_history(method_name, all_fitness_history, all_diversity_history)
        
        return results
    
    def _save_convergence_history(self, method_name: str, fitness_histories: List[List[float]], 
                                  diversity_histories: List[List[float]]):
        """
        Save convergence history to CSV file
        
        Args:
            method_name: Name of the method
            fitness_histories: List of fitness history for each run
            diversity_histories: List of diversity history for each run
        """
        # Find maximum generation length
        max_gens = max(len(hist) for hist in fitness_histories)
        
        # Prepare data for DataFrame
        data = {
            'generation': list(range(1, max_gens + 1))
        }
        
        # Add fitness data for each run
        for run_idx, fitness_hist in enumerate(fitness_histories):
            # Pad with last value if shorter
            padded_hist = list(fitness_hist) + [fitness_hist[-1]] * (max_gens - len(fitness_hist))
            data[f'fitness_run_{run_idx + 1}'] = padded_hist[:max_gens]
        
        # Add diversity data for each run
        for run_idx, diversity_hist in enumerate(diversity_histories):
            # Pad with last value if shorter
            padded_hist = list(diversity_hist) + [diversity_hist[-1]] * (max_gens - len(diversity_hist))
            data[f'diversity_run_{run_idx + 1}'] = padded_hist[:max_gens]
        
        # Calculate statistics
        fitness_array = np.array([list(fh) + [fh[-1]] * (max_gens - len(fh)) 
                                  for fh in fitness_histories])[:, :max_gens]
        diversity_array = np.array([list(dh) + [dh[-1]] * (max_gens - len(dh)) 
                                    for dh in diversity_histories])[:, :max_gens]
        
        data['fitness_mean'] = np.mean(fitness_array, axis=0).tolist()
        data['fitness_std'] = np.std(fitness_array, axis=0).tolist()
        data['fitness_min'] = np.min(fitness_array, axis=0).tolist()
        data['fitness_max'] = np.max(fitness_array, axis=0).tolist()
        
        data['diversity_mean'] = np.mean(diversity_array, axis=0).tolist()
        data['diversity_std'] = np.std(diversity_array, axis=0).tolist()
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Clean method name for filename
        clean_method_name = method_name.replace('/', '_').replace('\\', '_')
        filename = f"{clean_method_name}_convergence.csv"
        filepath = os.path.join(self.history_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"  Convergence history saved to: {filepath}")
    
    def compare_methods(self, configs: List[Tuple[str, GAConfig]]) -> Dict[str, ComparisonMetrics]:
        """
        Compare multiple GA configurations
        
        Args:
            configs: List of (method_name, config) tuples
            
        Returns:
            Dictionary mapping method names to aggregated metrics
        """
        comparison_results = {}
        
        for method_name, config in configs:
            print(f"\n{'='*60}")
            print(f"Evaluating method: {method_name}")
            print(f"{'='*60}")
            
            experiment_results = self.run_experiment(config, method_name=method_name)
            aggregated = self.metrics_calc.aggregate_metrics(experiment_results)
            aggregated.method_name = method_name
            
            comparison_results[method_name] = aggregated
            
            print(f"\nResults for {method_name}:")
            print(f"  Mean Fitness: {aggregated.mean_fitness:.2f} ± {aggregated.std_fitness:.2f}")
            print(f"  Mean Runtime: {aggregated.mean_runtime:.2f}s ± {aggregated.std_runtime:.2f}s")
            print(f"  Best Fitness: {aggregated.best_fitness:.2f}")
        
        return comparison_results

