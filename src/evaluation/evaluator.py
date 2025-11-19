"""
Evaluator for running multiple experiments and collecting results
Supports both sequential and parallel execution
"""

import numpy as np
import pandas as pd
import os
import sys
import time
from typing import List, Dict, Tuple, Optional
from ..ga.genetic_algorithm import GeneticAlgorithm, GAConfig, GAResult
from ..data.solomon_parser import VRPInstance
from .metrics import ExperimentMetrics, ComparisonMetrics, MetricsCalculator
from .checkpoint import CheckpointManager
from .parallel_executor import ParallelExperimentExecutor


class ExperimentEvaluator:
    """Run multiple experiments and evaluate performance"""
    
    def __init__(self, instance: VRPInstance, n_runs: int = 10, save_history: bool = True, 
                 history_dir: Optional[str] = None, checkpoint_manager: Optional[CheckpointManager] = None,
                 parallel: bool = False, n_jobs: Optional[int] = None):
        self.instance = instance
        self.n_runs = n_runs
        self.metrics_calc = MetricsCalculator()
        self.save_history = save_history
        self.history_dir = history_dir or "results/convergence"
        self.checkpoint_manager = checkpoint_manager
        self.parallel = parallel
        self.n_jobs = n_jobs
        
        if self.save_history:
            os.makedirs(self.history_dir, exist_ok=True)
        
        # Initialize parallel executor if needed
        if self.parallel:
            checkpoint_file = getattr(checkpoint_manager, 'checkpoint_file', 'results/checkpoint.json') if checkpoint_manager else 'results/checkpoint.json'
            self.parallel_executor = ParallelExperimentExecutor(
                instance, n_runs, n_jobs=n_jobs, checkpoint_file=checkpoint_file
            )
        else:
            self.parallel_executor = None
    
    def run_experiment(self, config: GAConfig, method_name: str = "") -> List[ExperimentMetrics]:
        """
        Run multiple independent runs of GA with given configuration
        Supports checkpoint/resume functionality and parallel execution
        
        Args:
            config: GA configuration
            method_name: Name of the method (for saving history)
            
        Returns:
            List of experiment metrics for each run
        """
        if self.parallel and self.parallel_executor:
            print(f"  Using parallel execution mode...")
            sys.stdout.flush()
            return self._run_experiment_parallel(config, method_name)
        else:
            print(f"  Using sequential execution mode ({self.n_runs} runs)...")
            sys.stdout.flush()
            return self._run_experiment_sequential(config, method_name)
    
    def _run_experiment_parallel(self, config: GAConfig, method_name: str) -> List[ExperimentMetrics]:
        """Run experiments in parallel"""
        results = self.parallel_executor.run_method_parallel(config, method_name)
        
        # Note: Convergence history saving for parallel mode would need to be handled differently
        # For now, we skip it in parallel mode (can be added later if needed)
        
        return results
    
    def _run_experiment_sequential(self, config: GAConfig, method_name: str) -> List[ExperimentMetrics]:
        """
        Run multiple independent runs sequentially (original implementation)
        """
        # Check checkpoint
        instance_name = self.instance.name
        completed_runs = set()
        
        if self.checkpoint_manager:
            # Check if method is already complete
            if self.checkpoint_manager.is_method_complete(instance_name, method_name, self.n_runs):
                print(f"  Method {method_name} already completed (checkpoint)")
                if instance_name in self.checkpoint_manager.state['partial_progress']:
                    method_progress = self.checkpoint_manager.state['partial_progress'][instance_name].get(method_name, {})
                    if 'status' in method_progress and method_progress['status'] == 'complete':
                        print(f"  Skipping {method_name} (already complete)")
                        return []
            
            # Get completed runs
            completed_runs = self.checkpoint_manager.get_completed_runs(instance_name, method_name)
            if completed_runs:
                print(f"  Resuming {method_name}: {len(completed_runs)}/{self.n_runs} runs already completed")
        
        results = []
        all_fitness_history = []
        all_diversity_history = []
        
        for run in range(self.n_runs):
            run_key = run + 1  # 1-indexed for checkpoint
            
            # Skip if already completed
            if run_key in completed_runs:
                print(f"  Run {run_key}/{self.n_runs} (already completed, skipping)")
                continue
            
            print(f"Run {run_key}/{self.n_runs}...")
            sys.stdout.flush()
            
            try:
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
                    solution_quality=0.0,
                    num_routes=len(result.best_routes),
                    total_distance=result.best_fitness
                )
                
                results.append(metrics)
                
                # Save checkpoint after each run
                if self.checkpoint_manager:
                    checkpoint_data = {
                        'fitness': float(metrics.fitness),
                        'runtime': float(metrics.runtime),
                        'convergence_generation': int(metrics.convergence_generation),
                        'diversity': float(metrics.diversity),
                    }
                    self.checkpoint_manager.mark_run_complete(
                        instance_name, method_name, run_key, checkpoint_data
                    )
                
            except Exception as e:
                print(f"  Error in run {run_key}: {e}")
                continue
        
        total_completed = len(results) + len(completed_runs)
        
        # Save convergence history if we have all runs
        if self.save_history and method_name and all_fitness_history and total_completed >= self.n_runs:
            self._save_convergence_history(method_name, all_fitness_history, all_diversity_history)
        
        # Mark method as complete if all runs done
        if self.checkpoint_manager and total_completed >= self.n_runs:
            self.checkpoint_manager.mark_method_complete(instance_name, method_name)
            if len(results) == 0 and len(completed_runs) >= self.n_runs:
                return []
        
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
        Skips methods that are already complete in checkpoint
        
        Args:
            configs: List of (method_name, config) tuples
            
        Returns:
            Dictionary mapping method names to aggregated metrics
        """
        comparison_results = {}
        instance_name = self.instance.name
        
        for method_name, config in configs:
            # Check if method is already complete
            if self.checkpoint_manager and \
               self.checkpoint_manager.is_method_complete(instance_name, method_name, self.n_runs):
                print(f"\n{'='*60}")
                print(f"Skipping {method_name} (already completed - {self.n_runs}/{self.n_runs} runs)")
                print(f"{'='*60}")
                # Try to load from saved results JSON (will be aggregated in summary)
                # Method is complete so we don't need to re-run
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating method: {method_name}")
            print(f"{'='*60}")
            sys.stdout.flush()
            
            experiment_results = self.run_experiment(config, method_name=method_name)
            
            # Skip if no results AND method is marked complete (all runs were cached)
            if not experiment_results:
                if self.checkpoint_manager and \
                   self.checkpoint_manager.is_method_complete(instance_name, method_name, self.n_runs):
                    # Method complete, results should be in saved JSON file
                    continue
                else:
                    # Method incomplete but no results - might be error, skip for now
                    print(f"  Warning: No results for {method_name}, skipping")
                    continue
            
            aggregated = self.metrics_calc.aggregate_metrics(experiment_results)
            aggregated.method_name = method_name
            
            comparison_results[method_name] = aggregated
            
            print(f"\nResults for {method_name}:")
            print(f"  Mean Fitness: {aggregated.mean_fitness:.2f} ± {aggregated.std_fitness:.2f}")
            print(f"  Mean Runtime: {aggregated.mean_runtime:.2f}s ± {aggregated.std_runtime:.2f}s")
            print(f"  Best Fitness: {aggregated.best_fitness:.2f}")
        
        return comparison_results
    
    def cleanup(self):
        """Cleanup resources (especially parallel executor)"""
        if self.parallel_executor:
            self.parallel_executor.cleanup()

