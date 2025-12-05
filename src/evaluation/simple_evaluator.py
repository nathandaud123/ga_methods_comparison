"""
Simple Evaluator for GA Method Comparison
- Runs each combination 5 times
- Saves generation history for each run
- Calculates average per generation from 5 runs
- Supports checkpoint/resume and parallel execution
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
from ..ga.genetic_algorithm import GeneticAlgorithm, GAConfig, GAResult
from ..data.solomon_parser import VRPInstance
from .simple_checkpoint import SimpleCheckpoint


def _run_single_ga_worker(args: Tuple) -> Tuple[int, float, List[float], List[float]]:
    """
    Worker function for parallel execution
    Returns: (run_num, best_fitness, fitness_history, diversity_history)
    """
    instance_data, config_dict, method_name, run_num, seed = args
    
    try:
        # Reconstruct instance
        from ..data.solomon_parser import VRPInstance, Customer
        
        depot_data = instance_data['depot']
        depot = Customer(
            id=depot_data['id'],
            x=depot_data['x'],
            y=depot_data['y'],
            demand=depot_data['demand'],
            ready_time=depot_data.get('ready_time', 0.0),
            due_time=depot_data.get('due_time', 0.0),
            service_time=depot_data.get('service_time', 0.0),
        )
        
        customers = [
            Customer(
                id=c['id'],
                x=c['x'],
                y=c['y'],
                demand=c['demand'],
                ready_time=c.get('ready_time', 0.0),
                due_time=c.get('due_time', 0.0),
                service_time=c.get('service_time', 0.0),
            )
            for c in instance_data['customers']
        ]
        
        instance = VRPInstance(
            name=instance_data['name'],
            num_customers=instance_data['num_customers'],
            depot=depot,
            customers=customers,
            vehicle_capacity=instance_data['vehicle_capacity'],
            type=instance_data.get('type', 'Unknown')
        )
        
        # Reconstruct config
        from ..ga.genetic_algorithm import GAConfig
        config = GAConfig(**config_dict)
        
        # Set seed
        np.random.seed(seed)
        
        # Run GA
        ga = GeneticAlgorithm(instance, config)
        result = ga.run()
        
        return (run_num, result.best_fitness, result.fitness_history, result.diversity_history)
    except Exception as e:
        print(f"Error in run {run_num}: {e}")
        import traceback
        traceback.print_exc()
        return (run_num, float('inf'), [], [])


class SimpleEvaluator:
    """Simple evaluator that runs each combination 5 times and saves generation averages"""
    
    def __init__(self, instance: VRPInstance, n_runs: int = 5, 
                 output_dir: str = "results", checkpoint: Optional[SimpleCheckpoint] = None,
                 n_jobs: Optional[int] = None):
        self.instance = instance
        self.n_runs = n_runs
        self.output_dir = output_dir
        self.instance_dir = os.path.join(output_dir, instance.name)
        os.makedirs(self.instance_dir, exist_ok=True)
        self.checkpoint = checkpoint
        self.n_jobs = n_jobs  # None = sequential, 1 = sequential, >1 = parallel
    
    def evaluate_combination(self, method_name: str, config: GAConfig) -> Optional[Dict]:
        """
        Evaluate a single combination by running it n_runs times
        
        Args:
            method_name: Name of the method combination
            config: GA configuration
            
        Returns:
            Dictionary with results, or None if already completed
        """
        # Check checkpoint
        if self.checkpoint and self.checkpoint.is_method_complete(self.instance.name, method_name):
            print(f"  Skipping {method_name} (already completed)")
            return None
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        start_time = time.time()
        
        # Store fitness and diversity history for each run
        all_fitness_histories = []
        all_diversity_histories = []
        final_fitnesses = []
        
        # Check if parallel execution
        use_parallel = self.n_jobs is not None and self.n_jobs > 1
        
        if use_parallel:
            # Parallel execution
            print(f"  Running {self.n_runs} runs in parallel ({self.n_jobs} workers)...")
            sys.stdout.flush()
            
            # Prepare instance data for serialization
            instance_data = {
                'name': self.instance.name,
                'num_customers': self.instance.num_customers,
                'depot': {
                    'id': self.instance.depot.id,
                    'x': self.instance.depot.x,
                    'y': self.instance.depot.y,
                    'demand': self.instance.depot.demand,
                    'ready_time': getattr(self.instance.depot, 'ready_time', 0.0),
                    'due_time': getattr(self.instance.depot, 'due_time', 0.0),
                    'service_time': getattr(self.instance.depot, 'service_time', 0.0),
                },
                'customers': [
                    {
                        'id': c.id,
                        'x': c.x,
                        'y': c.y,
                        'demand': c.demand,
                        'ready_time': getattr(c, 'ready_time', 0.0),
                        'due_time': getattr(c, 'due_time', 0.0),
                        'service_time': getattr(c, 'service_time', 0.0),
                    }
                    for c in self.instance.customers
                ],
                'vehicle_capacity': self.instance.vehicle_capacity,
                'type': self.instance.type
            }
            
            config_dict = {
                'representation': config.representation,
                'selection_method': config.selection_method,
                'crossover_method': config.crossover_method,
                'mutation_method': config.mutation_method,
                'population_size': config.population_size,
                'max_generations': config.max_generations,
                'elitism_rate': config.elitism_rate,
                'crossover_rate': config.crossover_rate,
                'mutation_rate': config.mutation_rate,
                'tournament_size': config.tournament_size,
                'verbose': False  # Disable verbose in parallel mode
            }
            
            # Prepare tasks
            tasks = []
            for run_num in range(1, self.n_runs + 1):
                seed = hash(f"{self.instance.name}_{method_name}_{run_num}") % (2**31)
                tasks.append((instance_data, config_dict, method_name, run_num, seed))
            
            # Execute in parallel
            with mp.Pool(processes=self.n_jobs) as pool:
                results = pool.map(_run_single_ga_worker, tasks)
            
            # Process results
            for run_num, best_fitness, fitness_history, diversity_history in results:
                all_fitness_histories.append(fitness_history)
                all_diversity_histories.append(diversity_history)
                final_fitnesses.append(best_fitness)
                print(f"  Run {run_num}/{self.n_runs} completed: Best = {best_fitness:.2f}")
                sys.stdout.flush()
        else:
            # Sequential execution
            for run_num in range(1, self.n_runs + 1):
                print(f"  Run {run_num}/{self.n_runs}...", end=" ", flush=True)
                
                # Set unique seed for reproducibility
                seed = hash(f"{self.instance.name}_{method_name}_{run_num}") % (2**31)
                np.random.seed(seed)
                
                # Run GA
                ga = GeneticAlgorithm(self.instance, config)
                result = ga.run()
                
                # Store histories
                all_fitness_histories.append(result.fitness_history)
                all_diversity_histories.append(result.diversity_history)
                final_fitnesses.append(result.best_fitness)
                
                print(f"Best: {result.best_fitness:.2f}")
                sys.stdout.flush()
        
        # Calculate average per generation
        max_generations = max(len(h) for h in all_fitness_histories)
        
        # Pad shorter histories with last value
        padded_fitness = []
        padded_diversity = []
        
        for fitness_hist, diversity_hist in zip(all_fitness_histories, all_diversity_histories):
            # Pad fitness
            if len(fitness_hist) < max_generations:
                padded_fitness.append(fitness_hist + [fitness_hist[-1]] * (max_generations - len(fitness_hist)))
            else:
                padded_fitness.append(fitness_hist[:max_generations])
            
            # Pad diversity
            if len(diversity_hist) < max_generations:
                padded_diversity.append(diversity_hist + [diversity_hist[-1]] * (max_generations - len(diversity_hist)))
            else:
                padded_diversity.append(diversity_hist[:max_generations])
        
        # Calculate averages per generation
        fitness_array = np.array(padded_fitness)
        diversity_array = np.array(padded_diversity)
        
        average_fitness_history = np.mean(fitness_array, axis=0).tolist()
        average_diversity_history = np.mean(diversity_array, axis=0).tolist()
        
        # Calculate statistics
        best_fitness = np.min(final_fitnesses)
        mean_fitness = np.mean(final_fitnesses)
        std_fitness = np.std(final_fitnesses)
        
        runtime = time.time() - start_time
        
        # Save detailed results
        self._save_detailed_results(method_name, all_fitness_histories, all_diversity_histories,
                                   average_fitness_history, average_diversity_history)
        
        # Mark as complete in checkpoint
        if self.checkpoint:
            self.checkpoint.mark_method_complete(self.instance.name, method_name)
        
        print(f"  Average final fitness: {mean_fitness:.2f} ± {std_fitness:.2f}")
        print(f"  Best fitness: {best_fitness:.2f}")
        print(f"  Runtime: {runtime:.2f}s")
        sys.stdout.flush()
        
        return {
            'method_name': method_name,
            'average_fitness_history': average_fitness_history,
            'average_diversity_history': average_diversity_history,
            'best_fitness': float(best_fitness),
            'mean_fitness': float(mean_fitness),
            'std_fitness': float(std_fitness),
            'runtime': float(runtime),
            'n_runs': self.n_runs
        }
    
    def _save_detailed_results(self, method_name: str, 
                               all_fitness_histories: List[List[float]],
                               all_diversity_histories: List[List[float]],
                               average_fitness_history: List[float],
                               average_diversity_history: List[float]):
        """Save detailed results to CSV"""
        max_generations = len(average_fitness_history)
        
        # Prepare data for DataFrame
        data = {
            'generation': list(range(1, max_generations + 1))
        }
        
        # Add individual run histories
        for run_idx, (fitness_hist, diversity_hist) in enumerate(zip(all_fitness_histories, all_diversity_histories)):
            # Pad if needed
            if len(fitness_hist) < max_generations:
                fitness_hist = fitness_hist + [fitness_hist[-1]] * (max_generations - len(fitness_hist))
            if len(diversity_hist) < max_generations:
                diversity_hist = diversity_hist + [diversity_hist[-1]] * (max_generations - len(diversity_hist))
            
            data[f'fitness_run_{run_idx + 1}'] = fitness_hist[:max_generations]
            data[f'diversity_run_{run_idx + 1}'] = diversity_hist[:max_generations]
        
        # Add averages
        data['fitness_average'] = average_fitness_history
        data['diversity_average'] = average_diversity_history
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_file = os.path.join(self.instance_dir, f"{method_name}_convergence.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"  Saved convergence history to {csv_file}")

