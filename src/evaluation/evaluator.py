"""
Evaluator for running multiple experiments and collecting results
Supports both sequential and parallel execution
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import hashlib
import random
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
                 parallel: bool = False, n_jobs: Optional[int] = None, parallel_methods: bool = False):
        self.instance = instance
        self.n_runs = n_runs
        self.metrics_calc = MetricsCalculator()
        self.save_history = save_history
        self.history_dir = history_dir or "results/convergence"
        self.checkpoint_manager = checkpoint_manager
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.parallel_methods = parallel_methods  # Enable parallel execution for multiple methods
        
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
                # Set unique random seed for this run to ensure different results
                seed_string = f"{self.instance.name}_{method_name}_{run_key}"
                seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16) % (2**31)
                np.random.seed(seed)
                random.seed(seed)
                
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
        Supports parallel execution for multiple methods if enabled
        
        Args:
            configs: List of (method_name, config) tuples
            
        Returns:
            Dictionary mapping method names to aggregated metrics
        """
        # Use parallel methods execution if enabled
        if self.parallel_methods and self.parallel and self.parallel_executor:
            return self._compare_methods_parallel(configs)
        else:
            return self._compare_methods_sequential(configs)
    
    def _compare_methods_sequential(self, configs: List[Tuple[str, GAConfig]]) -> Dict[str, ComparisonMetrics]:
        """Compare methods sequentially (original implementation)"""
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
    
    def _compare_methods_parallel(self, configs: List[Tuple[str, GAConfig]]) -> Dict[str, ComparisonMetrics]:
        """
        Compare multiple methods in parallel using batched execution
        - Process methods in batches (e.g., 12 methods at a time)
        - Each method runs 5 runs in parallel (run numbers 1-5)
        - Total: 12 methods × 5 runs = 60 workers optimal
        - Uses global pool but organizes tasks per method to ensure run numbers 1-5
        """
        import multiprocessing as mp
        from .parallel_executor import _run_single_experiment
        
        comparison_results = {}
        instance_name = self.instance.name
        
        # Filter out completed methods first
        pending_configs = []
        for method_name, config in configs:
            if self.checkpoint_manager and \
               self.checkpoint_manager.is_method_complete(instance_name, method_name, self.n_runs):
                print(f"\n{'='*60}")
                print(f"Skipping {method_name} (already completed - {self.n_runs}/{self.n_runs} runs)")
                print(f"{'='*60}")
                continue
            pending_configs.append((method_name, config))
        
        if not pending_configs:
            print("\nAll methods already completed!")
            return comparison_results
        
        # Calculate batch size: methods per batch
        # With n_jobs=60 and n_runs=5, we can run 12 methods simultaneously
        # (12 methods × 5 runs = 60 workers)
        methods_per_batch = max(1, self.n_jobs // self.n_runs) if self.n_jobs else 12
        
        print(f"\n{'='*80}")
        print(f"PARALLEL METHODS EXECUTION: {len(pending_configs)} methods")
        print(f"Batch size: {methods_per_batch} methods per batch")
        print(f"Each method: {self.n_runs} runs in parallel (run numbers 1-5)")
        print(f"Total workers per batch: {methods_per_batch * self.n_runs} (max {self.n_jobs})")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        checkpoint_file = getattr(self.checkpoint_manager, 'checkpoint_file', 'results/checkpoint.json') if self.checkpoint_manager else 'results/checkpoint.json'
        
        # Initialize set to track complete methods (for skipping warnings)
        self._complete_methods = set()
        
        # Process methods in batches
        num_batches = (len(pending_configs) + methods_per_batch - 1) // methods_per_batch
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * methods_per_batch
            end_idx = min(start_idx + methods_per_batch, len(pending_configs))
            batch_configs = pending_configs[start_idx:end_idx]
            
            print(f"\n{'='*80}")
            print(f"BATCH {batch_idx + 1}/{num_batches}: Processing {len(batch_configs)} methods")
            print(f"  Methods: {[name for name, _ in batch_configs]}")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            
            # Prepare all tasks for this batch: organized per method with run numbers 1-5
            batch_tasks = []
            method_results_map = {}
            
            for method_name, config in batch_configs:
                # Check completed runs for this method
                if self.checkpoint_manager:
                    completed_runs = self.checkpoint_manager.get_completed_runs(instance_name, method_name)
                    is_complete = self.checkpoint_manager.is_method_complete(instance_name, method_name, self.n_runs)
                else:
                    completed_runs = set()
                    is_complete = False
                
                # If method is already complete, load results from checkpoint
                if is_complete and len(completed_runs) >= self.n_runs:
                    # Load results from checkpoint
                    checkpoint_results = []
                    # Reload checkpoint state to get latest data
                    self.checkpoint_manager.state = self.checkpoint_manager.load()
                    
                    print(f"\n  [DEBUG] Attempting to load results for {method_name}:")
                    print(f"    - Method marked as complete: {is_complete}")
                    print(f"    - Completed runs in checkpoint: {len(completed_runs)}/{self.n_runs}")
                    sys.stdout.flush()
                    
                    has_partial_progress = instance_name in self.checkpoint_manager.state.get('partial_progress', {})
                    print(f"    - Has partial_progress for instance: {has_partial_progress}")
                    sys.stdout.flush()
                    
                    if has_partial_progress:
                        method_progress = self.checkpoint_manager.state['partial_progress'][instance_name].get(method_name, {})
                        method_progress_exists = method_progress is not None and method_progress != {}
                        print(f"    - Method progress found: {method_progress_exists}")
                        sys.stdout.flush()
                        
                        if isinstance(method_progress, dict):
                            print(f"    - Method progress type: dict")
                            print(f"    - Method progress keys: {list(method_progress.keys())}")
                            sys.stdout.flush()
                            
                            # Check if it's summary format (status='complete')
                            if 'status' in method_progress and method_progress['status'] == 'complete':
                                print(f"    - Format: Summary (status='complete')")
                                print(f"    - Completed runs in summary: {method_progress.get('completed_runs', 0)}")
                                sys.stdout.flush()
                                
                                # Method is complete but data might be in summary format
                                # Try to load from convergence CSV file instead
                                if self.save_history and self.history_dir:
                                    convergence_file = os.path.join(
                                        self.history_dir,
                                        f"{method_name}_convergence.csv"
                                    )
                                    print(f"    - Trying convergence file: {convergence_file}")
                                    print(f"    - Convergence file exists: {os.path.exists(convergence_file)}")
                                    sys.stdout.flush()
                                    
                                    if os.path.exists(convergence_file):
                                        try:
                                            df = pd.read_csv(convergence_file)
                                            print(f"    - Convergence file loaded: {len(df)} rows")
                                            sys.stdout.flush()
                                            
                                            if len(df) > 0:
                                                last_row = df.iloc[-1]
                                                fitness_cols = [col for col in df.columns if col.startswith('fitness_run_')]
                                                print(f"    - Fitness columns found: {len(fitness_cols)} ({fitness_cols})")
                                                sys.stdout.flush()
                                                
                                                if fitness_cols:
                                                    for col in fitness_cols:
                                                        fitness = float(last_row[col])
                                                        diversity_col = col.replace('fitness_run_', 'diversity_run_')
                                                        diversity = float(last_row[diversity_col]) if diversity_col in df.columns else 0.0
                                                        metrics = ExperimentMetrics(
                                                            fitness=fitness,
                                                            runtime=0.0,  # Not available in convergence file
                                                            convergence_generation=0,
                                                            diversity=diversity,
                                                            solution_quality=0.0,
                                                            num_routes=0,
                                                            total_distance=fitness
                                                        )
                                                        checkpoint_results.append(metrics)
                                                    print(f"    - Loaded {len(checkpoint_results)} runs from convergence file")
                                                    sys.stdout.flush()
                                                else:
                                                    print(f"    - WARNING: No fitness_run_ columns found in convergence file!")
                                                    sys.stdout.flush()
                                            else:
                                                print(f"    - WARNING: Convergence file is empty!")
                                                sys.stdout.flush()
                                        except Exception as e:
                                            print(f"    - ERROR loading convergence file: {e}")
                                            import traceback
                                            traceback.print_exc()
                                            sys.stdout.flush()
                                    else:
                                        print(f"    - WARNING: Convergence file does not exist!")
                                        sys.stdout.flush()
                                else:
                                    print(f"    - WARNING: save_history={self.save_history}, history_dir={self.history_dir}")
                                    sys.stdout.flush()
                            else:
                                # Load from checkpoint data (individual runs)
                                print(f"    - Format: Individual runs")
                                run_keys = sorted([k for k in method_progress.keys() if k.isdigit()], key=int)
                                print(f"    - Run keys found: {run_keys}")
                                sys.stdout.flush()
                                
                                for run_key in run_keys:
                                    run_data = method_progress[run_key]
                                    if isinstance(run_data, dict):
                                        metrics = ExperimentMetrics(
                                            fitness=run_data.get('fitness', 0.0),
                                            runtime=run_data.get('runtime', 0.0),
                                            convergence_generation=run_data.get('convergence_generation', 0),
                                            diversity=run_data.get('diversity', 0.0),
                                            solution_quality=0.0,
                                            num_routes=0,
                                            total_distance=run_data.get('fitness', 0.0)
                                        )
                                        checkpoint_results.append(metrics)
                                
                                print(f"    - Loaded {len(checkpoint_results)} runs from checkpoint data")
                                sys.stdout.flush()
                        else:
                            print(f"    - WARNING: Method progress is not a dict! Type: {type(method_progress)}")
                            sys.stdout.flush()
                    else:
                        print(f"    - WARNING: No partial_progress found for instance {instance_name}")
                        sys.stdout.flush()
                    
                    if checkpoint_results and len(checkpoint_results) >= self.n_runs:
                        method_results_map[method_name] = checkpoint_results[:self.n_runs]  # Take only n_runs
                        print(f"  ✓ Successfully loaded {len(checkpoint_results)} completed runs for {method_name} from checkpoint")
                        sys.stdout.flush()
                    else:
                        print(f"  ✗ FAILED to load results for {method_name}:")
                        print(f"    - Checkpoint results loaded: {len(checkpoint_results)}")
                        print(f"    - Required: {self.n_runs}")
                        print(f"    - Reason: Data may be in summary format or convergence file missing")
                        sys.stdout.flush()
                        # Method is complete but we couldn't load results - that's OK, results are in saved JSON
                        # Set empty list but mark that method is complete (we'll skip warning in aggregation)
                        method_results_map[method_name] = []
                        self._complete_methods.add(method_name)
                    continue  # Skip adding tasks for this method
                
                method_results_map[method_name] = []
                
                # Prepare instance data for serialization (shared across runs of same method)
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
                }
                
                # Add tasks for incomplete runs (run numbers 1-5)
                for run in range(1, self.n_runs + 1):
                    if run not in completed_runs:
                        batch_tasks.append((
                            instance_data,
                            config_dict,
                            method_name,
                            run,  # Run number 1-5 (sequential per method)
                            checkpoint_file
                        ))
            
            if batch_tasks:
                print(f"  Executing {len(batch_tasks)} tasks ({len(batch_configs)} methods × {self.n_runs} runs)")
                sys.stdout.flush()
                
                # Execute all tasks in this batch using global pool
                completed = 0
                total = len(batch_tasks)
                last_update = time.time()
                
                with mp.Pool(processes=self.n_jobs) as pool:
                    async_results = pool.map_async(_run_single_experiment, batch_tasks)
                    
                    # Wait with progress updates
                    while not async_results.ready():
                        time.sleep(2)
                        elapsed = time.time() - last_update
                        if elapsed >= 10:
                            print(f"  ... still running ({total} tasks in progress) ...")
                            sys.stdout.flush()
                            last_update = time.time()
                    
                    print(f"  All {total} tasks completed, collecting results...")
                    sys.stdout.flush()
                    
                    # Collect results
                    for method_name_result, run_num, metrics, error in async_results.get():
                        if error:
                            print(f"  Error in {method_name_result} run {run_num}: {error}")
                            sys.stdout.flush()
                            continue
                        if metrics:
                            if method_name_result not in method_results_map:
                                method_results_map[method_name_result] = []
                            method_results_map[method_name_result].append(metrics)
                        completed += 1
                        if completed % max(1, total // 10) == 0:
                            print(f"  Collected {completed}/{total} results...")
                            sys.stdout.flush()
            else:
                print("  All runs in this batch already completed (loaded from checkpoint)")
                sys.stdout.flush()
            
            # Aggregate results per method for this batch
            for method_name, results in method_results_map.items():
                if not results:
                    # Check if method is complete (either from flag or checkpoint)
                    is_complete_flag = hasattr(self, '_complete_methods') and method_name in self._complete_methods
                    is_complete_checkpoint = False
                    
                    if self.checkpoint_manager:
                        # Reload checkpoint state to ensure we have latest data
                        self.checkpoint_manager.state = self.checkpoint_manager.load()
                        is_complete_checkpoint = self.checkpoint_manager.is_method_complete(instance_name, method_name, self.n_runs)
                    
                    if is_complete_flag or is_complete_checkpoint:
                        # Method is complete but we couldn't load results - that's OK, results are in saved JSON
                        # Skip warning, results will be loaded from saved JSON file in main.py
                        continue
                    
                    print(f"  Warning: No results for {method_name}, skipping")
                    continue
                
                aggregated = self.metrics_calc.aggregate_metrics(results)
                aggregated.method_name = method_name
                comparison_results[method_name] = aggregated
                
                print(f"\nResults for {method_name}:")
                print(f"  Mean Fitness: {aggregated.mean_fitness:.2f} ± {aggregated.std_fitness:.2f}")
                print(f"  Mean Runtime: {aggregated.mean_runtime:.2f}s ± {aggregated.std_runtime:.2f}s")
                print(f"  Best Fitness: {aggregated.best_fitness:.2f}")
                sys.stdout.flush()
        
        return comparison_results
    
    def cleanup(self):
        """Cleanup resources (especially parallel executor)"""
        if self.parallel_executor:
            self.parallel_executor.cleanup()

