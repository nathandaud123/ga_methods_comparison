"""
Parallel execution support for GA experiments with thread-safe checkpointing
"""

import os
import json
import time
import multiprocessing as mp
from multiprocessing import Lock, Manager
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from functools import partial

from ..ga.genetic_algorithm import GeneticAlgorithm, GAConfig, GAResult
from ..data.solomon_parser import VRPInstance
from .metrics import ExperimentMetrics
from .checkpoint import CheckpointManager


class ThreadSafeCheckpointManager:
    """
    Thread-safe checkpoint manager using file locking
    Prevents race conditions in parallel execution
    """
    
    def __init__(self, checkpoint_file: str, lock: Lock):
        self.checkpoint_file = checkpoint_file
        self.lock = lock
        self.checkpoint_dir = os.path.dirname(checkpoint_file)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._state = None
        self._load()
    
    def _load(self):
        """Load checkpoint state (thread-safe)"""
        with self.lock:
            if os.path.exists(self.checkpoint_file):
                try:
                    with open(self.checkpoint_file, 'r') as f:
                        self._state = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self._state = self._init_state()
            else:
                self._state = self._init_state()
    
    def _save(self):
        """Save checkpoint state (thread-safe)"""
        with self.lock:
            try:
                # Write to temp file first, then rename (atomic operation)
                temp_file = self.checkpoint_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(self._state, f, indent=2)
                # Atomic rename (works on most systems)
                if os.path.exists(self.checkpoint_file):
                    os.remove(self.checkpoint_file)
                os.rename(temp_file, self.checkpoint_file)
            except IOError as e:
                print(f"Warning: Could not save checkpoint: {e}")
    
    def _init_state(self) -> Dict:
        """Initialize empty checkpoint state"""
        return {
            'completed_instances': [],
            'completed_methods': {},
            'partial_progress': {}
        }
    
    def is_method_complete(self, instance_name: str, method_name: str, 
                          expected_runs: int) -> bool:
        """Check if method is complete (thread-safe read)"""
        with self.lock:
            self._load()  # Reload to get latest state
            
            if instance_name not in self._state['completed_methods']:
                return False
            
            completed_methods = self._state['completed_methods'][instance_name]
            if method_name not in completed_methods:
                return False
            
            # Check if all runs are complete
            if instance_name in self._state['partial_progress']:
                method_progress = self._state['partial_progress'][instance_name].get(method_name, {})
                if isinstance(method_progress, dict) and 'status' in method_progress:
                    return method_progress['status'] == 'complete'
                return len(method_progress) >= expected_runs
            
            return method_name in completed_methods
    
    def get_completed_runs(self, instance_name: str, method_name: str) -> set:
        """Get completed runs (thread-safe read)"""
        with self.lock:
            self._load()
            
            if instance_name not in self._state['partial_progress']:
                return set()
            if method_name not in self._state['partial_progress'][instance_name]:
                return set()
            
            method_progress = self._state['partial_progress'][instance_name][method_name]
            if isinstance(method_progress, dict) and 'status' in method_progress:
                if method_progress['status'] == 'complete':
                    return set(range(1, method_progress.get('completed_runs', 0) + 1))
            
            run_keys = [int(k) for k in method_progress.keys() if k.isdigit()]
            return set(run_keys)
    
    def mark_run_complete(self, instance_name: str, method_name: str, 
                         run_number: int, result: Dict):
        """Mark run as complete (thread-safe write)"""
        with self.lock:
            self._load()  # Reload to get latest state
            
            if instance_name not in self._state['partial_progress']:
                self._state['partial_progress'][instance_name] = {}
            if method_name not in self._state['partial_progress'][instance_name]:
                self._state['partial_progress'][instance_name][method_name] = {}
            
            self._state['partial_progress'][instance_name][method_name][str(run_number)] = result
            self._save()
    
    def mark_method_complete(self, instance_name: str, method_name: str):
        """Mark method as complete (thread-safe write)"""
        with self.lock:
            self._load()
            
            if instance_name not in self._state['completed_methods']:
                self._state['completed_methods'][instance_name] = []
            
            if method_name not in self._state['completed_methods'][instance_name]:
                self._state['completed_methods'][instance_name].append(method_name)
            
            # Clean up partial progress
            if instance_name in self._state['partial_progress']:
                if method_name in self._state['partial_progress'][instance_name]:
                    method_progress = self._state['partial_progress'][instance_name][method_name]
                    num_runs = len([k for k in method_progress.keys() if k.isdigit()])
                    self._state['partial_progress'][instance_name][method_name] = {
                        'completed_runs': num_runs,
                        'status': 'complete'
                    }
            
            self._save()


def _run_single_experiment(args: Tuple) -> Tuple[str, int, Optional[ExperimentMetrics], Optional[Exception]]:
    """
    Run a single GA experiment (worker function for parallel execution)
    
    Args:
        args: Tuple of (instance_data, config_dict, method_name, run_number, 
                       checkpoint_file, checkpoint_lock)
    
    Returns:
        Tuple of (method_name, run_number, metrics, error)
    """
    try:
        (instance_data, config_dict, method_name, run_number, 
         checkpoint_file, checkpoint_lock) = args
        
        # Reconstruct instance
        from ..data.solomon_parser import VRPInstance, Customer
        
        # Reconstruct depot
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
        
        # Reconstruct customers
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
        
        # Thread-safe checkpoint manager
        checkpoint_mgr = ThreadSafeCheckpointManager(checkpoint_file, checkpoint_lock)
        
        # Check if already completed
        instance_name = instance.name
        completed_runs = checkpoint_mgr.get_completed_runs(instance_name, method_name)
        if run_number in completed_runs:
            return (method_name, run_number, None, None)  # Already done
        
        # Run GA
        ga = GeneticAlgorithm(instance, config)
        result = ga.run()
        
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
        
        # Save checkpoint (thread-safe)
        checkpoint_data = {
            'fitness': float(metrics.fitness),
            'runtime': float(metrics.runtime),
            'convergence_generation': int(metrics.convergence_generation),
            'diversity': float(metrics.diversity),
        }
        checkpoint_mgr.mark_run_complete(instance_name, method_name, run_number, checkpoint_data)
        
        return (method_name, run_number, metrics, None)
        
    except Exception as e:
        return (method_name, run_number, None, e)


class ParallelExperimentExecutor:
    """
    Execute GA experiments in parallel using multiprocessing
    Thread-safe checkpointing and result collection
    """
    
    def __init__(self, instance: VRPInstance, n_runs: int = 10, 
                 n_jobs: Optional[int] = None, checkpoint_file: str = "results/checkpoint.json"):
        self.instance = instance
        self.n_runs = n_runs
        self.n_jobs = n_jobs or max(1, mp.cpu_count() - 1)  # Leave one core free
        self.checkpoint_file = checkpoint_file
        
        # Create shared lock for checkpoint file
        self.manager = Manager()
        self.checkpoint_lock = self.manager.Lock()
        
        # Prepare instance data for serialization
        self.instance_data = {
            'name': instance.name,
            'num_customers': instance.num_customers,
            'depot': {
                'id': instance.depot.id,
                'x': instance.depot.x,
                'y': instance.depot.y,
                'demand': instance.depot.demand,
                'ready_time': getattr(instance.depot, 'ready_time', 0.0),
                'due_time': getattr(instance.depot, 'due_time', 0.0),
                'service_time': getattr(instance.depot, 'service_time', 0.0),
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
                for c in instance.customers
            ],
            'vehicle_capacity': instance.vehicle_capacity,
            'type': instance.type
        }
    
    def run_method_parallel(self, config: GAConfig, method_name: str) -> List[ExperimentMetrics]:
        """
        Run all runs for a method in parallel
        
        Args:
            config: GA configuration
            method_name: Method name
            
        Returns:
            List of experiment metrics
        """
        # Check checkpoint first (single-threaded check)
        checkpoint_mgr = ThreadSafeCheckpointManager(self.checkpoint_file, self.checkpoint_lock)
        instance_name = self.instance.name
        
        # Check if method is already complete
        if checkpoint_mgr.is_method_complete(instance_name, method_name, self.n_runs):
            print(f"  Method {method_name} already completed (checkpoint)")
            return []
        
        # Get completed runs
        completed_runs = checkpoint_mgr.get_completed_runs(instance_name, method_name)
        
        # Prepare tasks (only for incomplete runs)
        tasks = []
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
        
        for run in range(1, self.n_runs + 1):
            if run not in completed_runs:
                tasks.append((
                    self.instance_data,
                    config_dict,
                    method_name,
                    run,
                    self.checkpoint_file,
                    self.checkpoint_lock
                ))
        
        if not tasks:
            print(f"  All runs for {method_name} already completed")
            return []
        
        print(f"  Running {len(tasks)} runs in parallel ({self.n_jobs} workers)...")
        
        # Execute in parallel
        results = []
        with mp.Pool(processes=self.n_jobs) as pool:
            async_results = pool.map_async(_run_single_experiment, tasks)
            
            # Wait for completion with progress updates
            completed = 0
            total = len(tasks)
            while not async_results.ready():
                time.sleep(1)
                # Can't easily get progress from map_async, so just wait
                pass
            
            # Collect results
            for method_name_result, run_num, metrics, error in async_results.get():
                if error:
                    print(f"  Error in {method_name_result} run {run_num}: {error}")
                    continue
                if metrics:
                    results.append(metrics)
                completed += 1
        
        # Mark method as complete if all runs done
        final_completed = checkpoint_mgr.get_completed_runs(instance_name, method_name)
        if len(final_completed) >= self.n_runs:
            checkpoint_mgr.mark_method_complete(instance_name, method_name)
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'manager'):
            self.manager.shutdown()

