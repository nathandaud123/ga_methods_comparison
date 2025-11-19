"""
Parallel execution support for GA experiments with thread-safe checkpointing
"""

import os
import json
import sys
import time
import uuid
import shutil
import hashlib
import random
import multiprocessing as mp
from multiprocessing import Lock, Manager
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from functools import partial

# Cross-platform file locking
try:
    import msvcrt  # Windows
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

try:
    import fcntl  # Unix/Linux
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from ..ga.genetic_algorithm import GeneticAlgorithm, GAConfig, GAResult
from ..data.solomon_parser import VRPInstance
from .metrics import ExperimentMetrics
from .checkpoint import CheckpointManager


class FileLock:
    """Cross-platform file lock using file-based locking"""
    
    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self.lock_dir = os.path.dirname(lock_file)
        if self.lock_dir:
            os.makedirs(self.lock_dir, exist_ok=True)
        self._lock_handle = None
    
    def __enter__(self):
        max_retries = 30
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                if HAS_MSVCRT:
                    # Windows file locking - need to open in binary mode for locking
                    self._lock_handle = open(self.lock_file, 'wb+')
                    try:
                        msvcrt.locking(self._lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                    except IOError:
                        self._lock_handle.close()
                        self._lock_handle = None
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        raise
                elif HAS_FCNTL:
                    # Unix file locking
                    self._lock_handle = open(self.lock_file, 'w')
                    try:
                        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except IOError:
                        self._lock_handle.close()
                        self._lock_handle = None
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        raise
                else:
                    # Fallback: simple file existence check with retry
                    if os.path.exists(self.lock_file):
                        time.sleep(retry_delay)
                        continue
                    self._lock_handle = open(self.lock_file, 'w')
                    self._lock_handle.write(str(os.getpid()))
                    self._lock_handle.flush()
                return self
            except (IOError, OSError) as e:
                if self._lock_handle:
                    try:
                        self._lock_handle.close()
                    except:
                        pass
                    self._lock_handle = None
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_handle:
            try:
                if HAS_MSVCRT:
                    msvcrt.locking(self._lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif HAS_FCNTL:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                self._lock_handle.close()
                if os.path.exists(self.lock_file):
                    try:
                        os.remove(self.lock_file)
                    except:
                        pass
            except:
                pass


class ThreadSafeCheckpointManager:
    """
    Thread-safe checkpoint manager using file locking
    Prevents race conditions in parallel execution
    """
    
    def __init__(self, checkpoint_file: str, lock_file: Optional[str] = None):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_dir = os.path.dirname(checkpoint_file)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Use separate lock file for cross-platform compatibility
        self.lock_file = lock_file or (checkpoint_file + '.lock')
        self._state = None
        self._load(use_lock=False)  # Initial load doesn't need lock
    
    def _load(self, use_lock: bool = True):
        """Load checkpoint state (thread-safe)"""
        def _do_load():
            if os.path.exists(self.checkpoint_file):
                try:
                    with open(self.checkpoint_file, 'r') as f:
                        self._state = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self._state = self._init_state()
            else:
                self._state = self._init_state()
        
        if use_lock:
            with FileLock(self.lock_file):
                _do_load()
        else:
            _do_load()
    
    def _save(self, use_lock: bool = True):
        """Save checkpoint state (thread-safe)"""
        def _do_save():
            max_retries = 10
            base_delay = 0.05
            
            for attempt in range(max_retries):
                try:
                    # Write to temp file first, then rename (atomic operation)
                    temp_file = self.checkpoint_file + '.tmp'
                    
                    # Use unique temp file name to avoid conflicts
                    unique_temp = f"{temp_file}.{uuid.uuid4().hex[:8]}"
                    
                    try:
                        with open(unique_temp, 'w') as f:
                            json.dump(self._state, f, indent=2)
                            f.flush()
                            os.fsync(f.fileno())  # Force write to disk
                    except (IOError, OSError, PermissionError) as e:
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))  # Exponential backoff
                            continue
                        raise
                    
                    # Atomic rename (works on most systems)
                    # On Windows, need to handle file in use errors
                    if os.path.exists(self.checkpoint_file):
                        try:
                            # Try to remove old file with retry
                            for rm_attempt in range(3):
                                try:
                                    os.remove(self.checkpoint_file)
                                    break
                                except (IOError, OSError, PermissionError):
                                    if rm_attempt < 2:
                                        time.sleep(0.05)
                                        continue
                                    raise
                        except (IOError, OSError, PermissionError) as e:
                            # File might be locked by another process, retry
                            if os.path.exists(unique_temp):
                                try:
                                    os.remove(unique_temp)
                                except:
                                    pass
                            if attempt < max_retries - 1:
                                time.sleep(base_delay * (2 ** attempt))  # Exponential backoff
                                continue
                            raise
                    
                    try:
                        # Use shutil.move for better cross-platform support
                        shutil.move(unique_temp, self.checkpoint_file)
                    except (IOError, OSError, PermissionError) as e:
                        # Rename failed, try to clean up temp file
                        if os.path.exists(unique_temp):
                            try:
                                os.remove(unique_temp)
                            except:
                                pass
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))  # Exponential backoff
                            continue
                        raise
                    return True  # Success
                except (IOError, OSError, PermissionError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    # Last attempt failed, print warning but don't crash
                    print(f"Warning: Could not save checkpoint after {max_retries} attempts: {e}", file=sys.stderr)
                    return False
        
        if use_lock:
            try:
                with FileLock(self.lock_file):
                    return _do_save()
            except Exception as e:
                print(f"Warning: File lock error during save: {e}", file=sys.stderr)
                return False
        else:
            return _do_save()
    
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
        with FileLock(self.lock_file):
            self._load(use_lock=False)  # Reload to get latest state (already locked)
            
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
        with FileLock(self.lock_file):
            self._load(use_lock=False)  # Already locked
            
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
        with FileLock(self.lock_file):
            self._load(use_lock=False)  # Reload to get latest state (already locked)
            
            if instance_name not in self._state['partial_progress']:
                self._state['partial_progress'][instance_name] = {}
            if method_name not in self._state['partial_progress'][instance_name]:
                self._state['partial_progress'][instance_name][method_name] = {}
            
            self._state['partial_progress'][instance_name][method_name][str(run_number)] = result
            saved = self._save(use_lock=False)  # Already locked
            if not saved:
                # If save failed, at least we tried - state is in memory
                pass
    
    def mark_method_complete(self, instance_name: str, method_name: str):
        """Mark method as complete (thread-safe write)"""
        with FileLock(self.lock_file):
            self._load(use_lock=False)  # Already locked
            
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
            
            saved = self._save(use_lock=False)  # Already locked
            if not saved:
                # If save failed, at least we tried - state is in memory
                pass


def _run_single_experiment(args: Tuple) -> Tuple[str, int, Optional[ExperimentMetrics], Optional[Exception]]:
    """
    Run a single GA experiment (worker function for parallel execution)
    
    Args:
        args: Tuple of (instance_data, config_dict, method_name, run_number, 
                       checkpoint_file)
    
    Returns:
        Tuple of (method_name, run_number, metrics, error)
    """
    try:
        (instance_data, config_dict, method_name, run_number, 
         checkpoint_file) = args
        
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
        
        # Thread-safe checkpoint manager (using file-based locking)
        checkpoint_mgr = ThreadSafeCheckpointManager(checkpoint_file)
        
        # Check if already completed
        instance_name = instance.name
        completed_runs = checkpoint_mgr.get_completed_runs(instance_name, method_name)
        if run_number in completed_runs:
            return (method_name, run_number, None, None)  # Already done
        
        # Set unique random seed for this run to ensure different results
        # Use hash of (instance_name, method_name, run_number) for reproducibility
        seed_string = f"{instance_name}_{method_name}_{run_number}"
        seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16) % (2**31)
        np.random.seed(seed)
        random.seed(seed)
        
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
        # Don't let checkpoint save failure prevent returning results
        try:
            checkpoint_data = {
                'fitness': float(metrics.fitness),
                'runtime': float(metrics.runtime),
                'convergence_generation': int(metrics.convergence_generation),
                'diversity': float(metrics.diversity),
            }
            checkpoint_mgr.mark_run_complete(instance_name, method_name, run_number, checkpoint_data)
        except Exception as checkpoint_error:
            # Log but don't fail - results are more important than checkpoint
            print(f"Warning: Failed to save checkpoint for {method_name} run {run_number}: {checkpoint_error}", file=sys.stderr)
        
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
        checkpoint_mgr = ThreadSafeCheckpointManager(self.checkpoint_file)
        instance_name = self.instance.name
        
        # Check if method is already complete
        if checkpoint_mgr.is_method_complete(instance_name, method_name, self.n_runs):
            print(f"  Method {method_name} already completed (checkpoint)")
            sys.stdout.flush()
            return []
        
        # Get completed runs
        completed_runs = checkpoint_mgr.get_completed_runs(instance_name, method_name)
        if completed_runs:
            print(f"  Found {len(completed_runs)}/{self.n_runs} completed runs in checkpoint")
            sys.stdout.flush()
        
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
                    self.checkpoint_file
                ))
        
        if not tasks:
            print(f"  All runs for {method_name} already completed")
            sys.stdout.flush()
            return []
        
        print(f"  Running {len(tasks)} runs in parallel ({self.n_jobs} workers)...")
        sys.stdout.flush()
        
        # Execute in parallel
        results = []
        with mp.Pool(processes=self.n_jobs) as pool:
            async_results = pool.map_async(_run_single_experiment, tasks)
            
            # Wait for completion with progress updates
            completed = 0
            total = len(tasks)
            last_update = time.time()
            while not async_results.ready():
                time.sleep(2)  # Check every 2 seconds
                elapsed = time.time() - last_update
                if elapsed >= 10:  # Print progress every 10 seconds
                    print(f"  ... still running ({total} tasks in progress) ...")
                    sys.stdout.flush()
                    last_update = time.time()
            
            print(f"  All {total} runs completed, collecting results...")
            sys.stdout.flush()
            
            # Collect results
            for method_name_result, run_num, metrics, error in async_results.get():
                if error:
                    print(f"  Error in {method_name_result} run {run_num}: {error}")
                    sys.stdout.flush()
                    continue
                if metrics:
                    results.append(metrics)
                completed += 1
                if completed % max(1, total // 5) == 0:  # Print every 20% progress
                    print(f"  Collected {completed}/{total} results...")
                    sys.stdout.flush()
        
        # Mark method as complete if all runs done
        final_completed = checkpoint_mgr.get_completed_runs(instance_name, method_name)
        if len(final_completed) >= self.n_runs:
            checkpoint_mgr.mark_method_complete(instance_name, method_name)
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'manager'):
            self.manager.shutdown()

