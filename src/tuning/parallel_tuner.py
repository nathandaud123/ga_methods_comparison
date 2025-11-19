"""
Parallel Optuna Tuning for multiple methods simultaneously
Each method gets its own CPU core for tuning
"""

import multiprocessing as mp
import optuna
import pandas as pd
import os
import sys
import time
from typing import Dict, Any, Tuple, Optional
from ..ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ..data.solomon_parser import VRPInstance
import numpy as np


def _run_single_tuning(args: Tuple) -> Tuple[str, Dict[str, Any], Optional[Exception]]:
    """
    Run Optuna tuning for a single method (worker function for parallel execution)
    
    Args:
        args: Tuple of (instance_data, method_name, representation, selection_method,
                       crossover_method, mutation_method, n_trials, timeout, tuning_dir)
    
    Returns:
        Tuple of (method_name, tuning_result_dict, error)
    """
    try:
        (instance_data, method_name, representation, selection_method,
         crossover_method, mutation_method, n_trials, timeout, tuning_dir) = args
        
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
        
        # Define objective function for this method
        def objective(trial: optuna.Trial) -> float:
            # Suggest hyperparameters
            population_size = trial.suggest_int('population_size', 50, 200)
            max_generations = trial.suggest_int('max_generations', 100, 500)
            crossover_rate = trial.suggest_float('crossover_rate', 0.6, 0.95)
            mutation_rate = trial.suggest_float('mutation_rate', 0.05, 0.3)
            tournament_size = trial.suggest_int('tournament_size', 2, 10)
            elitism_rate = trial.suggest_float('elitism_rate', 0.05, 0.2)
            
            # Create config
            config = GAConfig(
                population_size=population_size,
                max_generations=max_generations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                elitism_rate=elitism_rate,
                tournament_size=tournament_size,
                selection_method=selection_method,
                crossover_method=crossover_method,
                mutation_method=mutation_method,
                representation=representation,
                verbose=False
            )
            
            # Run GA
            ga = GeneticAlgorithm(instance, config)
            result = ga.run()
            
            return result.best_fitness
        
        # Create study and run optimization
        study = optuna.create_study(
            direction='minimize',
            study_name=f"{instance.name}_{method_name}_tuning"
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False  # Disable progress bar in parallel mode
        )
        
        # Collect all trial data
        trial_history = []
        for trial in study.trials:
            trial_info = {
                'trial_number': trial.number,
                'state': 'COMPLETE' if trial.state == optuna.trial.TrialState.COMPLETE else str(trial.state),
                'value': trial.value if trial.value is not None else float('inf'),
                'population_size': trial.params.get('population_size'),
                'max_generations': trial.params.get('max_generations'),
                'crossover_rate': trial.params.get('crossover_rate'),
                'mutation_rate': trial.params.get('mutation_rate'),
                'tournament_size': trial.params.get('tournament_size'),
                'elitism_rate': trial.params.get('elitism_rate'),
            }
            trial_history.append(trial_info)
        
        # Save tuning history to CSV
        if trial_history:
            df = pd.DataFrame(trial_history)
            df['is_best'] = df['value'] == df['value'].min()
            df['cumulative_best'] = df['value'].cummin()
            df['improvement'] = df['cumulative_best'].diff().fillna(0)
            df = df.sort_values('trial_number').reset_index(drop=True)
            
            filename = f"{instance.name}_{method_name}_optuna_tuning.csv"
            filepath = os.path.join(tuning_dir, filename)
            df.to_csv(filepath, index=False)
        
        return (method_name, {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'trial_history': trial_history
        }, None)
        
    except Exception as e:
        return (method_name, {}, e)


def run_parallel_tuning(
    instance: VRPInstance,
    method_combinations: list,
    n_trials: int = 1,
    timeout: int = 3600,
    n_jobs: int = 150,
    tuning_dir: str = "results/tuning"
) -> Dict[str, Dict[str, Any]]:
    """
    Run Optuna tuning for multiple methods in parallel
    
    Args:
        instance: VRP instance
        method_combinations: List of (method_name, GAConfig) tuples
        n_trials: Number of trials per method
        timeout: Timeout per method (seconds)
        n_jobs: Number of parallel workers
        tuning_dir: Directory to save tuning results
    
    Returns:
        Dictionary mapping method_name to tuning results
    """
    os.makedirs(tuning_dir, exist_ok=True)
    
    # Prepare instance data for serialization
    instance_data = {
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
    
    # Prepare tasks
    tasks = []
    for method_name, method_config in method_combinations:
        tasks.append((
            instance_data,
            method_name,
            method_config.representation,
            method_config.selection_method,
            method_config.crossover_method,
            method_config.mutation_method,
            n_trials,
            timeout,
            tuning_dir
        ))
    
    print(f"\nRunning parallel tuning for {len(tasks)} methods with {n_jobs} workers...")
    print(f"Each method: {n_trials} trial(s)")
    sys.stdout.flush()
    
    # Run in parallel
    results = {}
    completed = 0
    total = len(tasks)
    last_update = time.time()
    
    with mp.Pool(processes=n_jobs) as pool:
        async_results = pool.map_async(_run_single_tuning, tasks)
        
        # Wait with progress updates
        while not async_results.ready():
            time.sleep(2)
            elapsed = time.time() - last_update
            if elapsed >= 10:
                print(f"  ... tuning in progress ({total} methods) ...")
                sys.stdout.flush()
                last_update = time.time()
        
        # Collect results
        for method_name, tuning_result, error in async_results.get():
            if error:
                print(f"  Error tuning {method_name}: {error}")
                sys.stdout.flush()
                continue
            
            if tuning_result:
                results[method_name] = tuning_result
                completed += 1
                if completed % max(1, total // 10) == 0:
                    print(f"  Completed tuning: {completed}/{total} methods")
                    sys.stdout.flush()
    
    print(f"\nParallel tuning complete: {completed}/{total} methods")
    sys.stdout.flush()
    
    return results

