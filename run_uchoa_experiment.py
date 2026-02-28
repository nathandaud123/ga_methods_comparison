
import os
import sys
import pandas as pd
import numpy as np
import yaml
import argparse
from typing import List, Dict

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.data.uchoa_parser import UchoaParser
from src.ga.genetic_algorithm import GAConfig
from src.evaluation.simple_evaluator import SimpleEvaluator
from src.evaluation.simple_checkpoint import SimpleCheckpoint

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_uchoa_experiment(sampled_csv_path: str, uchoa_base_path: str, results_dir: str, config_file: str, instance_name: str = None):
    # Load GA Config
    config = load_config(config_file)
    ga_params = config.get('ga', {})
    
    POPULATION_SIZE = ga_params.get('population_size', 80)
    MAX_GENERATIONS = ga_params.get('max_generations', 500)
    CROSSOVER_RATE = ga_params.get('crossover_rate', 0.9)
    MUTATION_RATE = ga_params.get('mutation_rate', 0.3)
    TOURNAMENT_SIZE = ga_params.get('tournament_size', 3)
    
    # Evaluation Config
    eval_config = config.get('evaluation', {})
    N_RUNS = eval_config.get('n_runs', 5)
    N_JOBS = eval_config.get('n_jobs', 0) # Auto-detect CPUs if 0
    
    # Read sampled instances
    print(f"Reading sampled instances from {sampled_csv_path}...")
    df = pd.read_csv(sampled_csv_path)
    uchoa_df = df[df['Dataset'] == 'Uchoa']
    
    instances = uchoa_df['InstanceName'].tolist()
    
    if instance_name:
        # Filter strictly for this instance, accepting with or without .vrp
        target = instance_name if instance_name.endswith('.vrp') else f"{instance_name}.vrp"
        instances = [inst for inst in instances if inst == target]
        if not instances:
            print(f"Instance {instance_name} not found in the sampled instances.")
            # Optionally just add it anyway to allow running outside sample list:
            instances = [target]
        print(f"Running specifically for instance: {target}")
    else:
        print(f"Found {len(instances)} Uchoa instances to process in sample list.")

    # Define Combinations
    combinations = []
    
    # Permutation
    combinations.extend([
        {'rep': 'permutation', 'sel': s, 'cross': c, 'mut': m, 'name': f"permutation_{s}_{c}_{m}"}
        for s in ['stochastic_universal', 'tournament', 'roulette_wheel']
        for c in ['scx', 'pos', 'ox']
        for m in ['swap', 'scramble', 'inversion']
    ])

    # Binary
    combinations.extend([
        {'rep': 'binary', 'sel': s, 'cross': c, 'mut': m, 'name': f"binary_{s}_{c}_{m}"}
        for s in ['stochastic_universal', 'tournament', 'roulette_wheel']
        for c in ['uniform', 'shuffle', 'multi_point']
        for m in ['bit_flip', 'uniform', 'interchanging']
    ])

    # Real Valued
    combinations.extend([
        {'rep': 'real_valued', 'sel': s, 'cross': c, 'mut': m, 'name': f"real_valued_{s}_{c}_{m}"}
        for s in ['stochastic_universal', 'tournament', 'roulette_wheel']
        for c in ['sbx', 'blx_alpha', 'flat']
        for m in ['gaussian', 'polynomial', 'uniform']
    ])
    
    print(f"Total combinations per instance: {len(combinations)}")
    
    # Initialize global checkpoint
    os.makedirs(results_dir, exist_ok=True)
    if instance_name:
        checkpoint_file = os.path.join(results_dir, f'checkpoint_{instances[0]}.json')
    else:
        checkpoint_file = os.path.join(results_dir, 'checkpoint.json')
    checkpoint = SimpleCheckpoint(checkpoint_file)
    
    for filename in instances:
        file_path = os.path.join(uchoa_base_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: Instance file {file_path} not found. Skipping.")
            continue
            
        print(f"\nProcessing instance: {filename}")
        
        # Parse Instance
        try:
            instance = UchoaParser.parse(file_path)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue
            
        # Check completion
        if checkpoint.is_instance_complete(instance.name):
            print(f"Instance {instance.name} already completed. Skipping.")
            continue
            
        # Initialize Evaluator
        evaluator = SimpleEvaluator(
            instance=instance,
            n_runs=N_RUNS,
            output_dir=results_dir,
            checkpoint=checkpoint,
            n_jobs=N_JOBS
        )
        
        # Run combinations
        for config_dict in combinations:
            method_name = config_dict['name']
            
            ga_config = GAConfig(
                population_size=POPULATION_SIZE,
                max_generations=MAX_GENERATIONS,
                crossover_rate=CROSSOVER_RATE,
                mutation_rate=MUTATION_RATE,
                elitism_rate=0.0,
                tournament_size=TOURNAMENT_SIZE,
                selection_method=config_dict['sel'],
                crossover_method=config_dict['cross'],
                mutation_method=config_dict['mut'],
                representation=config_dict['rep'],
                verbose=False
            )
            
            try:
                evaluator.evaluate_combination(method_name, ga_config)
            except Exception as e:
                print(f"Error evaluating {method_name} on {instance.name}: {e}")
                import traceback
                traceback.print_exc()

        # Mark instance complete
        checkpoint.mark_instance_complete(instance.name)
        print(f"Completed instance {instance.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA comparison on Uchoa instances")
    parser.add_argument("--sampled_csv", type=str, default="data/sampled_instances.csv", help="Path to sampled_instances.csv")
    parser.add_argument("--instances_dir", type=str, default="data/Uchoa", help="Path to folder containing Uchoa instances files (.vrp)")
    parser.add_argument("--results_dir", type=str, default="results_uchoa", help="Output directory for results")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--instance", type=str, default=None, help="Process only this specific instance (e.g. X-n819-k171)")
    
    args = parser.parse_args()
    
    run_uchoa_experiment(
        sampled_csv_path=args.sampled_csv,
        uchoa_base_path=args.instances_dir,
        results_dir=args.results_dir,
        config_file=args.config,
        instance_name=args.instance
    )
