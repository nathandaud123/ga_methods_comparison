"""
Main script for GA Method Comparison Study (Simplified Version)
- All combinations of representation, selection, crossover, mutation
- Each combination runs 5 times
- Saves generation history for each run
- Calculates and saves average per generation
- Tests all combinations on all instances
"""

import argparse
import yaml
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from src.data.solomon_parser import SolomonParser
from src.ga.genetic_algorithm import GAConfig
from src.evaluation.simple_evaluator import SimpleEvaluator
from src.evaluation.parallel_executor import FileLock


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_output_dirs(config: dict):
    """Create output directories"""
    output_config = config.get('output', {})
    results_dir = output_config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)


def _save_result_to_json(results_file: str, method_name: str, result: dict):
    """
    Thread-safe function to save a single method result to JSON file.
    Loads existing results, adds new result, and saves back.
    """
    lock_file = results_file + '.lock'
    
    try:
        with FileLock(lock_file):
            # Load existing results
            existing_results = {}
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        existing_results = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_results = {}
            
            # Add/update the new result
            existing_results[method_name] = result
            
            # Save back to file
            with open(results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            
            print(f"  ✓ Saved result for {method_name} to {results_file}")
    except Exception as e:
        print(f"  Warning: Could not save result for {method_name}: {e}")


def generate_method_combinations(config: dict) -> List[Tuple[str, GAConfig]]:
    """
    Generate all method combinations to test
    
    Returns:
        List of (method_name, GAConfig) tuples
    """
    combinations = []
    
    representations = config.get('representations', ['permutation'])
    selection_methods = config.get('selection_methods', ['tournament'])
    crossover_configs = config.get('crossover_methods', {})
    mutation_configs = config.get('mutation_methods', {})
    
    ga_base = config.get('ga', {})
    
    for rep in representations:
        crossovers = crossover_configs.get(rep, [])
        mutations = mutation_configs.get(rep, [])
        
        for sel in selection_methods:
            for cross in crossovers:
                for mut in mutations:
                    method_name = f"{rep}_{sel}_{cross}_{mut}"
                    
                    ga_config = GAConfig(
                        population_size=ga_base.get('population_size', 100),
                        max_generations=ga_base.get('max_generations', 500),
                        crossover_rate=ga_base.get('crossover_rate', 0.8),
                        mutation_rate=ga_base.get('mutation_rate', 0.1),
                        elitism_rate=0.0,  # No elitism
                        tournament_size=ga_base.get('tournament_size', 3),
                        selection_method=sel,
                        crossover_method=cross,
                        mutation_method=mut,
                        representation=rep,
                        verbose=config.get('output', {}).get('verbose', False)
                    )
                    
                    combinations.append((method_name, ga_config))
    
    return combinations


def discover_instances(base_path: str) -> List[str]:
    """Discover all CSV instances from subfolders"""
    instances = []
    
    for subfolder in ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']:
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.exists(subfolder_path):
            csv_files = [f for f in os.listdir(subfolder_path) 
                        if f.endswith(('.csv', '.CSV'))]
            for csv_file in sorted(csv_files):
                instances.append(os.path.join(subfolder_path, csv_file))
    
    return instances


def run_comparison_study(config: dict, instance_name: str = None):
    """
    Run complete comparison study
    
    Args:
        config: Configuration dictionary
        instance_name: Optional instance name to process (if None, process all instances)
    """
    print("="*80)
    print("Genetic Algorithm Method Comparison Study (Simplified)")
    print("="*80)
    if instance_name:
        print(f"Processing single instance: {instance_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Create output directories
    create_output_dirs(config)
    
    # Initialize checkpoint
    from src.evaluation.simple_checkpoint import SimpleCheckpoint
    output_config = config.get('output', {})
    results_dir = output_config.get('results_dir', 'results')
    
    # Use instance-specific checkpoint if instance_name is provided
    if instance_name:
        checkpoint_file = os.path.join(results_dir, f'checkpoint_{instance_name}.json')
        print(f"Using instance-specific checkpoint: {checkpoint_file}")
    else:
        checkpoint_file = os.path.join(results_dir, 'checkpoint.json')
    
    checkpoint = SimpleCheckpoint(checkpoint_file)
    
    # Get parallel config
    n_jobs = config.get('evaluation', {}).get('n_jobs', None)
    if n_jobs is None:
        n_jobs = 1  # Sequential by default
    elif n_jobs == 0:
        import multiprocessing as mp
        n_jobs = max(1, mp.cpu_count() - 1)  # Auto: use all cores except one
    
    if n_jobs > 1:
        print(f"\nParallel execution enabled: {n_jobs} workers")
    else:
        print(f"\nSequential execution (n_jobs=1)")
    
    # Load dataset
    dataset_config = config.get('dataset', {})
    base_path = dataset_config.get('base_path', 'data/solomon')
    
    # Discover instances
    print("\nDiscovering instances...")
    all_instances = discover_instances(base_path)
    if not all_instances:
        print("No instances found!")
        return
    
    # Filter by instance_name if provided
    if instance_name:
        # Find instance by name (case-insensitive, with or without extension)
        target_name = instance_name.upper()
        instances = []
        for inst_path in all_instances:
            inst_file_name = os.path.basename(inst_path)
            inst_name = os.path.splitext(inst_file_name)[0].upper()
            if inst_name == target_name:
                instances.append(inst_path)
                break
        
        if not instances:
            print(f"ERROR: Instance '{instance_name}' not found!")
            print(f"Available instances: {[os.path.splitext(os.path.basename(p))[0] for p in all_instances[:10]]}...")
            return
        print(f"Processing single instance: {instance_name}")
    else:
        instances = all_instances
        print(f"Found {len(instances)} instances to process")
    
    # Generate method combinations
    print("\nGenerating method combinations...")
    method_combinations = generate_method_combinations(config)
    print(f"Total combinations: {len(method_combinations)}")
    
    # Get evaluation config
    n_runs = config.get('evaluation', {}).get('n_runs', 5)
    
    # Process each instance
    all_results = {}
    
    for instance_path in instances:
        instance_path = os.path.normpath(instance_path)
        
        if not os.path.exists(instance_path):
            print(f"Warning: Instance file not found: {instance_path}")
            continue
        
        instance_file_name = os.path.basename(instance_path)
        instance_name = os.path.splitext(instance_file_name)[0]
        
        print(f"\n{'='*80}")
        print(f"Processing instance: {instance_name}")
        print(f"File: {instance_file_name}")
        print(f"{'='*80}")
        
        # Parse instance
        parser = SolomonParser()
        instance = parser.parse(instance_path)
        
        # Check if instance already complete
        if checkpoint.is_instance_complete(instance.name):
            print(f"\n{'='*80}")
            print(f"Skipping instance: {instance.name} (already completed)")
            print(f"{'='*80}")
            # Try to load existing results
            results_file = os.path.join(results_dir, instance.name, f"{instance.name}_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    instance_results = json.load(f)
                    all_results[instance.name] = instance_results
                    print(f"  Loaded existing results from {results_file}")
            continue
        
        print(f"Instance: {instance.name}")
        print(f"Type: {instance.type}")
        print(f"Customers: {instance.num_customers}")
        print(f"Capacity: {instance.vehicle_capacity}")
        
        # Get completed methods for this instance
        completed_methods = checkpoint.get_completed_methods(instance.name)
        if completed_methods:
            print(f"Resuming: {len(completed_methods)}/{len(method_combinations)} methods already completed")
        
        # Initialize evaluator
        evaluator = SimpleEvaluator(instance, n_runs=n_runs, output_dir=results_dir,
                                   checkpoint=checkpoint, n_jobs=n_jobs)
        
        # Results file path
        results_file = os.path.join(results_dir, instance.name, f"{instance.name}_results.json")
        
        # Try to load existing results first
        instance_results = {}
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    instance_results = json.load(f)
            except:
                pass
        
        for method_name, ga_config in method_combinations:
            # Skip if already completed and have results
            if checkpoint.is_method_complete(instance.name, method_name):
                if method_name in instance_results:
                    continue  # Already have results
            
            try:
                result = evaluator.evaluate_combination(method_name, ga_config)
                if result:  # None if already completed
                    # Save immediately after each method completes
                    _save_result_to_json(results_file, method_name, result)
                    instance_results[method_name] = result
            except Exception as e:
                print(f"  ERROR evaluating {method_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Check if all methods complete
        completed_count = len([m for m in method_combinations 
                              if checkpoint.is_method_complete(instance.name, m[0])])
        if completed_count >= len(method_combinations):
            checkpoint.mark_instance_complete(instance.name)
            print(f"\n✓ Instance {instance.name} completed: {completed_count}/{len(method_combinations)} methods")
        
        # Final summary (results already saved incrementally)
        print(f"\nResults file: {results_file}")
        all_results[instance.name] = instance_results
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Processed {len(all_results)} instances")
    print(f"Total combinations tested: {len(method_combinations)}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GA Method Comparison Study (Simplified)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--instance', type=str, default=None,
                       help='Process only this instance (e.g., C101, C102). If not specified, process all instances.')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run comparison study
    run_comparison_study(config, instance_name=args.instance)


if __name__ == '__main__':
    main()

