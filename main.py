"""
Main script for GA Method Comparison Study
"""

import argparse
import yaml
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple

from src.data.solomon_parser import SolomonParser
from src.ga.genetic_algorithm import GAConfig, GeneticAlgorithm
from src.evaluation.evaluator import ExperimentEvaluator
from src.evaluation.metrics import ComparisonMetrics
from src.evaluation.checkpoint import CheckpointManager
from src.evaluation.task_distributor import TaskDistributor
from src.visualization.route_plotter import RoutePlotter
from src.visualization.result_plotter import ResultPlotter
from src.tuning.optuna_tuner import OptunaTuner


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_output_dirs(config: dict):
    """Create output directories"""
    output_config = config.get('output', {})
    dirs = [
        output_config.get('results_dir', 'results'),
        output_config.get('experiments_dir', 'results/experiments'),
        output_config.get('plots_dir', 'results/plots'),
        output_config.get('routes_dir', 'results/routes'),
        output_config.get('convergence_dir', 'results/convergence'),
        output_config.get('tuning_dir', 'results/tuning'),
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


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
                        elitism_rate=ga_base.get('elitism_rate', 0.1),
                        tournament_size=ga_base.get('tournament_size', 3),
                        selection_method=sel,
                        crossover_method=cross,
                        mutation_method=mut,
                        representation=rep,
                        verbose=config.get('output', {}).get('verbose', True)
                    )
                    
                    combinations.append((method_name, ga_config))
    
    return combinations


def run_comparison_study(config: dict):
    """Run complete comparison study"""
    print("="*80)
    print("Genetic Algorithm Method Comparison Study")
    print("="*80)
    
    # Create output directories
    create_output_dirs(config)
    
    # Get output config (needed later for summary)
    output_config = config.get('output', {})
    
    # Initialize checkpoint manager
    checkpoint_file = output_config.get('checkpoint_file', 'results/checkpoint.json')
    checkpoint_manager = CheckpointManager(checkpoint_file)
    
    # Initialize task distributor for multi-device execution
    device_config = config.get('device', {})
    device_id = device_config.get('device_id', None)
    total_devices = device_config.get('total_devices', 1)
    task_distributor = None
    
    if device_id and total_devices > 1:
        task_distributor = TaskDistributor(device_id, total_devices, checkpoint_file)
        print(f"\n{'='*80}")
        print(f"MULTI-DEVICE MODE: Device {device_id} of {total_devices}")
        print(f"{'='*80}")
        summary = task_distributor.get_summary()
        print(f"Assigned instances: {summary['assigned_instances']}")
        print(f"Instances: {summary['instances'][:5]}..." if len(summary['instances']) > 5 else f"Instances: {summary['instances']}")
        print(f"{'='*80}\n")
    
    # Show checkpoint status
    if checkpoint_manager.state['completed_instances'] or checkpoint_manager.state['completed_methods']:
        print(f"\n{'='*80}")
        print("CHECKPOINT STATUS")
        print(f"{'='*80}")
        summary = checkpoint_manager.get_progress_summary()
        print(f"Completed instances: {summary['completed_instances']}")
        print(f"Instances with progress: {summary['instances_with_progress']}")
        for inst, num_methods in summary['completed_methods_by_instance'].items():
            print(f"  - {inst}: {num_methods} methods completed")
        print(f"{'='*80}\n")
    
    # Load dataset
    dataset_config = config.get('dataset', {})
    base_path = dataset_config.get('base_path', 'data/solomon')
    instances = dataset_config.get('instances', [])
    
    # If instances specified as simple names, try to find in subfolders
    if instances:
        # Expand instances to include subfolder paths
        expanded_instances = []
        for instance_name in instances:
            instance_name_clean = instance_name.replace('.txt', '').replace('.csv', '')
            
            # Try different possible locations
            possible_paths = [
                os.path.join(base_path, instance_name),  # Direct path
                os.path.join(base_path, instance_name_clean + '.csv'),  # CSV in root
                os.path.join(base_path, instance_name_clean + '.txt'),  # TXT in root
            ]
            
            # Try subfolder structure (C1/C101.csv, R1/R101.csv, etc)
            if instance_name_clean[0] in ['C', 'R']:
                folder_num = instance_name_clean[1] if len(instance_name_clean) > 1 and instance_name_clean[1].isdigit() else '1'
                folder_name = instance_name_clean[0] + folder_num
                possible_paths.extend([
                    os.path.join(base_path, folder_name, instance_name_clean + '.csv'),
                    os.path.join(base_path, folder_name, instance_name_clean + '.txt'),
                ])
            
            # Find first existing path
            found_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path:
                expanded_instances.append(found_path)
            else:
                print(f"Warning: Instance file not found: {instance_name}")
                print(f"  Tried paths: {possible_paths[:3]}...")
        
        instances = expanded_instances if expanded_instances else None
    
    # If no instances specified, try to auto-discover from subfolders
    if not instances:
        print("No instances specified. Auto-discovering ALL datasets from subfolders...")
        instances = []
        
        # Look for CSV files in subfolders
        for subfolder in ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']:
            subfolder_path = os.path.join(base_path, subfolder)
            if os.path.exists(subfolder_path):
                csv_files = [f for f in os.listdir(subfolder_path) 
                            if f.endswith(('.csv', '.CSV'))]
                # Include ALL CSV files (no limit)
                for csv_file in sorted(csv_files):
                    instances.append(os.path.join(subfolder_path, csv_file))
        
        if not instances:
            print("No instances found. Using default test instance.")
            instances = ['test_instance.txt']
        else:
            print(f"Found {len(instances)} instances to process")
    
    # Filter instances if multi-device mode
    if task_distributor:
        original_count = len(instances)
        instance_names = [os.path.basename(os.path.normpath(ip)).split('.')[0] for ip in instances]
        filtered_names = task_distributor.filter_instances(instance_names)
        # Filter instance_paths to match filtered names
        instances = [ip for ip in instances 
                    if os.path.basename(os.path.normpath(ip)).split('.')[0] in filtered_names]
        print(f"\n{'='*80}")
        print(f"DEVICE {device_id}: Filtered {len(instances)}/{original_count} instances")
        print(f"{'='*80}\n")
    
    # Process each instance
    all_results = {}
    processed_instances = 0
    
    for instance_path in instances:
        # Normalize path for cross-platform compatibility
        instance_path = os.path.normpath(instance_path)
        
        if not os.path.exists(instance_path):
            print(f"Warning: Instance file not found: {instance_path}")
            continue
        
        instance_file_name = os.path.basename(instance_path)
        
        # Parse instance
        parser = SolomonParser()
        instance = parser.parse(instance_path)
        
        # Check if instance is already complete (from any device)
        if checkpoint_manager.is_instance_complete(instance.name):
            print(f"\n{'='*80}")
            print(f"Skipping instance: {instance.name} (already completed by another device)")
            print(f"{'='*80}")
            # Load existing results instead of recomputing
            results_file = os.path.join(
                output_config.get('experiments_dir', 'results/experiments'),
                instance.name,
                f"{instance.name}_results.json"
            )
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results_dict = json.load(f)
                    # Convert back to ComparisonMetrics (simplified)
                    all_results[instance.name] = results_dict
            continue
        
        # Additional safety: Check if this instance is assigned to this device
        if task_distributor:
            assigned_instances = task_distributor.get_assigned_instances()
            if instance.name not in assigned_instances:
                print(f"\n{'='*80}")
                print(f"Skipping instance: {instance.name} (not assigned to this device)")
                print(f"{'='*80}")
                continue
        
        print(f"\n{'='*80}")
        print(f"Processing instance: {instance_file_name}")
        print(f"Path: {instance_path}")
        print(f"{'='*80}")
        print(f"Instance: {instance.name}")
        print(f"Type: {instance.type}")
        print(f"Customers: {instance.num_customers}")
        print(f"Capacity: {instance.vehicle_capacity}")
        
        # Generate method combinations
        method_combinations = generate_method_combinations(config)
        print(f"\nTotal method combinations to test: {len(method_combinations)}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        import sys
        sys.stdout.flush()  # Force flush output
        
        # Setup directories for this instance
        instance_results_dir = os.path.join(output_config.get('experiments_dir', 'results/experiments'), instance.name)
        instance_plots_dir = os.path.join(output_config.get('plots_dir', 'results/plots'), instance.name)
        instance_routes_dir = os.path.join(output_config.get('routes_dir', 'results/routes'), instance.name)
        instance_convergence_dir = os.path.join(output_config.get('convergence_dir', 'results/convergence'), instance.name)
        instance_tuning_dir = os.path.join(output_config.get('tuning_dir', 'results/tuning'), instance.name)
        
        os.makedirs(instance_results_dir, exist_ok=True)
        os.makedirs(instance_plots_dir, exist_ok=True)
        os.makedirs(instance_routes_dir, exist_ok=True)
        os.makedirs(instance_convergence_dir, exist_ok=True)
        os.makedirs(instance_tuning_dir, exist_ok=True)
        
        # Optuna tuning (if enabled)
        optuna_config = config.get('optuna', {})
        if optuna_config.get('enabled', False):
            print("\nRunning Optuna parameter tuning...")
            # Tune for first method combination as example
            if method_combinations:
                method_name, base_config = method_combinations[0]
                tuner = OptunaTuner(
                    instance,
                    representation=base_config.representation,
                    selection_method=base_config.selection_method,
                    crossover_method=base_config.crossover_method,
                    mutation_method=base_config.mutation_method,
                    n_trials=optuna_config.get('n_trials', 50),
                    timeout=optuna_config.get('timeout', 3600),
                    save_history=optuna_config.get('save_history', True),
                    history_dir=instance_tuning_dir
                )
                tuning_result = tuner.tune(
                    study_name=f"{instance.name}_tuning",
                    instance_name=instance.name
                )
                # Update config with best parameters
                best_config = tuner.get_best_config(tuning_result['best_params'])
                method_combinations[0] = (method_name, best_config)
        
        # Run experiments
        eval_config = config.get('evaluation', {})
        save_history = eval_config.get('save_convergence_history', True)
        parallel = eval_config.get('parallel', False)
        n_jobs = eval_config.get('n_jobs', None)
        
        evaluator = ExperimentEvaluator(
            instance,
            n_runs=eval_config.get('n_runs', 10),
            save_history=save_history,
            history_dir=instance_convergence_dir,
            checkpoint_manager=checkpoint_manager,
            parallel=parallel,
            n_jobs=n_jobs
        )
        
        comparison_results = evaluator.compare_methods(method_combinations)
        
        # Cleanup parallel executor resources
        evaluator.cleanup()
        
        # Load existing results if available (for summary)
        results_file = os.path.join(instance_results_dir, f"{instance.name}_results.json")
        existing_results = {}
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    existing_results = json.load(f)
            except:
                pass
        
        # Merge with new results
        if comparison_results:
            # Update with new results
            for method, metrics in comparison_results.items():
                existing_results[method] = {
                    'mean_fitness': float(metrics.mean_fitness),
                    'std_fitness': float(metrics.std_fitness),
                    'mean_runtime': float(metrics.mean_runtime),
                    'std_runtime': float(metrics.std_runtime),
                    'best_fitness': float(metrics.best_fitness),
                    'worst_fitness': float(metrics.worst_fitness),
                }
            all_results[instance.name] = comparison_results
            processed_instances += 1
        elif existing_results:
            # No new results but have existing - use those for summary
            all_results[instance.name] = existing_results
        
        # Mark instance as complete if all methods are done
        total_methods = len(method_combinations)
        completed_methods = checkpoint_manager.state['completed_methods'].get(instance.name, [])
        if len(completed_methods) >= total_methods:
            checkpoint_manager.mark_instance_complete(instance.name)
            print(f"\n[OK] Instance {instance.name} completed: {len(completed_methods)}/{total_methods} methods")
        
        # Save results (update with any new data)
        if comparison_results or existing_results:
            # Merge results
            results_dict = existing_results.copy() if existing_results else {}
            
            # Add/update with new results
            if comparison_results:
                for method, metrics in comparison_results.items():
                    results_dict[method] = {
                        'mean_fitness': float(metrics.mean_fitness),
                        'std_fitness': float(metrics.std_fitness),
                        'mean_runtime': float(metrics.mean_runtime),
                        'std_runtime': float(metrics.std_runtime),
                        'best_fitness': float(metrics.best_fitness),
                        'worst_fitness': float(metrics.worst_fitness),
                    }
            
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"\nResults saved to {results_file} ({len(results_dict)} methods)")
        
        # Visualizations
        if config.get('evaluation', {}).get('save_plots', True):
            plotter = ResultPlotter()
            
            # Comparison bar chart
            plotter.plot_comparison_bar(
                comparison_results,
                metric='mean_fitness',
                save_path=os.path.join(instance_plots_dir, f"{instance.name}_fitness_comparison.png"),
                show=False
            )
            
            plotter.plot_comparison_bar(
                comparison_results,
                metric='mean_runtime',
                save_path=os.path.join(instance_plots_dir, f"{instance.name}_runtime_comparison.png"),
                show=False
            )
            
            # Heatmap
            plotter.plot_comparison_heatmap(
                comparison_results,
                save_path=os.path.join(instance_plots_dir, f"{instance.name}_heatmap.png"),
                show=False
            )
        
        # Visualize best route
        if config.get('evaluation', {}).get('save_routes', True):
            # Find best method
            best_method = min(comparison_results.items(), key=lambda x: x[1].mean_fitness)[0]
            best_config = next(config for name, config in method_combinations if name == best_method)
            
            # Run once more to get best route
            ga = GeneticAlgorithm(instance, best_config)
            result = ga.run()
            
            route_plotter = RoutePlotter(instance)
            route_plotter.plot_routes(
                result.best_routes,
                title=f"Best Solution - {instance.name} ({best_method})",
                save_path=os.path.join(instance_routes_dir, f"{instance.name}_best_route.png"),
                show=False
            )
    
    # Summary across all instances
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL INSTANCES")
    print(f"{'='*80}")
    
    if not all_results:
        print("\nNo results to summarize. Please ensure:")
        print("1. Solomon benchmark datasets are downloaded")
        print("2. Dataset files are placed in data/solomon/ directory")
        print("3. Instance names in config.yaml match the actual file names")
        print("\nYou can use test_instance.txt for testing:")
        print("  python run_experiment.py")
        return
    
    # Aggregate results
    summary_results = {}
    for instance_name, results in all_results.items():
        for method_name, metrics in results.items():
            if method_name not in summary_results:
                summary_results[method_name] = {
                    'fitnesses': [],
                    'runtimes': [],
                }
            summary_results[method_name]['fitnesses'].append(metrics.mean_fitness)
            summary_results[method_name]['runtimes'].append(metrics.mean_runtime)
    
    # Calculate averages
    summary_metrics = {}
    for method_name, data in summary_results.items():
        summary_metrics[method_name] = ComparisonMetrics(
            method_name=method_name,
            mean_fitness=np.mean(data['fitnesses']),
            std_fitness=np.std(data['fitnesses']),
            mean_runtime=np.mean(data['runtimes']),
            std_runtime=np.std(data['runtimes']),
            mean_convergence=0.0,
            std_convergence=0.0,
            mean_diversity=0.0,
            std_diversity=0.0,
            best_fitness=np.min(data['fitnesses']),
            worst_fitness=np.max(data['fitnesses']),
            success_rate=1.0
        )
    
    # Print summary
    print("\nAverage Performance Across All Instances:")
    print(f"{'Method':<40} {'Mean Fitness':<15} {'Mean Runtime':<15}")
    print("-" * 70)
    for method_name, metrics in sorted(summary_metrics.items(), 
                                      key=lambda x: x[1].mean_fitness):
        print(f"{method_name:<40} {metrics.mean_fitness:>12.2f} {metrics.mean_runtime:>12.2f}s")
    
    # Save summary
    summary_file = os.path.join(output_config.get('results_dir', 'results'), 'summary.json')
    summary_dict = {
        method: {
            'mean_fitness': float(m.mean_fitness),
            'std_fitness': float(m.std_fitness),
            'mean_runtime': float(m.mean_runtime),
            'std_runtime': float(m.std_runtime),
        }
        for method, m in summary_metrics.items()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    print("\n" + "="*80)
    print("Comparison study completed!")
    print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GA Method Comparison Study')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run comparison study
    run_comparison_study(config)


if __name__ == '__main__':
    main()

