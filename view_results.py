"""
Script untuk melihat hasil experiment
"""
import json
import os
import pandas as pd
from pathlib import Path

def view_checkpoint_results(instance_name="C101"):
    """View results from checkpoint"""
    checkpoint_file = "results/checkpoint.json"
    
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
        return
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT RESULTS FOR {instance_name}")
    print(f"{'='*80}\n")
    
    # Check completed methods
    completed_methods = checkpoint.get('completed_methods', {}).get(instance_name, [])
    print(f"Completed methods: {len(completed_methods)}")
    
    # Check partial progress
    partial_progress = checkpoint.get('partial_progress', {}).get(instance_name, {})
    
    print(f"\n{'Method Name':<50} {'Status':<15} {'Runs':<10}")
    print("-" * 80)
    
    for method_name in completed_methods:
        method_progress = partial_progress.get(method_name, {})
        if isinstance(method_progress, dict) and 'status' in method_progress:
            status = method_progress.get('status', 'unknown')
            runs = method_progress.get('completed_runs', 0)
        else:
            # Old format - count runs
            runs = len([k for k in method_progress.keys() if k.isdigit()]) if method_progress else 0
            status = 'complete' if runs > 0 else 'unknown'
        
        print(f"{method_name:<50} {status:<15} {runs}/5")
    
    # Show detailed results for methods with data
    print(f"\n{'='*80}")
    print("DETAILED RESULTS (from checkpoint data)")
    print(f"{'='*80}\n")
    
    for method_name in completed_methods[:5]:  # Show first 5
        method_progress = partial_progress.get(method_name, {})
        if isinstance(method_progress, dict) and 'status' in method_progress:
            continue  # Skip if only status
        
        runs_data = {k: v for k, v in method_progress.items() if k.isdigit()}
        if runs_data:
            print(f"\n{method_name}:")
            for run_num in sorted(runs_data.keys(), key=int):
                run_data = runs_data[run_num]
                print(f"  Run {run_num}: Fitness={run_data.get('fitness', 'N/A'):.2f}, "
                      f"Runtime={run_data.get('runtime', 'N/A'):.2f}s")

def view_convergence_results(instance_name="C101", method_name=None):
    """View convergence history from CSV"""
    convergence_dir = f"results/convergence/{instance_name}"
    
    if not os.path.exists(convergence_dir):
        print(f"Convergence directory not found: {convergence_dir}")
        return
    
    csv_files = list(Path(convergence_dir).glob("*_convergence.csv"))
    
    if method_name:
        csv_files = [f for f in csv_files if method_name in f.name]
    
    if not csv_files:
        print(f"No convergence files found for {instance_name}")
        return
    
    print(f"\n{'='*80}")
    print(f"CONVERGENCE HISTORY FOR {instance_name}")
    print(f"{'='*80}\n")
    
    for csv_file in csv_files[:3]:  # Show first 3
        df = pd.read_csv(csv_file)
        method = csv_file.stem.replace('_convergence', '')
        print(f"\n{method}:")
        print(f"  Generations: {len(df)}")
        if 'fitness_mean' in df.columns:
            final_fitness = df['fitness_mean'].iloc[-1]
            best_fitness = df['fitness_min'].min()
            print(f"  Final Mean Fitness: {final_fitness:.2f}")
            print(f"  Best Fitness: {best_fitness:.2f}")

if __name__ == "__main__":
    import sys
    
    instance = sys.argv[1] if len(sys.argv) > 1 else "C101"
    method = sys.argv[2] if len(sys.argv) > 2 else None
    
    view_checkpoint_results(instance)
    view_convergence_results(instance, method)
    
    print(f"\n{'='*80}")
    print("FILE LOCATIONS:")
    print(f"{'='*80}")
    print(f"Checkpoint: results/checkpoint.json")
    print(f"Convergence CSV: results/convergence/{instance}/")
    print(f"Results JSON: results/experiments/{instance}/{instance}_results.json")
    print(f"Plots: results/plots/{instance}/")
    print(f"Routes: results/routes/{instance}/")

