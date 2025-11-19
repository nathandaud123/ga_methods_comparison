"""
Script untuk clean checkpoint C102 yang memiliki run numbers tidak sequential
dan fitness yang sama semua (dari run sebelum random seed di-fix)
"""
import json
import os
import sys

def clean_c102_checkpoint(checkpoint_file="results/checkpoint.json"):
    """Clean C102 checkpoint - remove runs with non-sequential numbers or identical fitness"""
    
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
        return
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    print("="*80)
    print("CLEANING C102 CHECKPOINT")
    print("="*80)
    
    instance_name = "C102"
    
    # Backup checkpoint
    backup_file = checkpoint_file + '.backup'
    if os.path.exists(checkpoint_file):
        import shutil
        shutil.copy2(checkpoint_file, backup_file)
        print(f"\n✅ Backup created: {backup_file}")
    
    # Check if C102 exists in partial_progress
    if instance_name in checkpoint.get('partial_progress', {}):
        partial_progress = checkpoint['partial_progress'][instance_name]
        
        print(f"\nFound C102 in partial_progress:")
        print(f"  Methods: {list(partial_progress.keys())}")
        
        # Clean each method
        for method_name, method_data in list(partial_progress.items()):
            if isinstance(method_data, dict):
                # Check for non-sequential run numbers or identical fitness
                run_keys = [k for k in method_data.keys() if k.isdigit()]
                
                if run_keys:
                    # Check if run numbers are sequential (1-5)
                    run_numbers = sorted([int(k) for k in run_keys])
                    expected_runs = list(range(1, 6))  # 1-5
                    
                    # Check if fitness values are all identical
                    fitness_values = []
                    for run_key in run_keys:
                        if isinstance(method_data[run_key], dict):
                            fitness_values.append(method_data[run_key].get('fitness'))
                    
                    all_same_fitness = len(set(f for f in fitness_values if f is not None)) <= 1
                    non_sequential = run_numbers != expected_runs[:len(run_numbers)]
                    
                    if non_sequential or all_same_fitness:
                        print(f"\n  ❌ {method_name}:")
                        if non_sequential:
                            print(f"     Non-sequential run numbers: {run_numbers} (expected 1-5)")
                        if all_same_fitness:
                            print(f"     All fitness values identical: {fitness_values[0] if fitness_values else 'N/A'}")
                        
                        # Remove this method's progress
                        del checkpoint['partial_progress'][instance_name][method_name]
                        print(f"     ✅ Removed from checkpoint")
                    else:
                        print(f"\n  ✅ {method_name}: OK (sequential runs, different fitness)")
        
        # Remove C102 if no methods left
        if not checkpoint['partial_progress'][instance_name]:
            del checkpoint['partial_progress'][instance_name]
            print(f"\n✅ Removed C102 from partial_progress (no valid methods)")
    
    # Remove from completed_methods if exists
    if instance_name in checkpoint.get('completed_methods', {}):
        del checkpoint['completed_methods'][instance_name]
        print(f"✅ Removed C102 from completed_methods")
    
    # Remove from completed_instances if exists
    if instance_name in checkpoint.get('completed_instances', []):
        checkpoint['completed_instances'].remove(instance_name)
        print(f"✅ Removed C102 from completed_instances")
    
    # Save cleaned checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ Checkpoint cleaned and saved!")
    print(f"{'='*80}")

if __name__ == "__main__":
    clean_c102_checkpoint()

