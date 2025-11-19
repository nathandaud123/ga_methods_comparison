"""
Test script untuk verify generate_results_from_checkpoint bekerja
"""
from src.evaluation.checkpoint import CheckpointManager
from main import generate_results_from_checkpoint
import os

# Test generate results untuk C101
print("="*80)
print("TEST: Generate Results from Checkpoint")
print("="*80)

cm = CheckpointManager()
instance_name = "C101"
results_dir = f"results/experiments/{instance_name}"

# Check if instance is complete
is_complete = cm.is_instance_complete(instance_name)
print(f"\nInstance {instance_name} complete: {is_complete}")

if is_complete:
    completed_methods = cm.state.get('completed_methods', {}).get(instance_name, [])
    print(f"Completed methods: {len(completed_methods)}")
    
    # Generate results
    print(f"\nGenerating results from checkpoint/convergence files...")
    results = generate_results_from_checkpoint(cm, instance_name, results_dir)
    
    print(f"\n✅ Generated {len(results)} methods")
    if results:
        print(f"\nSample methods:")
        for i, (method, metrics) in enumerate(list(results.items())[:3]):
            print(f"  {i+1}. {method}")
            print(f"     Mean Fitness: {metrics['mean_fitness']:.2f} ± {metrics['std_fitness']:.2f}")
            print(f"     Best: {metrics['best_fitness']:.2f}, Worst: {metrics['worst_fitness']:.2f}")
        
        # Check if results file was created
        results_file = os.path.join(results_dir, f"{instance_name}_results.json")
        if os.path.exists(results_file):
            print(f"\n✅ Results file created: {results_file}")
        else:
            print(f"\n❌ Results file NOT created: {results_file}")
    else:
        print("\n❌ No results generated!")
else:
    print(f"\nInstance {instance_name} is not complete yet")

print("\n" + "="*80)

