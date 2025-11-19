"""
Script untuk fix checkpoint yang salah mark instance sebagai complete
"""
import json
import os
import sys

def fix_checkpoint(checkpoint_file="results/checkpoint.json"):
    """Remove instances from completed_instances if they don't have completed methods"""
    
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
        return
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    print("="*80)
    print("FIXING CHECKPOINT")
    print("="*80)
    
    completed_instances = checkpoint.get('completed_instances', [])
    completed_methods = checkpoint.get('completed_methods', {})
    partial_progress = checkpoint.get('partial_progress', {})
    
    print(f"\nCurrent completed_instances: {completed_instances}")
    print(f"Instances with completed_methods: {list(completed_methods.keys())}")
    
    # Find instances that are marked complete but have no methods
    invalid_instances = []
    for instance in completed_instances:
        methods = completed_methods.get(instance, [])
        progress = partial_progress.get(instance, {})
        
        # Check if instance has any actual progress
        has_progress = len(methods) > 0 or len(progress) > 0
        
        if not has_progress:
            invalid_instances.append(instance)
            print(f"\n❌ {instance}: Marked complete but has NO methods or progress")
        else:
            print(f"\n✅ {instance}: Has {len(methods)} completed methods")
    
    if invalid_instances:
        print(f"\n{'='*80}")
        print(f"Removing {len(invalid_instances)} invalid instances from completed_instances:")
        for inst in invalid_instances:
            print(f"  - {inst}")
        
        # Remove invalid instances
        checkpoint['completed_instances'] = [inst for inst in completed_instances if inst not in invalid_instances]
        
        # Save fixed checkpoint
        backup_file = checkpoint_file + '.backup'
        if os.path.exists(checkpoint_file):
            import shutil
            shutil.copy2(checkpoint_file, backup_file)
            print(f"\n✅ Backup created: {backup_file}")
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"\n✅ Fixed checkpoint saved!")
        print(f"   Removed: {invalid_instances}")
        print(f"   Remaining completed_instances: {checkpoint['completed_instances']}")
    else:
        print(f"\n✅ All completed instances are valid!")
    
    print("="*80)

if __name__ == "__main__":
    fix_checkpoint()

