"""
Verify that task distribution ensures no overlap between devices
"""

import yaml
from src.evaluation.task_distributor import TaskDistributor


def verify_no_overlap():
    """Verify that devices don't work on same instances"""
    print("="*80)
    print("VERIFYING NO OVERLAP BETWEEN DEVICES")
    print("="*80)
    print()
    
    # Load all instances
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config.get('dataset', {})
    base_path = dataset_config.get('base_path', 'data/solomon')
    
    import os
    from pathlib import Path
    
    all_instances = []
    for subfolder in ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']:
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.exists(subfolder_path):
            csv_files = list(Path(subfolder_path).glob('*.csv'))
            for csv_file in csv_files:
                instance_name = csv_file.stem
                all_instances.append(instance_name)
    
    all_instances = sorted(all_instances)
    print(f"Total instances: {len(all_instances)}")
    print()
    
    # Setup for 2 devices
    laptop_distributor = TaskDistributor("laptop", 2)
    aicenter_distributor = TaskDistributor("aicenter", 2)
    
    laptop_instances = laptop_distributor.assign_instances_by_modulo(all_instances)
    aicenter_instances = aicenter_distributor.assign_instances_by_modulo(all_instances)
    
    print("="*80)
    print("DEVICE ASSIGNMENTS")
    print("="*80)
    print()
    print(f"📱 LAPTOP: {len(laptop_instances)} instances")
    print(f"   {laptop_instances[:10]}..." if len(laptop_instances) > 10 else f"   {laptop_instances}")
    print()
    print(f"🖥️  AI CENTER: {len(aicenter_instances)} instances")
    print(f"   {aicenter_instances[:10]}..." if len(aicenter_instances) > 10 else f"   {aicenter_instances}")
    print()
    
    # Check for overlap
    laptop_set = set(laptop_instances)
    aicenter_set = set(aicenter_instances)
    overlap = laptop_set.intersection(aicenter_set)
    
    print("="*80)
    print("OVERLAP CHECK")
    print("="*80)
    print()
    
    if overlap:
        print("❌ ERROR: OVERLAP DETECTED!")
        print(f"   Overlapping instances: {overlap}")
        print("   This should NOT happen!")
    else:
        print("✅ NO OVERLAP - Perfect!")
        print("   Laptop and AI Center work on different instances")
        print("   No conflicts guaranteed!")
    
    print()
    print("="*80)
    print("COVERAGE CHECK")
    print("="*80)
    print()
    
    all_assigned = laptop_set.union(aicenter_set)
    all_instances_set = set(all_instances)
    missing = all_instances_set - all_assigned
    
    if missing:
        print(f"⚠️  WARNING: {len(missing)} instances not assigned")
        print(f"   Missing: {list(missing)[:10]}...")
    else:
        print("✅ ALL INSTANCES COVERED")
        print(f"   Total: {len(all_instances)} instances")
        print(f"   Laptop: {len(laptop_instances)} instances")
        print(f"   AI Center: {len(aicenter_instances)} instances")
        print(f"   Sum: {len(laptop_instances) + len(aicenter_instances)} instances")
    
    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    
    if not overlap and not missing:
        print("✅ PERFECT DISTRIBUTION!")
        print("   - No overlap between devices")
        print("   - All instances covered")
        print("   - Safe to run simultaneously!")
    elif overlap:
        print("❌ OVERLAP DETECTED - Fix needed!")
    elif missing:
        print("⚠️  Some instances not assigned - Check distribution method")
    
    print("="*80)


if __name__ == '__main__':
    verify_no_overlap()

