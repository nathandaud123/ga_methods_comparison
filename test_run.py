"""
Quick test to verify everything is ready to run
"""

import yaml
import os
from pathlib import Path

print("="*80)
print("PRE-RUN CHECK")
print("="*80)
print()

# Check config
print("1. Checking config.yaml...")
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✅ config.yaml found and valid")
    
    # Check device config
    device_config = config.get('device', {})
    device_id = device_config.get('device_id')
    total_devices = device_config.get('total_devices', 1)
    
    if device_id and total_devices > 1:
        print(f"   📱 Multi-device mode: {device_id} of {total_devices}")
    else:
        print(f"   📱 Single device mode")
    
    # Check evaluation config
    eval_config = config.get('evaluation', {})
    parallel = eval_config.get('parallel', False)
    n_jobs = eval_config.get('n_jobs')
    n_runs = eval_config.get('n_runs', 5)
    
    print(f"   ⚡ Parallel: {parallel}")
    if parallel:
        if n_jobs:
            print(f"   ⚡ n_jobs: {n_jobs} workers")
        else:
            print(f"   ⚡ n_jobs: Auto-detect")
    print(f"   📊 n_runs: {n_runs} runs per method")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

print()

# Check datasets
print("2. Checking datasets...")
dataset_config = config.get('dataset', {})
base_path = dataset_config.get('base_path', 'data/solomon')
instances = dataset_config.get('instances', [])

if not instances:
    # Auto-discover
    all_instances = []
    for subfolder in ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']:
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.exists(subfolder_path):
            csv_files = list(Path(subfolder_path).glob('*.csv'))
            all_instances.extend([f.stem for f in csv_files])
    
    print(f"   ✅ Found {len(all_instances)} instances (auto-discover)")
else:
    print(f"   ✅ {len(instances)} instances specified")

print()

# Check dependencies
print("3. Checking dependencies...")
try:
    import numpy
    print("   ✅ numpy")
except:
    print("   ❌ numpy missing")

try:
    import pandas
    print("   ✅ pandas")
except:
    print("   ❌ pandas missing")

try:
    import matplotlib
    print("   ✅ matplotlib")
except:
    print("   ❌ matplotlib missing")

try:
    import optuna
    print("   ✅ optuna")
except:
    print("   ⚠️  optuna missing (optional)")

print()

# Check checkpoint
print("4. Checking checkpoint...")
checkpoint_file = config.get('output', {}).get('checkpoint_file', 'results/checkpoint.json')
if os.path.exists(checkpoint_file):
    print(f"   ✅ Checkpoint exists: {checkpoint_file}")
    import json
    with open(checkpoint_file, 'r') as f:
        cp = json.load(f)
    print(f"   📊 Completed instances: {len(cp.get('completed_instances', []))}")
    print(f"   📊 Instances with progress: {len(cp.get('completed_methods', {}))}")
else:
    print(f"   ℹ️  No checkpoint yet (will be created)")

print()
print("="*80)
print("READY TO RUN!")
print("="*80)
print()
print("Jalankan dengan:")
print("  python main.py --config config.yaml")
print()
print("Eksperimen akan:")
if device_id and total_devices > 1:
    print(f"  - Run sebagai device: {device_id}")
    print(f"  - Work pada instances yang di-assign")
if parallel:
    print(f"  - Menggunakan parallel execution ({n_jobs or 'auto'} workers)")
print(f"  - {n_runs} runs per method")
print(f"  - Auto-resume dari checkpoint jika ada")
print()

