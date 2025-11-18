"""Quick status check for experiment"""
import os
import json
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("EXPERIMENT STATUS CHECK")
print("=" * 70)
print()

# Check checkpoint
cp_file = Path("results/checkpoint.json")
if cp_file.exists():
    cp_mtime = datetime.fromtimestamp(cp_file.stat().st_mtime)
    age_min = (datetime.now() - cp_mtime).total_seconds() / 60
    
    print(f"✅ Checkpoint file exists")
    print(f"   Last updated: {cp_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Age: {age_min:.1f} minutes")
    
    if age_min < 5:
        print("   Status: 🟢 ACTIVE (recently updated)")
    elif age_min < 60:
        print("   Status: 🟡 MAY BE RUNNING (updated within hour)")
    else:
        print("   Status: 🔴 NOT RUNNING (old update)")
    
    # Load and show progress
    with open(cp_file, 'r') as f:
        cp = json.load(f)
    
    print()
    print(f"Progress:")
    print(f"  Completed instances: {len(cp.get('completed_instances', []))}")
    print(f"  Instances with partial progress: {len(cp.get('partial_progress', {}))}")
    
    if cp.get('completed_methods'):
        for inst, methods in cp['completed_methods'].items():
            print(f"    - {inst}: {len(methods)} methods completed")
else:
    print("❌ No checkpoint file found")

print()

# Check log files
log_dir = Path("results/logs")
if log_dir.exists():
    logs = list(log_dir.glob("experiment_*.log"))
    if logs:
        latest = max(logs, key=lambda p: p.stat().st_mtime)
        size_kb = latest.stat().st_size / 1024
        log_mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        age_min = (datetime.now() - log_mtime).total_seconds() / 60
        
        print(f"📝 Latest log: {latest.name}")
        print(f"   Size: {size_kb:.2f} KB")
        print(f"   Last updated: {log_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Age: {age_min:.1f} minutes")
        
        if age_min < 5 and size_kb > 0:
            print("   Status: 🟢 ACTIVE")
        elif age_min < 60:
            print("   Status: 🟡 CHECK")
        else:
            print("   Status: 🔴 OLD")
    else:
        print("❌ No log files found")
else:
    print("❌ Log directory not found")

print()
print("=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

if cp_file.exists():
    age_min = (datetime.now() - datetime.fromtimestamp(cp_file.stat().st_mtime)).total_seconds() / 60
    if age_min > 60:
        print("❌ Experiment is NOT running (checkpoint is old)")
        print("   Start with: python main.py --config config.yaml")
    elif age_min < 5:
        print("✅ Experiment appears to be RUNNING (checkpoint recently updated)")
    else:
        print("⚠️  Experiment status unclear - check manually")
        print("   Run: python main.py --config config.yaml")
else:
    print("❌ No checkpoint - experiment has not started")
    print("   Start with: python main.py --config config.yaml")

print("=" * 70)

