"""
Check experiment progress
"""

import os
from pathlib import Path

def check_progress():
    """Check how many instances and methods have been completed"""
    results_dir = Path("results")
    
    # Count completed instances
    experiments_dir = results_dir / "experiments"
    convergence_dir = results_dir / "convergence"
    
    completed_instances = set()
    
    if experiments_dir.exists():
        completed_instances.update([d.name for d in experiments_dir.iterdir() if d.is_dir()])
    
    if convergence_dir.exists():
        completed_instances.update([d.name for d in convergence_dir.iterdir() if d.is_dir()])
    
    print("="*60)
    print("Experiment Progress")
    print("="*60)
    
    print(f"\nCompleted Instances: {len(completed_instances)}")
    if completed_instances:
        print("  Instances:")
        for instance in sorted(completed_instances):
            # Count methods
            instance_conv_dir = convergence_dir / instance
            if instance_conv_dir.exists():
                csv_files = list(instance_conv_dir.glob("*_convergence.csv"))
                print(f"    - {instance}: {len(csv_files)} methods completed")
    
    # Check for summary
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        print(f"\n✅ Summary file exists: {summary_file}")
    
    # Check log files
    log_dir = results_dir / "logs"
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            print(f"\n📝 Latest log: {latest_log.name}")
            print(f"   Size: {latest_log.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    check_progress()

