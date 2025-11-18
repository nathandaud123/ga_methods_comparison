"""
Monitor experiment progress in real-time
Shows checkpoint status and latest log output
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

def get_latest_log():
    """Get the latest experiment log file"""
    log_dir = Path("results/logs")
    if not log_dir.exists():
        return None
    
    log_files = list(log_dir.glob("experiment_*.log"))
    if not log_files:
        return None
    
    # Get most recent log file
    latest = max(log_files, key=lambda p: p.stat().st_mtime)
    return latest

def tail_log(file_path, n_lines=20):
    """Get last n lines of log file"""
    if not file_path or not file_path.exists():
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            return lines[-n_lines:]
    except:
        return []

def load_checkpoint():
    """Load checkpoint file"""
    checkpoint_file = Path("results/checkpoint.json")
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except:
        return None

def format_size(size_bytes):
    """Format file size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def monitor():
    """Monitor experiment progress"""
    print("=" * 60)
    print("Experiment Monitor")
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring\n")
    
    last_checkpoint_time = None
    
    try:
        while True:
            # Clear screen (works on most terminals)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 60)
            print(f"Experiment Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            # Checkpoint status
            checkpoint = load_checkpoint()
            if checkpoint:
                print("\n📊 Checkpoint Status:")
                print(f"  Completed Instances: {len(checkpoint.get('completed_instances', []))}")
                if checkpoint.get('completed_instances'):
                    for inst in checkpoint['completed_instances']:
                        methods = checkpoint.get('completed_methods', {}).get(inst, [])
                        print(f"    - {inst}: {len(methods)} methods completed")
                
                partial = checkpoint.get('partial_progress', {})
                if partial:
                    print(f"  Instances with Partial Progress: {len(partial)}")
                    for inst, methods in partial.items():
                        print(f"    - {inst}: {len(methods)} methods in progress")
                
                # Check if checkpoint was updated
                checkpoint_file = Path("results/checkpoint.json")
                if checkpoint_file.exists():
                    mtime = checkpoint_file.stat().st_mtime
                    if last_checkpoint_time and mtime > last_checkpoint_time:
                        print("  ✅ Checkpoint updated!")
                    last_checkpoint_time = mtime
            else:
                print("\n📊 Checkpoint Status: No checkpoint file yet")
            
            # Log file status
            log_file = get_latest_log()
            if log_file:
                size = log_file.stat().st_size
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                print(f"\n📝 Latest Log: {log_file.name}")
                print(f"   Size: {format_size(size)}")
                print(f"   Last Updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Show last few lines
                print("\n📄 Last 10 lines of log:")
                print("-" * 60)
                lines = tail_log(log_file, 10)
                for line in lines:
                    print(line.rstrip())
            else:
                print("\n📝 No log file found yet")
            
            print("\n" + "=" * 60)
            print("Refreshing in 5 seconds... (Ctrl+C to stop)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == '__main__':
    monitor()

