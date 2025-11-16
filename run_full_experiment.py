"""
Script to run full experiment in background with progress tracking
"""

import subprocess
import sys
import os
from datetime import datetime

def run_experiment_with_logging():
    """Run experiment with output logging"""
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    print(f"Starting full experiment...")
    print(f"Log file: {log_file}")
    print(f"This will take several hours. Press Ctrl+C to stop.")
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                [sys.executable, "main.py", "--config", "config.yaml"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            
        print(f"\nExperiment completed! Log saved to: {log_file}")
        
    except KeyboardInterrupt:
        print(f"\nExperiment interrupted. Partial log saved to: {log_file}")
        if process:
            process.terminate()

if __name__ == '__main__':
    run_experiment_with_logging()

