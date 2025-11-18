"""
Clear checkpoint to restart experiment from beginning
"""

import sys
from src.evaluation.checkpoint import CheckpointManager

if __name__ == '__main__':
    checkpoint_file = "results/checkpoint.json"
    
    if len(sys.argv) > 1:
        instance_name = sys.argv[1]
        manager = CheckpointManager(checkpoint_file)
        manager.reset_instance(instance_name)
        print(f"Checkpoint reset for instance: {instance_name}")
    else:
        response = input("Clear entire checkpoint? This will reset all progress. (yes/no): ")
        if response.lower() == 'yes':
            manager = CheckpointManager(checkpoint_file)
            manager.clear()
            print("Checkpoint cleared. Experiment will start from beginning.")
        else:
            print("Cancelled.")



