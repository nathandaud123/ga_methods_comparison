"""
Simple checkpoint management for resuming experiments
"""

import json
import os
from typing import Dict, Set


class SimpleCheckpoint:
    """Simple checkpoint manager for tracking completed combinations"""
    
    def __init__(self, checkpoint_file: str = "results/checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_dir = os.path.dirname(checkpoint_file)
        if self.checkpoint_dir:  # Only create dir if path is not empty
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.state = self.load()
    
    def load(self) -> Dict:
        """Load checkpoint state"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    state = json.load(f)
                print(f"[Checkpoint] Loaded: {len(state.get('completed_instances', []))} instances completed")
                return state
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load checkpoint: {e}")
                return self._init_state()
        return self._init_state()
    
    def save(self):
        """Save checkpoint state"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def _init_state(self) -> Dict:
        """Initialize empty checkpoint state"""
        return {
            'completed_instances': [],
            'completed_methods': {}  # {instance_name: [method_names]}
        }
    
    def is_instance_complete(self, instance_name: str) -> bool:
        """Check if instance is completely processed"""
        return instance_name in self.state['completed_instances']
    
    def is_method_complete(self, instance_name: str, method_name: str) -> bool:
        """Check if method is complete for an instance"""
        if instance_name not in self.state['completed_methods']:
            return False
        return method_name in self.state['completed_methods'][instance_name]
    
    def mark_method_complete(self, instance_name: str, method_name: str):
        """Mark a method as complete"""
        if instance_name not in self.state['completed_methods']:
            self.state['completed_methods'][instance_name] = []
        
        if method_name not in self.state['completed_methods'][instance_name]:
            self.state['completed_methods'][instance_name].append(method_name)
            self.save()
    
    def mark_instance_complete(self, instance_name: str):
        """Mark an instance as completely processed"""
        if instance_name not in self.state['completed_instances']:
            self.state['completed_instances'].append(instance_name)
            self.save()
    
    def get_completed_methods(self, instance_name: str) -> Set[str]:
        """Get set of completed methods for an instance"""
        return set(self.state['completed_methods'].get(instance_name, []))

