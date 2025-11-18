"""
Checkpoint management for experiment resuming
"""

import json
import os
from typing import Dict, Set, Optional
from pathlib import Path


class CheckpointManager:
    """Manage experiment checkpoints for resuming"""
    
    def __init__(self, checkpoint_file: str = "results/checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_dir = os.path.dirname(checkpoint_file)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.state: Dict = self.load()
    
    def load(self) -> Dict:
        """Load checkpoint state from file"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    state = json.load(f)
                num_instances = len(state.get('completed_methods', {}))
                print(f"[OK] Checkpoint loaded: {num_instances} instances with progress")
                return state
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load checkpoint: {e}")
                return self._init_state()
        return self._init_state()
    
    def save(self):
        """Save checkpoint state to file (thread-safe for multi-device)"""
        try:
            # Use atomic write: write to temp file, then rename
            import tempfile
            import shutil
            
            temp_file = self.checkpoint_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            
            # Atomic rename (works on most systems)
            if os.path.exists(self.checkpoint_file):
                # On Windows, need to remove first
                try:
                    os.remove(self.checkpoint_file)
                except:
                    pass
            shutil.move(temp_file, self.checkpoint_file)
        except IOError as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def _init_state(self) -> Dict:
        """Initialize empty checkpoint state"""
        return {
            'completed_instances': [],
            'completed_methods': {},  # {instance_name: [method_names]}
            'partial_progress': {}    # {instance_name: {method_name: {run: result}}}
        }
    
    def is_instance_complete(self, instance_name: str) -> bool:
        """Check if instance is completely processed"""
        return instance_name in self.state['completed_instances']
    
    def is_method_complete(self, instance_name: str, method_name: str, 
                          expected_runs: int) -> bool:
        """Check if method is completely processed for an instance"""
        if instance_name not in self.state['completed_methods']:
            return False
        
        completed_methods = self.state['completed_methods'][instance_name]
        if method_name not in completed_methods:
            return False
        
        # Check if all runs are complete
        if instance_name in self.state['partial_progress']:
            method_progress = self.state['partial_progress'][instance_name].get(method_name, {})
            return len(method_progress) >= expected_runs
        
        return method_name in completed_methods
    
    def get_completed_runs(self, instance_name: str, method_name: str) -> Set[int]:
        """Get set of completed run numbers for a method"""
        if instance_name not in self.state['partial_progress']:
            return set()
        if method_name not in self.state['partial_progress'][instance_name]:
            return set()
        method_progress = self.state['partial_progress'][instance_name][method_name]
        # If status is complete, all runs are done
        if isinstance(method_progress, dict) and 'status' in method_progress:
            if method_progress['status'] == 'complete':
                return set(range(1, method_progress.get('completed_runs', 0) + 1))
        # Otherwise, get actual run keys
        run_keys = [int(k) for k in method_progress.keys() if k.isdigit()]
        return set(run_keys)
    
    def mark_run_complete(self, instance_name: str, method_name: str, 
                         run_number: int, result: Dict):
        """Mark a single run as complete"""
        if instance_name not in self.state['partial_progress']:
            self.state['partial_progress'][instance_name] = {}
        if method_name not in self.state['partial_progress'][instance_name]:
            self.state['partial_progress'][instance_name][method_name] = {}
        
        self.state['partial_progress'][instance_name][method_name][str(run_number)] = result
        self.save()  # Save after each run
    
    def mark_method_complete(self, instance_name: str, method_name: str):
        """Mark a method as completely processed"""
        if instance_name not in self.state['completed_methods']:
            self.state['completed_methods'][instance_name] = []
        
        if method_name not in self.state['completed_methods'][instance_name]:
            self.state['completed_methods'][instance_name].append(method_name)
        
        # Clean up partial progress for this method (keep summary only)
        if instance_name in self.state['partial_progress']:
            if method_name in self.state['partial_progress'][instance_name]:
                # Keep only summary info
                method_progress = self.state['partial_progress'][instance_name][method_name]
                num_runs = len(method_progress)
                # Store summary instead of individual runs to save space
                self.state['partial_progress'][instance_name][method_name] = {
                    'completed_runs': num_runs,
                    'status': 'complete'
                }
        
        self.save()
    
    def mark_instance_complete(self, instance_name: str):
        """Mark an instance as completely processed"""
        if instance_name not in self.state['completed_instances']:
            self.state['completed_instances'].append(instance_name)
        self.save()
    
    def get_progress_summary(self) -> Dict:
        """Get summary of checkpoint progress"""
        return {
            'completed_instances': len(self.state['completed_instances']),
            'instances_with_progress': len(self.state['completed_methods']),
            'completed_methods_by_instance': {
                inst: len(methods) 
                for inst, methods in self.state['completed_methods'].items()
            }
        }
    
    def clear(self):
        """Clear checkpoint (use with caution)"""
        self.state = self._init_state()
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        print("Checkpoint cleared")
    
    def reset_instance(self, instance_name: str):
        """Reset checkpoint for a specific instance (to rerun it)"""
        if instance_name in self.state['completed_instances']:
            self.state['completed_instances'].remove(instance_name)
        if instance_name in self.state['completed_methods']:
            del self.state['completed_methods'][instance_name]
        if instance_name in self.state['partial_progress']:
            del self.state['partial_progress'][instance_name]
        self.save()
        print(f"Checkpoint reset for instance: {instance_name}")

