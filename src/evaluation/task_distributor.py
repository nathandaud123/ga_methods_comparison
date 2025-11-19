"""
Task distribution for multi-device parallel execution
Ensures each device works on different instances/methods to avoid conflicts
"""

import json
import os
from typing import List, Dict, Set, Optional
from pathlib import Path


class TaskDistributor:
    """
    Distribute tasks across multiple devices to avoid conflicts
    Each device gets assigned specific instances/methods to work on
    """
    
    def __init__(self, device_id: str, total_devices: int, 
                 checkpoint_file: str = "results/checkpoint.json"):
        """
        Initialize task distributor
        
        Args:
            device_id: Unique identifier for this device (e.g., "device1", "laptop", "server1")
            total_devices: Total number of devices that will run in parallel
            checkpoint_file: Path to shared checkpoint file
        """
        self.device_id = device_id
        self.total_devices = total_devices
        self.checkpoint_file = checkpoint_file
        self.device_config_file = f"results/device_{device_id}_config.json"
        
        # Load or create device config
        self.device_config = self._load_device_config()
    
    def _load_device_config(self) -> Dict:
        """Load device-specific configuration"""
        if os.path.exists(self.device_config_file):
            try:
                with open(self.device_config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Create new config
        return {
            'device_id': self.device_id,
            'total_devices': self.total_devices,
            'assigned_instances': [],
            'assigned_method_patterns': []
        }
    
    def _save_device_config(self):
        """Save device-specific configuration"""
        os.makedirs(os.path.dirname(self.device_config_file), exist_ok=True)
        with open(self.device_config_file, 'w') as f:
            json.dump(self.device_config, f, indent=2)
    
    def assign_instances_by_modulo(self, all_instances: List[str]) -> List[str]:
        """
        Assign instances to this device using modulo distribution
        Device 0 gets instances 0, N, 2N, ...
        Device 1 gets instances 1, N+1, 2N+1, ...
        etc.
        
        Args:
            all_instances: List of all instance names
            
        Returns:
            List of instances assigned to this device
        """
        device_index = self._get_device_index()
        assigned = []
        
        for idx, instance in enumerate(sorted(all_instances)):
            if idx % self.total_devices == device_index:
                assigned.append(instance)
        
        self.device_config['assigned_instances'] = assigned
        self._save_device_config()
        
        return assigned
    
    def assign_instances_by_type(self, all_instances: List[str], 
                                 instance_types: List[str]) -> List[str]:
        """
        Assign instances by type (e.g., device 1 gets C types, device 2 gets R types)
        
        Args:
            all_instances: List of all instance names
            instance_types: List of instance type prefixes (e.g., ['C', 'R', 'RC'])
            
        Returns:
            List of instances assigned to this device
        """
        device_index = self._get_device_index()
        
        # Group instances by type
        instances_by_type = {t: [] for t in instance_types}
        for instance in all_instances:
            instance_type = instance[0] if instance else 'R'
            if instance_type in instances_by_type:
                instances_by_type[instance_type].append(instance)
        
        # Assign types to devices in round-robin
        assigned = []
        for type_idx, (inst_type, instances) in enumerate(instances_by_type.items()):
            if type_idx % self.total_devices == device_index:
                assigned.extend(sorted(instances))
        
        self.device_config['assigned_instances'] = assigned
        self._save_device_config()
        
        return assigned
    
    def assign_instances_manual(self, assigned_instances: List[str]):
        """
        Manually assign specific instances to this device
        
        Args:
            assigned_instances: List of instance names to assign
        """
        self.device_config['assigned_instances'] = assigned_instances
        self._save_device_config()
    
    def assign_methods_by_modulo(self, all_methods: List[str]) -> List[str]:
        """
        Assign methods to this device using modulo distribution
        
        Args:
            all_methods: List of all method names
            
        Returns:
            List of methods assigned to this device
        """
        device_index = self._get_device_index()
        assigned = []
        
        for idx, method in enumerate(sorted(all_methods)):
            if idx % self.total_devices == device_index:
                assigned.append(method)
        
        self.device_config['assigned_method_patterns'] = assigned
        self._save_device_config()
        
        return assigned
    
    def filter_instances(self, all_instances: List[str]) -> List[str]:
        """
        Filter instances to only those assigned to this device
        
        Args:
            all_instances: List of all instance names
            
        Returns:
            Filtered list of instances for this device
        """
        assigned = self.device_config.get('assigned_instances', [])
        
        if not assigned:
            # Auto-assign if not set
            return self.assign_instances_by_modulo(all_instances)
        
        # Return intersection
        return [inst for inst in all_instances if inst in assigned]
    
    def filter_methods(self, all_methods: List[str]) -> List[str]:
        """
        Filter methods to only those assigned to this device
        
        Args:
            all_methods: List of all method names
            
        Returns:
            Filtered list of methods for this device
        """
        assigned_patterns = self.device_config.get('assigned_method_patterns', [])
        
        if not assigned_patterns:
            # Auto-assign if not set
            return self.assign_methods_by_modulo(all_methods)
        
        # Return intersection
        return [method for method in all_methods if method in assigned_patterns]
    
    def _get_device_index(self) -> int:
        """Get numeric index for this device"""
        # Try to extract number from device_id
        import re
        match = re.search(r'\d+', self.device_id)
        if match:
            return int(match.group()) - 1  # device1 -> 0, device2 -> 1, etc.
        
        # Special handling for common device names
        device_id_lower = self.device_id.lower()
        if 'laptop' in device_id_lower or device_id_lower == 'laptop':
            return 0
        elif 'aicenter' in device_id_lower or 'ai-center' in device_id_lower or 'dgx' in device_id_lower:
            return 1
        elif 'server' in device_id_lower:
            # Extract server number if exists
            match = re.search(r'\d+', self.device_id)
            if match:
                return int(match.group()) - 1
            return 1  # Default server to index 1
        
        # Fallback: consistent hash based on device_id
        # Use a deterministic hash that's consistent across runs
        hash_value = abs(hash(self.device_id))
        return hash_value % self.total_devices
    
    def get_assigned_instances(self) -> List[str]:
        """Get list of instances assigned to this device"""
        return self.device_config.get('assigned_instances', [])
    
    def get_summary(self) -> Dict:
        """Get summary of device assignment"""
        return {
            'device_id': self.device_id,
            'total_devices': self.total_devices,
            'assigned_instances': len(self.device_config.get('assigned_instances', [])),
            'instances': self.device_config.get('assigned_instances', [])
        }


def create_device_configs(total_devices: int, all_instances: List[str],
                         distribution_method: str = "modulo",
                         output_dir: str = "results"):
    """
    Create device configuration files for all devices
    
    Args:
        total_devices: Total number of devices
        all_instances: List of all instance names
        distribution_method: "modulo" or "type"
        output_dir: Directory to save config files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for device_idx in range(total_devices):
        device_id = f"device{device_idx + 1}"
        distributor = TaskDistributor(device_id, total_devices)
        
        if distribution_method == "modulo":
            assigned = distributor.assign_instances_by_modulo(all_instances)
        elif distribution_method == "type":
            instance_types = list(set([inst[0] for inst in all_instances if inst]))
            assigned = distributor.assign_instances_by_type(all_instances, instance_types)
        else:
            assigned = distributor.assign_instances_by_modulo(all_instances)
        
        print(f"{device_id}: {len(assigned)} instances assigned")
        print(f"  Examples: {assigned[:3]}..." if len(assigned) > 3 else f"  {assigned}")

