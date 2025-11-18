"""
Setup script for multi-device execution
Creates device configuration files for distributing workload across devices
"""

import argparse
import yaml
import os
from pathlib import Path
from src.evaluation.task_distributor import create_device_configs, TaskDistributor


def load_all_instances(config_path: str) -> list:
    """Load all instances from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config.get('dataset', {})
    base_path = dataset_config.get('base_path', 'data/solomon')
    instances = dataset_config.get('instances', [])
    
    # If empty, discover all CSV files
    if not instances:
        all_instances = []
        for subfolder in ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']:
            subfolder_path = os.path.join(base_path, subfolder)
            if os.path.exists(subfolder_path):
                csv_files = list(Path(subfolder_path).glob('*.csv'))
                for csv_file in csv_files:
                    instance_name = csv_file.stem
                    all_instances.append(instance_name)
        return sorted(all_instances)
    
    # Extract instance names from paths
    instance_names = []
    for inst in instances:
        if isinstance(inst, str):
            # Extract name from path
            name = os.path.basename(inst).split('.')[0]
            instance_names.append(name)
    
    return sorted(instance_names)


def main():
    parser = argparse.ArgumentParser(description='Setup multi-device execution')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--total-devices', type=int, required=True,
                       help='Total number of devices')
    parser.add_argument('--method', type=str, default='modulo',
                       choices=['modulo', 'type'],
                       help='Distribution method: modulo or type')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Multi-Device Setup")
    print("="*80)
    print(f"Total devices: {args.total_devices}")
    print(f"Distribution method: {args.method}")
    print()
    
    # Load all instances
    all_instances = load_all_instances(args.config)
    print(f"Total instances: {len(all_instances)}")
    print()
    
    # Create device configs
    print("Creating device configurations...")
    create_device_configs(
        total_devices=args.total_devices,
        all_instances=all_instances,
        distribution_method=args.method,
        output_dir="results"
    )
    
    print()
    print("="*80)
    print("Device configurations created!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Copy the project to each device")
    print("2. On each device, edit config.yaml:")
    print("   device:")
    print(f"     device_id: \"device1\"  # Change to device1, device2, etc.")
    print(f"     total_devices: {args.total_devices}")
    print("3. Run on each device:")
    print("   python main.py --config config.yaml")
    print()
    print("Each device will automatically work on its assigned instances!")


if __name__ == '__main__':
    main()

