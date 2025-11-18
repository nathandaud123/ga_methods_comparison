"""
Quick setup untuk 2 devices: Laptop + AI Center UB DGX UB
AI Center akan menggunakan semua core-nya (parallel execution)
"""

import yaml
import os
from pathlib import Path
from src.evaluation.task_distributor import TaskDistributor


def load_all_instances(config_path: str = 'config.yaml') -> list:
    """Load all instances from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config.get('dataset', {})
    base_path = dataset_config.get('base_path', 'data/solomon')
    
    # Auto-discover all CSV files
    all_instances = []
    for subfolder in ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']:
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.exists(subfolder_path):
            csv_files = list(Path(subfolder_path).glob('*.csv'))
            for csv_file in csv_files:
                instance_name = csv_file.stem
                all_instances.append(instance_name)
    
    return sorted(all_instances)


def main():
    print("="*80)
    print("Setup untuk 2 Devices: Laptop + AI Center UB DGX UB")
    print("="*80)
    print()
    
    # Load all instances
    all_instances = load_all_instances()
    print(f"Total instances ditemukan: {len(all_instances)}")
    print()
    
    # Setup untuk 2 devices
    total_devices = 2
    
    # Device 1: Laptop
    laptop_distributor = TaskDistributor("laptop", total_devices)
    laptop_instances = laptop_distributor.assign_instances_by_modulo(all_instances)
    
    # Device 2: AI Center
    aicenter_distributor = TaskDistributor("aicenter", total_devices)
    aicenter_instances = aicenter_distributor.assign_instances_by_modulo(all_instances)
    
    print("="*80)
    print("DISTRIBUSI WORKLOAD")
    print("="*80)
    print()
    print(f"📱 LAPTOP:")
    print(f"   Instances: {len(laptop_instances)}")
    print(f"   Examples: {laptop_instances[:5]}")
    if len(laptop_instances) > 5:
        print(f"   ... dan {len(laptop_instances) - 5} instances lainnya")
    print()
    print(f"🖥️  AI CENTER UB DGX UB:")
    print(f"   Instances: {len(aicenter_instances)}")
    print(f"   Examples: {aicenter_instances[:5]}")
    if len(aicenter_instances) > 5:
        print(f"   ... dan {len(aicenter_instances) - 5} instances lainnya")
    print()
    print("="*80)
    print("KONFIGURASI")
    print("="*80)
    print()
    print("📱 LAPTOP - Edit config.yaml:")
    print("```yaml")
    print("device:")
    print("  device_id: \"laptop\"")
    print("  total_devices: 2")
    print()
    print("evaluation:")
    print("  parallel: true  # Gunakan semua core laptop")
    print("  n_jobs: null    # Auto-detect")
    print("```")
    print()
    print("🖥️  AI CENTER - Edit config.yaml:")
    print("```yaml")
    print("device:")
    print("  device_id: \"aicenter\"")
    print("  total_devices: 2")
    print()
    print("evaluation:")
    print("  parallel: true  # Gunakan SEMUA core AI Center")
    print("  n_jobs: null    # Auto-detect (akan pakai semua core yang tersedia)")
    print("# Atau set manual jika tahu jumlah core:")
    print("# n_jobs: 64  # Contoh: jika ada 64 cores")
    print("```")
    print()
    print("="*80)
    print("CARA MENJALANKAN")
    print("="*80)
    print()
    print("📱 LAPTOP:")
    print("  python main.py --config config.yaml")
    print()
    print("🖥️  AI CENTER:")
    print("  python main.py --config config.yaml")
    print()
    print("="*80)
    print("HASIL")
    print("="*80)
    print()
    print("✅ Laptop akan kerja pada instances yang di-assign")
    print("✅ AI Center akan kerja pada instances yang berbeda")
    print("✅ AI Center akan menggunakan SEMUA core-nya (parallel execution)")
    print("✅ Tidak ada konflik - setiap device kerja pada instance berbeda")
    print("✅ Progress ter-track di checkpoint yang sama")
    print()
    print("="*80)


if __name__ == '__main__':
    main()

