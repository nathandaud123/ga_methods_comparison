"""
Merge multiple instance-specific checkpoint files into a single checkpoint.json

This script is useful when running experiments on multiple instances
in parallel (e.g., different terminals), where each instance uses its own
checkpoint file (checkpoint_<instance>.json).

Usage:
    python merge_checkpoints.py [--results-dir results]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Set, List


def find_checkpoint_files(results_dir: str) -> List[str]:
    """Find all checkpoint_*.json files"""
    checkpoint_files = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return checkpoint_files
    
    # Find all checkpoint_*.json files
    for checkpoint_file in results_path.glob('checkpoint_*.json'):
        checkpoint_files.append(str(checkpoint_file))
    
    return sorted(checkpoint_files)


def load_checkpoint(checkpoint_file: str) -> Dict:
    """Load checkpoint from file"""
    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {checkpoint_file}: {e}")
        return None


def merge_checkpoints(checkpoint_files: List[str]) -> Dict:
    """Merge multiple checkpoint files into one"""
    merged = {
        'completed_instances': [],
        'completed_methods': {}
    }
    
    instance_names = set()
    
    for checkpoint_file in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_file)
        print(f"Loading {checkpoint_name}...")
        
        state = load_checkpoint(checkpoint_file)
        if state is None:
            continue
        
        # Extract instance name from checkpoint filename
        # checkpoint_C101.json -> C101
        instance_name = checkpoint_name.replace('checkpoint_', '').replace('.json', '')
        instance_names.add(instance_name)
        
        # Merge completed_methods
        if 'completed_methods' in state:
            for inst_name, methods in state['completed_methods'].items():
                if inst_name not in merged['completed_methods']:
                    merged['completed_methods'][inst_name] = []
                
                # Add methods that aren't already in the list
                existing_methods = set(merged['completed_methods'][inst_name])
                for method in methods:
                    if method not in existing_methods:
                        merged['completed_methods'][inst_name].append(method)
        
        # Merge completed_instances
        if 'completed_instances' in state:
            for inst_name in state['completed_instances']:
                if inst_name not in merged['completed_instances']:
                    merged['completed_instances'].append(inst_name)
    
    return merged, instance_names


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Merge instance-specific checkpoint files into a single checkpoint.json'
    )
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory containing checkpoint files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output checkpoint file (default: results/checkpoint.json)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Checkpoint Merger")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print("="*80)
    
    # Find all checkpoint files
    checkpoint_files = find_checkpoint_files(args.results_dir)
    
    if not checkpoint_files:
        print("\nNo checkpoint_*.json files found!")
        print(f"Looking in: {os.path.abspath(args.results_dir)}")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoint file(s):")
    for cf in checkpoint_files:
        print(f"  - {os.path.basename(cf)}")
    
    # Merge checkpoints
    print("\nMerging checkpoints...")
    merged_state, instance_names = merge_checkpoints(checkpoint_files)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(args.results_dir, 'checkpoint.json')
    
    # Save merged checkpoint
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_state, f, indent=2)
    
    print(f"\n✓ Merged checkpoint saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Instances processed: {len(instance_names)}")
    print(f"  - Instances completed: {len(merged_state['completed_instances'])}")
    print(f"  - Total methods tracked: {sum(len(methods) for methods in merged_state['completed_methods'].values())}")
    
    # Show breakdown by instance
    if merged_state['completed_methods']:
        print(f"\nMethods completed per instance:")
        for inst_name in sorted(merged_state['completed_methods'].keys()):
            methods = merged_state['completed_methods'][inst_name]
            is_complete = inst_name in merged_state['completed_instances']
            status = "✓ COMPLETE" if is_complete else f"{len(methods)} methods"
            print(f"  - {inst_name}: {status}")
    
    print("="*80)


if __name__ == '__main__':
    main()

