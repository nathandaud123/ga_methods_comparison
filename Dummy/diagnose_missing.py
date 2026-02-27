
import os
import json
import pandas as pd

RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"

def diagnose():
    method_counts = {}
    all_methods = set()
    instance_method_map = {}

    print("Scanning results...")
    for instance_name in os.listdir(RESULTS_DIR):
        instance_path = os.path.join(RESULTS_DIR, instance_name)
        if not os.path.isdir(instance_path):
            continue
            
        json_file = os.path.join(instance_path, f"{instance_name}_results.json")
        if not os.path.exists(json_file):
            continue
            
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
                methods_in_file = set(results.keys())
                all_methods.update(methods_in_file)
                instance_method_map[instance_name] = methods_in_file
                
                # print(f"{instance_name}: found {len(methods_in_file)} methods")
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    total_methods = len(all_methods)
    print(f"\nTotal unique methods found across ALL files: {total_methods}")
    
    # Check intersection
    intersection = all_methods.copy()
    for methods in instance_method_map.values():
        intersection &= methods
        
    print(f"Methods present in ALL {len(instance_method_map)} instances: {len(intersection)}")
    
    # Identify missing culprits
    missing_info = []
    for m in all_methods:
        missing_in = []
        for instance, methods in instance_method_map.items():
            if m not in methods:
                missing_in.append(instance)
        
        if missing_in:
            missing_info.append((m, len(missing_in)))
            
    missing_info.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 methods with most missing instances:")
    for m, count in missing_info[:10]:
        print(f"  - {m}: missing in {count} instances")

    print("\nRecommendation: If count is small, we can impute with 'Max Fitness' (Worst Rank).")

if __name__ == "__main__":
    diagnose()
