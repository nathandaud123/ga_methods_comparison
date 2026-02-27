import os
import pandas as pd
import glob
import re

# Define representative instances (one from each class to save time)
# If fast enough, we could do more, but let's start with these.
INSTANCES = ['C101', 'R101', 'RC101', 'C201', 'R201', 'RC201']
BASE_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"

data = []

print("Starting analysis on representative instances...")

for instance in INSTANCES:
    instance_path = os.path.join(BASE_DIR, instance)
    if not os.path.exists(instance_path):
        print(f"Warning: {instance_path} not found.")
        continue
    
    # Files are like: binary_boltzmann_multi_point_uniform_convergence.csv
    # Pattern: [encoding]_[selection]_[crossover]_[mutation]_convergence.csv
    # But some operator names have underscores (e.g. multi_point).
    # We need a robust parsing strategy.
    
    files = glob.glob(os.path.join(instance_path, "*_convergence.csv"))
    
    for f in files:
        filename = os.path.basename(f)
        combination_name = filename.replace("_convergence.csv", "")
        
        # Heuristic parsing based on known Encodings
        encoding = "unknown"
        rest = combination_name
        if combination_name.startswith("binary_"):
            encoding = "binary"
            rest = combination_name[7:]
        elif combination_name.startswith("real_valued_"):
            encoding = "real_valued"
            rest = combination_name[12:]
        elif combination_name.startswith("permutation_"):
            encoding = "permutation"
            rest = combination_name[12:]
            
        # This parsing is tricky without fixed delimiters. 
        # Let's just track the "Combination Name" for ranking first.
        
        try:
            # Read only the last few lines to get final convergence
            # Read full file is OK, they are small (500 lines)
            df = pd.read_csv(f)
            if 'fitness_average' in df.columns:
                final_fitness = df['fitness_average'].iloc[-1]
                final_diversity = df['diversity_average'].iloc[-1]
                
                data.append({
                    'Instance': instance,
                    'Combination': combination_name,
                    'Encoding': encoding,
                    'FinalFitness': final_fitness,
                    'FinalDiversity': final_diversity
                })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Create DataFrame
df_results = pd.DataFrame(data)

# Calculate Rankings per Instance
df_results['Rank'] = df_results.groupby('Instance')['FinalFitness'].rank(method='min')

# 1. Top Combinations Overall (Average Rank across 6 instances)
top_combinations = df_results.groupby('Combination').agg({
    'Rank': 'mean',
    'FinalFitness': 'mean',
    'Encoding': 'first' # Assuming constant
}).sort_values('Rank').head(15)

print("\n--- Top 15 Combinations by Average Rank ---")
print(top_combinations.to_markdown())

# 2. Performance by Encoding (Boxplot data equivalent)
encoding_stats = df_results.groupby('Encoding')['Rank'].describe()
print("\n--- Rank Statistics by Encoding ---")
print(encoding_stats.to_markdown())

# Save for user
top_combinations.to_csv("top_combinations_summary.csv")
df_results.to_csv("analyzed_sample_results.csv")
