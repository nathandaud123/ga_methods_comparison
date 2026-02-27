import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\paper_assets"

print("Generating detailed breakdown for all 56 instances...")

# 1. Load Data (Re-scanning efficiently)
all_data = []
instance_folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]

# We only care about the TOP 15 combinations to keep the chart readable
# Ideally, we should load 'table_top50_combinations.csv' to know which ones are top
top_50_path = os.path.join(OUTPUT_DIR, "table_top50_combinations.csv")

if os.path.exists(top_50_path):
    top_performers = pd.read_csv(top_50_path)['Combination'].head(15).tolist()
else:
    print("Top 50 file not found, please ensure previous step ran correctly.")
    exit()

print(f"Focusing on Top 15 Combinations: {len(top_performers)} methods")

count = 0
for instance in instance_folders:
    instance_path = os.path.join(RESULTS_DIR, instance)
    
    # Only check files for top performers to save IO
    for combo in top_performers:
        file_path = os.path.join(instance_path, f"{combo}_convergence.csv")
        try:
            if os.path.exists(file_path):
                # Read just the last line for speed
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(',')
                        # Assuming fitness_average is the 2nd to last column (index 11 usually)
                        # csv header: generation,fitness_run_1... fitness_average, diversity_average
                        final_fitness = float(last_line[-2]) 
                        
                        all_data.append({
                            'Instance': instance,
                            'Combination': combo,
                            'FinalFitness': final_fitness
                        })
        except Exception as e:
            pass
            
    count += 1
    if count % 10 == 0: print(f"scanned {count} instances...")

df = pd.DataFrame(all_data)

# Rank per instance
df['Rank'] = df.groupby('Instance')['FinalFitness'].rank(method='min')

# Pivot for Heatmap: Rows=Methods, Cols=Instances
# This will be very wide (56 columns). We might split it into 6 subplots or one giant one.
# Let's try groupings by Class for better layout.

problem_classes = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']

# Create a figure with 6 subplots (rows) to show detailed breakdown clearly
fig, axes = plt.subplots(6, 1, figsize=(20, 24), sharey=True)

method_order = top_performers # Consistent y-axis order

for i, p_class in enumerate(problem_classes):
    # Filter columns that belong to this class
    class_instances = sorted([inst for inst in df['Instance'].unique() if inst.startswith(p_class)])
    
    if not class_instances: continue
    
    subset = df[df['Instance'].isin(class_instances)]
    pivot = subset.pivot(index='Combination', columns='Instance', values='Rank')
    
    # Reindex to ensure consistent row order and column sorting
    pivot = pivot.reindex(index=method_order)
    pivot = pivot[class_instances] # Ensure column order C101, C102...
    
    sns.heatmap(pivot, ax=axes[i], annot=True, fmt=".0f", cmap="YlGnBu_r", cbar=i==0)
    axes[i].set_title(f"Performance Rank Breakdown: Class {p_class} ({len(class_instances)} Instances)", fontsize=14, loc='left')
    axes[i].set_ylabel("")
    axes[i].set_xlabel("")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_appendix_full_breakdown.png"), dpi=150) # Moderate DPI for giant image
plt.close()

print("Full breakdown heatmap generated.")
