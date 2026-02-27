import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\paper_assets"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def parse_combination(name):
    """Parses the combination string into components."""
    parts = name.split('_')
    # This is heuristic based on your naming convention
    # Known representations
    rep = "unknown"
    if name.startswith("binary"): rep = "binary"
    elif name.startswith("real_valued"): rep = "real_valued"
    elif name.startswith("permutation"): rep = "permutation"
    
    # Simple extraction (can be improved if naming is very strictly structured)
    return rep, name

def get_problem_class(instance_name):
    """Classifies instance into C1, C2, R1, R2, RC1, RC2"""
    if instance_name.startswith("C1"): return "C1"
    if instance_name.startswith("C2"): return "C2"
    if instance_name.startswith("R1"): return "R1"
    if instance_name.startswith("R2"): return "R2"
    if instance_name.startswith("RC1"): return "RC1"
    if instance_name.startswith("RC2"): return "RC2"
    return "Other"

print("Scanning all results... this might take a moment.")

# 1. Aggregating Data
all_data = []
# We define specific representative instances for the Convergence Plot
rep_instances = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
convergence_data = {} # Store full history for selected instances

# List all instance folders
instance_folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]

count = 0
for instance in instance_folders:
    instance_path = os.path.join(RESULTS_DIR, instance)
    csv_files = glob.glob(os.path.join(instance_path, "*_convergence.csv"))
    
    problem_class = get_problem_class(instance)
    
    for f in csv_files:
        try:
            # Quick read for final values
            # We assume the file is small enough to read quickly
            df = pd.read_csv(f)
            if df.empty: continue
            
            final_row = df.iloc[-1]
            combination_name = os.path.basename(f).replace("_convergence.csv", "")
            representation, _ = parse_combination(combination_name)
            
            # Store summary
            all_data.append({
                'Instance': instance,
                'Class': problem_class,
                'Combination': combination_name,
                'Representation': representation,
                'FinalFitness': final_row['fitness_average'],
                'FinalDiversity': final_row['diversity_average'] if 'diversity_average' in final_row else 0,
                'Generations': len(df)
            })
            
            # Store convergence history ONLY for representatives and specific methods
            # To avoid memory explosion, we only store for Top performers later, 
            # but since we don't know who wins yet, we might re-read later or store specific ones now.
            # Let's store representative subset now for creating the plot immediately.
            if instance in rep_instances:
                # We save all, but will filter for plotting.
                if instance not in convergence_data: convergence_data[instance] = []
                convergence_data[instance].append({
                    'Combination': combination_name,
                    'Representation': representation,
                    'History': df['fitness_average'].values[::10] # Sample every 10th gen to save space
                })
                
        except Exception as e:
            # print(f"Skipping {f}: {e}")
            pass
            
    count += 1
    if count % 5 == 0:
        print(f"Processed {count}/{len(instance_folders)} instances...")

df = pd.DataFrame(all_data)
print(f"Total records processed: {len(df)}")

# --- Pre-calculation for Ranking ---
# We rank within each Instance to normalize scales
df['Rank'] = df.groupby('Instance')['FinalFitness'].rank(method='min')

# --- VISUALIZATION 1: Global Performance Boxplot (Representation) ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='Representation', y='Rank', data=df, showfliers=False)
plt.title("Distribution of Rankings Comparison by Representation Scheme\n(Lower Rank is Better)")
plt.ylabel("Rank (across 354 combinations)")
plt.xlabel("Representation Scheme")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_representation_boxplot.png"), dpi=300)
plt.close()

# --- VISUALIZATION 2: Heatmap Top Methods vs Problem Class ---
# Get Global Top 15 Methods
top_methods = df.groupby('Combination')['Rank'].mean().sort_values().head(10).index
pivot_data = df[df['Combination'].isin(top_methods)].groupby(['Combination', 'Class'])['Rank'].mean().reset_index()
pivot_table = pivot_data.pivot(index='Combination', columns='Class', values='Rank')

# Sort by average rank
pivot_table['Avg'] = pivot_table.mean(axis=1)
pivot_table = pivot_table.sort_values('Avg')
pivot_table = pivot_table.drop('Avg', axis=1)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu_r", linewidths=.5)
plt.title("Heatmap: Stability of Top 10 Methods Across Problem Classes (Average Rank)")
plt.ylabel("Method Combination")
plt.xlabel("Solomon Problem Class")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_robustness_heatmap.png"), dpi=300)
plt.close()

# --- VISUALIZATION 3: Convergence Panel (6 Representative Instances) ---
# We pick the Best Permutation, Best Real, Best Binary for each instance to plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, instance in enumerate(rep_instances):
    if instance not in convergence_data: continue
    
    # Identify best performers for this specific instance
    inst_df = df[df['Instance'] == instance]
    best_perm = inst_df[inst_df['Representation'] == 'permutation'].nsmallest(1, 'FinalFitness')['Combination'].values[0]
    best_real = inst_df[inst_df['Representation'] == 'real_valued'].nsmallest(1, 'FinalFitness')['Combination'].values[0]
    best_bin = inst_df[inst_df['Representation'] == 'binary'].nsmallest(1, 'FinalFitness')['Combination'].values[0]
    
    targets = [best_perm, best_real, best_bin]
    labels = ['Best Permutation', 'Best Real-Valued', 'Best Binary']
    colors = ['green', 'orange', 'blue']
    
    ax = axes[idx]
    
    for hist in convergence_data[instance]:
        if hist['Combination'] in targets:
            color = colors[targets.index(hist['Combination'])]
            label = labels[targets.index(hist['Combination'])]
            # X-axis is sampled (x10)
            x_axis = np.arange(len(hist['History'])) * 10 
            ax.plot(x_axis, hist['History'], label=label, color=color, linewidth=2)
            
    ax.set_title(f"Convergence: {instance} ({get_problem_class(instance)})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Total Distance)")
    if idx == 0: ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_convergence_panel.png"), dpi=300)
plt.close()

# --- TABLE GENERATION (CSV for Paper) ---
# 1. Big Summary Table
summary_table = df.groupby('Combination').agg({
    'Rank': 'mean',
    'FinalFitness': 'mean',
    'Representation': 'first'
}).sort_values('Rank')

summary_table.head(50).to_csv(os.path.join(OUTPUT_DIR, "table_top50_combinations.csv"))

# 2. Friedman-like Rank Table by Class
class_ranks = df.groupby(['Combination', 'Class'])['Rank'].mean().unstack()
class_ranks['Global_Rank'] = class_ranks.mean(axis=1)
class_ranks = class_ranks.sort_values('Global_Rank')
class_ranks.head(20).to_csv(os.path.join(OUTPUT_DIR, "table_class_stability.csv"))

print("Analysis Complete. Artifacts saved to 'paper_assets' folder.")
