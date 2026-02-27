import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\paper_assets"
TOP_N_PER_REP = 20

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Generating Stratified Heatmap (Top {TOP_N_PER_REP} per Representation)...")

# 1. Load Summary Data to Identify Leaders
# We need to re-scan mostly because table_top50 only has global top.
# WE need top per rep.
# Let's do a quick scan of "All Combinations" summary if possible.
# Since we didn't save a "table_all_combinations.csv", let's reconstruct rank from file scan.
# Faster approach: Scan one instance (e.g. RC201) to get list of all methods, 
# then rely on the global rank logic we established. 
# Actually, we need to know WHICH methods are Permutation/Binary/Real.

all_data_scan = []
instance_dirs = sorted([f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))])

print("Scanning for method names and types...")
# Just scan one folder to get the universe of methods
sample_inst_path = os.path.join(RESULTS_DIR, "RC101") # Use a dense one
sample_files = glob.glob(os.path.join(sample_inst_path, "*_convergence.csv"))

methods_meta = []
for f in sample_files:
    combo_name = os.path.basename(f).replace("_convergence.csv", "")
    
    rep = "unknown"
    if combo_name.startswith("binary"): rep = "binary"
    elif combo_name.startswith("real_valued"): rep = "real_valued"
    elif combo_name.startswith("permutation"): rep = "permutation"
    
    methods_meta.append({'Combination': combo_name, 'Representation': rep})

df_methods = pd.DataFrame(methods_meta)

# Now we need their RANKS. To be accurate, we really should scan all 56 instances again
# OR trust a smaller sample. For Q1 paper, let's scan all to be safe. 
# It took ~60s last time. Worth it.

print("Rescanning global performance for accurate ranking...")
ranking_data = []

counts = 0
for instance in instance_dirs:
    inst_path = os.path.join(RESULTS_DIR, instance)
    csvs = glob.glob(os.path.join(inst_path, "*_convergence.csv"))
    
    for f in csvs:
        try:
            # Optimize: Read just last line
            with open(f, 'r') as file_handle:
                lines = file_handle.readlines()
                if len(lines) > 1:
                    vals = lines[-1].strip().split(',')
                    # fitness is usually 2nd from last
                    fitness = float(vals[-2])
                    combo = os.path.basename(f).replace("_convergence.csv", "")
                    
                    ranking_data.append({
                        'Instance': instance,
                        'Combination': combo,
                        'Fitness': fitness
                    })
        except:
            pass
    counts +=1
    if counts % 10 == 0: print(f"Scanned {counts} instances...")

df_rank = pd.DataFrame(ranking_data)

# Calculate Rank per Instance
df_rank['Rank'] = df_rank.groupby('Instance')['Fitness'].rank(method='min')

# Calculate Global Mean Rank
global_ranks = df_rank.groupby('Combination')['Rank'].mean().reset_index()
global_ranks = global_ranks.merge(df_methods, on='Combination', how='left')
global_ranks = global_ranks.sort_values('Rank')

# Select Top N per Representation
top_perm = global_ranks[global_ranks['Representation'] == 'permutation'].head(TOP_N_PER_REP)
top_real = global_ranks[global_ranks['Representation'] == 'real_valued'].head(TOP_N_PER_REP)
top_bin  = global_ranks[global_ranks['Representation'] == 'binary'].head(TOP_N_PER_REP)

combined_top = pd.concat([top_perm, top_real, top_bin])
target_methods = combined_top['Combination'].tolist()

print(f"Selected {len(target_methods)} methods for visualization.")

# 3. Pivot Data for Heatmap
# We need Fitness values relative to the BEST known for that instance.
# Better metric: % GAP from Best Known (in this experiment).
# Formula: (Fitness - BestFitnessInExperiment) / BestFitnessInExperiment
# This is better than MinMax for comparing different classes.

# Calculate Best Fitness per instance
best_per_instance = df_rank.groupby('Instance')['Fitness'].min().to_dict()

# Filter df_rank for target methods
heatmap_df = df_rank[df_rank['Combination'].isin(target_methods)].copy()

# Calculate Gap
heatmap_df['Gap'] = heatmap_df.apply(lambda row: (row['Fitness'] - best_per_instance[row['Instance']]) / best_per_instance[row['Instance']], axis=1)

pivot_gap = heatmap_df.pivot(index='Combination', columns='Instance', values='Gap')

# Reorder Rows: Permutation 1-20, then Real 1-20, then Binary 1-20
pivot_gap = pivot_gap.reindex(target_methods)

# Reorder Columns: Solomon Order
def solomon_sort_key_col(name):
    import re
    match = re.match(r"([A-Z]+)(\d)(\d+)", name)
    if match:
        cat, type_num, id_num = match.groups()
        cat_order = {'C': 1, 'R': 2, 'RC': 3}
        return (cat_order.get(cat, 9), int(type_num), int(id_num))
    return (9, 0, 0)

sorted_cols = sorted(pivot_gap.columns, key=solomon_sort_key_col)
pivot_gap = pivot_gap[sorted_cols]

# 4. Plotting
fig, axes = plt.subplots(3, 1, figsize=(20, 24), gridspec_kw={'height_ratios': [1, 1, 1]})

# Common Color Limit: 0% to say 100% gap? 
# Permutation will be near 0. Binary might be 200%.
# Let's cap colorbar at 1.0 (100% gap) to keep contrast in good range.
vmax_val = 1.0 

# Plot Permutation
sns.heatmap(pivot_gap.iloc[0:TOP_N_PER_REP], ax=axes[0], cmap="RdYlGn_r", 
            vmin=0, vmax=vmax_val, cbar=False, linewidths=0.5, linecolor='gray')
axes[0].set_title(f"Top {TOP_N_PER_REP} Permutation-Based Configurations (Avg Rank: {top_perm['Rank'].mean():.1f})", fontsize=14, loc='left')
axes[0].set_xlabel("")
axes[0].set_ylabel("")

# Plot Real-Valued
sns.heatmap(pivot_gap.iloc[TOP_N_PER_REP:2*TOP_N_PER_REP], ax=axes[1], cmap="RdYlGn_r", 
            vmin=0, vmax=vmax_val, cbar=False, linewidths=0.5, linecolor='gray')
axes[1].set_title(f"Top {TOP_N_PER_REP} Real-Valued Configurations (Avg Rank: {top_real['Rank'].mean():.1f})", fontsize=14, loc='left')
axes[1].set_xlabel("")
axes[1].set_ylabel("")

# Plot Binary
# Add Colorbar only to the last one or right side?
# Let's add separate cbar ax if needed, or just let last one have it.
sns.heatmap(pivot_gap.iloc[2*TOP_N_PER_REP:], ax=axes[2], cmap="RdYlGn_r", 
            vmin=0, vmax=vmax_val, cbar=True, cbar_kws={'label': 'Optimality Gap (>1.0 = >100% worse than best)', 'orientation': 'horizontal', 'pad': 0.2}, 
            linewidths=0.5, linecolor='gray')
axes[2].set_title(f"Top {TOP_N_PER_REP} Binary Configurations (Avg Rank: {top_bin['Rank'].mean():.1f})", fontsize=14, loc='left')
axes[2].set_xlabel("Solomon Benchmark Instances", fontsize=12)
axes[2].set_ylabel("")

plt.suptitle("Stratified Performance Landscape: Best Methods per Representation Scheme\n(Metric: Gap from Best Known Solution)", fontsize=18, y=0.92)

# Improve x-axis labels rotation
for ax in axes:
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', labelsize=9)

plt.subplots_adjust(hspace=0.4)
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_stratified_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Stratified Heatmap generated: fig5_stratified_heatmap.png")
