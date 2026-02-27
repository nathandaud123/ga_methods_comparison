import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\paper_assets"
TOP_N = 30

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Generating optimized heatmap for top {TOP_N} configurations...")

# 1. Identify Top 30 Methods using Global Average Rank
# Re-load or Re-calculate global rankings efficiently
# Ideally we use the 'table_top50_combinations.csv' if it exists and is trusted
top_table_path = os.path.join(OUTPUT_DIR, "table_top50_combinations.csv")

if os.path.exists(top_table_path):
    print("Loading top combinations from existing analysis...")
    top_df = pd.read_csv(top_table_path)
    # Sort by Rank to be sure 1 is top
    top_df = top_df.sort_values("Rank")
    top_30_methods = top_df['Combination'].head(TOP_N).tolist()
else:
    print("Top 50 table not found! Cannot proceed without ranking info.")
    exit()

# 2. Extract Data for these 30 methods across all 56 instances
# Structure: Rows = Methods, Cols = Instances
matrix_data = []

# Get list of all instances sorted naturally (C101... RC208)
instance_folders = sorted([f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))])

# Custom sorter for Solomon instances to ensure C1->C2->R1->R2->RC1->RC2 order
def solomon_sort_key(name):
    # Extract alpha and numeric parts
    import re
    match = re.match(r"([A-Z]+)(\d)(\d+)", name)
    if match:
        cat, type_num, id_num = match.groups()
        # Order: C, R, RC mapping
        cat_order = {'C': 1, 'R': 2, 'RC': 3}
        return (cat_order.get(cat, 9), int(type_num), int(id_num))
    return (9, 0, 0) # Support fallback

instance_folders.sort(key=solomon_sort_key)

print(f"Scanning {len(instance_folders)} instances for {len(top_30_methods)} methods...")

data_records = []

for instance in instance_folders:
    instance_path = os.path.join(RESULTS_DIR, instance)
    
    for method in top_30_methods:
        file_path = os.path.join(instance_path, f"{method}_convergence.csv")
        try:
            if os.path.exists(file_path):
                # Read just the last avg fitness
                # We can use pandas for robustness or simple read for speed. 
                # Let's use pandas just to be safe about columns
                df = pd.read_csv(file_path)
                if not df.empty:
                    fit_val = df['fitness_average'].iloc[-1]
                    data_records.append({
                        'Method': method,
                        'Instance': instance,
                        'Fitness': fit_val
                    })
        except:
            pass

df_heatmap = pd.DataFrame(data_records)

# 3. Pivot and Normalize
# Pivot: Index=Method, Cols=Instance, Values=Fitness
pivot_raw = df_heatmap.pivot(index='Method', columns='Instance', values='Fitness')

# Reorder Index based on the Global Rank (top_30_methods list order)
pivot_raw = pivot_raw.reindex(top_30_methods)

# Reorder Columns based on Solomon sort
pivot_raw = pivot_raw.reindex(columns=instance_folders)

# CRITICAL STEP: Normalization per Instance (Column)
# Because Fitness scales range from ~800 (C1) to ~3000 (RC2), 
# we must normalize to compare "Performance Quality" visually.
# Min-Max Scaling per column: (Value - Min) / (Max - Min)
# 0 (Best/Min Fitness) -> Green, 1 (Worst/Max Fitness) -> Red
pivot_norm = pivot_raw.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

# Fill NAs if any specific run failed (optional, usually 0 or 0.5)
pivot_norm = pivot_norm.fillna(1.0) # Assume worst if missing

# 4. Plotting
plt.figure(figsize=(20, 12))

# Use a Green-Yellow-Red colormap (reversed so 0/Green is best, 1/Red is worst)
# 'RdYlGn_r' : Red (High/Bad) to Green (Low/Good). 
# Wait, fitness is minimized. So Low Value = Good.
# In RdYlGn: Green is high, Red is low.
# So we want 'RdYlGn' but mapped to inverted values? 
# Easier: Use 'RdYlGn_r'. High Value (1.0, bad) -> Red. Low Value (0.0, good) -> Green.
sns.heatmap(pivot_norm, cmap='RdYlGn_r', linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'Normalized Fitness Gap (0=Best in Column, 1=Worst)'})

plt.title(f"Performance Heatmap of Top {TOP_N} GA Configurations on Solomon Benchmarks", fontsize=16, pad=20)
plt.xlabel("Solomon Instances (grouped by type)", fontsize=12)
plt.ylabel("GA Configuration (Ranked Top to Bottom)", fontsize=12)

# Improve labelling
# Simplify Y labels: remove 'permutation_' prefix if present to save space?
# Let's keep full for precision as requested.
plt.yticks(rotation=0, fontsize=10)
plt.xticks(rotation=90, fontsize=8)

# Add grouping vertical lines manually? 
# The grid lines from seaborn (linewidths) handle cell separation.
# But visually separating C1 / C2 / R1 might be nice. 
# We can find indices where category changes.
curr_cat = ""
for i, col in enumerate(pivot_norm.columns):
    cat = col[:2] # C1, R1, RC...
    # simple heuristic: first 2 chars
    if instance_folders[i].startswith("RC"): cat = "RC"
    elif instance_folders[i].startswith("R"): cat = "R" + instance_folders[i][1]
    elif instance_folders[i].startswith("C"): cat = "C" + instance_folders[i][1]
    
    if i > 0 and cat != curr_cat:
        plt.axvline(i, color='black', linewidth=2)
    curr_cat = cat

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_perplexity_style_heatmap.png"), dpi=300)
plt.close()

print("Heatmap generated: fig4_perplexity_style_heatmap.png")
