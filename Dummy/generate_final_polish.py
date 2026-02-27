import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Config
RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\final_polished_assets"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Starting FINAL POLISH Analysis...")

# --- DATA SCANNING (Need full data for boxplots) --- 
print("Scanning full dataset for instance-level distributions...")
all_data = []
all_instances = sorted([f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))])

def parse_combo(name):
    rep = "unknown"
    if name.startswith("binary"): rep = "binary"
    elif name.startswith("real_valued"): rep = "real_valued"
    elif name.startswith("permutation"): rep = "permutation"
    
    rest = name.replace(rep + "_", "")
    selections = ['roulette_wheel', 'tournament', 'rank', 'stochastic_universal', 'elitism', 'boltzmann', 'stairwise']
    sel = "unknown"
    for s in selections:
        if rest.startswith(s):
            sel = s
            rest = rest.replace(s + "_", "")
            break
            
    crossovers = ['single_point', 'two_point', 'multi_point', 'uniform', 'shuffle', 'arithmetic', 
                  'sbx', 'blx_alpha', 'flat', 
                  'pmx', 'ox', 'cx', 'obx', 'pos', 'erx', 'scx']
    cross = "unknown"
    crossovers.sort(key=len, reverse=True)
    for c in crossovers:
        if rest.startswith(c):
            cross = c
            rest = rest.replace(c + "_", "")
            break
            
    mut = rest 
    return rep, sel, cross, mut

df_rows = []
for instance in all_instances:
    path = os.path.join(RESULTS_DIR, instance)
    csvs = glob.glob(os.path.join(path, "*_convergence.csv"))
    
    for f in csvs:
        try:
            with open(f, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1:
                    last = lines[-1].strip().split(',')
                    fit = float(last[-2])
                    combo = os.path.basename(f).replace("_convergence.csv", "")
                    rep, sel, cross, mut = parse_combo(combo)
                    
                    df_rows.append({
                        'Instance': instance,
                        'Representation': rep,
                        'Selection': sel,
                        'Crossover': cross,
                        'Mutation': mut,
                        'Combination': combo,
                        'Fitness': fit
                    })
        except:
            pass

df_full = pd.DataFrame(df_rows)
print(f"Loaded {len(df_full)} data points.")

# --- REVISION NO 2: Boxplot Breakdown (2 Parts, 28 instances each) ---
print("Generating Fig 2 Revised: Per-Instance Boxplot comparison...")

# Sort instances naturally
# They are already sorted by name C101...
# But let's fix color scheme
palette = {'permutation': '#2ca02c', 'real_valued': '#ff7f0e', 'binary': '#1f77b4'} # Green, Orange, Blue

def plot_instance_boxplots(data, filename_suffix, instances_subset):
    plt.figure(figsize=(24, 10))
    
    # Filter data
    subset = data[data['Instance'].isin(instances_subset)]
    
    # Use Scaled Fitness per Instance for visualization? Or Raw?
    # User said "kayak gini" referring to previous plot which was Scaled.
    # But Raw fitness varies too much (800 vs 3000). Scaled is necessary for Boxplot across instances in one chart.
    # Calculate scale per instance
    min_max = subset.groupby('Instance')['Fitness'].agg(['min', 'max'])
    subset = subset.merge(min_max, on='Instance')
    subset['ScaledFitness'] = (subset['Fitness'] - subset['min']) / (subset['max'] - subset['min'])
    
    sns.boxplot(x='Instance', y='ScaledFitness', hue='Representation', data=subset, 
                palette=palette, showfliers=False, width=0.7)
    
    # Visual cues for boundaries
    # Draw vertical lines between classes
    current_class = instances_subset[0][:2]
    for i, inst in enumerate(instances_subset):
        cls = inst[:2]
        if cls != current_class:
            plt.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.5)
            current_class = cls
            
    plt.title(f"Performance Distribution by Representation per Instance (Scaled Gap 0-1) - {filename_suffix}", fontsize=16)
    plt.ylabel("Scaled Optimality Gap", fontsize=14)
    plt.xlabel("Instance ID", fontsize=14)
    plt.legend(loc='upper right', title="Representation")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fig2_Boxplot_Comparison_{filename_suffix}.png"), dpi=300)
    plt.close()

# Split 56 instances into 2 chunks
chunk1 = all_instances[:28]
chunk2 = all_instances[28:]

plot_instance_boxplots(df_full, "Part1_C1-R1", chunk1)
plot_instance_boxplots(df_full, "Part2_R1-RC2", chunk2)


# --- REVISION NO 4: Enhanced 3D Plot (Permutation Only focused) ---
print("Generating Fig 4 Revised: High-Contrast 3D Plot...")

# We need average fitness per operator combo for Permutation
perm_data = df_full[df_full['Representation'] == 'permutation'].copy()
avg_perm = perm_data.groupby(['Selection', 'Crossover', 'Mutation'])['Fitness'].mean().reset_index()

# 1. Map categories to int
sel_order = sorted(avg_perm['Selection'].unique())
cross_order = sorted(avg_perm['Crossover'].unique())
mut_order = sorted(avg_perm['Mutation'].unique())

sel_map = {x: i for i, x in enumerate(sel_order)}
cross_map = {x: i for i, x in enumerate(cross_order)}
mut_map = {x: i for i, x in enumerate(mut_order)}

avg_perm['X'] = avg_perm['Selection'].map(sel_map)
avg_perm['Y'] = avg_perm['Crossover'].map(cross_map)
avg_perm['Z'] = avg_perm['Mutation'].map(mut_map)

# 2. Plotting
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Normalize fitness for sizing
# We want Best (Low Fitness) -> Big Size, Top Color
fit_vals = avg_perm['Fitness']
fit_norm = (fit_vals - fit_vals.min()) / (fit_vals.max() - fit_vals.min())
# Size: Invert norm (0->1, 1->0) so Best is Big
sizes = (1 - fit_norm) * 500 + 50 # Range 50 to 550

# Scatter with "depth" perception
# Color map: 'viridis_r' (Yellow=Bad, Purple=Good)? Or RdYlGn_r (Green=Good)
sc = ax.scatter(avg_perm['X'], avg_perm['Y'], avg_perm['Z'], 
                s=sizes, c=fit_vals, cmap='RdYlGn_r', 
                edgecolors='black', linewidth=0.5, alpha=0.9)

# 3. Add Drop Lines for Best Points (Top 10%) to make them stand out
threshold = np.percentile(fit_vals, 10) # Top 10% fitness
best_points = avg_perm[avg_perm['Fitness'] <= threshold]

for _, row in best_points.iterrows():
    ax.plot([row['X'], row['X']], [row['Y'], row['Y']], [0, row['Z']], 
            color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Labels
def nice_label(s): return s.replace("_", " ").title()

ax.set_xticks(range(len(sel_order)))
ax.set_xticklabels([nice_label(x) for x in sel_order], rotation=20, ha='right', fontsize=9)

ax.set_yticks(range(len(cross_order)))
ax.set_yticklabels([nice_label(x) for x in cross_order], rotation=-10, fontsize=9)

ax.set_zticks(range(len(mut_order)))
ax.set_zticklabels([nice_label(x) for x in mut_order], fontsize=9)

ax.set_xlabel('Selection', labelpad=15)
ax.set_ylabel('Crossover', labelpad=15)
ax.set_zlabel('Mutation', labelpad=10)
ax.set_title("3D Operator Performance Space (Permutation)\nLarger/Green = Better Performance", fontsize=14)

# Colorbar
cbar = plt.colorbar(sc, shrink=0.5, aspect=10)
cbar.set_label('Average Total Distance (Fitness)')

plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_3D_Enhanced_Permutation.png"), dpi=300)
plt.close()


# --- REVISION NO 6: Top 15 Heatmap Sorted Left-to-Right ---
print("Generating Fig 6 Revised: Sorted Top 15 Heatmap...")

# 1. Identify Top 15 Global
method_ranks = df_full.groupby('Combination')['Fitness'].mean().reset_index()
top_15 = method_ranks.sort_values('Fitness').head(15) # Smallest fitness first (Best) -> Rank 1
top_15_names = top_15['Combination'].tolist()

# 2. Filter & Pivot
subset_hm = df_full[df_full['Combination'].isin(top_15_names)].copy()

def fancy_label(name):
    r, s, c, m = parse_combo(name)
    r = r.capitalize()
    s = s.replace("_"," ").title()
    c = c.replace("_"," ").title()
    m = m.replace("_"," ").title()
    return f"{r}\n{s}\n{c}\n{m}"

subset_hm['Label'] = subset_hm['Combination'].apply(fancy_label)

pivot_hm = subset_hm.pivot(index='Instance', columns='Label', values='Fitness')

# 3. Sort Columns by Global Average Fitness (Best Left -> Worst Right)
# Calculate column means
col_means = pivot_hm.mean().sort_values() # Ascending (Small is Best)
pivot_hm = pivot_hm[col_means.index] # Reorder columns

# 4. Normalize Rows (Instance) for visual comparison
pivot_norm = pivot_hm.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# Plot
plt.figure(figsize=(18, 16))
sns.heatmap(pivot_norm, cmap='RdYlGn_r', linewidths=0.5, linecolor='lightgray',
            cbar_kws={'label': 'Relative Performance (Green=Best in Row)'})

plt.title("Top 15 Algorithms Sorted by Global Effectiveness (Best on Left)", fontsize=16)
plt.xlabel("Algorithm Configuration", fontsize=14)
plt.ylabel("Instance", fontsize=14)
plt.xticks(rotation=0, fontsize=10)

plt.savefig(os.path.join(OUTPUT_DIR, "Fig6_Heatmap_Top15_Sorted.png"), dpi=300, bbox_inches='tight')
plt.close()

print("All Final Polish Assets Generated.")
