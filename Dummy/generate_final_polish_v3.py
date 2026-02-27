import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Config
RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\final_polished_v3_no_title"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Starting FINAL POLISH V3 Analysis...")

# ... (Data Loading Logic Same as Before) ...
df_rows = []
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
                  'sbx', 'blx_alpha', 'flat', 'pmx', 'ox', 'cx', 'obx', 'pos', 'erx', 'scx']
    cross = "unknown"
    crossovers.sort(key=len, reverse=True)
    for c in crossovers:
        if rest.startswith(c):
            cross = c
            rest = rest.replace(c + "_", "")
            break
    mut = rest 
    return rep, sel, cross, mut

print("Scanning data...")
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
                    df_rows.append({'Instance': instance,'Representation': rep,'Selection': sel,'Crossover': cross,'Mutation': mut,'Combination': combo,'Fitness': fit})
        except: pass

df_full = pd.DataFrame(df_rows)
print("Data Loaded.")

# --- COMMON STYLE SETTINGS ---
TITLE_FONT_SIZE = 0 # No Title
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 11
palette = {'permutation': '#2ca02c', 'real_valued': '#ff7f0e', 'binary': '#1f77b4'}

def nice_label(s): return s.replace("_", " ").title()

# --- REVISION 1: Boxplot per Class Group ---
print("Generating Fig 2 V3: Class-based Boxplots (No Title, Bold Labels)...")

def plot_class_boxplot_v3(class_prefix, filename_suffix):
    subset = df_full[df_full['Instance'].str.startswith(class_prefix)].copy()
    subset = subset.sort_values('Instance')
    
    min_max = subset.groupby('Instance')['Fitness'].agg(['min', 'max'])
    subset = subset.merge(min_max, on='Instance')
    subset['ScaledFitness'] = (subset['Fitness'] - subset['min']) / (subset['max'] - subset['min'])
    
    plt.figure(figsize=(20, 8))
    sns.boxplot(x='Instance', y='ScaledFitness', hue='Representation', data=subset, 
                palette=palette, showfliers=False, width=0.7)
    
    unique_inst = subset['Instance'].unique()
    if len(unique_inst) > 0:
        sub_class = unique_inst[0][:2]
        for i, inst in enumerate(unique_inst):
            curr_sub = inst[:2]
            if curr_sub != sub_class:
                plt.axvline(i - 0.5, color='black', linestyle='-', alpha=0.3, linewidth=2)
                sub_class = curr_sub
                
    # No Title
    plt.ylabel("Scaled Optimality Gap", fontsize=LABEL_FONT_SIZE, weight='bold')
    plt.xlabel("", fontsize=LABEL_FONT_SIZE) # Empty x-label often preferred if ticks are self-explanatory
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    # Legend cleanup
    plt.legend(loc='upper right', title="Representation", title_fontsize=12, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fig2_Boxplot_Class_{filename_suffix}.png"), dpi=300)
    plt.close()

plot_class_boxplot_v3(tuple(['C']), "C")
plot_class_boxplot_v3(tuple(['R']), "R")
plot_class_boxplot_v3(tuple(['RC']), "RC")


# --- REVISION 2: 3D Plot Layout Fix (Extreme Spacing) ---
print("Generating Fig 4 V3: 3D Plots (Space Fix)...")

def plot_3d_v3(rep_name):
    data_rep = df_full[df_full['Representation'] == rep_name].copy()
    avg_perm = data_rep.groupby(['Selection', 'Crossover', 'Mutation'])['Fitness'].mean().reset_index()

    sel_order = sorted(avg_perm['Selection'].unique())
    cross_order = sorted(avg_perm['Crossover'].unique())
    mut_order = sorted(avg_perm['Mutation'].unique())

    sel_map = {x: i for i, x in enumerate(sel_order)}
    cross_map = {x: i for i, x in enumerate(cross_order)}
    mut_map = {x: i for i, x in enumerate(mut_order)}

    avg_perm['X'] = avg_perm['Selection'].map(sel_map)
    avg_perm['Y'] = avg_perm['Crossover'].map(cross_map)
    avg_perm['Z'] = avg_perm['Mutation'].map(mut_map)

    fit_vals = avg_perm['Fitness']
    fit_norm = (fit_vals - fit_vals.min()) / (fit_vals.max() - fit_vals.min())
    sizes = (1 - fit_norm) * 700 + 50 

    fig = plt.figure(figsize=(18, 14)) # Even bigger
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(avg_perm['X'], avg_perm['Y'], avg_perm['Z'], 
                    s=sizes, c=fit_vals, cmap='RdYlGn_r', 
                    edgecolors='black', linewidth=0.5, alpha=0.9)

    # Drop lines
    threshold = np.percentile(fit_vals, 15)
    best_points = avg_perm[avg_perm['Fitness'] <= threshold]
    for _, row in best_points.iterrows():
        ax.plot([row['X'], row['X']], [row['Y'], row['Y']], [0, row['Z']], 
                color='gray', linestyle='--', linewidth=0.5, alpha=0.4)

    # Padding and Rotation - Aggressive
    ax.set_xticks(range(len(sel_order)))
    # Rotate almost vertical to avoid overlap
    ax.set_xticklabels([nice_label(x) for x in sel_order], rotation=50, ha='right', fontsize=10)
    ax.tick_params(axis='x', pad=15) # Push ticks out

    ax.set_yticks(range(len(cross_order)))
    ax.set_yticklabels([nice_label(x) for x in cross_order], rotation=-20, ha='left', fontsize=10)
    ax.tick_params(axis='y', pad=15)

    ax.set_zticks(range(len(mut_order)))
    ax.set_zticklabels([nice_label(x) for x in mut_order], fontsize=10)
    ax.tick_params(axis='z', pad=15)

    # Bold Axis Labels with Huge Padding
    ax.set_xlabel('Selection Method', labelpad=40, fontsize=14, weight='bold') # 35-40 pad
    ax.set_ylabel('Crossover Operator', labelpad=40, fontsize=14, weight='bold')
    ax.set_zlabel('Mutation Strategy', labelpad=25, fontsize=14, weight='bold')
    
    # No Title.

    # Camera Angle - slightly higher to see depth
    ax.view_init(elev=30, azim=-60)

    # Colorbar - Thicker/More Visible
    cbar = plt.colorbar(sc, shrink=0.7, aspect=15, pad=0.1) # pad moves it right
    cbar.set_label('Average Total Distance', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(OUTPUT_DIR, f"Fig4_3D_{rep_name}_V3.png"), dpi=300, bbox_inches='tight')
    plt.close()

for rep in ['permutation', 'binary', 'real_valued']:
    try:
        plot_3d_v3(rep)
    except: pass


# --- REVISION 3: Heatmap Spacing Fix V3 ---
print("Generating Fig 6 V3: Larger Text Heatmap...")

method_ranks = df_full.groupby('Combination')['Fitness'].mean().reset_index()
top_15 = method_ranks.sort_values('Fitness').head(15)
top_15_names = top_15['Combination'].tolist()

subset_hm = df_full[df_full['Combination'].isin(top_15_names)].copy()

name_map = {
    'roulette_wheel': 'Roulette', 'stochastic_universal': 'SUS', 
    'single_point': '1-Point', 'multi_point': 'M-Point', 'two_point': '2-Point',
    'real_valued': 'Real', 'permutation': 'Perm', 'binary': 'Bin'
}

def fancy_label_short(name):
    r, s, c, m = parse_combo(name)
    r = name_map.get(r, r.capitalize())
    s = name_map.get(s, s.replace("_"," ").title())
    c = name_map.get(c, c.replace("_"," ").title())
    m = name_map.get(m, m.replace("_"," ").title())
    return f"{r}\n{s}\n{c}\n{m}"

subset_hm['Label'] = subset_hm['Combination'].apply(fancy_label_short)
pivot_hm = subset_hm.pivot(index='Instance', columns='Label', values='Fitness')
sorted_cols = [fancy_label_short(x) for x in top_15_names]
pivot_hm = pivot_hm[sorted_cols]
pivot_norm = pivot_hm.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# Wider Figure + Larger Font
plt.figure(figsize=(26, 18)) 
sns.heatmap(pivot_norm, cmap='RdYlGn_r', linewidths=1, linecolor='white',
            cbar_kws={'label': 'Normalized Performance'})

# Larger fonts (+2 size roughly)
plt.xticks(rotation=0, fontsize=13) # Was 11
plt.yticks(fontsize=11) # Was 9
plt.xlabel("Algorithm Configuration", fontsize=16, weight='bold', labelpad=20)
plt.ylabel("Solomon Benchmark Instance", fontsize=16, weight='bold', labelpad=20)
# No Title

# Fix Colorbar Label
cbar = plt.gca().collections[0].colorbar
cbar.set_label('Normalized Performance (Green = Best)', fontsize=14, weight='bold')
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig6_Heatmap_Top15_V3.png"), dpi=300)
plt.close()

print("Final Polish V3 Complete.")
