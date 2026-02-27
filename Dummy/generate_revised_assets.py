import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Config
RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\revised_paper_assets"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Starting REVISED Analysis...")

# --- DATA LOADING (Re-used for speed) ---
# We need to re-scan to get raw metrics and history for 56 plots
all_results = []
instance_folders = sorted([f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))])

# Convergence storage: {instance_name: { 'perm': [hist], 'bin': [hist], 'real': [hist] }}
# Storing only "Best of Class" convergence to save RAM/Time
convergence_data = {} 

def parse_combo(name):
    # Same parser as before
    parts = name.split('_')
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

def get_class(inst):
    return inst[:2] if inst[:2] in ['C1', 'C2', 'R1', 'R2'] else inst[:3]

print("Scanning for convergence data (Best per Rep)...")

# First pass: find best combo per rep per instance to know which file to read history
# This is slow if we read everything.
# Let's read summary first.
# Oh, we don't have summary file with 354 rows per instance.
# We must scan. 

# Efficient scan:
# For each instance, we track best fitness seen so far for P, B, R.
# Then we store the file path.
# Then we read history only for those 3 winners.

best_files = {} # {instance: {'permutation': path, ...}}

count = 0
for instance in instance_folders:
    path = os.path.join(RESULTS_DIR, instance)
    csvs = glob.glob(os.path.join(path, "*_convergence.csv"))
    
    best_local = {'permutation': (float('inf'), None), 'binary': (float('inf'), None), 'real_valued': (float('inf'), None)}
    
    for f in csvs:
        combo = os.path.basename(f).replace("_convergence.csv", "")
        rep, sel, cross, mut = parse_combo(combo)
        
        try:
            # Read last line
            with open(f, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1:
                    last = lines[-1].strip().split(',')
                    fit = float(last[-2])
                    
                    all_results.append({
                        'Instance': instance,
                        'Class': get_class(instance),
                        'Combination': combo,
                        'Representation': rep,
                        'Selection': sel,
                        'Crossover': cross,
                        'Mutation': mut,
                        'Fitness': fit
                    })
                    
                    if fit < best_local[rep][0]:
                        best_local[rep] = (fit, f)
        except:
            pass
            
    # Store winners for history reading
    best_files[instance] = best_local
    
    count += 1
    if count % 10 == 0: print(f"Scanned {count} instances...")

df = pd.DataFrame(all_results)

# --- REVISION 2: Convergence Grid (56 Subplots) ---
# Req: 56 plots. 7 rows x 8 cols = 56. (Ideal)
print("Generating Fig 2 Revised: 56-Grid Convergence Plot...")

fig, axes = plt.subplots(7, 8, figsize=(24, 18)) # Huge figure
axes = axes.flatten()

# Prepare history data
# This is IO intensive, we read 56 * 3 files.
for i, instance in enumerate(instance_folders):
    if i >= 56: break # Safety
    ax = axes[i]
    
    # Read history for the 3 winners
    paths = best_files[instance]
    colors = {'permutation': 'green', 'real_valued': 'orange', 'binary': 'blue'}
    
    for rep, (fit, path) in paths.items():
        if path:
            try:
                hist_df = pd.read_csv(path)
                # Plot
                ax.plot(hist_df['generation'], hist_df['fitness_average'], color=colors[rep], linewidth=1)
            except:
                pass
                
    ax.set_title(instance, fontsize=9, pad=2)
    ax.tick_params(axis='both', which='major', labelsize=7)
    if i % 8 != 0: ax.set_ylabel("") # Only left col
    if i < 48: ax.set_xlabel("") # Only bottom row
    ax.grid(True, linestyle=':', alpha=0.5)

# Legend only on first
axes[0].legend(['Perm', 'Real', 'Bin'], fontsize=6)
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Convergence_56Grid.png"), dpi=300)
plt.close()


# --- REVISION 3: Tables Top 20 per Rep (Raw Fitness) ---
print("Generating Table 2-4 Revised: Raw Fitness...")
avg_per_combo = df.groupby(['Representation', 'Selection', 'Crossover', 'Mutation', 'Combination'])['Fitness'].mean().reset_index()

for rep in ['permutation', 'binary', 'real_valued']:
    top20 = avg_per_combo[avg_per_combo['Representation'] == rep].sort_values('Fitness').head(20)
    top20.to_csv(os.path.join(OUTPUT_DIR, f"Table_{rep}_Top20_Raw.csv"), index=False)


# --- REVISION 4: 3D Scatter (X=Sel, Y=Cross, Z=Mut) ---
print("Generating Fig 4 Revised: 3D Plots (Specific Axis)...")

# Helper to format names nicely
def nice_name(s):
    return s.replace("_", " ").title()

def plot_3d_revised(rep_name):
    data_rep = avg_per_combo[avg_per_combo['Representation'] == rep_name].copy()
    
    # Map strings to integers for placement
    sel_cats = sorted(data_rep['Selection'].unique())
    cross_cats = sorted(data_rep['Crossover'].unique())
    mut_cats = sorted(data_rep['Mutation'].unique())
    
    sel_map = {name: i for i, name in enumerate(sel_cats)}
    cross_map = {name: i for i, name in enumerate(cross_cats)}
    mut_map = {name: i for i, name in enumerate(mut_cats)}
    
    data_rep['X'] = data_rep['Selection'].map(sel_map)
    data_rep['Y'] = data_rep['Crossover'].map(cross_map)
    data_rep['Z'] = data_rep['Mutation'].map(mut_map)
    
    # Size/Color by Fitness
    # Normalized fitness for color: Good(Low)=Green/1, Bad(High)=Red/0
    fit_min, fit_max = data_rep['Fitness'].min(), data_rep['Fitness'].max()
    # Invert so big size = good? Or color map.
    # Req: "tata letak diatur lagi". 
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter
    # We use color for fitness (performance).
    sc = ax.scatter(data_rep['X'], data_rep['Y'], data_rep['Z'],
                    c=data_rep['Fitness'], cmap='RdYlGn_r', s=100, edgecolors='k', alpha=0.8)
    
    # Custom Ticks
    ax.set_xticks(range(len(sel_cats)))
    ax.set_xticklabels([nice_name(s) for s in sel_cats], rotation=45, ha='right')
    
    ax.set_yticks(range(len(cross_cats)))
    ax.set_yticklabels([nice_name(s) for s in cross_cats], rotation=-15)
    
    ax.set_zticks(range(len(mut_cats)))
    ax.set_zticklabels([nice_name(s) for s in mut_cats])
    
    ax.set_xlabel('Selection')
    ax.set_ylabel('Crossover')
    ax.set_zlabel('Mutation')
    ax.set_title(f"3D Operator Space: {nice_name(rep_name)}\n(Color = Fitness: Green=Best, Red=Worst)")
    
    # Fix spacing
    ax.dist = 12 # Zoom out slightly
    
    cbar = plt.colorbar(sc, shrink=0.6)
    cbar.set_label('Average Total Distance')
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fig4_3D_{rep_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

for rep in ['permutation', 'binary', 'real_valued']:
    try:
        plot_3d_revised(rep)
    except Exception as e:
        print(f"Error plotting 3D {rep}: {e}")


# --- REVISION 5: Excel Breakdown No Scaled ---
print("Generating Table 5 Revised: Raw Fitness Breakdown...")
def create_pivot_raw(group_cols, name):
    pivot = df.groupby(group_cols + ['Class'])['Fitness'].mean().unstack()
    # Add mean global
    pivot['Global_Avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Global_Avg')
    return pivot

create_pivot_raw(['Representation'], 'Rep').to_csv(os.path.join(OUTPUT_DIR, "5a_Rep_Raw.csv"))
create_pivot_raw(['Representation', 'Selection'], 'Sel').to_csv(os.path.join(OUTPUT_DIR, "5b_Sel_Raw.csv"))
create_pivot_raw(['Representation', 'Crossover'], 'Cross').to_csv(os.path.join(OUTPUT_DIR, "5c_Cross_Raw.csv"))
create_pivot_raw(['Representation', 'Mutation'], 'Mut').to_csv(os.path.join(OUTPUT_DIR, "5d_Mut_Raw.csv"))


# --- REVISION 6: Heatmap Transposed & Renamed ---
print("Generating Fig 6 Revised: Heatmap Transposed...")
# Top 20 Global from Raw Fitness
top_20_names = avg_per_combo.sort_values('Fitness').head(20)['Combination'].tolist()
subset_hm = df[df['Combination'].isin(top_20_names)].copy()

# Rename Combo: p_x_y_z -> P\nX\nY\nZ
def fancy_label(name):
    r, s, c, m = parse_combo(name)
    # Capitalize first letter
    r = r.capitalize()
    s = s.replace("_"," ").title()
    c = c.replace("_"," ").title()
    m = m.replace("_"," ").title()
    return f"{r}\n{s}\n{c}\n{m}"

subset_hm['Label'] = subset_hm['Combination'].apply(fancy_label)

# Pivot: Index=Instance, Cols=Algo
pivot_hm = subset_hm.pivot(index='Instance', columns='Label', values='Fitness')

# Normalize per row (Instance) because C1 vs R1 scale diff
pivot_norm = pivot_hm.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

plt.figure(figsize=(20, 16)) # High height for 56 rows
sns.heatmap(pivot_norm, cmap='RdYlGn_r', linewidths=0.5, linecolor='lightgray',
            cbar_kws={'label': 'Normalized Quality (Green=Best Algorithm for Instance)'})

# Labels
plt.xlabel("Genetic Algorithm Configuration", fontsize=14, labelpad=15)
plt.ylabel("Solomon Benchmark Instance", fontsize=14)
plt.xticks(rotation=0, fontsize=10) # Horizontal labels as requested "P enter X..."
plt.yticks(fontsize=9)
plt.title("Algorithm Performance Spectrum (Top 20)", fontsize=16)

plt.savefig(os.path.join(OUTPUT_DIR, "Fig6_Heatmap_Transposed.png"), dpi=300, bbox_inches='tight')
plt.close()

print("All Revised Assets Generated.")
