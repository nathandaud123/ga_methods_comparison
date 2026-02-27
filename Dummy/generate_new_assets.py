import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Config
RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\new_paper_assets"

print("Starting FULL Analysis for New Paper Assets...")

# 1. SCANNING DATA (The Heavy Lifting)
all_results = []
instance_folders = sorted([f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))])

def parse_combo(name):
    parts = name.split('_')
    rep = "unknown"
    if name.startswith("binary"): rep = "binary"
    elif name.startswith("real_valued"): rep = "real_valued"
    elif name.startswith("permutation"): rep = "permutation"
    
    # Try to heuristic parse Selection, Crossover, Mutation
    # This is tricky because names have underscores. 
    # Assumption: Structure is [rep]_[selection]_[crossover]_[mutation]
    # But "roulette_wheel" has underscore. "multi_point" has underscore.
    
    # Let's clean the rep prefix first
    rest = name.replace(rep + "_", "")
    
    # Selection dictionary
    selections = ['roulette_wheel', 'tournament', 'rank', 'stochastic_universal', 'elitism', 'boltzmann', 'stairwise']
    sel = "unknown"
    for s in selections:
        if rest.startswith(s):
            sel = s
            rest = rest.replace(s + "_", "")
            break
            
    # Crossover dictionary
    crossovers = ['single_point', 'two_point', 'multi_point', 'uniform', 'shuffle', 'arithmetic', # binary
                  'sbx', 'blx_alpha', 'flat', # real
                  'pmx', 'ox', 'cx', 'obx', 'pos', 'erx', 'scx'] # permutation
    cross = "unknown"
    # Sort by length desc to match 'multi_point' before 'point' if existed
    crossovers.sort(key=len, reverse=True)
    for c in crossovers:
        if rest.startswith(c):
            cross = c
            rest = rest.replace(c + "_", "")
            # Clean potential trailing underscore if mutation follows
            break
            
    mut = rest # The rest is mutation
    
    return rep, sel, cross, mut

# Sorting helper for Solomon
def get_class(inst):
    if inst.startswith("C1"): return "C1"
    if inst.startswith("C2"): return "C2"
    if inst.startswith("R1"): return "R1"
    if inst.startswith("R2"): return "R2"
    if inst.startswith("RC1"): return "RC1"
    if inst.startswith("RC2"): return "RC2"
    return "Other"

count = 0
for instance in instance_folders:
    path = os.path.join(RESULTS_DIR, instance)
    csvs = glob.glob(os.path.join(path, "*_convergence.csv"))
    
    for f in csvs:
        try:
            # Optimize: read headers and last line
            with open(f, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1:
                    last_vals = lines[-1].strip().split(',')
                    fit = float(last_vals[-2]) # fitness_average
                    
                    combo = os.path.basename(f).replace("_convergence.csv", "")
                    rep, sel, cross, mut = parse_combo(combo)
                    
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
        except:
            pass
            
    count += 1
    if count % 10 == 0: print(f"Scanned {count} instances...")

df = pd.DataFrame(all_results)
print(f"Loaded {len(df)} records.")

# --- ANALYSES & OUTPUTS ---

# 1. TABLE: Best For All Instance (Row 56)
print("Generating Table 1: Best per Instance...")
idx = df.groupby('Instance')['Fitness'].idxmin()
best_per_instance = df.loc[idx].sort_values('Instance')
best_per_instance.to_csv(os.path.join(OUTPUT_DIR, "1_Best_Method_per_Instance.csv"), index=False)

# 2. VISUALIZATION: Boxplot Global (Scaled)
print("Generating Fig 1: Scaled Boxplot...")
# Scale fitness: (X - min) / (max - min) per instance
min_max = df.groupby('Instance')['Fitness'].agg(['min', 'max'])
df = df.merge(min_max, on='Instance')
df['ScaledFitness'] = (df['Fitness'] - df['min']) / (df['max'] - df['min'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Representation', y='ScaledFitness', data=df, showfliers=False)
plt.title("Global Performance Distribution (Scaled Fitness 0-1)")
plt.ylabel("Scaled Gap (0=Best Known, 1=Worst)")
plt.savefig(os.path.join(OUTPUT_DIR, "2_Global_Representation_Boxplot.png"), dpi=300)
plt.close()

# 3. TABLE: Top 20 per Representation
print("Generating Table 2-4: Top 20 per Rep...")
avg_per_combo = df.groupby(['Representation', 'Selection', 'Crossover', 'Mutation', 'Combination']).agg(
    AvgFitness=('Fitness', 'mean'),
    AvgScaled=('ScaledFitness', 'mean') # Better for ranking
).reset_index()

for rep in ['permutation', 'binary', 'real_valued']:
    top20 = avg_per_combo[avg_per_combo['Representation'] == rep].sort_values('AvgScaled').head(20)
    top20.to_csv(os.path.join(OUTPUT_DIR, f"3_Top20_{rep}.csv"), index=False)

# 4. VISUALIZATION 3D Scatter (Average Fitness Space)
print("Generating Fig 2: 3D Plots...")
# We aggregate by AvgFitness for each S/C/M combo. 
# Mapping categorical to numbers for plotting
def plot_3d(rep_name):
    data_rep = avg_per_combo[avg_per_combo['Representation'] == rep_name]
    
    # Factorize
    data_rep = data_rep.copy()
    data_rep['Sel_Code'], _ = pd.factorize(data_rep['Selection'])
    data_rep['Cross_Code'], _ = pd.factorize(data_rep['Crossover'])
    data_rep['Mut_Code'], _ = pd.factorize(data_rep['Mutation'])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter: x=Sel, y=Cross, z=AvgScaledFitness
    # Color by Mutation
    sc = ax.scatter(data_rep['Sel_Code'], data_rep['Cross_Code'], data_rep['AvgScaled'], 
                    c=data_rep['Mut_Code'], cmap='viridis', s=50)
    
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Crossover Operator')
    ax.set_zlabel('Avg Scaled Cost (Lower is Better)')
    ax.set_title(f"3D Design Space: {rep_name}")
    plt.colorbar(sc, label='Mutation Type (Coded)')
    plt.savefig(os.path.join(OUTPUT_DIR, f"4_3D_Plot_{rep_name}.png"), dpi=300)
    plt.close()

for rep in ['permutation', 'binary', 'real_valued']:
    try:
        plot_3d(rep)
    except:
        pass

# 5. TABLE EXCEL BREAKDOWN: Rep/Sel/Cross/Mut vs C1/C2/R1...
print("Generating Table 5: Detailed Breakdown Matrix...")
# Pivot table logic
# We need groupings: [Rep], [Rep+Sel], [Rep+Cross], [Rep+Mut]
matrix_dfs = []

# Function to pivot avg scaled fitness per class
def create_pivot(group_cols, name):
    pivot = df.groupby(group_cols + ['Class'])['ScaledFitness'].mean().unstack()
    pivot['Category'] = name
    return pivot

# Representation Table
t1 = create_pivot(['Representation'], 'Representation')
t1.to_csv(os.path.join(OUTPUT_DIR, "5a_Breakdown_Representation.csv"))

# Selection Table
t2 = create_pivot(['Representation', 'Selection'], 'Selection')
t2.to_csv(os.path.join(OUTPUT_DIR, "5b_Breakdown_Selection.csv"))

# Crossover Table
t3 = create_pivot(['Representation', 'Crossover'], 'Crossover')
t3.to_csv(os.path.join(OUTPUT_DIR, "5c_Breakdown_Crossover.csv"))

# Mutation Table
t4 = create_pivot(['Representation', 'Mutation'], 'Mutation')
t4.to_csv(os.path.join(OUTPUT_DIR, "5d_Breakdown_Mutation.csv"))

# 6. HEATMAP Top 20 Algorithms (x=Instance, y=Algo)
print("Generating Fig 3: Top 20 Heatmap...")
top_20_global_names = avg_per_combo.sort_values('AvgScaled').head(20)['Combination'].tolist()
subset_hm = df[df['Combination'].isin(top_20_global_names)]

pivot_hm = subset_hm.pivot(index='Combination', columns='Instance', values='ScaledFitness')
# Sort index by rank
pivot_hm = pivot_hm.reindex(top_20_global_names)

plt.figure(figsize=(16, 8))
sns.heatmap(pivot_hm, cmap='RdYlGn_r', linewidths=0.1, linecolor='white')
plt.title("Performance Heatmap: Top 20 Global Algorithms across 56 Instances")
plt.xlabel("Solomon Instance")
plt.savefig(os.path.join(OUTPUT_DIR, "6_Heatmap_Top20.png"), dpi=300)
plt.close()

print("All Assets Generated Successfully in 'new_paper_assets'.")
