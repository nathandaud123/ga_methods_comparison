import os
import pandas as pd
import glob
import numpy as np

# Config
RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\final_submission_assets"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Scanning all result files for Breakdown Tables with Representation...")

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

# Scan Loop
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
print(f"Data Loaded: {len(df_full)} records.")

# Helper to generate and save table
def save_table(group_cols, filename):
    # Group by provided columns (list)
    stats = df_full.groupby(group_cols)['Fitness'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats = stats.sort_values(by=group_cols if isinstance(group_cols, list) and group_cols[0] == 'Representation' else 'mean') 
    
    # Sort by 'mean' fitness to show best on top, 
    # but if grouped by Representation, maybe sort by Rep then Mean? 
    # Actually, User generally wants performance rank. Let's sort by 'mean' (ascending, as lower is better)
    stats = stats.sort_values('mean')

    # Dynamic Column Renaming
    cols = list(group_cols) if isinstance(group_cols, list) else [group_cols]
    cols.extend(['Sample Count', 'Average Fitness', 'Std Dev', 'Best Fitness', 'Worst Fitness'])
    stats.columns = cols
    
    # Rounding
    stats = stats.round(2)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    stats.to_csv(save_path, index=False)
    print(f"Saved: {filename}")

# 1. Representation Table
save_table(['Representation'], '5a_Breakdown_Representation.csv')

# 2. Selection Table: NOW SPLIT BY REPRESENTATION
save_table(['Representation', 'Selection'], '5b_Breakdown_Selection.csv')

# 3. Crossover Table: Split by Representation
save_table(['Representation', 'Crossover'], '5c_Breakdown_Crossover.csv')

# 4. Mutation Table: Split by Representation
save_table(['Representation', 'Mutation'], '5d_Breakdown_Mutation.csv')

print("\nBreakdown tables generation complete.")
