import os
import pandas as pd
import numpy as np

# Config
DATA_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\data\solomon"

def parse_solomon_stats(file_path):
    # Determine type based on filename
    name = os.path.basename(file_path).replace(".csv", "")
    if name.startswith("C1"): ctype = "C1"
    elif name.startswith("C2"): ctype = "C2"
    elif name.startswith("R1"): ctype = "R1"
    elif name.startswith("R2"): ctype = "R2"
    elif name.startswith("RC1"): ctype = "RC1"
    elif name.startswith("RC2"): ctype = "RC2"
    else: return None

    # Read CSV
    # Assuming CSV format as seen in C101.csv: 
    # CUST NO.,XCOORD.,YCOORD.,DEMAND,READY TIME,DUE DATE,SERVICE TIME
    try:
        df = pd.read_csv(file_path)
    except:
        return None
        
    # Standardize cols
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Extract columns (Skip Depot which is usually first row with Demand 0 or ID 0/1)
    # In C101 example, Row 1 (Index 0) is Depot (Demand 0).
    # We want stats for CUSTOMERS only usually.
    # Let's separate Depot
    depot = df.iloc[0]
    customers = df.iloc[1:]
    
    # Columns of interest - Ensure Numeric
    # Force convert to numeric, handling potential string issues
    for col in ['DEMAND', 'DUE DATE', 'READY TIME', 'SERVICE TIME', 'XCOORD.', 'YCOORD.']:
        customers[col] = pd.to_numeric(customers[col], errors='coerce')

    demands = customers['DEMAND']
    time_windows = customers['DUE DATE'] - customers['READY TIME'] # Width
    x_coords = customers['XCOORD.']
    y_coords = customers['YCOORD.']
    
    # Calculate Spatial Density (Avg distance to depot?)
    # or Spread (Std Dev of X and Y)
    
    return {
        'Type': ctype,
        'Capacity': 0, # Placeholder, not in CSV usually
        'Num_Cust': len(customers),
        'Avg_Demand': demands.mean(),
        'Std_Demand': demands.std(),
        'Avg_TW_Width': time_windows.mean(),
        'Std_TW_Width': time_windows.std(),
        'Avg_Service': customers['SERVICE TIME'].mean(),
        'X_Spread': x_coords.std(),
        'Y_Spread': y_coords.std()
    }

print("Scanning Solomon instances for numerical stats...")

stats_list = []
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith(".csv"):
            path = os.path.join(root, f)
            res = parse_solomon_stats(path)
            if res:
                stats_list.append(res)

df = pd.DataFrame(stats_list)

# Group by Type and aggregate
# We want Avg (Mean) of the file-level Means, and maybe Std of them.
# Or just the Mean of attributes.
summary = df.groupby('Type').agg({
    'Num_Cust': 'mean', # Should be 100
    'Avg_Demand': ['mean', 'std'],
    'Avg_TW_Width': ['mean', 'std'],
    'Avg_Service': 'mean',
    'X_Spread': 'mean', # measure of clustering/spread
    # 'Y_Spread': 'mean' 
}).round(1)

print("\n--- Summary Table Data ---")
print(summary)

# Helper to format for user
# We will manually construct the Markdown table from this output
summary.to_csv("solomon_stats_summary.csv")
print("Saved stats to solomon_stats_summary.csv")
