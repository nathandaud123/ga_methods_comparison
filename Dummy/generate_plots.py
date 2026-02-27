import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys

# Define directories
base_dir = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\data\solomon"
output_dir = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\plot"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Find all CSV files recursively
csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
print(f"Found {len(csv_files)} CSV files.")

if not csv_files:
    print("No CSV files found in " + base_dir)
    sys.exit(1)

for file_path in csv_files:
    try:
        # Read CSV
        # Use simple read first, trimming spaces in headers
        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        
        # Check if required columns exist
        required_cols = ['XCOORD.', 'YCOORD.', 'DEMAND']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file_path}: Missing columns. Found: {df.columns}")
            continue

        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        plt.figure(figsize=(10, 8))
        
        x = df['XCOORD.']
        y = df['YCOORD.']
        demand = df['DEMAND']
        
        # Plot all points
        # Color by demand to show "capacity or whatever info"
        sc = plt.scatter(x, y, c=demand, cmap='viridis', s=100, edgecolors='k', alpha=0.7)
        
        # Add labels to points (Cust No)
        if 'CUST NO.' in df.columns:
            for i, txt in enumerate(df['CUST NO.']):
                # Annotate every point? Might be too crowded. 
                # Let's annotate 20% or just the depot and a few others?
                # User asked to just plot them.
                # Let's simple annotate the depot (first one).
                pass
        
        # Highlight Depot (assumed to be the first row)
        plt.scatter(x.iloc[0], y.iloc[0], c='red', s=200, marker='*', label='Depot', edgecolors='white')
        
        plt.title(f"Solomon Instance: {name_no_ext}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.colorbar(sc, label='Demand')
        plt.legend()
        plt.grid(False) # "Jangan pakai grid"
        
        save_path = os.path.join(output_dir, f"{name_no_ext}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Generated plot: {save_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Done.")
