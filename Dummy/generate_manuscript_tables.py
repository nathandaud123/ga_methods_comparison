
import pandas as pd
import os

ASSETS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\final_submission_assets"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\statistic"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def format_method_name(name):
    # Optional: Make method names prettier if needed, e.g., replace underscores with hyphens or Title Case
    # For now, keep as is or just replace underscores for readability
    return name.replace('_', '-')

def generate_manuscript_tables():
    print("Generating manuscript tables in", OUTPUT_DIR)
    
    # --- Load Data ---
    friedman_df = pd.read_csv(os.path.join(ASSETS_DIR, "statistical_friedman_summary.csv"))
    ranks_df = pd.read_csv(os.path.join(ASSETS_DIR, "statistical_ranks.csv"))
    nemenyi_df = pd.read_csv(os.path.join(ASSETS_DIR, "statistical_nemenyi.csv"))
    effect_df = pd.read_csv(os.path.join(ASSETS_DIR, "statistical_effect_sizes.csv"))

    # --- Table [X]. Friedman Test Summary ---
    # Just formatted cleanly
    # Expected cols: Statistic, Value
    friedman_df.columns = ['Statistic', 'Value']
    friedman_df.to_csv(os.path.join(OUTPUT_DIR, "Table_X_Friedman_Summary.csv"), index=False)
    print("Created Table X")

    # --- Table [Y]. Average Friedman Ranks (Top 15) ---
    # Filter top 15
    top_15 = ranks_df.sort_values("Avg_Rank").head(15).copy()
    top_15['Method'] = top_15['Method'].apply(format_method_name)
    top_15 = top_15[['Method', 'Avg_Rank', 'Mean_Fitness', 'Std_Fitness']]
    # Round
    top_15['Avg_Rank'] = top_15['Avg_Rank'].round(2)
    top_15['Mean_Fitness'] = top_15['Mean_Fitness'].round(2)
    top_15['Std_Fitness'] = top_15['Std_Fitness'].round(2)
    
    top_15.to_csv(os.path.join(OUTPUT_DIR, "Table_Y_Average_Ranks_Top15.csv"), index=False)
    print("Created Table Y")

    # --- Table [Z]. Nemenyi Post-Hoc (Top 15 + Significance) ---
    # Should correspond to the methods in Table Y generally.
    # Nemenyi matches "Best" vs others.
    # Filter nemenyi rows where 'Method' is in our Top 15 list (excluding the Best itself which isn't in the comparison list usually)
    
    best_method = ranks_df.iloc[0]['Method']
    top_15_names = ranks_df.sort_values("Avg_Rank").head(16)['Method'].tolist() # unique names
    
    # Filter Nemenyi to only show comparisons relevant to the top players
    nemenyi_filtered = nemenyi_df[nemenyi_df['Method'].isin(top_15_names)].copy()
    
    nemenyi_filtered['Method'] = nemenyi_filtered['Method'].apply(format_method_name)
    nemenyi_filtered['Comparison'] = nemenyi_filtered['Comparison'].apply(lambda x: x.replace('_', '-'))
    
    # Select cols
    # Comparison, Avg_Rank (of the method), Rank_Difference, CD, Significant
    nemenyi_final = nemenyi_filtered[['Method', 'Avg_Rank', 'Rank_Difference', 'CD', 'Significant']]
    nemenyi_final['Avg_Rank'] = nemenyi_final['Avg_Rank'].round(2)
    nemenyi_final['Rank_Difference'] = nemenyi_final['Rank_Difference'].round(2)
    nemenyi_final['CD'] = nemenyi_final['CD'].round(2)
    
    emenyi_filename = f"Table_Z_Nemenyi_PostHoc_vs_{format_method_name(best_method)}.csv"
    # Rename specifically to generic Table Z for user convenience
    nemenyi_final.to_csv(os.path.join(OUTPUT_DIR, "Table_Z_Nemenyi_PostHoc.csv"), index=False)
    print("Created Table Z")

    # --- Table [W]. Cohen's d Effect Sizes (Top 15) ---
    # Join with ranks to get context or just filter by top 15 names
    effect_filtered = effect_df[effect_df['Method'].isin(top_15_names)].copy()
    effect_filtered['Method'] = effect_filtered['Method'].apply(format_method_name)
    
    # Sort by Cohens_d (usually small to large if we look at top ranks, actually top ranks have small effect size difference from best)
    # Let's sort to match Rank order
    rank_map = {format_method_name(m): r for m, r in zip(ranks_df['Method'], ranks_df['Avg_Rank'])}
    effect_filtered['Rank_Ref'] = effect_filtered['Method'].map(rank_map)
    effect_filtered = effect_filtered.sort_values('Rank_Ref')
    
    effect_final = effect_filtered[['Method', 'Cohens_d', 'Magnitude']]
    effect_final['Cohens_d'] = effect_final['Cohens_d'].round(2)
    
    effect_final.to_csv(os.path.join(OUTPUT_DIR, "Table_W_Cohens_d.csv"), index=False)
    print("Created Table W")

if __name__ == "__main__":
    generate_manuscript_tables()
