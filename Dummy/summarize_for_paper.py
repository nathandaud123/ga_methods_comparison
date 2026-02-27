
import pandas as pd
import os

ASSETS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\final_submission_assets"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\paper_tables"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def parse_method_name(name):
    parts = name.split('_')
    # Heuristics to parse standard names: representation_selection_crossover_mutation
    # This might need adjustment based on exact naming convention used in the json
    # Example: permutation_roulette_wheel_pmx_swap
    
    # Representations: permutation, binary, real_valued
    rep = parts[0]
    if rep == "real": rep = "real_valued" # handle real_valued split
    
    return rep

def summarize_results():
    print("Generating summaries suitable for paper...")
    
    # Load Ranks (Main source of truth)
    ranks_df = pd.read_csv(os.path.join(ASSETS_DIR, "statistical_ranks.csv"))
    nemenyi_df = pd.read_csv(os.path.join(ASSETS_DIR, "statistical_nemenyi.csv"))
    
    # 1. Top 20 Methods (The winners)
    top_20 = ranks_df.head(20).copy()
    top_20['Rank'] = range(1, 21)
    top_20 = top_20[['Rank', 'Method', 'Avg_Rank', 'Mean_Fitness', 'Std_Fitness']]
    top_20['Avg_Rank'] = top_20['Avg_Rank'].round(2)
    top_20['Mean_Fitness'] = top_20['Mean_Fitness'].round(1)
    top_20['Std_Fitness'] = top_20['Std_Fitness'].round(1)
    
    top_20.to_csv(os.path.join(OUTPUT_DIR, "Table_Top20_Methods.csv"), index=False)
    print("Generated Table_Top20_Methods.csv")

    # 2. Aggregated Performance by Representation
    # We need to tag each row with its representation
    ranks_df['Representation'] = ranks_df['Method'].apply(lambda x: x.split('_')[0])
    # Correction for 'real_valued' which might get split as 'real'
    ranks_df.loc[ranks_df['Method'].str.startswith('real_valued'), 'Representation'] = 'real_valued'
    
    rep_stats = ranks_df.groupby('Representation').agg({
        'Avg_Rank': ['mean', 'min', 'count'],
        'Mean_Fitness': 'mean'
    }).reset_index()
    rep_stats.columns = ['Representation', 'Avg_Rank_Mean', 'Best_Rank', 'Count', 'Global_Mean_Fitness']
    rep_stats = rep_stats.sort_values('Avg_Rank_Mean')
    
    rep_stats.to_csv(os.path.join(OUTPUT_DIR, "Table_Representation_Comparison.csv"), index=False)
    print("Generated Table_Representation_Comparison.csv")

    # 3. Aggregated Performance by Operators (Regex/Substring search)
    # This is rough but effective. We define known operators and search for them in method names.
    operators = {
        'Crossover': ['pmx', 'ox', 'cx', 'erx', 'scx', 'uniform', 'two_point', 'single_point', 'multi_point', 'shuffle', 'sbx', 'flat', 'blx_alpha'],
        'Mutation': ['swap', 'insert', 'inversion', 'scramble', 'displacement', 'interchanging', 'bit_flip', 'polynomial', 'gaussian', 'uniform']
        # Selection is harder to isolate safely without stricter naming, but let's try common ones
        # 'Selection': ['roulette_wheel', 'tournament', 'rank', 'stochastic_universal', 'boltzmann', 'elitism']
    }
    
    op_data = []
    
    for op_type, op_list in operators.items():
        for op in op_list:
            # Filter methods containing this operator name
            # Be careful with overlapping names (e.g. 'uniform' is in both cross and mut, and binary rep)
            # We assume position or distinct naming helps, but raw average is a good start.
            
            # Strict check: 
            # Crossover is usually 3rd component? method_rep_sel_CROSS_mut
            # Mutation is last?
            # Let's just do "contains" for broad overview, but exclude conflicting contexts if possible.
            
            matches = ranks_df[ranks_df['Method'].str.contains(op)]
            if not matches.empty:
                op_data.append({
                    'Type': op_type,
                    'Operator': op,
                    'Avg_Rank': matches['Avg_Rank'].mean(),
                    'Methods_Count': len(matches),
                    'Best_Rank_With_Op': matches['Avg_Rank'].min()
                })
                
    op_df = pd.DataFrame(op_data).sort_values(['Type', 'Avg_Rank'])
    op_df['Avg_Rank'] = op_df['Avg_Rank'].round(2)
    op_df.to_csv(os.path.join(OUTPUT_DIR, "Table_Operator_Performance.csv"), index=False)
    print("Generated Table_Operator_Performance.csv")

    # 4. Nemenyi Significant Groups (Simplified)
    # Instead of a huge matrix, list methods that are statistically EQUAL to the Best Method.
    # From Nemenyi CSV, find rows where Significant = False
    
    rank_1_method = ranks_df.iloc[0]['Method']
    
    # Check nemenyi file for "Significant" column
    # The file compares Rank 1 vs Others
    # We want the list of methods where Significant == False
    
    # Filter nemenyi for comparisons involving the best method (Reference)
    # The file structure is: Comparison, Method, ... Significant, CD
    # Assume 'Method' col is the one being compared TO the best (if row 1 is reference)
    
    # Actually, my previous script generated comparison of "Best vs X".
    # So we just filter Significant == False
    insignificant_diff = nemenyi_df[nemenyi_df['Significant'] == False]
    
    # Method names
    statistically_tied = insignificant_diff['Method'].tolist()
    # Add the winner itself
    statistically_tied.insert(0, rank_1_method)
    
    tied_df = ranks_df[ranks_df['Method'].isin(statistically_tied)]
    tied_df = tied_df[['Method', 'Avg_Rank', 'Mean_Fitness']]
    
    tied_df.to_csv(os.path.join(OUTPUT_DIR, "Table_Statistically_Equivalent_Best.csv"), index=False)
    print(f"Generated Table_Statistically_Equivalent_Best.csv ({len(tied_df)} methods tied with top result)")

if __name__ == "__main__":
    summarize_results()
