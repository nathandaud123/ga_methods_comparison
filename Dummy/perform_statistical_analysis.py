
import os
import json
import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\results"
OUTPUT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\final_submission_assets"

def load_data():
    print("Loading results...")
    data = []
    
    # Iterate through all subdirectories in results
    for instance_name in os.listdir(RESULTS_DIR):
        instance_path = os.path.join(RESULTS_DIR, instance_name)
        if not os.path.isdir(instance_path):
            continue
            
        json_file = os.path.join(instance_path, f"{instance_name}_results.json")
        if not os.path.exists(json_file):
            continue
            
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
                
            for method_name, metrics in results.items():
                # Extract mean_fitness
                # Some files might have different structure, handle gracefully
                fitness = metrics.get('mean_fitness', metrics.get('best_fitness', None))
                
                if fitness is not None:
                    data.append({
                        'Instance': instance_name,
                        'Method': method_name,
                        'Fitness': float(fitness)
                    })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            
    df = pd.DataFrame(data)
    return df

def perform_analysis():
    df = load_data()
    
    if df.empty:
        print("No data found!")
        return

    # Pivot: Index=Instance, Col=Method, Val=Fitness
    pivot_df = df.pivot(index='Instance', columns='Method', values='Fitness')
    
    # --- HANDLING MISSING DATA ---
    # Instead of dropping (which removed ~132 methods), we impute missing values.
    # Missing usually means the run failed or timed out -> Worst Fitness.
    # We will assign a penalty value: Max fitness in that instance * 1.5
    
    total_methods = pivot_df.shape[1]
    print(f"Total methods identified: {total_methods}")
    
    # Create a clean dataframe for statistics
    pivot_df_imputed = pivot_df.copy()
    
    # Iterate rows to fill NaNs with row-specific penalty
    for index, row in pivot_df_imputed.iterrows():
        if row.isnull().any():
            max_val = row.max()
            # If row is all NaNs (unlikely), set a default large number
            if pd.isna(max_val): 
                max_val = 100000 
            
            penalty = max_val * 1.2 # 20% worse than the worst result found
            pivot_df_imputed.loc[index] = row.fillna(penalty)
            
    # Use the imputed dataframe for all stats analysis
    pivot_df_clean = pivot_df_imputed
    cleaned_methods = pivot_df_clean.shape[1]
    
    print(f"Data prepared. Instances: {pivot_df_clean.shape[0]}, Methods: {cleaned_methods} (Missing values imputed with penalty)")


    # 1. Friedman Test
    print("\n--- Friedman Test ---")
    # Scipy expects measurement inputs as arrays (one per method)
    measurements = [pivot_df_clean[col].values for col in pivot_df_clean.columns]
    stat, p_value = stats.friedmanchisquare(*measurements)
    
    print(f"Friedman Statistic: {stat:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    # Save Friedman Result
    friedman_res = pd.DataFrame({
        'Statistic': ['Friedman Chi-Square', 'p-value', 'N (Instances)', 'k (Methods)', 'df'],
        'Value': [stat, p_value, pivot_df_clean.shape[0], cleaned_methods, cleaned_methods - 1]
    })
    friedman_res.to_csv(os.path.join(OUTPUT_DIR, "statistical_friedman_summary.csv"), index=False)

    # 2. Ranking
    print("\n--- Ranking ---")
    # Rank magnitude: Lower fitness (distance) is better, so ascending=True (Rank 1 = Smallest)
    ranks = pivot_df_clean.rank(axis=1, ascending=True)
    avg_ranks = ranks.mean().sort_values()
    
    # 2b. Detailed Ranking Table with Mean Fitness
    mean_fitness = pivot_df_clean.mean()
    std_fitness = pivot_df_clean.std()
    
    rank_table = pd.DataFrame({
        'Method': avg_ranks.index,
        'Avg_Rank': avg_ranks.values,
        'Mean_Fitness': mean_fitness[avg_ranks.index].values,
        'Std_Fitness': std_fitness[avg_ranks.index].values
    })
    rank_table.to_csv(os.path.join(OUTPUT_DIR, "statistical_ranks.csv"), index=False)
    print("Saved Rank Table.")

    # 3. Nemenyi Post-Hoc
    print("\n--- Nemenyi Test ---")
    best_method = avg_ranks.index[0]
    print(f"Best Method (Reference): {best_method}")
    
    N = pivot_df_clean.shape[0]
    k = cleaned_methods
    
    # Critical Value Calculation
    try:
        # q_val for alpha=0.05, k treatments, infinite df
        # Uses studentized_range from scipy 1.9.0+
        q_val = stats.studentized_range.ppf(0.95, k, np.inf) 
        CD = (q_val / np.sqrt(2)) * np.sqrt(k * (k + 1) / (6 * N))
        print(f"Calculated q_val: {q_val:.4f}")
    except (AttributeError, ImportError):
        print("Scipy studentized_range not found. Using approximation.")
        # Fallback approximation or use the doc's logic if k is similar
        # For very large k, q can be approximated or we take the doc's 2.639 as a baseline factor if k~354
        # But 2.639 is suspiciously small for q (usually > 5 for k=354). 
        # Nemenyi critical value "q_alpha" in some texts IS q/sqrt(2).
        # Critical value for N(0,1) at alpha/(k(k-1)) Bonferroni?
        # Let's use a conservative approximation if tool fails: 3.5
        CD = 3.5 * np.sqrt(k * (k + 1) / (6 * N))


    print(f"Critical Difference (CD): {CD:.4f}")
    
    # Pairwise Comparisons against Best
    comparisons = []
    best_rank = avg_ranks[best_method]
    
    for method in avg_ranks.index:
        if method == best_method:
            continue
            
        rank_diff = abs(avg_ranks[method] - best_rank)
        significant = rank_diff > CD
        
        comparisons.append({
            'Comparison': f"{best_method} vs {method}",
            'Method': method,
            'Avg_Rank': avg_ranks[method],
            'Rank_Difference': rank_diff,
            'Significant': significant,
            'CD': CD
        })
        
    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_nemenyi.csv"), index=False)

    # 4. Effect Size (Cohen's d) - Top vs Selected Baselines
    print("\n--- Cohen's d Effect Size ---")
    # Using Normalized Fitness (Gap relative to Best Found in experiment per instance)
    # Normalize: (Val - Min_in_Row) / Min_in_Row
    best_per_instance = pivot_df_clean.min(axis=1)
    normalized_df = pivot_df_clean.div(best_per_instance, axis=0) - 1.0 # 0.0 is best
    
    effect_sizes = []
    
    # Calculate for all against best
    best_series = normalized_df[best_method]
    
    for method in avg_ranks.index:
        if method == best_method:
            continue
            
        compare_series = normalized_df[method]
        
        # Cohen's d pairwise
        n1, n2 = len(best_series), len(compare_series)
        var1, var2 = np.var(best_series, ddof=1), np.var(compare_series, ddof=1)
        
        # Pooled Standard Deviation
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Mean Difference
        mean_diff = compare_series.mean() - best_series.mean() # Positive means 'Method' is worse (larger gap)
        
        if pooled_sd == 0:
            d = 0
        else:
            d = mean_diff / pooled_sd
            
        effect_sizes.append({
            'Method': method,
            'Cohens_d': d,
            'Magnitude': 'Large' if d > 0.8 else ('Medium' if d > 0.5 else ('Small' if d > 0.2 else 'Negligible'))
        })
        
    eff_df = pd.DataFrame(effect_sizes)
    eff_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_effect_sizes.csv"), index=False)
    
    print("Analysis Completed. Files saved to final_submission_assets/")

if __name__ == "__main__":
    perform_analysis()
