import os
import sys
import numpy as np
import pandas as pd
import time
import argparse
import multiprocessing as mp
from math import exp
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.getcwd())

from src.data.solomon_parser import SolomonParser
from src.data.uchoa_parser import UchoaParser
from src.representation.permutation import PermutationRepresentation
from src.selection.selection_methods import get_selection_method
from src.crossover.permutation_crossover import get_permutation_crossover
from src.mutation.permutation_mutation import get_permutation_mutation

# --- APST GA Configuration ---
APST_METHODS = ['LR', 'SVM', 'RF', 'DT', 'XGB']

POPULATION_SIZE = 100
MAX_GENERATIONS = 500
BOOTSTRAP_GENERATIONS = MAX_GENERATIONS // 2
NUM_RUNS = 5

Pcmax = 0.9
Pcmin = 0.6
Pmmax = 0.3
Pmmin = 0.1
A = 2.0

class APSTGeneticAlgorithm:
    def __init__(self, instance, ml_method: str, use_gpu: bool = False):
        self.instance = instance
        self.ml_method = ml_method
        self.use_gpu = use_gpu
        self.dist_matrix = instance.get_distance_matrix()
        
        self.representation = PermutationRepresentation(self.instance)
        self.selection = get_selection_method("roulette_wheel")
        # Ensure crossover & mutation models match the best combination found
        self.crossover = get_permutation_crossover("scx")
        self.mutation = get_permutation_mutation("inversion")
        
        self.ml_crossover_model = None
        self.ml_mutation_model = None
        self.ml_trained = False
        
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        self.training_data = {
            'features': [],
            'pc_targets': [],
            'pm_targets': [],
            'performance': []
        }
        
    def _to_maximization_fitness(self, distance):
        return 1.0 / distance if distance > 0 else 0.0

    def mathematical_crossover_rate(self, parent1_fit, parent2_fit, favg, fmin):
        f_prime = max(parent1_fit, parent2_fit)
        if f_prime <= favg:
            if favg != f_prime:
                pc = (Pcmax - Pcmin) / (2 * (favg - f_prime)) + Pcmax
            else:
                pc = Pcmax
        else:
            if favg != fmin:
                try:
                    exponent = A * (2 * (favg - f_prime) / (favg - fmin) - 1)
                    # Cap exponent to avoid overflow
                    exponent = min(700, max(-700, exponent))
                    pc = Pcmax / (1 + exp(exponent))
                except:
                    pc = Pcmax
            else:
                pc = Pcmax
        return max(Pcmin, min(Pcmax, pc))

    def mathematical_mutation_rate(self, individual_fit, favg, fmin):
        f = individual_fit
        if f <= favg:
            if favg != f:
                pm = (Pmmax - Pmmin) / (2 * (favg - f)) + Pmmax
            else:
                pm = Pmmax
        else:
            if favg != fmin:
                try:
                    exponent = A * (2 * (favg - f) / (favg - fmin) - 1)
                    exponent = min(700, max(-700, exponent))
                    pm = Pmmax / (1 + exp(exponent))
                except:
                    pm = Pmmax
            else:
                pm = Pmmax
        return max(Pmmin, min(Pmmax, pm))

    def train_ml_models(self):
        if len(self.training_data['features']) < 10:
            return False
            
        X = np.array(self.training_data['features'])
        y_pc = np.array(self.training_data['pc_targets'])
        y_pm = np.array(self.training_data['pm_targets'])
        X_scaled = self.scaler.fit_transform(X)
        
        if self.use_gpu:
            try:
                import cuml
                GPU_AVAILABLE = True
            except ImportError:
                GPU_AVAILABLE = False
        else:
            GPU_AVAILABLE = False

        if self.ml_method == 'LR':
            if GPU_AVAILABLE:
                from cuml.linear_model import LinearRegression
                self.ml_crossover_model = LinearRegression()
                self.ml_mutation_model = LinearRegression()
            else:
                from sklearn.linear_model import LinearRegression
                self.ml_crossover_model = LinearRegression()
                self.ml_mutation_model = LinearRegression()
                
        elif self.ml_method == 'DT':
            from sklearn.tree import DecisionTreeRegressor
            self.ml_crossover_model = DecisionTreeRegressor(random_state=42)
            self.ml_mutation_model = DecisionTreeRegressor(random_state=42)
            
        elif self.ml_method == 'RF':
            if GPU_AVAILABLE:
                from cuml.ensemble import RandomForestRegressor
                self.ml_crossover_model = RandomForestRegressor(n_estimators=50, random_state=42)
                self.ml_mutation_model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.ml_crossover_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                self.ml_mutation_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                
        elif self.ml_method == 'SVM':
            if GPU_AVAILABLE:
                from cuml.svm import SVR
                self.ml_crossover_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                self.ml_mutation_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            else:
                from sklearn.svm import SVR
                self.ml_crossover_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                self.ml_mutation_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                
        elif self.ml_method == 'XGB':
            try:
                import xgboost as xgb
                tree_method = 'hist'
                if self.use_gpu:
                    tree_method = 'gpu_hist' # Older versions
                    # Modern xgboost uses device='cuda'
                
                # Check modern xgboost syntax support
                try:
                    self.ml_crossover_model = xgb.XGBRegressor(n_estimators=50, random_state=42, 
                                                              tree_method='hist', device='cuda' if self.use_gpu else 'cpu')
                    self.ml_mutation_model = xgb.XGBRegressor(n_estimators=50, random_state=42, 
                                                             tree_method='hist', device='cuda' if self.use_gpu else 'cpu')
                except:
                    # Fallback to older syntax
                    self.ml_crossover_model = xgb.XGBRegressor(n_estimators=50, random_state=42, tree_method=tree_method)
                    self.ml_mutation_model = xgb.XGBRegressor(n_estimators=50, random_state=42, tree_method=tree_method)
            except ImportError:
                print("XGBoost not installed. Falling back to Random Forest.")
                from sklearn.ensemble import RandomForestRegressor
                self.ml_crossover_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                self.ml_mutation_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
        self.ml_crossover_model.fit(X_scaled, y_pc)
        self.ml_mutation_model.fit(X_scaled, y_pm)
        self.ml_trained = True
        return True

    def predict_parameters(self, generation, favg, fmin, fmax, diversity, p1_fit, p2_fit):
        if not self.ml_trained:
            return None, None
            
        features = np.array([[favg, fmin, fmax, diversity, generation/MAX_GENERATIONS, max(p1_fit, p2_fit), min(p1_fit, p2_fit)]])
        try:
            features_scaled = self.scaler.transform(features)
            pc_pred = self.ml_crossover_model.predict(features_scaled)[0]
            pm_pred = self.ml_mutation_model.predict(features_scaled)[0]
            
            # Extract float if array returned
            if isinstance(pc_pred, (list, np.ndarray)): pc_pred = pc_pred[0]
            if isinstance(pm_pred, (list, np.ndarray)): pm_pred = pm_pred[0]
                
            return max(Pcmin, min(Pcmax, pc_pred)), max(Pmmin, min(Pmmax, pm_pred))
        except:
            return None, None

    def calculate_diversity(self, population):
        if len(population) < 2: return 0.0
        # Compute exact diversity fast via unique representations or simplified sample
        # Using a sample to avoid O(N^2)
        sample_size = min(30, len(population))
        pop_sample = [population[i] for i in np.random.choice(len(population), sample_size, replace=False)]
        
        total_diff, comparisons = 0, 0
        for i in range(len(pop_sample)):
            for j in range(i + 1, len(pop_sample)):
                total_diff += np.sum(pop_sample[i] != pop_sample[j])
                comparisons += 1
        return total_diff / comparisons if comparisons > 0 else 0.0

    def run(self):
        start_time = time.time()
        
        population = [self.representation.create_chromosome() for _ in range(POPULATION_SIZE)]
        distances = np.array([self.representation.calculate_fitness(chrom, self.dist_matrix) for chrom in population])
        
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]
        best_chromosome = population[best_idx].copy()
        
        fitness_history = []
        diversity_history = []
        
        for generation in range(MAX_GENERATIONS):
            fitness_vals = np.array([self._to_maximization_fitness(d) for d in distances])
            favg, fmin, fmax = np.mean(fitness_vals), np.min(fitness_vals), np.max(fitness_vals)
            diversity = self.calculate_diversity(population)
            
            if generation == BOOTSTRAP_GENERATIONS:
                self.train_ml_models()
                
            new_population = []
            
            while len(new_population) < POPULATION_SIZE:
                parent_indices = self.selection.select(population, distances, num_parents=2, minimize=True)
                p1, p2 = population[parent_indices[0]], population[parent_indices[1]]
                p1_fit, p2_fit = fitness_vals[parent_indices[0]], fitness_vals[parent_indices[1]]
                
                # Determine Rates
                if generation < BOOTSTRAP_GENERATIONS or not self.ml_trained:
                    pc = self.mathematical_crossover_rate(p1_fit, p2_fit, favg, fmin)
                else:
                    pc, _ = self.predict_parameters(generation, favg, fmin, fmax, diversity, p1_fit, p2_fit)
                    if pc is None:
                        pc = self.mathematical_crossover_rate(p1_fit, p2_fit, favg, fmin)
                        
                c1, c2 = self.crossover.crossover(p1, p2, crossover_rate=pc, dist_matrix=self.dist_matrix)
                
                # Mutate c1
                if generation < BOOTSTRAP_GENERATIONS or not self.ml_trained:
                    pm1 = self.mathematical_mutation_rate(p1_fit, favg, fmin)
                else:
                    _, pm1 = self.predict_parameters(generation, favg, fmin, fmax, diversity, p1_fit, p1_fit)
                    if pm1 is None: pm1 = self.mathematical_mutation_rate(p1_fit, favg, fmin)
                c1 = self.mutation.mutate(c1, mutation_rate=pm1)
                
                # Mutate c2
                if generation < BOOTSTRAP_GENERATIONS or not self.ml_trained:
                    pm2 = self.mathematical_mutation_rate(p2_fit, favg, fmin)
                else:
                    _, pm2 = self.predict_parameters(generation, favg, fmin, fmax, diversity, p2_fit, p2_fit)
                    if pm2 is None: pm2 = self.mathematical_mutation_rate(p2_fit, favg, fmin)
                c2 = self.mutation.mutate(c2, mutation_rate=pm2)
                
                c1 = self.representation.repair(c1)
                c2 = self.representation.repair(c2)
                
                # Collect Data
                if generation < BOOTSTRAP_GENERATIONS:
                    d1 = self.representation.calculate_fitness(c1, self.dist_matrix)
                    d2 = self.representation.calculate_fitness(c2, self.dist_matrix)
                    f1 = self._to_maximization_fitness(d1)
                    f2 = self._to_maximization_fitness(d2)
                    improvement1 = max(0, f1 - p1_fit)
                    improvement2 = max(0, f2 - p2_fit)
                    
                    self.training_data['features'].append([favg, fmin, fmax, diversity, generation/MAX_GENERATIONS, max(p1_fit, p2_fit), min(p1_fit, p2_fit)])
                    self.training_data['pc_targets'].append(pc)
                    self.training_data['pm_targets'].append(pm1)
                    self.training_data['performance'].append((improvement1 + improvement2)/2)
                    
                new_population.extend([c1, c2])
                
            population = new_population[:POPULATION_SIZE]
            distances = np.array([self.representation.calculate_fitness(chrom, self.dist_matrix) for chrom in population])
            
            current_best_idx = np.argmin(distances)
            if distances[current_best_idx] < best_distance:
                best_distance = distances[current_best_idx]
                best_chromosome = population[current_best_idx].copy()
                
            fitness_history.append(best_distance)
            diversity_history.append(diversity)
            
        runtime = time.time() - start_time
        return {
            'best_fitness': best_distance,
            'runtime': runtime,
            'fitness_history': fitness_history,
            'diversity_history': diversity_history
        }

def worker_func(task):
    instance_name, dataset_type, category, ml_method, solomon_base, uchoa_base, results_dir, use_gpu = task
    
    # Parse dataset
    is_solomon = (dataset_type == 'Solomon')
    if is_solomon:
        # Reconstruct path based on solomon base path plus category if available
        # Usually solomon files are inside base_path/C1/ etc or just flat.
        # Let's check if category subfolder exists
        cat_path = os.path.join(solomon_base, str(category))
        if os.path.exists(cat_path):
            parser_path = os.path.join(cat_path, f"{instance_name}.csv")
        else:
            parser_path = os.path.join(solomon_base, f"{instance_name}.csv")
            if not os.path.exists(parser_path):
                # Search recursively
                found = False
                for root, _, files in os.walk(solomon_base):
                    if f"{instance_name}.csv" in files:
                        parser_path = os.path.join(root, f"{instance_name}.csv")
                        found = True
                        break
                if not found:
                    print(f"File not found for {instance_name} in {solomon_base}")
                    return
    else:
        parser_path = os.path.join(uchoa_base, f"{instance_name}.vrp")
        if not os.path.exists(parser_path):
            # Check with .sol or just try to find it
            found = False
            for root, _, files in os.walk(uchoa_base):
                if f"{instance_name}.vrp" in files:
                    parser_path = os.path.join(root, f"{instance_name}.vrp")
                    found = True
                    break
            if not found:
                print(f"File not found for {instance_name} in {uchoa_base}")
                return
    
    try:
        if is_solomon:
            instance = SolomonParser.parse(parser_path)
        else:
            instance = UchoaParser.parse(parser_path)
    except Exception as e:
        print(f"Failed to parse {instance_name}: {e}")
        return
        
    print(f"Starting {ml_method} on {dataset_type} instance {instance_name} (GPU: {use_gpu})...")
    
    # Run NUM_RUNS times
    all_fitness = []
    all_diversity = []
    for r in range(NUM_RUNS):
        ga = APSTGeneticAlgorithm(instance, ml_method, use_gpu)
        res = ga.run()
        all_fitness.append(res['fitness_history'])
        all_diversity.append(res['diversity_history'])
        
    # Average across runs
    fitness_avg = np.mean(np.array(all_fitness), axis=0)
    diversity_avg = np.mean(np.array(all_diversity), axis=0)
    
    # Save CSV
    df = pd.DataFrame({'generation': range(1, MAX_GENERATIONS + 1)})
    for r in range(NUM_RUNS):
        df[f'fitness_run_{r+1}'] = all_fitness[r]
        df[f'diversity_run_{r+1}'] = all_diversity[r]
    df['fitness_average'] = fitness_avg
    df['diversity_average'] = diversity_avg
    
    # Save partitioned by dataset
    dataset_out_dir = os.path.join(results_dir, dataset_type, instance_name)
    os.makedirs(dataset_out_dir, exist_ok=True)
    out_file = os.path.join(dataset_out_dir, f"APST_GA_{ml_method}_convergence.csv")
    df.to_csv(out_file, index=False)
    print(f"Finished {ml_method} on {instance_name}. Best Average: {fitness_avg[-1]:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Run ML APST GA experiments.")
    parser.add_argument("--sampled_csv", type=str, default="data/sampled_instances.csv", help="Path to sampled_instances.csv")
    parser.add_argument("--solomon_dir", type=str, default="data/solomon", help="Path to Solomon benchmark data")
    parser.add_argument("--uchoa_dir", type=str, default="data/Uchoa", help="Path to Uchoa benchmark data")
    parser.add_argument("--results_dir", type=str, default="results_ml", help="Output directory for ML results")
    parser.add_argument("--methods", type=str, nargs="+", default=APST_METHODS, help="ML methods to test (e.g. LR SVM RF DT XGB)")
    parser.add_argument("--instance", type=str, default=None, help="Process only specific instance")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration where possible (cuML/XGBoost)")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel processes (1 for sequential, -1 for all cores)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sampled_csv):
        print(f"Sampled instances CSV not found at {args.sampled_csv}!")
        return
        
    df = pd.read_csv(args.sampled_csv)
    
    if args.instance:
        # Strip extensions if provided
        target = os.path.splitext(args.instance)[0]
        df = df[df['InstanceName'].str.startswith(target)]
        if df.empty:
            print(f"Instance {args.instance} not found in sample list.")
            return
            
    tasks = []
    for _, row in df.iterrows():
        filename = row['InstanceName']
        dataset = row['Dataset']
        category = row.get('Category', '')
        instance_name = os.path.splitext(filename)[0]
        
        for ml_method in args.methods:
            tasks.append((instance_name, dataset, category, ml_method, args.solomon_dir, args.uchoa_dir, args.results_dir, args.use_gpu))
            
    print(f"Total tasks planned: {len(tasks)}")

    num_workers = mp.cpu_count() if args.jobs == -1 else args.jobs
    # If GPU is enabled, we should probably restrict to 1 parallel process to avoid CUDA OOM, 
    # but let the user override with --jobs. By default it's 1.
    if args.use_gpu and args.jobs == -1:
         print("Warning: Running multiple GPU processes might cause memory errors.")
         
    if num_workers > 1:
        with mp.Pool(processes=min(num_workers, len(tasks))) as pool:
            pool.map(worker_func, tasks)
    else:
        for t in tasks:
            worker_func(t)
            
    print("All tasks finished.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
