import os
import sys
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
import glob
import random
from copy import deepcopy

# Add current dir to path to find src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from src.representation.permutation import PermutationRepresentation
except ImportError:
    print("Error: Could not find 'src' directory. Please run this script from the project root.")
    sys.exit(1)

# Parameters for Apple-to-Apple comparison
POPULATION_SIZE = 100
MAX_ITERATIONS = 500
NUM_RUNS = 5

class MetaheuristicBase:
    def __init__(self, instance):
        self.instance = instance
        self.dist_matrix = instance.get_distance_matrix()
        self.representation = PermutationRepresentation(self.instance)
        self.num_nodes = self.instance.num_customers
        
    def get_fitness(self, chrom):
        return self.representation.calculate_fitness(chrom, self.dist_matrix)

    def calculate_diversity(self, population):
        if len(population) < 2: return 0.0
        sample_size = min(len(population), 10)
        idx = np.random.choice(len(population), sample_size, replace=False)
        total_diff, comparisons = 0, 0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                total_diff += np.sum(population[idx[i]] != population[idx[j]])
                comparisons += 1
        return total_diff / comparisons if comparisons > 0 else 0.0

# --- 1. Simulated Annealing (SA) ---
class SimulatedAnnealing(MetaheuristicBase):
    def run(self):
        start_time = time.time()
        current = self.representation.create_chromosome()
        current_fit = self.get_fitness(current)
        best = current.copy()
        best_fit = current_fit
        T = 1000.0
        alpha = 0.99
        history, div_history = [], []
        for i in range(MAX_ITERATIONS):
            neighbor = current.copy()
            idx1, idx2 = random.sample(range(self.num_nodes), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            neighbor_fit = self.get_fitness(neighbor)
            delta = neighbor_fit - current_fit
            if delta < 0 or random.random() < np.exp(-delta / T):
                current, current_fit = neighbor, neighbor_fit
                if current_fit < best_fit:
                    best, best_fit = current.copy(), current_fit
            T *= alpha
            history.append(best_fit)
            div_history.append(0.0)
        return {'best_fitness': best_fit, 'runtime': time.time() - start_time, 'fitness_history': history, 'diversity_history': div_history}

# --- 2. Particle Swarm Optimization (PSO) ---
class PSO(MetaheuristicBase):
    def run(self):
        start_time = time.time()
        swarm = [self.representation.create_chromosome() for _ in range(POPULATION_SIZE)]
        pbest = [s.copy() for s in swarm]
        pbest_fit = [self.get_fitness(s) for s in swarm]
        gbest_idx = np.argmin(pbest_fit)
        gbest, gbest_fit = pbest[gbest_idx].copy(), pbest_fit[gbest_idx]
        w, c1, c2 = 0.7, 1.4, 1.4
        history, div_history = [], []
        for it in range(MAX_ITERATIONS):
            new_swarm = []
            for i in range(POPULATION_SIZE):
                current = swarm[i].copy()
                if random.random() < c1: current = self._move_towards(current, pbest[i])
                if random.random() < c2: current = self._move_towards(current, gbest)
                if random.random() < (1-w):
                    idx1, idx2 = random.sample(range(self.num_nodes), 2)
                    current[idx1], current[idx2] = current[idx2], current[idx1]
                new_swarm.append(current)
                fit = self.get_fitness(current)
                if fit < pbest_fit[i]:
                    pbest[i], pbest_fit[i] = current.copy(), fit
                    if fit < gbest_fit: gbest, gbest_fit = current.copy(), fit
            swarm = new_swarm
            history.append(gbest_fit)
            div_history.append(self.calculate_diversity(swarm))
        return {'best_fitness': gbest_fit, 'runtime': time.time() - start_time, 'fitness_history': history, 'diversity_history': div_history}

    def _move_towards(self, source, target):
        res = source.copy()
        for i in range(self.num_nodes):
            if res[i] != target[i]:
                idx = np.where(res == target[i])[0][0]
                res[i], res[idx] = res[idx], res[i]
                if random.random() > 0.5: break
        return res

# --- 3. Grey Wolf Optimizer (GWO) ---
class GWO(MetaheuristicBase):
    def run(self):
        start_time = time.time()
        wolves = [self.representation.create_chromosome() for _ in range(POPULATION_SIZE)]
        fits = [self.get_fitness(w) for w in wolves]
        indices = np.argsort(fits)
        alpha, beta, delta = wolves[indices[0]].copy(), wolves[indices[1]].copy(), wolves[indices[2]].copy()
        alpha_fit = fits[indices[0]]
        history, div_history = [], []
        for it in range(MAX_ITERATIONS):
            new_wolves = []
            a = 2 - it * (2 / MAX_ITERATIONS)
            for i in range(POPULATION_SIZE):
                r = random.random()
                if r < 0.5: new_pos = self._crossover(wolves[i], alpha)
                elif r < 0.8: new_pos = self._crossover(wolves[i], beta)
                else: new_pos = self._crossover(wolves[i], delta)
                if random.random() < (a/2):
                     idx1, idx2 = random.sample(range(self.num_nodes), 2)
                     new_pos[idx1], new_pos[idx2] = new_pos[idx2], new_pos[idx1]
                new_wolves.append(new_pos)
                fits[i] = self.get_fitness(new_pos)
            wolves = new_wolves
            indices = np.argsort(fits)
            alpha, beta, delta = wolves[indices[0]].copy(), wolves[indices[1]].copy(), wolves[indices[2]].copy()
            alpha_fit = fits[indices[0]]
            history.append(alpha_fit)
            div_history.append(self.calculate_diversity(wolves))
        return {'best_fitness': alpha_fit, 'runtime': time.time() - start_time, 'fitness_history': history, 'diversity_history': div_history}

    def _crossover(self, p1, p2):
        size = self.num_nodes
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = p1[a:b]
        p2_filtered = [item for item in p2 if item not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_filtered[idx]
                idx += 1
        return np.array(child)

# --- 4. Whale Optimization Algorithm (WOA) ---
class WOA(MetaheuristicBase):
    def run(self):
        start_time = time.time()
        whales = [self.representation.create_chromosome() for _ in range(POPULATION_SIZE)]
        fits = [self.get_fitness(w) for w in whales]
        leader_idx = np.argmin(fits)
        leader, leader_fit = whales[leader_idx].copy(), fits[leader_idx]
        history, div_history = [], []
        for it in range(MAX_ITERATIONS):
            a = 2 - it * (2 / MAX_ITERATIONS)
            for i in range(POPULATION_SIZE):
                r, p = random.random(), random.random()
                A = 2 * a * random.random() - a
                l = (random.random() * 2) - 1
                if p < 0.5:
                    if abs(A) < 1: whales[i] = self._move_towards(whales[i], leader, magnitude=abs(A))
                    else:
                        rand_whale = whales[random.randrange(POPULATION_SIZE)]
                        whales[i] = self._move_towards(whales[i], rand_whale, magnitude=abs(A))
                else:
                    whales[i] = self._move_towards(whales[i], leader, magnitude=np.exp(l))
                
                if random.random() < 0.05:
                    idx1, idx2 = random.sample(range(self.num_nodes), 2)
                    whales[i][idx1], whales[i][idx2] = whales[i][idx2], whales[i][idx1]

                fit = self.get_fitness(whales[i])
                fits[i] = fit
                if fit < leader_fit: leader, leader_fit = whales[i].copy(), fit
            history.append(leader_fit)
            div_history.append(self.calculate_diversity(whales))
        return {'best_fitness': leader_fit, 'runtime': time.time() - start_time, 'fitness_history': history, 'diversity_history': div_history}

    def _move_towards(self, source, target, magnitude):
        res = source.copy()
        num_swaps = max(1, int(self.num_nodes * min(magnitude, 0.5)))
        swaps = 0
        for i in range(self.num_nodes):
            if res[i] != target[i]:
                idx = np.where(res == target[i])[0][0]
                res[i], res[idx] = res[idx], res[i]
                swaps += 1
                if swaps >= num_swaps: break
        return res

# --- 5. Ant Colony Optimization (ACO) ---
class ACO(MetaheuristicBase):
    def run(self):
        start_time = time.time()
        pheromone = np.ones((self.num_nodes + 1, self.num_nodes + 1)) * 0.1
        alpha, beta, rho, Q = 1.0, 2.0, 0.1, 100.0
        best_fit = float('inf')
        history, div_history = [], []
        heuristic = 1.0 / (self.dist_matrix + 1e-10)
        for it in range(MAX_ITERATIONS):
            solutions, fits = [], []
            for ant in range(POPULATION_SIZE):
                unvisited, current_node, path = list(range(1, self.num_nodes + 1)), 0, []
                while unvisited:
                    probs = np.array([(pheromone[current_node][n]**alpha) * (heuristic[current_node][n]**beta) for n in unvisited])
                    probs /= probs.sum()
                    next_idx = np.random.choice(len(unvisited), p=probs)
                    chosen = unvisited.pop(next_idx)
                    path.append(chosen)
                    current_node = chosen
                chrom = np.array(path)
                fit = self.get_fitness(chrom)
                solutions.append(chrom); fits.append(fit)
                if fit < best_fit: best_fit = fit
            pheromone *= (1 - rho)
            for i in range(POPULATION_SIZE):
                path_full = [0] + list(solutions[i]) + [0]
                deposit = Q / fits[i]
                for k in range(len(path_full) - 1): pheromone[path_full[k]][path_full[k+1]] += deposit
            history.append(best_fit)
            div_history.append(self.calculate_diversity(solutions))
        return {'best_fitness': best_fit, 'runtime': time.time() - start_time, 'fitness_history': history, 'diversity_history': div_history}

# --- 6. Bee Colony Optimization (BCO) ---
class BCO(MetaheuristicBase):
    def run(self):
        start_time = time.time()
        num_bees = POPULATION_SIZE
        num_scouts = num_bees // 5
        num_employed = num_bees - num_scouts
        population = [self.representation.create_chromosome() for _ in range(num_bees)]
        fits = [self.get_fitness(p) for p in population]
        best_fit, history, div_history = np.min(fits), [], []
        trials, limit = [0] * num_bees, 20
        for it in range(MAX_ITERATIONS):
            for i in range(num_employed):
                new_sol = self._local_search(population[i])
                new_fit = self.get_fitness(new_sol)
                if new_fit < fits[i]: population[i], fits[i], trials[i] = new_sol, new_fit, 0
                else: trials[i] += 1
            prob = np.array([1.0/f for f in fits]); prob /= prob.sum()
            for _ in range(num_scouts):
                i = np.random.choice(num_bees, p=prob)
                new_sol = self._local_search(population[i])
                new_fit = self.get_fitness(new_sol)
                if new_fit < fits[i]: population[i], fits[i], trials[i] = new_sol, new_fit, 0
                else: trials[i] += 1
            for i in range(num_bees):
                if trials[i] > limit: population[i], fits[i], trials[i] = self.representation.create_chromosome(), self.get_fitness(population[i]), 0
            best_fit = min(best_fit, np.min(fits))
            history.append(best_fit)
            div_history.append(self.calculate_diversity(population))
        return {'best_fitness': best_fit, 'runtime': time.time() - start_time, 'fitness_history': history, 'diversity_history': div_history}

    def _local_search(self, chrom):
        res = chrom.copy()
        i, j = random.sample(range(self.num_nodes), 2)
        res[i], res[j] = res[j], res[i]
        return res

def worker_func(task, target_dir):
    instance_path, instance_name, dataset_type, algorithm = task
    out_dir = os.path.join(target_dir, f"{dataset_type}_Result", instance_name)
    out_file = os.path.join(out_dir, f"{algorithm}_convergence.csv")
    if os.path.exists(out_file): return

    try:
        if dataset_type == 'Solomon':
            from src.data.solomon_parser import SolomonParser
            instance = SolomonParser.parse(instance_path)
        else:
            from src.data.uchoa_parser import UchoaParser
            instance = UchoaParser.parse(instance_path)
    except Exception as e:
        print(f"Failed to parse {instance_name}: {e}"); return
        
    print(f"Starting {algorithm} on {instance_name}...")
    all_fitness, all_diversity = [], []
    for r in range(NUM_RUNS):
        if algorithm == 'SA': model = SimulatedAnnealing(instance)
        elif algorithm == 'PSO': model = PSO(instance)
        elif algorithm == 'GWO': model = GWO(instance)
        elif algorithm == 'WOA': model = WOA(instance)
        elif algorithm == 'ACO': model = ACO(instance)
        elif algorithm == 'BCO': model = BCO(instance)
        else: return
        res = model.run()
        all_fitness.append(res['fitness_history']); all_diversity.append(res['diversity_history'])
        
    df = pd.DataFrame({'generation': range(1, MAX_ITERATIONS + 1)})
    for r in range(NUM_RUNS):
        df[f'fitness_run_{r+1}'] = all_fitness[r]
        df[f'diversity_run_{r+1}'] = all_diversity[r]
    df['fitness_average'] = np.mean(all_fitness, axis=0)
    df['diversity_average'] = np.mean(all_diversity, axis=0)
    os.makedirs(out_dir, exist_ok=True); df.to_csv(out_file, index=False)
    print(f"Finished {algorithm} on {instance_name}. Best Avg: {df['fitness_average'].iloc[-1]:.2f}")

def main():
    target_dir = os.path.join(BASE_DIR, "results_metaheuristics")
    solomon_base_dir = os.path.join(BASE_DIR, "data", "solomon")
    uchoa_base_dir = os.path.join(BASE_DIR, "data", "uchoa")
    
    solomon_files = glob.glob(os.path.join(solomon_base_dir, "**", "*.csv"), recursive=True)
    uchoa_files = glob.glob(os.path.join(uchoa_base_dir, "*.vrp"))
    
    all_available = ['SA', 'PSO', 'GWO', 'WOA', 'ACO', 'BCO']
    if len(sys.argv) > 1 and sys.argv[1].upper() in all_available:
        algorithms = [sys.argv[1].upper()]
    else:
        algorithms = all_available
    
    tasks = []
    for f in solomon_files:
        name = os.path.splitext(os.path.basename(f))[0]
        for alg in algorithms: tasks.append((f, name, 'Solomon', alg))
    for f in uchoa_files:
        name = os.path.splitext(os.path.basename(f))[0]
        for alg in algorithms: tasks.append((f, name, 'Uchoa', alg))
        
    print(f"Total tasks: {len(tasks)} using {mp.cpu_count()} cores.")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(worker_func, [(t, target_dir) for t in tasks])

if __name__ == "__main__":
    mp.freeze_support(); main()
