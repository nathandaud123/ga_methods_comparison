# Genetic Algorithm Operator Comparison for VRPTW

A simplified experimental framework for comparing Genetic Algorithm (GA) operator configurations on the Vehicle Routing Problem with Time Windows (VRPTW) using the Solomon benchmark dataset (100 customers per instance).

## Main Features

1. **Full Operator Grid**: Exhaustive combinations of representation, selection, crossover, and mutation operators.
2. **Multiple Runs per Configuration**: Each operator configuration is evaluated over 5 independent runs for more reliable statistics.
3. **Per-Generation Logging**: Fitness and diversity values are recorded at every generation for each run.
4. **Run-Averaged Curves**: Per-generation averages across 5 runs are computed and stored.
5. **All Solomon Instances**: All selected Solomon instances can be evaluated with the same experimental pipeline.

## Experimental Workflow

1. **Generate Configurations**: Build the Cartesian product of representation × selection × crossover × mutation.
2. **Run 5 Times**: Execute each configuration 5 times with different random seeds.
3. **Log History**: Store fitness and diversity at each generation for every run.
4. **Compute Averages**: Aggregate per-generation averages over the 5 runs.
5. **Save Results**: Export results to CSV and JSON for downstream analysis and visualization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on all configured datasets (sequential)
python main.py --config config.yaml

# Run with parallel execution (e.g., 4 workers)
# Set in config.yaml: n_jobs: 4
python main.py --config config.yaml

# Run with auto-detected number of CPU cores
# Set in config.yaml: n_jobs: 0
python main.py --config config.yaml

# Run a specific instance only
python main.py --config config.yaml --instance C101
python main.py --config config.yaml --instance C102
```


## Multi-Terminal / Multi-Instance Execution

You can run different instances in parallel from separate terminals or machines.

### 1. Run Specific Instances

Each terminal runs a different instance:

```bash
# Terminal 1
python main.py --config config.yaml --instance C101

# Terminal 2
python main.py --config config.yaml --instance C102

# Terminal 3
python main.py --config config.yaml --instance C103
```

Notes:

- Each instance uses its own checkpoint file: `checkpoint_<instance>.json`.

```
- Results are stored under `results/<instance>/<instance>_results.json`.
```

- There are no conflicts because each instance has a separate checkpoint file.


### 2. Merge Checkpoints

After all instances finish, merge per-instance checkpoints:

```bash
# Merge all checkpoint_*.json into a single checkpoint.json
python merge_checkpoints.py --results-dir results

# Custom merged output file
python merge_checkpoints.py --results-dir results --output results/checkpoint_merged.json
```

The merge step:

- Combines all `checkpoint_<instance>.json` into one consolidated checkpoint.
- Merges `completed_methods` and `completed_instances` across all instances.
- Prints a summary per instance.


## Checkpointing \& Resume

The system supports safe interruption and resumption:

- **Auto-save**: State is saved after each completed configuration.
- **Auto-resume**: On restart, the system continues from unfinished configurations.
- **Skip completed**: Already finished configurations are automatically skipped.

Checkpoint files:

- **All-instances mode**: `results/checkpoint.json`
- **Single-instance mode** (`--instance`): `results/checkpoint_<instance>.json`
Examples: `checkpoint_C101.json`, `checkpoint_C102.json`

Each instance maintains its own checkpoint file, preventing overwrites.

## Parallel Execution

Parallelism is controlled via `config.yaml`:

```yaml
evaluation:
  n_jobs: null  # Sequential (default)
  # n_jobs: 1   # Sequential
  # n_jobs: 0   # Auto: use all CPU cores - 1
  # n_jobs: 4   # Use 4 parallel workers
  # n_jobs: 8   # Use 8 parallel workers
```

Behavior:

- `n_jobs: null` or `1`: Purely sequential execution.
- `n_jobs: 0`: Auto-detect CPU cores and use all cores minus one.
- `n_jobs: N` (N > 1): Use N parallel workers.

Example:

- With 5 runs per configuration and `n_jobs: 5`, all runs for that configuration can execute in parallel, giving an approximate N× speedup (subject to overhead and hardware limits).


## Output Structure

```text
results/
├── {instance_name}/
│   ├── {method_name}_convergence.csv   # Per-run and averaged histories
│   └── {instance_name}_results.json    # Summary statistics per method
└── checkpoint*.json                    # Global or per-instance checkpoints
```


### CSV Convergence Format

Each `{method_name}_convergence.csv` contains:

- `generation`: Generation index (1, 2, 3, ...).
- `fitness_run_1` … `fitness_run_5`: Fitness per run per generation.
- `diversity_run_1` … `diversity_run_5`: Diversity per run per generation.
- `fitness_average`: Mean fitness across 5 runs per generation.
- `diversity_average`: Mean diversity across 5 runs per generation.


### JSON Results Format

Each `{instance_name}_results.json` stores method-level summaries:

```json
{
  "method_name": {
    "method_name": "representation_selection_crossover_mutation",
    "average_fitness_history": [...],
    "average_diversity_history": [...],
    "best_fitness": 1234.56,
    "mean_fitness": 1234.56,
    "std_fitness": 12.34,
    "runtime": 123.45,
    "n_runs": 5
  }
}
```


## Configuration

Edit `config.yaml` to customize:

- GA parameters: population size, max generations, mutation rate, etc.
- Available operators: representations, selection methods, crossover operators, mutation operators.
- Number of runs per configuration (default: 5).
- Parallelism (`n_jobs`) and other evaluation settings.


## Example Console Output

```text
================================================================================
Processing instance: C101
================================================================================
Instance: C101
Type: C
Customers: 100
Capacity: 200.0

============================================================
Evaluating: permutation_tournament_pmx_swap
============================================================
  Run 1/5... Best: 1234.56
  Run 2/5... Best: 1235.12
  Run 3/5... Best: 1234.89
  Run 4/5... Best: 1235.45
  Run 5/5... Best: 1234.23
  Saved convergence history to results/C101/permutation_tournament_pmx_swap_convergence.csv
  Average final fitness: 1234.85 ± 0.45
  Best fitness: 1234.23
  Runtime: 45.67s
```


## Data Analysis

The stored data can be used for:

- Inspecting convergence behavior per configuration.
- Comparing performance across operator combinations and instances.
- Statistical significance tests (e.g., non-parametric tests, effect sizes).
- Plotting convergence curves, diversity trajectories, and interaction effects.


## Notes

- This repository focuses on a clean, reproducible GA operator comparison framework for VRPTW.
- Checkpointing and parallel execution are supported for efficient large-scale experiments.
- The design emphasizes complete and structured data collection to enable in-depth post-hoc analysis.
