# Genetic Algorithm Method Comparison Study

Comprehensive comparison study of Genetic Algorithm operators (representation, selection, crossover, mutation) for Vehicle Routing Problem using Solomon benchmark datasets.

## Project Structure

```
ga_method_comparison/
├── README.md
├── requirements.txt
├── config.yaml
├── config_test.yaml
├── main.py
├── run_experiment.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── solomon_parser.py
│   ├── representation/
│   │   ├── __init__.py
│   │   ├── binary.py
│   │   ├── real_valued.py
│   │   └── permutation.py
│   ├── selection/
│   │   ├── __init__.py
│   │   └── selection_methods.py
│   ├── crossover/
│   │   ├── __init__.py
│   │   ├── binary_crossover.py
│   │   ├── real_crossover.py
│   │   └── permutation_crossover.py
│   ├── mutation/
│   │   ├── __init__.py
│   │   ├── binary_mutation.py
│   │   ├── real_mutation.py
│   │   └── permutation_mutation.py
│   ├── ga/
│   │   ├── __init__.py
│   │   └── genetic_algorithm.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── route_plotter.py
│   │   └── result_plotter.py
│   └── tuning/
│       ├── __init__.py
│       └── optuna_tuner.py
├── data/
│   └── solomon/
│       ├── C101.txt
│       ├── C201.txt
│       ├── R101.txt
│       ├── R201.txt
│       ├── RC101.txt
│       └── RC201.txt
├── results/
│   ├── experiments/
│   │   └── {instance_name}/
│   │       └── {instance_name}_results.json
│   ├── plots/
│   │   └── {instance_name}/
│   │       ├── {instance_name}_fitness_comparison.png
│   │       ├── {instance_name}_runtime_comparison.png
│   │       └── {instance_name}_heatmap.png
│   ├── routes/
│   │   └── {instance_name}/
│   │       └── {instance_name}_best_route.png
│   ├── convergence/
│   │   └── {instance_name}/
│   │       └── {method_name}_convergence.csv
├── tuning/
│   │   └── {instance_name}/
│   │       ├── {instance_name}_optuna_tuning.csv
│   │       └── {instance_name}_optuna_tuning_summary.csv
│   └── summary.json
└── notebooks/
    └── analysis.ipynb
```

## Features

### Representations
- **Binary**: Bit string encoding
- **Real-valued**: Continuous value encoding
- **Permutation**: Order-based encoding (for VRP/TSP)

### Selection Methods
- Roulette Wheel Selection
- Tournament Selection
- Rank Selection
- Stochastic Universal Sampling (SUS)
- Elitism Selection
- Stairwise Selection (SWS)
- Boltzmann Selection

### Crossover Operators
- **Binary**: Single-point, Two-point, Multi-point, Uniform, Shuffle, Arithmetic
- **Real-valued**: SBX, BLX-α, Flat, SPX, UNDX, PCX
- **Permutation**: PMX, OX, CX, OBX, POS, ER/ERX, IX, SCX

### Mutation Operators
- **Binary**: Bit Flip, Uniform, Interchanging, Reversing
- **Real-valued**: Gaussian, Polynomial, Uniform, Non-uniform
- **Permutation**: Swap, Insert, Inversion, Scramble, Displacement, Exchange

## Installation

```bash
# Clone repository
git clone <repository-url>
cd ga_method_comparison

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Dataset Setup

Solomon benchmark datasets need to be downloaded separately. Place them in `data/solomon/` directory.

You can download from:
- [Solomon Benchmark](http://web.cba.neu.edu/~msolomon/problems.htm)
- Or use the provided download script (if available)

Expected structure:
```
data/solomon/
├── C101.txt
├── C201.txt
├── R101.txt
├── R201.txt
├── RC101.txt
└── RC201.txt
```

## Usage

### Basic Usage

```bash
python main.py --config config.yaml
```

### Custom Configuration

Edit `config.yaml` to customize:
- Dataset selection
- GA parameters
- Selection of operators to compare
- Optuna tuning parameters

### Running Specific Experiments

You can modify `config.yaml` to test specific combinations:

```yaml
representations:
  - "permutation"  # Focus on permutation only

selection_methods:
  - "tournament"
  - "roulette_wheel"

crossover_methods:
  permutation:
    - "pmx"
    - "ox"
```

### Parameter Tuning with Optuna

Enable Optuna tuning in `config.yaml`:

```yaml
optuna:
  enabled: true
  n_trials: 50
  timeout: 3600
```

## Output Structure

Results are organized by instance name for easy management:

```
results/
├── experiments/
│   └── {instance_name}/
│       └── {instance_name}_results.json
├── plots/
│   └── {instance_name}/
│       ├── {instance_name}_fitness_comparison.png
│       ├── {instance_name}_runtime_comparison.png
│       └── {instance_name}_heatmap.png
├── routes/
│   └── {instance_name}/
│       └── {instance_name}_best_route.png
├── convergence/
│   └── {instance_name}/
│       └── {method_name}_convergence.csv  # Contains generation-by-generation data
└── summary.json
```

### Convergence CSV Format

Each method's convergence history is saved as CSV with the following columns:
- `generation`: Generation number (1, 2, 3, ...)
- `fitness_run_1`, `fitness_run_2`, ...: Fitness value for each run
- `diversity_run_1`, `diversity_run_2`, ...: Diversity value for each run
- `fitness_mean`, `fitness_std`, `fitness_min`, `fitness_max`: Statistics across runs
- `diversity_mean`, `diversity_std`: Diversity statistics

You can use these CSV files to create custom convergence plots in your analysis.

### Optuna Tuning CSV Format

When Optuna tuning is enabled, detailed trial history is saved:
- `{instance_name}_optuna_tuning.csv`: Contains all trials with columns:
  - `trial_number`: Trial number
  - `state`: Trial state (COMPLETE, PRUNED, etc.)
  - `value`: Best fitness value from GA run
  - `population_size`, `max_generations`, `crossover_rate`, `mutation_rate`, `tournament_size`, `elitism_rate`: Hyperparameters
  - `runtime`: Execution time for this trial
  - `convergence_generation`: Generation when best solution was found
  - `final_diversity`: Average population diversity
  - `is_best`: Whether this trial achieved best value
  - `cumulative_best`: Best value up to this trial
  - `improvement`: Improvement over previous best

- `{instance_name}_optuna_tuning_summary.csv`: Summary statistics including:
  - Best/mean/std/min/max values
  - Best hyperparameters found
  - Mean runtime and trial counts

This allows complete documentation and analysis of the hyperparameter tuning process.

## Evaluation Metrics

- **Fitness/Cost**: Total distance/cost of solution
- **Runtime**: Execution time
- **Convergence Speed**: Generations to convergence
- **Population Diversity**: Diversity metrics
- **Solution Quality**: Gap from best known solution

## Citation

If you use this code in your research, please cite the relevant papers mentioned in the comparison study document.
