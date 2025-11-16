# Genetic Algorithm Method Comparison Study

Comprehensive comparison study of Genetic Algorithm operators (representation, selection, crossover, mutation) for Vehicle Routing Problem using Solomon benchmark datasets.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with all datasets (will take hours!)
python main.py --config config.yaml

# Or run with logging
python run_full_experiment.py
```

## 📊 Status

- ✅ **GitHub Repository**: [ga_methods_comparison](https://github.com/NathanDaud123/ga_methods_comparison)
- ✅ **56 Solomon Instances** ready (C1, C2, R1, R2, RC1, RC2)
- ✅ **354 Method Combinations** to test per instance
- ✅ All results saved to CSV/JSON

## 📁 Project Structure

```
ga_method_comparison/
├── README.md
├── requirements.txt
├── config.yaml              # Main config (all datasets)
├── config_test.yaml         # Quick test config
├── main.py
├── run_experiment.py        # Quick test
├── run_full_experiment.py   # Full run with logging
├── src/
│   ├── data/
│   │   └── solomon_parser.py    # Supports CSV format
│   ├── representation/          # Binary, Real-valued, Permutation
│   ├── selection/               # 7 methods
│   ├── crossover/               # 16+ methods
│   ├── mutation/                # 14+ methods
│   ├── ga/
│   │   └── genetic_algorithm.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluator.py         # Saves CSV convergence
│   ├── visualization/
│   │   ├── route_plotter.py
│   │   └── result_plotter.py
│   └── tuning/
│       └── optuna_tuner.py      # Saves CSV tuning history
├── data/solomon/
│   ├── C1/  (9 instances)
│   ├── C2/  (8 instances)
│   ├── R1/  (12 instances)
│   ├── R2/  (11 instances)
│   ├── RC1/ (8 instances)
│   └── RC2/ (8 instances)
└── results/
    ├── experiments/{instance}/    # JSON results
    ├── plots/{instance}/          # Comparison charts
    ├── routes/{instance}/         # Route visualizations
    ├── convergence/{instance}/    # CSV per method
    └── tuning/{instance}/         # CSV Optuna results
```

## 🎯 Features

### Representations
- **Binary**: Bit string encoding
- **Real-valued**: Continuous value encoding
- **Permutation**: Order-based encoding (primary for VRP)

### Selection Methods (7)
- Roulette Wheel Selection
- Tournament Selection
- Rank Selection
- Stochastic Universal Sampling (SUS)
- Elitism Selection
- Stairwise Selection (SWS)
- Boltzmann Selection

### Crossover Operators (16+)
- **Permutation**: PMX, OX, CX, OBX, POS, ERX, SCX
- **Binary**: Single-point, Two-point, Multi-point, Uniform, Shuffle, Arithmetic
- **Real-valued**: SBX, BLX-α, Flat

### Mutation Operators (14+)
- **Permutation**: Swap, Insert, Inversion, Scramble, Displacement, Exchange
- **Binary**: Bit Flip, Uniform, Interchanging, Reversing
- **Real-valued**: Gaussian, Polynomial, Uniform, Non-uniform

## 📝 Output Files

### Convergence CSV
Each method saves convergence history:
- `generation`: Generation number
- `fitness_run_1..N`: Fitness per run
- `diversity_run_1..N`: Diversity per run
- `fitness_mean/std/min/max`: Statistics
- `diversity_mean/std`: Diversity statistics

### Optuna Tuning CSV
When tuning enabled:
- `{instance}_optuna_tuning.csv`: All trials with parameters
- `{instance}_optuna_tuning_summary.csv`: Summary statistics

## ⚙️ Configuration

### Run All Datasets
```yaml
dataset:
  instances: []  # Auto-discover all CSV files
```

### Run Specific Instances
```yaml
dataset:
  instances:
    - "C101"
    - "R101"
    - "RC101"
```

### Adjust Parameters
```yaml
ga:
  population_size: 100
  max_generations: 500
  n_runs: 5  # Independent runs per method
```

## 📊 Expected Runtime

- **56 instances** × **354 methods** × **5 runs** = **99,120 GA executions**
- Estimated: ~3-5 days on typical hardware
- Each GA run: ~1-2 minutes (500 generations, 100 population)

## 🔍 Analysis

All results are saved in structured format for easy analysis:
- JSON for programmatic access
- CSV for Excel/Python analysis
- PNG for visual inspection

## 📚 Citation

If you use this code, please cite relevant papers from the comparison study document.

## 🔗 Links

- GitHub: https://github.com/NathanDaud123/ga_methods_comparison
- Solomon Benchmark: http://web.cba.neu.edu/~msolomon/problems.htm
