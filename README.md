# Genetic Algorithm Method Comparison Study

Comprehensive comparison study of Genetic Algorithm operators (representation, selection, crossover, mutation) for Vehicle Routing Problem using Solomon benchmark datasets.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with all datasets (supports checkpoint/resume!)
python main.py --config config.yaml

# Or use the Python background script
python run_full_experiment.py

# Or use PowerShell script
.\start_experiment.ps1

# Check progress
python check_progress.py

# Quick status check
python status_check.py

# Monitor in real-time
python monitor_experiment.py

# Clear checkpoint to restart
python clear_checkpoint.py
```

## ⚡ Parallel Execution (HPC Support)

**NEW**: Sistem sekarang mendukung **parallel execution** untuk high-performance computing!

### Single Device (Multi-core)
- ✅ **Thread-safe checkpointing** - No race conditions
- ✅ **Automatic core detection** - Uses all available CPU cores
- ✅ **Linear speedup** - ~N cores = ~Nx faster
- ✅ **Safe file I/O** - File locking prevents conflicts

**Enable in `config.yaml`:**
```yaml
evaluation:
  parallel: true  # Enable parallel execution
  n_jobs: null    # Auto (CPU_COUNT - 1), or set manually (e.g., 8)
```

### Multi-Device (Distributed)
- ✅ **True distributed execution** - Multiple devices, no conflicts
- ✅ **Automatic workload distribution** - Each device gets different instances
- ✅ **Shared checkpoint** - Progress tracked across all devices
- ✅ **Linear speedup** - N devices = ~Nx faster

**Setup:**
```bash
# 1. Setup device configs
python setup_multi_device.py --total-devices 3

# 2. Configure each device in config.yaml
device:
  device_id: "device1"  # Unique per device
  total_devices: 3

# 3. Run on each device
python main.py --config config.yaml
```

See `PARALLEL_USAGE.md` for single-device guide.
See `MULTI_DEVICE_SETUP.md` for multi-device guide.
See `SETUP_2_DEVICES.md` for quick setup (Laptop + AI Center).
See `QUICK_ANSWER.md` for quick answer: "Apakah aman run bersamaan?"

## ✅ Checkpoint & Resume Feature

**IMPORTANT**: The system now supports checkpoint/resume functionality!

- ✅ **Auto-save checkpoint** after each GA run
- ✅ **Resume from last checkpoint** if interrupted
- ✅ **Skip completed methods** automatically  
- ✅ **Resume partial runs** - continues from last incomplete run
- ✅ **Safe to terminate** - laptop can sleep/hibernate anytime
- ✅ **No data loss** - results saved separately from checkpoint

### How It Works

1. **Checkpoint saved automatically** after each GA execution
2. **If terminated** (Ctrl+C, sleep, etc.), simply restart:
   ```bash
   python main.py --config config.yaml
   ```
3. **System automatically**:
   - ✅ Skips completed instances
   - ✅ Skips completed methods  
   - ✅ Resumes from last incomplete run
   - ✅ Shows checkpoint status on startup

### Checkpoint Management

```bash
# Check progress
python check_progress.py

# Clear checkpoint for specific instance
python clear_checkpoint.py C101

# Clear entire checkpoint (restart from beginning)
python clear_checkpoint.py
```

Checkpoint file: `results/checkpoint.json`

## 📊 Status

- ✅ **GitHub Repository**: [ga_methods_comparison](https://github.com/NathanDaud123/ga_methods_comparison)
- ✅ **56 Solomon Instances** ready (C1, C2, R1, R2, RC1, RC2)
- ✅ **354 Method Combinations** to test per instance
- ✅ **Checkpoint/Resume** support
- ✅ All results saved to CSV/JSON

## 📄 Paper Structure Template

A comprehensive paper structure template is available in `PAPER_STRUCTURE.md`:
- Complete outline from Title to Conclusion
- Section-by-section guidelines
- Writing tips and recommendations
- Table and figure suggestions
- Word count guidelines

Perfect for organizing your research paper on GA operator comparison!

## 📁 Project Structure

```
ga_method_comparison/
├── README.md
├── requirements.txt
├── config.yaml              # Main config (all datasets)
├── config_test.yaml         # Quick test config
├── main.py                  # Main experiment runner (with checkpoint)
├── check_progress.py        # Check experiment progress
├── clear_checkpoint.py      # Clear checkpoint
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
│   │   ├── evaluator.py         # Saves CSV convergence + checkpoint
│   │   └── checkpoint.py        # Checkpoint management
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
    ├── checkpoint.json          # Checkpoint file (auto-created)
    ├── experiments/{instance}/  # JSON results
    ├── plots/{instance}/        # Comparison charts
    ├── routes/{instance}/       # Route visualizations
    ├── convergence/{instance}/  # CSV per method
    └── tuning/{instance}/       # CSV Optuna results
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

### Checkpoint File
- `results/checkpoint.json`: Tracks completed instances, methods, and runs
- Auto-saved after each run
- Safe to terminate anytime

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

**With checkpoint, you can safely interrupt and resume anytime!**

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
