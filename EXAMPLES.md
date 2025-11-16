# Usage Examples

## Example 1: Quick Comparison

Compare permutation representation with different selection methods:

```python
from src.data.solomon_parser import SolomonParser
from src.ga.genetic_algorithm import GAConfig, GeneticAlgorithm
from src.evaluation.evaluator import ExperimentEvaluator

# Load instance
parser = SolomonParser()
instance = parser.parse('data/solomon/C101.txt')

# Define configurations to compare
configs = [
    ('tournament_pmx', GAConfig(
        selection_method='tournament',
        crossover_method='pmx',
        mutation_method='swap',
        representation='permutation'
    )),
    ('roulette_ox', GAConfig(
        selection_method='roulette_wheel',
        crossover_method='ox',
        mutation_method='swap',
        representation='permutation'
    )),
]

# Run comparison
evaluator = ExperimentEvaluator(instance, n_runs=5)
results = evaluator.compare_methods(configs)
```

## Example 2: Single Run with Visualization

```python
from src.data.solomon_parser import SolomonParser
from src.ga.genetic_algorithm import GAConfig, GeneticAlgorithm
from src.visualization.route_plotter import RoutePlotter

# Load instance
parser = SolomonParser()
instance = parser.parse('data/solomon/R101.txt')

# Configure GA
config = GAConfig(
    population_size=100,
    max_generations=200,
    selection_method='tournament',
    crossover_method='pmx',
    mutation_method='swap',
    representation='permutation'
)

# Run GA
ga = GeneticAlgorithm(instance, config)
result = ga.run()

# Visualize
plotter = RoutePlotter(instance)
plotter.plot_routes(result.best_routes, title='Best Solution')
```

## Example 3: Parameter Tuning

```python
from src.data.solomon_parser import SolomonParser
from src.tuning.optuna_tuner import OptunaTuner

# Load instance
parser = SolomonParser()
instance = parser.parse('data/solomon/C101.txt')

# Setup tuner
tuner = OptunaTuner(
    instance,
    representation='permutation',
    selection_method='tournament',
    crossover_method='pmx',
    mutation_method='swap',
    n_trials=30
)

# Run tuning
result = tuner.tune()

# Get best configuration
best_config = tuner.get_best_config(result['best_params'])
print(f"Best fitness: {result['best_value']:.2f}")
```

## Example 4: Custom Evaluation Metrics

```python
from src.evaluation.metrics import MetricsCalculator, ExperimentMetrics

# Calculate convergence speed
fitness_history = [100, 95, 90, 88, 87.5, 87.2, 87.1, 87.05]
convergence_gen = MetricsCalculator.calculate_convergence_speed(fitness_history)
print(f"Converged at generation: {convergence_gen}")

# Calculate solution quality
obtained = 1200.5
best_known = 1100.0
gap = MetricsCalculator.calculate_solution_quality(obtained, best_known)
print(f"Gap from best known: {gap:.2f}%")
```

