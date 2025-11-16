"""
Quick experiment runner for testing
Run this to test the implementation with a small experiment
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.solomon_parser import SolomonParser
from src.ga.genetic_algorithm import GAConfig, GeneticAlgorithm
from src.visualization.route_plotter import RoutePlotter
from src.visualization.result_plotter import ResultPlotter
import numpy as np


def create_test_instance():
    """Create a simple test instance if Solomon dataset is not available"""
    # This is a minimal VRP instance for testing
    test_file = "data/solomon/test_instance.txt"
    os.makedirs("data/solomon", exist_ok=True)
    
    if not os.path.exists(test_file):
        with open(test_file, 'w') as f:
            f.write("""CUSTOMER
CUSTOMER NO.	XCOORD.	YCOORD.	DEMAND	READY TIME	DUE DATE	SERVICE TIME
0	40	50	0	0	1236	0
1	45	68	10	912	967	90
2	45	70	30	825	870	90
3	42	66	10	65	146	90
4	42	68	10	727	782	90
5	42	65	10	15	67	90
6	40	69	20	621	702	90
7	40	66	20	170	225	90
8	38	68	20	255	324	90
9	38	70	10	534	605	90
10	35	66	10	357	410	90
""")
        print(f"Created test instance: {test_file}")
    
    return test_file


def run_quick_test():
    """Run a quick test of the GA implementation"""
    print("="*60)
    print("Quick GA Test")
    print("="*60)
    
    # Try to load a real instance, or create test instance
    test_file = create_test_instance()
    
    try:
        parser = SolomonParser()
        instance = parser.parse(test_file)
        print(f"Loaded instance: {instance.name}")
        print(f"Customers: {instance.num_customers}")
    except Exception as e:
        print(f"Error loading instance: {e}")
        print("Please ensure Solomon benchmark files are in data/solomon/")
        return
    
    # Configure GA
    config = GAConfig(
        population_size=50,
        max_generations=100,
        selection_method='tournament',
        crossover_method='pmx',
        mutation_method='swap',
        representation='permutation',
        verbose=True
    )
    
    print("\nRunning GA...")
    ga = GeneticAlgorithm(instance, config)
    result = ga.run()
    
    print(f"\nResults:")
    print(f"  Best Fitness: {result.best_fitness:.2f}")
    print(f"  Runtime: {result.runtime:.2f}s")
    print(f"  Number of Routes: {len(result.best_routes)}")
    print(f"  Convergence Generation: {result.convergence_generation}")
    
    # Visualize
    print("\nGenerating route visualization...")
    plotter = RoutePlotter(instance)
    os.makedirs("results/routes", exist_ok=True)
    plotter.plot_routes(
        result.best_routes,
        title=f"Test Solution - {instance.name}",
        save_path="results/routes/test_route.png",
        show=False
    )
    
    print("\nTest completed successfully!")
    print("Check results/routes/test_route.png for visualization")


if __name__ == '__main__':
    run_quick_test()

