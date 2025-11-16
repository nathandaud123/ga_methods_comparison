"""Quick test to check if all imports work"""
import sys

print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import optuna
    print("✓ optuna")
except ImportError as e:
    print(f"✗ optuna: {e}")

try:
    from src.data.solomon_parser import SolomonParser
    print("✓ SolomonParser")
except ImportError as e:
    print(f"✗ SolomonParser: {e}")

try:
    from src.ga.genetic_algorithm import GAConfig, GeneticAlgorithm
    print("✓ GA classes")
except ImportError as e:
    print(f"✗ GA classes: {e}")

print("\nAll critical imports tested!")

