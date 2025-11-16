"""
Setup script for GA Method Comparison Study
"""

from setuptools import setup, find_packages

setup(
    name="ga-method-comparison",
    version="1.0.0",
    description="Comprehensive comparison study of Genetic Algorithm operators",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "optuna>=3.0.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
)

