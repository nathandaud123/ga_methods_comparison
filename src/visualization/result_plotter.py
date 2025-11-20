"""
Result Visualization for GA Comparison
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
from ..evaluation.metrics import ComparisonMetrics


class ResultPlotter:
    """Visualize GA comparison results"""
    
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_convergence(self, fitness_histories: Dict[str, List[float]],
                        save_path: str = None, show: bool = True):
        """
        Plot convergence curves for multiple methods
        
        Args:
            fitness_histories: Dictionary mapping method names to fitness history
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for method_name, history in fitness_histories.items():
            generations = range(1, len(history) + 1)
            ax.plot(generations, history, label=method_name, linewidth=2)
        
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Best Fitness', fontweight='bold')
        ax.set_title('Convergence Comparison')
        ax.legend()
        ax.grid(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison_bar(self, metrics: Dict[str, ComparisonMetrics],
                           metric: str = 'mean_fitness',
                           save_path: str = None, show: bool = True):
        """
        Plot bar chart comparing methods
        
        Args:
            metrics: Dictionary mapping method names to ComparisonMetrics
            metric: Metric to plot ('mean_fitness', 'mean_runtime', etc.)
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        method_names = list(metrics.keys())
        values = []
        errors = []
        
        for name in method_names:
            m = metrics[name]
            if metric == 'mean_fitness':
                values.append(m.mean_fitness)
                errors.append(m.std_fitness)
            elif metric == 'mean_runtime':
                values.append(m.mean_runtime)
                errors.append(m.std_runtime)
            elif metric == 'mean_convergence':
                values.append(m.mean_convergence)
                errors.append(m.std_convergence)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        x_pos = np.arange(len(method_names))
        bars = ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)
        
        ax.set_xlabel('Method', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(f'Comparison: {metric.replace("_", " ").title()}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.grid(False)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison_heatmap(self, metrics: Dict[str, ComparisonMetrics],
                               save_path: str = None, show: bool = True):
        """
        Plot heatmap comparing multiple metrics
        
        Args:
            metrics: Dictionary mapping method names to ComparisonMetrics
            save_path: Path to save figure
            show: Whether to display plot
        """
        # Prepare data
        data = []
        method_names = list(metrics.keys())
        metric_names = ['Mean Fitness', 'Mean Runtime', 'Mean Convergence', 'Best Fitness']
        
        for name in method_names:
            m = metrics[name]
            # Normalize values for heatmap (0-1 scale)
            row = [
                m.mean_fitness / max([metrics[n].mean_fitness for n in method_names]),
                m.mean_runtime / max([metrics[n].mean_runtime for n in method_names]),
                m.mean_convergence / max([metrics[n].mean_convergence for n in method_names]),
                m.best_fitness / max([metrics[n].best_fitness for n in method_names])
            ]
            data.append(row)
        
        df = pd.DataFrame(data, index=method_names, columns=metric_names)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Normalized Value'})
        ax.set_title('Method Comparison Heatmap')
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Methods', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_boxplot_comparison(self, all_results: Dict[str, List[float]],
                               metric_name: str = 'Fitness',
                               save_path: str = None, show: bool = True):
        """
        Plot boxplot comparing distributions
        
        Args:
            all_results: Dictionary mapping method names to list of values
            metric_name: Name of metric
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data = [all_results[name] for name in all_results.keys()]
        labels = list(all_results.keys())
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_xlabel('Method', fontweight='bold')
        ax.set_title(f'{metric_name} Distribution Comparison')
        ax.grid(False)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Boxplot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

