"""
Optuna-based Parameter Tuning for GA
"""

import optuna
import pandas as pd
import os
from typing import Dict, Any, Callable, Optional, List
from ..ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ..data.solomon_parser import VRPInstance
import numpy as np


class OptunaTuner:
    """Optuna-based hyperparameter tuning for GA"""
    
    def __init__(self, instance: VRPInstance, 
                 representation: str = "permutation",
                 selection_method: str = "tournament",
                 crossover_method: str = "pmx",
                 mutation_method: str = "swap",
                 n_trials: int = 50,
                 timeout: int = 3600,
                 save_history: bool = True,
                 history_dir: Optional[str] = None):
        self.instance = instance
        self.representation = representation
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.n_trials = n_trials
        self.timeout = timeout
        self.save_history = save_history
        self.history_dir = history_dir or "results/tuning"
        if self.save_history:
            os.makedirs(self.history_dir, exist_ok=True)
        
        # Store trial history
        self.trial_history: List[Dict[str, Any]] = []
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Best fitness value
        """
        # Suggest hyperparameters
        population_size = trial.suggest_int('population_size', 50, 200)
        max_generations = trial.suggest_int('max_generations', 100, 500)
        crossover_rate = trial.suggest_float('crossover_rate', 0.6, 0.95)
        mutation_rate = trial.suggest_float('mutation_rate', 0.05, 0.3)
        tournament_size = trial.suggest_int('tournament_size', 2, 10)
        elitism_rate = trial.suggest_float('elitism_rate', 0.05, 0.2)
        
        # Create config
        config = GAConfig(
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_rate=elitism_rate,
            tournament_size=tournament_size,
            selection_method=self.selection_method,
            crossover_method=self.crossover_method,
            mutation_method=self.mutation_method,
            representation=self.representation,
            verbose=False  # Disable verbose during tuning
        )
        
        # Run GA
        ga = GeneticAlgorithm(self.instance, config)
        result = ga.run()
        
        # Store trial information
        if self.save_history:
            trial_info = {
                'trial_number': trial.number,
                'state': 'COMPLETE',  # Trial state (COMPLETE if we reach here)
                'value': result.best_fitness,
                'population_size': population_size,
                'max_generations': max_generations,
                'crossover_rate': crossover_rate,
                'mutation_rate': mutation_rate,
                'tournament_size': tournament_size,
                'elitism_rate': elitism_rate,
                'runtime': result.runtime,
                'convergence_generation': result.convergence_generation,
                'final_diversity': np.mean(result.diversity_history) if result.diversity_history else 0.0,
            }
            self.trial_history.append(trial_info)
        
        return result.best_fitness
    
    def tune(self, study_name: str = "ga_tuning", instance_name: str = "") -> Dict[str, Any]:
        """
        Run Optuna optimization
        
        Args:
            study_name: Name for Optuna study
            instance_name: Instance name for saving results
            
        Returns:
            Dictionary with best parameters and value
        """
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name
        )
        
        # Reset trial history
        self.trial_history = []
        
        print(f"Starting Optuna optimization with {self.n_trials} trials...")
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        print(f"\nOptimization finished!")
        print(f"Best value: {study.best_value:.2f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save tuning history to CSV
        if self.save_history and self.trial_history:
            self._save_tuning_history(study_name, instance_name)
        
        return {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'study': study,
            'trial_history': self.trial_history
        }
    
    def _save_tuning_history(self, study_name: str, instance_name: str):
        """
        Save Optuna tuning history to CSV
        
        Args:
            study_name: Name of the study
            instance_name: Instance name for filename
        """
        if not self.trial_history:
            return
        
        # Create DataFrame from trial history
        df = pd.DataFrame(self.trial_history)
        
        # Add additional statistics
        df['is_best'] = df['value'] == df['value'].min()
        df['cumulative_best'] = df['value'].cummin()
        df['improvement'] = df['cumulative_best'].diff().fillna(0)
        
        # Sort by trial number
        df = df.sort_values('trial_number').reset_index(drop=True)
        
        # Determine filename
        if instance_name:
            filename = f"{instance_name}_optuna_tuning.csv"
        else:
            clean_study_name = study_name.replace('/', '_').replace('\\', '_')
            filename = f"{clean_study_name}_tuning.csv"
        
        filepath = os.path.join(self.history_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"\nTuning history saved to: {filepath}")
        print(f"  Total trials: {len(df)}")
        print(f"  Best trial: {df.loc[df['value'].idxmin(), 'trial_number']}")
        print(f"  Best value: {df['value'].min():.2f}")
        
        # Also save summary statistics
        summary_filepath = filepath.replace('.csv', '_summary.csv')
        summary_data = {
            'metric': [
                'best_value', 'mean_value', 'std_value', 'min_value', 'max_value',
                'best_population_size', 'best_max_generations', 'best_crossover_rate',
                'best_mutation_rate', 'best_tournament_size', 'best_elitism_rate',
                'mean_runtime', 'total_trials', 'completed_trials'
            ],
            'value': [
                df['value'].min(),
                df['value'].mean(),
                df['value'].std(),
                df['value'].min(),
                df['value'].max(),
                df.loc[df['value'].idxmin(), 'population_size'],
                df.loc[df['value'].idxmin(), 'max_generations'],
                df.loc[df['value'].idxmin(), 'crossover_rate'],
                df.loc[df['value'].idxmin(), 'mutation_rate'],
                df.loc[df['value'].idxmin(), 'tournament_size'],
                df.loc[df['value'].idxmin(), 'elitism_rate'],
                df['runtime'].mean(),
                len(df),
                len(df[df['state'] == 'COMPLETE']) if 'state' in df.columns else len(df)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_filepath, index=False)
        print(f"  Summary saved to: {summary_filepath}")
    
    def get_best_config(self, best_params: Dict[str, Any]) -> GAConfig:
        """
        Create GAConfig from best parameters
        
        Args:
            best_params: Best parameters from Optuna
            
        Returns:
            GAConfig with optimized parameters
        """
        return GAConfig(
            population_size=best_params['population_size'],
            max_generations=best_params.get('max_generations', 500),
            crossover_rate=best_params['crossover_rate'],
            mutation_rate=best_params['mutation_rate'],
            elitism_rate=best_params.get('elitism_rate', 0.1),
            tournament_size=best_params.get('tournament_size', 3),
            selection_method=self.selection_method,
            crossover_method=self.crossover_method,
            mutation_method=self.mutation_method,
            representation=self.representation,
            verbose=True
        )

