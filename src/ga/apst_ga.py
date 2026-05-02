import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional
from math import exp

# ML Libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class APST_GA:
    """
    Implementation of Adaptive Parameter Selection and Tuning Genetic Algorithm (APST-GA).
    APST-GA supports multiple ML models to dynamically predict crossover (Pc) 
    and mutation (Pm) rates.
    
    Supported ML Methods:
    - 'DT': Decision Tree Regressor (Default, Fast, Interpretable)
    - 'RF': Random Forest Regressor (Robust, Handles non-linear well)
    - 'SVM': Support Vector Regression (Good for high-dimensional features)
    - 'LR': Linear Regression (Simple, Low computational overhead)
    """
    
    def __init__(self, 
                 pc_max: float = 0.9, pc_min: float = 0.6, 
                 pm_max: float = 0.3, pm_min: float = 0.1,
                 bootstrap_gens: int = 250,
                 max_gens: int = 500,
                 ml_method: str = 'DT'):
        
        # Parameter Ranges
        self.pc_max = pc_max
        self.pc_min = pc_min
        self.pm_max = pm_max
        self.pm_min = pm_min
        
        # APST Specific Configuration
        self.bootstrap_gens = bootstrap_gens
        self.max_gens = max_gens
        self.ml_method = ml_method.upper() # 'DT', 'RF', 'SVM', 'LR'
        self.A = 2.0
        
        # ML Models and Data
        self.ml_crossover_model = None
        self.ml_mutation_model = None
        self.ml_trained = False
        self.scaler = StandardScaler()
        
        self.training_data = {
            'features': [],
            'pc_targets': [],
            'pm_targets': [],
            'performance': []
        }

    def _get_ml_model(self):
        """Helper to instantiate the selected ML model."""
        if self.ml_method == 'DT':
            return DecisionTreeRegressor(random_state=42)
        elif self.ml_method == 'RF':
            return RandomForestRegressor(n_estimators=50, random_state=42)
        elif self.ml_method == 'SVM':
            return SVR(kernel='rbf', C=1.0)
        elif self.ml_method == 'LR':
            return LinearRegression()
        else:
            print(f"Unknown ml_method '{self.ml_method}', defaulting to DT.")
            return DecisionTreeRegressor(random_state=42)

    def _to_maximization_fitness(self, distance: float) -> float:
        return 1.0 / distance if distance > 0 else 0.0

    def mathematical_pc(self, d1: float, d2: float, d_avg: float, d_min: float) -> float:
        """Initial sigmoid-based heuristic."""
        d_prime = max(d1, d2)
        if d_prime <= d_avg:
            if d_avg != d_min:
                try:
                    exponent = self.A * ((2 * (d_avg - d_prime) / (d_avg - d_min)) - 1)
                    exponent = min(700, max(-700, exponent))
                    pc = ((self.pc_max - self.pc_min) / (1 + exp(exponent))) + self.pc_min
                except: pc = self.pc_min
            else: pc = self.pc_min
        else: pc = self.pc_max
        return max(self.pc_min, min(self.pc_max, pc))

    def mathematical_pm(self, d: float, d_avg: float, d_min: float) -> float:
        """Initial sigmoid-based heuristic."""
        if d <= d_avg:
            if d_avg != d_min:
                try:
                    exponent = self.A * ((2 * (d_avg - d) / (d_avg - d_min)) - 1)
                    exponent = min(700, max(-700, exponent))
                    pm = ((self.pm_max - self.pm_min) / (1 + exp(exponent))) + self.pm_min
                except: pm = self.pm_min
            else: pm = self.pm_min
        else: pm = self.pm_max
        return max(self.pm_min, min(self.pm_max, pm))

    def train_models(self):
        """Trains models using the selected ML method."""
        if len(self.training_data['features']) < 10:
            return False
            
        X = np.array(self.training_data['features'])
        y_pc = np.array(self.training_data['pc_targets'])
        y_pm = np.array(self.training_data['pm_targets'])
        X_scaled = self.scaler.fit_transform(X)
        
        # Instantiate and train models
        self.ml_crossover_model = self._get_ml_model()
        self.ml_mutation_model = self._get_ml_model()
            
        self.ml_crossover_model.fit(X_scaled, y_pc)
        self.ml_mutation_model.fit(X_scaled, y_pm)
        self.ml_trained = True
        return True

    def predict_rates(self, gen: int, d_avg: float, d_min: float, d_max: float, 
                      diversity: float, d1: float, d2: float) -> Tuple[float, float]:
        if not self.ml_trained: return None, None
            
        features = np.array([[d_avg, d_min, d_max, diversity, gen/self.max_gens, max(d1, d2), min(d1, d2)]])
        try:
            features_scaled = self.scaler.transform(features)
            pc_pred = self.ml_crossover_model.predict(features_scaled)[0]
            pm_pred = self.ml_mutation_model.predict(features_scaled)[0]
            
            return max(self.pc_min, min(self.pc_max, pc_pred)), max(self.pm_min, min(self.pm_max, pm_pred))
        except: return None, None

    def collect_training_data(self, d_avg, d_min, d_max, diversity, gen, d1, d2, pc, pm, 
                             c1_dist, c2_dist):
        f1_child, f2_child = self._to_maximization_fitness(c1_dist), self._to_maximization_fitness(c2_dist)
        f1_parent, f2_parent = self._to_maximization_fitness(d1), self._to_maximization_fitness(d2)
        improvement = ((f1_child - f1_parent) + (f2_child - f2_parent)) / 2
        
        self.training_data['features'].append([d_avg, d_min, d_max, diversity, gen/self.max_gens, max(d1, d2), min(d1, d2)])
        self.training_data['pc_targets'].append(pc)
        self.training_data['pm_targets'].append(pm)
        self.training_data['performance'].append(max(0, improvement))

# --- Contoh Inisialisasi Berbagai Model untuk Tesis ---
if __name__ == "__main__":
    # Pilih model: 'DT', 'RF', 'SVM', atau 'LR'
    selected_method = 'RF' 
    
    apst = APST_GA(ml_method=selected_method)
    
    print(f"APST-GA initialized using: {apst.ml_method}")
    print(f"Model Object: {type(apst._get_ml_model()).__name__}")
