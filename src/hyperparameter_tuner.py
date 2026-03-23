"""
Hyperparameter Tuning Module
Automated hyperparameter optimization using GridSearch and RandomSearch
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Performs hyperparameter tuning for ML models
    """
    
    def __init__(self, n_iter: int = 20, cv: int = 3, random_state: int = 42):
        """
        Initialize tuner
        
        Args:
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            random_state: Random seed
        """
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.best_params = {}
        self.best_models = {}
        self.tuning_results = {}
    
    def get_default_param_grids(self) -> Dict[str, Dict]:
        """
        Get default parameter grids for common models
        
        Returns:
            Dictionary of parameter grids
        """
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
        }
    
    def tune_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        param_grid: Optional[Dict] = None,
        search_type: str = 'random',
        scoring: str = 'f1'
    ) -> Any:
        """
        Tune a single model
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            param_grid: Parameter grid (None = use defaults)
            search_type: 'grid' or 'random'
            scoring: Scoring metric
            
        Returns:
            Best model
        """
        print(f"\n  Tuning {model_name}...")
        
        # Get parameter grid
        if param_grid is None:
            all_grids = self.get_default_param_grids()
            param_grid = all_grids.get(model_name, {})
        
        if not param_grid:
            print(f"    ⚠️  No parameter grid defined for {model_name}, skipping")
            return model
        
        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                model,
                param_grid,
                cv=self.cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
        else:  # random search
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Store results
        self.best_params[model_name] = search.best_params_
        self.best_models[model_name] = search.best_estimator_
        
        self.tuning_results[model_name] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }
        
        print(f"    ✓ Best {scoring}: {search.best_score_:.4f}")
        print(f"    ✓ Best params: {search.best_params_}")
        
        return search.best_estimator_
    
    def tune_all_models(
        self,
        models_dict: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grids: Optional[Dict[str, Dict]] = None,
        search_type: str = 'random',
        scoring: str = 'f1'
    ) -> Dict[str, Any]:
        """
        Tune all models
        
        Args:
            models_dict: Dictionary of models
            X_train: Training features
            y_train: Training target
            param_grids: Dictionary of parameter grids
            search_type: 'grid' or 'random'
            scoring: Scoring metric
            
        Returns:
            Dictionary of best models
        """
        print(f"\n{'='*60}")
        print(f"Hyperparameter Tuning ({search_type.upper()} SEARCH)")
        print(f"{'='*60}")
        
        tuned_models = {}
        
        for model_name, model in models_dict.items():
            param_grid = param_grids.get(model_name) if param_grids else None
            
            best_model = self.tune_model(
                model, X_train, y_train, model_name,
                param_grid, search_type, scoring
            )
            
            tuned_models[model_name] = best_model
        
        print(f"\n✓ Tuning complete for {len(tuned_models)} models")
        
        return tuned_models
    
    def get_tuning_summary(self) -> pd.DataFrame:
        """
        Get summary of tuning results
        
        Returns:
            DataFrame with tuning summary
        """
        summary_data = []
        
        for model_name, results in self.tuning_results.items():
            summary_data.append({
                'model': model_name,
                'best_score': results['best_score'],
                'best_params': str(results['best_params'])
            })
        
        return pd.DataFrame(summary_data)