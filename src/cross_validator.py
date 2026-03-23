"""
Cross-Validation Module
Provides robust model validation with multiple folds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


class CrossValidator:
    """
    Performs cross-validation on models
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize CrossValidator
        
        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_results = {}
    
    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a single model
        
        Args:
            model: Trained model instance
            X: Features
            y: Target
            model_name: Name of the model
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with CV results
        """
        # Define scoring metrics
        if task_type == 'classification':
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, zero_division=0),
                'recall': make_scorer(recall_score, zero_division=0),
                'f1': make_scorer(f1_score, zero_division=0)
            }
            cv_splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            scoring = {
                'mse': make_scorer(mean_squared_error, greater_is_better=False),
                'mae': make_scorer(mean_absolute_error, greater_is_better=False),
                'r2': make_scorer(r2_score)
            }
            cv_splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        print(f"  Cross-validating {model_name}...")
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Process results
        results = {
            'model': model_name,
            'n_folds': self.n_folds
        }
        
        # Calculate mean and std for each metric
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            # Handle negative scores (MSE, MAE)
            if metric in ['mse', 'mae']:
                test_scores = -test_scores
                train_scores = -train_scores
            
            results[f'{metric}_mean'] = np.mean(test_scores)
            results[f'{metric}_std'] = np.std(test_scores)
            results[f'{metric}_train_mean'] = np.mean(train_scores)
            results[f'{metric}_scores'] = test_scores.tolist()
        
        # Check for overfitting
        if task_type == 'classification':
            train_acc = results['accuracy_train_mean']
            test_acc = results['accuracy_mean']
            results['overfit_gap'] = train_acc - test_acc
            results['is_overfitting'] = results['overfit_gap'] > 0.1
        
        self.cv_results[model_name] = results
        
        print(f"    ✓ {model_name}: Mean F1 = {results.get('f1_mean', 0):.4f} (±{results.get('f1_std', 0):.4f})")
        
        return results
    
    def cross_validate_all(
        self,
        models_dict: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = 'classification'
    ) -> pd.DataFrame:
        """
        Cross-validate all models
        
        Args:
            models_dict: Dictionary of {model_name: model_instance}
            X: Features
            y: Target
            task_type: Type of ML task
            
        Returns:
            DataFrame with CV results
        """
        print(f"\n{'='*60}")
        print(f"Cross-Validation ({self.n_folds} folds)")
        print(f"{'='*60}")
        
        all_results = []
        
        for model_name, model in models_dict.items():
            result = self.cross_validate_model(model, X, y, model_name, task_type)
            all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        return results_df
    
    def get_cv_summary(self) -> pd.DataFrame:
        """
        Get summary of cross-validation results
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.cv_results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, results in self.cv_results.items():
            row = {
                'model': model_name,
                'n_folds': results['n_folds']
            }
            
            # Add mean scores
            for key, value in results.items():
                if key.endswith('_mean') and not key.endswith('_train_mean'):
                    metric_name = key.replace('_mean', '')
                    row[metric_name] = value
                    
                    # Add std if available
                    std_key = f"{metric_name}_std"
                    if std_key in results:
                        row[f"{metric_name}_std"] = results[std_key]
            
            # Add overfitting info if available
            if 'is_overfitting' in results:
                row['overfitting'] = results['is_overfitting']
                row['overfit_gap'] = results['overfit_gap']
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def compare_with_holdout(
        self,
        cv_results: pd.DataFrame,
        holdout_results: pd.DataFrame,
        metric: str = 'f1'
    ) -> pd.DataFrame:
        """
        Compare CV results with holdout test results
        
        Args:
            cv_results: DataFrame from cross_validate_all
            holdout_results: DataFrame from ModelTrainer.get_results_summary
            metric: Metric to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison = []
        
        for _, cv_row in cv_results.iterrows():
            model_name = cv_row['model']
            
            # Find matching holdout result
            holdout_match = holdout_results[holdout_results['model'] == model_name]
            
            if len(holdout_match) > 0:
                holdout_score = holdout_match[metric].iloc[0]
                cv_mean = cv_row[f'{metric}_mean'] if f'{metric}_mean' in cv_row else cv_row[metric]
                cv_std = cv_row[f'{metric}_std'] if f'{metric}_std' in cv_row else 0
                
                comparison.append({
                    'model': model_name,
                    f'{metric}_cv_mean': cv_mean,
                    f'{metric}_cv_std': cv_std,
                    f'{metric}_holdout': holdout_score,
                    'difference': holdout_score - cv_mean,
                    'within_1std': abs(holdout_score - cv_mean) <= cv_std
                })
        
        return pd.DataFrame(comparison)