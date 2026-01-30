"""
Model Training Pipeline
Handles training multiple ML models and storing results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Trains multiple models and tracks performance metrics
    """
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        """
        Initialize ModelTrainer
        
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.trained_models = {}
        
        # Initialize models based on task type
        if task_type == 'classification':
            self.models = {
                'logistic_regression': LogisticRegression(
                    random_state=random_state,
                    max_iter=1000
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric='logloss'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=random_state
                )
            }
        else:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            
            self.models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    random_state=random_state,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=random_state
                )
            }
    
    def train_single_model(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Any:
        """
        Train a single model
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model
    
    def evaluate_classification(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def evaluate_regression(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        print(f"{model_name} - R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        dataset_name: str = 'dataset'
    ) -> Dict[str, Dict]:
        """
        Train all models and evaluate them
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            dataset_name: Name to identify this dataset (e.g., 'original', 'enriched')
            
        Returns:
            Dictionary of results for each model
        """
        results = {}
        
        for model_name in self.models.keys():
            # Train model
            model = self.train_single_model(model_name, X_train, y_train)
            
            # Evaluate model
            if self.task_type == 'classification':
                metrics = self.evaluate_classification(model, X_test, y_test, model_name)
            else:
                metrics = self.evaluate_regression(model, X_test, y_test, model_name)
            
            # Store results
            result_key = f"{dataset_name}_{model_name}"
            results[result_key] = {
                'model': model,
                'metrics': metrics,
                'model_name': model_name,
                'dataset': dataset_name
            }
        
        self.results.update(results)
        return results
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target
            model_name: Name of model to validate
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        model = self.models[model_name]
        
        if self.task_type == 'classification':
            scoring = 'f1'
        else:
            scoring = 'r2'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all results as DataFrame
        
        Returns:
            DataFrame with all model results
        """
        summary_data = []
        
        for key, result in self.results.items():
            row = {
                'model': result['model_name'],
                'dataset': result['dataset'],
                **result['metrics']
            }
            # Remove confusion matrix from summary table
            if 'confusion_matrix' in row:
                del row['confusion_matrix']
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)