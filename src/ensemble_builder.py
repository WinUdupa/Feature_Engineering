"""
Ensemble Model Builder
Creates ensemble models using voting and stacking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class EnsembleBuilder:
    """
    Builds ensemble models from trained base models
    """
    
    def __init__(self):
        self.ensemble_models = {}
        self.ensemble_results = {}
    
    def create_voting_ensemble(
        self,
        models_dict: Dict[str, Any],
        voting: str = 'soft'
    ) -> VotingClassifier:
        """
        Create voting ensemble
        
        Args:
            models_dict: Dictionary of trained models
            voting: 'hard' or 'soft' voting
            
        Returns:
            VotingClassifier
        """
        estimators = [(name, model) for name, model in models_dict.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        print(f"  ✓ Created {voting} voting ensemble with {len(estimators)} models")
        
        return ensemble
    
    def create_stacking_ensemble(
        self,
        models_dict: Dict[str, Any],
        final_estimator: Any = None
    ) -> StackingClassifier:
        """
        Create stacking ensemble
        
        Args:
            models_dict: Dictionary of base models
            final_estimator: Meta-learner (None = LogisticRegression)
            
        Returns:
            StackingClassifier
        """
        estimators = [(name, model) for name, model in models_dict.items()]
        
        if final_estimator is None:
            final_estimator = LogisticRegression(random_state=42)
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )
        
        print(f"  ✓ Created stacking ensemble with {len(estimators)} base models")
        
        return ensemble
    
    def train_and_evaluate_ensemble(
        self,
        ensemble: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        ensemble_name: str
    ) -> Dict[str, float]:
        """
        Train and evaluate ensemble model
        
        Args:
            ensemble: Ensemble model
            X_train, X_test: Train/test features
            y_train, y_test: Train/test targets
            ensemble_name: Name of ensemble
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        print(f"\n  Training {ensemble_name}...")
        ensemble.fit(X_train, y_train)
        
        y_pred = ensemble.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        self.ensemble_models[ensemble_name] = ensemble
        self.ensemble_results[ensemble_name] = metrics
        
        print(f"    ✓ {ensemble_name} - F1: {metrics['f1']:.4f}")
        
        return metrics