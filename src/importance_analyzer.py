"""
Feature Importance Analysis
Extracts and analyzes feature importance from trained models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import shap
import warnings
warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance across different model types
    """
    
    def __init__(self):
        self.importance_results = {}
    
    def get_tree_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str
    ) -> pd.DataFrame:
        """
        Extract importance from tree-based models
        
        Args:
            model: Trained tree-based model
            feature_names: List of feature names
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_,
            'model': model_name
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_linear_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str
    ) -> pd.DataFrame:
        """
        Extract importance from linear models (using coefficient magnitude)
        
        Args:
            model: Trained linear model
            feature_names: List of feature names
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'coef_'):
            return pd.DataFrame()
        
        # For binary classification, coef_ is 1D, for multiclass it's 2D
        coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coef),
            'model': model_name
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_shap_importance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        model_name: str,
        max_samples: int = 100
    ) -> pd.DataFrame:
        """
        Calculate SHAP-based feature importance
        
        Args:
            model: Trained model
            X_test: Test dataset
            model_name: Name of the model
            max_samples: Maximum samples to use for SHAP (for performance)
            
        Returns:
            DataFrame with SHAP importance
        """
        try:
            # Use subset for performance
            X_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
            
            # Choose appropriate explainer
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_sample)
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Take positive class for binary
            
            importance_df = pd.DataFrame({
                'feature': X_test.columns.tolist(),
                'shap_importance': np.abs(shap_values).mean(axis=0),
                'model': model_name
            }).sort_values('shap_importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error calculating SHAP for {model_name}: {e}")
            return pd.DataFrame()
    
    def analyze_model(
        self,
        model: Any,
        model_name: str,
        feature_names: List[str],
        X_test: Optional[pd.DataFrame] = None,
        use_shap: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive importance analysis for a single model
        
        Args:
            model: Trained model
            model_name: Name of the model
            feature_names: List of feature names
            X_test: Test data (required for SHAP)
            use_shap: Whether to calculate SHAP values
            
        Returns:
            Dictionary with different importance measures
        """
        results = {}
        
        # Try tree-based importance
        tree_imp = self.get_tree_importance(model, feature_names, model_name)
        if not tree_imp.empty:
            results['tree_importance'] = tree_imp
        
        # Try linear importance
        linear_imp = self.get_linear_importance(model, feature_names, model_name)
        if not linear_imp.empty:
            results['linear_importance'] = linear_imp
        
        # SHAP importance (optional)
        if use_shap and X_test is not None:
            shap_imp = self.get_shap_importance(model, X_test, model_name)
            if not shap_imp.empty:
                results['shap_importance'] = shap_imp
        
        return results
    
    def analyze_all_models(
        self,
        models_dict: Dict[str, Any],
        feature_names: List[str],
        X_test: Optional[pd.DataFrame] = None,
        use_shap: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Analyze importance across all models
        
        Args:
            models_dict: Dictionary of {model_name: trained_model}
            feature_names: List of feature names
            X_test: Test data
            use_shap: Whether to use SHAP
            
        Returns:
            Nested dictionary of importance results
        """
        all_results = {}
        
        for model_name, model in models_dict.items():
            print(f"Analyzing {model_name}...")
            results = self.analyze_model(
                model, model_name, feature_names, X_test, use_shap
            )
            all_results[model_name] = results
        
        self.importance_results = all_results
        return all_results
    
    def get_top_features(
        self,
        n: int = 10,
        importance_type: str = 'tree_importance'
    ) -> pd.DataFrame:
        """
        Get top N features across all models
        
        Args:
            n: Number of top features
            importance_type: Type of importance to use
            
        Returns:
            DataFrame with top features
        """
        all_importance = []
        
        for model_name, results in self.importance_results.items():
            if importance_type in results:
                df = results[importance_type].copy()
                df['model'] = model_name
                all_importance.append(df)
        
        if not all_importance:
            return pd.DataFrame()
        
        combined = pd.concat(all_importance, ignore_index=True)
        
        # Average importance across models
        avg_importance = combined.groupby('feature').agg({
            'importance': 'mean' if 'importance' in combined.columns else 'shap_importance'
        }).reset_index()
        
        return avg_importance.sort_values(
            by=avg_importance.columns[1], 
            ascending=False
        ).head(n)
    
    def compare_feature_sets(
        self,
        original_features: List[str],
        new_features: List[str]
    ) -> Dict[str, List[str]]:
        """
        Compare importance of original vs new features
        
        Args:
            original_features: List of original feature names
            new_features: List of newly generated feature names
            
        Returns:
            Dictionary categorizing features by importance
        """
        top_features = self.get_top_features(n=20)
        
        if top_features.empty:
            return {'top_original': [], 'top_new': [], 'low_impact': []}
        
        top_feature_names = top_features['feature'].tolist()
        
        return {
            'top_original': [f for f in top_feature_names if f in original_features],
            'top_new': [f for f in top_feature_names if f in new_features],
            'low_impact': [f for f in new_features if f not in top_feature_names]
        }