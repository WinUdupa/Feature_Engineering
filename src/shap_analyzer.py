"""
SHAP Analysis Module
Detailed SHAP-based interpretability analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """
    Performs SHAP analysis for model interpretability
    """
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        self.shap_figures = {}
    
    def create_explainer(
        self,
        model: Any,
        X_background: pd.DataFrame,
        model_name: str
    ):
        """
        Create SHAP explainer for a model
        
        Args:
            model: Trained model
            X_background: Background dataset for explainer
            model_name: Name of the model
        """
        print(f"  Creating SHAP explainer for {model_name}...")
        
        try:
            # Choose appropriate explainer
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                explainer = shap.TreeExplainer(model)
            else:
                # Linear models
                explainer = shap.LinearExplainer(model, X_background)
            
            self.explainers[model_name] = explainer
            print(f"    ✓ Explainer created")
            
        except Exception as e:
            print(f"    ⚠️  Could not create explainer: {e}")
    
    def calculate_shap_values(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        max_samples: int = 100
    ):
        """
        Calculate SHAP values
        
        Args:
            model_name: Name of the model
            X_test: Test dataset
            max_samples: Maximum samples to explain
        """
        if model_name not in self.explainers:
            print(f"  ⚠️  No explainer for {model_name}")
            return
        
        print(f"  Calculating SHAP values for {model_name}...")
        
        try:
            explainer = self.explainers[model_name]
            
            # Use subset for performance
            X_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class for binary
            
            self.shap_values[model_name] = {
                'values': shap_values,
                'data': X_sample,
                'feature_names': X_test.columns.tolist()
            }
            
            print(f"    ✓ SHAP values calculated ({X_sample.shape[0]} samples)")
            
        except Exception as e:
            print(f"    ⚠️  Error calculating SHAP: {e}")
    
    def plot_shap_summary(
        self,
        model_name: str,
        plot_type: str = 'dot',
        figsize: tuple = (10, 8)
    ) -> Optional[plt.Figure]:
        """
        Create SHAP summary plot
        
        Args:
            model_name: Name of the model
            plot_type: 'dot', 'bar', or 'violin'
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.shap_values:
            print(f"  ⚠️  No SHAP values for {model_name}")
            return None
        
        shap_data = self.shap_values[model_name]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'dot':
            shap.summary_plot(
                shap_data['values'],
                shap_data['data'],
                feature_names=shap_data['feature_names'],
                show=False
            )
        elif plot_type == 'bar':
            shap.summary_plot(
                shap_data['values'],
                shap_data['data'],
                feature_names=shap_data['feature_names'],
                plot_type='bar',
                show=False
            )
        
        plt.title(f'SHAP Summary - {model_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        self.shap_figures[f'{model_name}_summary'] = fig
        
        return fig
    
    def plot_shap_waterfall(
        self,
        model_name: str,
        sample_idx: int = 0,
        figsize: tuple = (10, 6)
    ) -> Optional[plt.Figure]:
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            model_name: Name of the model
            sample_idx: Index of sample to explain
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.shap_values:
            return None
        
        shap_data = self.shap_values[model_name]
        
        # Create explanation object
        explainer = self.explainers[model_name]
        explanation = shap.Explanation(
            values=shap_data['values'][sample_idx],
            base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
            data=shap_data['data'].iloc[sample_idx],
            feature_names=shap_data['feature_names']
        )
        
        fig = plt.figure(figsize=figsize)
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall - {model_name} (Sample {sample_idx})', fontsize=12)
        plt.tight_layout()
        
        self.shap_figures[f'{model_name}_waterfall_{sample_idx}'] = fig
        
        return fig
    
    def get_feature_importance_from_shap(
        self,
        model_name: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values
        
        Args:
            model_name: Name of the model
            top_n: Number of top features
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.shap_values:
            return pd.DataFrame()
        
        shap_data = self.shap_values[model_name]
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_data['values']).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': shap_data['feature_names'],
            'shap_importance': importance
        }).sort_values('shap_importance', ascending=False).head(top_n)
        
        return importance_df