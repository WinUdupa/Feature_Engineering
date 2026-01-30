"""
Performance Comparison Analysis
Compares baseline vs enriched model performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class ComparisonAnalyzer:
    """
    Compares performance between baseline and enriched models
    """
    
    def __init__(self, results_dict: Dict):
        """
        Initialize with results from ModelTrainer
        
        Args:
            results_dict: Dictionary of model results
        """
        self.results = results_dict
        self.comparison_df = None
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of all metrics
        
        Returns:
            DataFrame comparing all models and datasets
        """
        comparison_data = []
        
        for key, result in self.results.items():
            metrics = result['metrics'].copy()
            
            # Remove non-numeric fields
            if 'confusion_matrix' in metrics:
                del metrics['confusion_matrix']
            
            row = {
                'model': result['model_name'],
                'dataset': result['dataset'],
                **metrics
            }
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df
    
    def calculate_improvements(
        self,
        metric: str = 'f1'
    ) -> pd.DataFrame:
        """
        Calculate improvement from original to enriched features
        
        Args:
            metric: Metric to calculate improvement for
            
        Returns:
            DataFrame with improvement statistics
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        improvements = []
        
        # Get unique models
        models = self.comparison_df['model'].unique()
        
        for model_name in models:
            # Get original and enriched metrics
            original = self.comparison_df[
                (self.comparison_df['model'] == model_name) & 
                (self.comparison_df['dataset'] == 'original')
            ]
            enriched = self.comparison_df[
                (self.comparison_df['model'] == model_name) & 
                (self.comparison_df['dataset'] == 'enriched')
            ]
            
            if len(original) > 0 and len(enriched) > 0 and metric in original.columns:
                original_val = original[metric].iloc[0]
                enriched_val = enriched[metric].iloc[0]
                
                absolute_improvement = enriched_val - original_val
                relative_improvement = (absolute_improvement / original_val) * 100 if original_val != 0 else 0
                
                improvements.append({
                    'model': model_name,
                    'metric': metric,
                    'original': original_val,
                    'enriched': enriched_val,
                    'absolute_improvement': absolute_improvement,
                    'relative_improvement_%': relative_improvement,
                    'improved': absolute_improvement > 0
                })
        
        return pd.DataFrame(improvements)
    
    def get_best_model(
        self,
        dataset: str = 'enriched',
        metric: str = 'f1'
    ) -> Tuple[str, float]:
        """
        Get best performing model for a dataset
        
        Args:
            dataset: 'original' or 'enriched'
            metric: Metric to optimize
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        subset = self.comparison_df[self.comparison_df['dataset'] == dataset]
        
        if metric not in subset.columns:
            return None, None
        
        best_idx = subset[metric].idxmax()
        best_model = subset.loc[best_idx, 'model']
        best_score = subset.loc[best_idx, metric]
        
        return best_model, best_score
    
    def statistical_significance_test(
        self,
        model_name: str,
        metric: str = 'f1'
    ) -> Dict:
        """
        Perform t-test to check if improvement is statistically significant
        
        Note: This requires cross-validation scores, simplified version here
        
        Args:
            model_name: Name of model to test
            metric: Metric to test
            
        Returns:
            Dictionary with test results
        """
        # Placeholder: would need CV scores for proper test
        # For now, return basic comparison
        
        improvements = self.calculate_improvements(metric)
        model_improvement = improvements[improvements['model'] == model_name]
        
        if len(model_improvement) == 0:
            return {'significant': False, 'p_value': None}
        
        improvement = model_improvement['absolute_improvement'].iloc[0]
        
        # Simplified: consider >1% improvement as "significant"
        return {
            'significant': abs(improvement) > 0.01,
            'improvement': improvement,
            'note': 'Simplified significance test - use CV for rigorous testing'
        }
    
    def generate_summary(self) -> Dict:
        """
        Generate executive summary of comparisons
        
        Returns:
            Dictionary with summary statistics
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        # Determine metric based on available columns
        available_metrics = [col for col in self.comparison_df.columns 
                           if col not in ['model', 'dataset']]
        primary_metric = 'f1' if 'f1' in available_metrics else available_metrics[0]
        
        improvements = self.calculate_improvements(primary_metric)
        
        return {
            'total_models_tested': len(self.comparison_df['model'].unique()),
            'datasets_compared': self.comparison_df['dataset'].unique().tolist(),
            'primary_metric': primary_metric,
            'models_improved': improvements['improved'].sum(),
            'average_improvement': improvements['relative_improvement_%'].mean(),
            'best_original_model': self.get_best_model('original', primary_metric)[0],
            'best_enriched_model': self.get_best_model('enriched', primary_metric)[0],
            'max_improvement': improvements['relative_improvement_%'].max(),
            'improvements_by_model': improvements.to_dict('records')
        }