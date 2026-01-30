"""
Visualization Module
Creates charts and graphs for model performance and feature importance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import io
import base64


class PerformanceVisualizer:
    """
    Creates visualizations for model performance analysis
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        self.figures = {}
    
    def plot_metrics_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = ['accuracy', 'f1', 'precision', 'recall'],
        figsize: tuple = (14, 8)
    ) -> plt.Figure:
        """
        Create bar chart comparing metrics across models and datasets
        
        Args:
            comparison_df: DataFrame with model comparison results
            metrics: List of metrics to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            print("No metrics available to plot")
            return None
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Pivot data for grouped bar chart
            pivot_data = comparison_df.pivot(
                index='model', 
                columns='dataset', 
                values=metric
            )
            
            pivot_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel(metric.capitalize(), fontsize=10)
            ax.legend(title='Dataset', loc='lower right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.05])
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.figures['metrics_comparison'] = fig
        return fig
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        figsize: tuple = (10, 8),
        highlight_new_features: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Plot feature importance as horizontal bar chart
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            figsize: Figure size
            highlight_new_features: List of new feature names to highlight
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get top features
        top_features = importance_df.head(top_n).copy()
        
        # Determine column name for importance
        importance_col = 'importance' if 'importance' in top_features.columns else top_features.columns[1]
        
        # Create color array
        colors = []
        if highlight_new_features:
            colors = ['#ff6b6b' if feat in highlight_new_features else '#4ecdc4' 
                     for feat in top_features['feature']]
        else:
            colors = '#4ecdc4'
        
        # Create horizontal bar chart
        ax.barh(top_features['feature'], top_features[importance_col], color=colors)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend if highlighting
        if highlight_new_features:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff6b6b', label='New Features'),
                Patch(facecolor='#4ecdc4', label='Original Features')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
        
        # Invert y-axis to show highest importance at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        self.figures['feature_importance'] = fig
        return fig
    
    def plot_improvement_heatmap(
    self,
    improvements_df: pd.DataFrame,
    figsize: tuple = (10, 6)
    ) -> plt.Figure:
        """
        Create heatmap showing improvement across models and metrics
    
        Args:
            improvements_df: DataFrame with improvement data
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
    
        # Prepare data for heatmap
        if 'model' in improvements_df.columns and 'relative_improvement_%' in improvements_df.columns:
            # Create a simple DataFrame with model and improvement
            heatmap_data = improvements_df.set_index('model')[['relative_improvement_%']]
            heatmap_data.columns = ['Improvement %']
        else:
            heatmap_data = improvements_df
    
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Improvement %'},
            linewidths=0.5,
            linecolor='white'
        )
    
        ax.set_title('Model Performance Improvement', fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Model', fontsize=11)
    
        # Rotate y-axis labels for better readability
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
        plt.tight_layout()
        self.figures['improvement_heatmap'] = fig
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        figsize: tuple = (12, 10)
    ) -> plt.Figure:
        """
        Plot correlation matrix for features
        
        Args:
            df: DataFrame with features
            features: Specific features to include (None = all numeric)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Select features
        if features:
            plot_df = df[features]
        else:
            plot_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr_matrix = plot_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        self.figures['correlation_matrix'] = fig
        return fig
    
    def plot_model_comparison_radar(
        self,
        comparison_df: pd.DataFrame,
        model_names: Optional[List[str]] = None,
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
        figsize: tuple = (10, 10)
    ) -> plt.Figure:
        """
        Create radar chart comparing models across metrics
        
        Args:
            comparison_df: DataFrame with model comparison
            model_names: Specific models to include
            metrics: Metrics to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Filter for enriched dataset only
        plot_df = comparison_df[comparison_df['dataset'] == 'enriched'].copy()
        
        if model_names:
            plot_df = plot_df[plot_df['model'].isin(model_names)]
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in plot_df.columns]
        
        if not available_metrics:
            print("No metrics available for radar plot")
            return None
        
        # Number of variables
        num_vars = len(available_metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot each model
        for idx, row in plot_df.iterrows():
            values = row[available_metrics].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
            ax.fill(angles, values, alpha=0.15)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in available_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison (Enriched Features)', 
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        self.figures['model_radar'] = fig
        return fig
    
    def save_all_figures(self, output_dir: str = 'reports', dpi: int = 100):
        """
        Save all generated figures
        
        Args:
            output_dir: Directory to save figures
            dpi: Resolution for saved images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            filepath = os.path.join(output_dir, f'{name}.png')
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
    
    def figure_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string for HTML embedding
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()
        return f"data:image/png;base64,{image_base64}"
    
    def close_all(self):
        """Close all figures to free memory"""
        plt.close('all')
        self.figures = {}