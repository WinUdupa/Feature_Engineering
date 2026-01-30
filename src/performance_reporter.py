"""
Performance Report Generator
Creates comprehensive HTML reports with visualizations
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import os


class PerformanceReporter:
    """
    Generates comprehensive performance reports
    """
    
    def __init__(self):
        self.report_data = {}
        self.html_content = ""
    
    def generate_report(
        self,
        comparison_df: pd.DataFrame,
        improvements_df: pd.DataFrame,
        feature_importance_df: pd.DataFrame,
        summary_dict: Dict,
        original_features: List[str],
        new_features: List[str],
        visualizations: Optional[Dict] = None
    ) -> Dict:
        """
        Generate comprehensive report
        
        Args:
            comparison_df: Model comparison DataFrame
            improvements_df: Improvements DataFrame
            feature_importance_df: Feature importance DataFrame
            summary_dict: Summary statistics dictionary
            original_features: List of original feature names
            new_features: List of new feature names
            visualizations: Dictionary of base64-encoded visualizations
            
        Returns:
            Dictionary with report data
        """
        self.report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': summary_dict,
            'comparison_table': comparison_df,
            'improvements': improvements_df,
            'feature_importance': feature_importance_df,
            'original_features': original_features,
            'new_features': new_features,
            'visualizations': visualizations or {}
        }
        
        return self.report_data
    
    def create_executive_summary(self) -> str:
        """
        Create executive summary section
        
        Returns:
            HTML string for executive summary
        """
        summary = self.report_data['summary']
        
        html = f"""
        <div class="summary-section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>{summary['total_models_tested']}</h3>
                    <p>Models Tested</p>
                </div>
                <div class="summary-card">
                    <h3>{summary['models_improved']}/{summary['total_models_tested']}</h3>
                    <p>Models Improved</p>
                </div>
                <div class="summary-card">
                    <h3>{summary['average_improvement']:.2f}%</h3>
                    <p>Avg Improvement</p>
                </div>
                <div class="summary-card">
                    <h3>{summary['max_improvement']:.2f}%</h3>
                    <p>Max Improvement</p>
                </div>
            </div>
            
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Best Original Model:</strong> {summary['best_original_model']}</li>
                    <li><strong>Best Enriched Model:</strong> {summary['best_enriched_model']}</li>
                    <li><strong>Primary Metric:</strong> {summary['primary_metric'].upper()}</li>
                    <li><strong>New Features Added:</strong> {len(self.report_data['new_features'])}</li>
                </ul>
            </div>
        </div>
        """
        return html
    
    def create_comparison_table_html(self) -> str:
        """
        Create HTML table for model comparison
        
        Returns:
            HTML string for comparison table
        """
        df = self.report_data['comparison_table']
        
        html = f"""
        <div class="table-section">
            <h2>📈 Model Performance Comparison</h2>
            {df.to_html(index=False, classes='comparison-table', float_format='%.4f')}
        </div>
        """
        return html
    
    def create_improvements_section(self) -> str:
        """
        Create improvements section
        
        Returns:
            HTML string for improvements
        """
        df = self.report_data['improvements']
        
        html = f"""
        <div class="improvements-section">
            <h2>🚀 Performance Improvements</h2>
            {df[['model', 'original', 'enriched', 'absolute_improvement', 'relative_improvement_%']].to_html(
                index=False, 
                classes='improvements-table',
                float_format='%.4f'
            )}
        </div>
        """
        return html
    
    def create_feature_importance_section(self) -> str:
        """
        Create feature importance section
        
        Returns:
            HTML string for feature importance
        """
        df = self.report_data['feature_importance'].head(15)
        
        # Highlight new features
        new_features = self.report_data['new_features']
        
        html = f"""
        <div class="feature-section">
            <h2>🌟 Top 15 Important Features</h2>
            <p class="feature-note">
                <span class="new-feature-badge">●</span> New Features 
                <span class="original-feature-badge">●</span> Original Features
            </p>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        importance_col = 'importance' if 'importance' in df.columns else df.columns[1]
        
        for idx, row in df.iterrows():
            feature_type = 'new' if row['feature'] in new_features else 'original'
            badge_class = 'new-feature-badge' if feature_type == 'new' else 'original-feature-badge'
            
            html += f"""
                    <tr>
                        <td>{idx + 1}</td>
                        <td><span class="{badge_class}">●</span> {row['feature']}</td>
                        <td>{row[importance_col]:.4f}</td>
                        <td>{feature_type.capitalize()}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        return html
    
    def create_visualizations_section(self) -> str:
        """
        Create visualizations section
        
        Returns:
            HTML string for visualizations
        """
        viz = self.report_data['visualizations']
        
        html = """
        <div class="visualizations-section">
            <h2>📊 Visualizations</h2>
        """
        
        for name, image_data in viz.items():
            title = name.replace('_', ' ').title()
            html += f"""
            <div class="visualization">
                <h3>{title}</h3>
                <img src="{image_data}" alt="{title}" style="max-width: 100%; height: auto;">
            </div>
            """
        
        html += "</div>"
        return html
    
    def generate_html_report(self, output_path: str = 'reports/performance_report.html'):
        """
        Generate complete HTML report
        
        Args:
            output_path: Path to save HTML report
        """
        css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 40px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                border-radius: 10px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 4px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }
            h2 {
                color: #34495e;
                margin-top: 40px;
                margin-bottom: 20px;
                border-left: 5px solid #3498db;
                padding-left: 15px;
            }
            h3 {
                color: #7f8c8d;
            }
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .summary-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .summary-card h3 {
                margin: 0;
                font-size: 36px;
                color: white;
            }
            .summary-card p {
                margin: 10px 0 0 0;
                font-size: 14px;
                opacity: 0.9;
            }
            .key-findings {
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }
            .key-findings ul {
                list-style-type: none;
                padding-left: 0;
            }
            .key-findings li {
                padding: 8px 0;
                border-bottom: 1px solid #bdc3c7;
            }
            .key-findings li:last-child {
                border-bottom: none;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th {
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }
            td {
                padding: 10px 12px;
                border-bottom: 1px solid #ecf0f1;
            }
            tr:hover {
                background-color: #f8f9fa;
            }
            .new-feature-badge {
                color: #e74c3c;
                font-size: 18px;
                margin-right: 5px;
            }
            .original-feature-badge {
                color: #3498db;
                font-size: 18px;
                margin-right: 5px;
            }
            .feature-note {
                background-color: #fff3cd;
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
                margin: 15px 0;
            }
            .visualization {
                margin: 30px 0;
                padding: 20px;
                background-color: #fafafa;
                border-radius: 8px;
            }
            .timestamp {
                color: #7f8c8d;
                font-size: 14px;
                text-align: right;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #ecf0f1;
            }
        </style>
        """
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Feature Engineering Performance Report</title>
            {css}
        </head>
        <body>
            <div class="container">
                <h1>🤖 GenAI-Assisted Feature Engineering Report</h1>
                
                {self.create_executive_summary()}
                {self.create_comparison_table_html()}
                {self.create_improvements_section()}
                {self.create_feature_importance_section()}
                {self.create_visualizations_section()}
                
                <div class="timestamp">
                    <p>Report generated: {self.report_data['timestamp']}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"✓ HTML Report saved: {output_path}")
        return output_path