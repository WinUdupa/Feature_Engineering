"""
Complete integration test with visualizations and reporting
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_trainer import ModelTrainer
from src.importance_analyzer import FeatureImportanceAnalyzer
from src.comparison_analyzer import ComparisonAnalyzer
from src.visualizations import PerformanceVisualizer
from src.performance_reporter import PerformanceReporter


def test_complete_pipeline_with_report():
    """
    Test complete pipeline including visualizations and report generation
    """
    print("\n" + "="*70)
    print("COMPLETE INTEGRATION TEST: ML Pipeline + Visualizations + Report")
    print("="*70)
    
    # Generate data
    print("\n[1] Generating dataset...")
    X, y = make_classification(n_samples=500, n_features=8, n_informative=5, random_state=42)
    
    original_features = [f'feature_{i}' for i in range(8)]
    df_original = pd.DataFrame(X, columns=original_features)
    df_original['target'] = y
    
    # Add engineered features
    df_enriched = df_original.copy()
    df_enriched['feat_squared'] = df_enriched['feature_0'] ** 2
    df_enriched['feat_ratio'] = df_enriched['feature_1'] / (df_enriched['feature_2'] + 1)
    df_enriched['feat_interaction'] = df_enriched['feature_0'] * df_enriched['feature_3']
    
    new_features = ['feat_squared', 'feat_ratio', 'feat_interaction']
    print(f"✓ Created dataset with {len(new_features)} new features")
    
    # Split data
    print("\n[2] Splitting data...")
    X_original = df_original.drop('target', axis=1)
    X_enriched = df_enriched.drop('target', axis=1)
    y = df_original['target']
    
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42
    )
    X_enrich_train, X_enrich_test, _, _ = train_test_split(
        X_enriched, y, test_size=0.2, random_state=42
    )
    
    # Train models
    print("\n[3] Training models...")
    trainer = ModelTrainer(task_type='classification')
    trainer.train_and_evaluate(X_orig_train, X_orig_test, y_train, y_test, 'original')
    trainer.train_and_evaluate(X_enrich_train, X_enrich_test, y_train, y_test, 'enriched')
    print("✓ Trained 8 models")
    
    # Analysis
    print("\n[4] Running analysis...")
    comparator = ComparisonAnalyzer(trainer.results)
    comparison_df = comparator.create_comparison_table()
    improvements_df = comparator.calculate_improvements('f1')
    summary = comparator.generate_summary()
    
    analyzer = FeatureImportanceAnalyzer()
    enriched_models = {
        k.replace('enriched_', ''): v['model']
        for k, v in trainer.results.items() if 'enriched' in k
    }
    analyzer.analyze_all_models(enriched_models, X_enrich_test.columns.tolist(), X_enrich_test)
    importance_df = analyzer.get_top_features(n=15)
    print("✓ Analysis complete")
    
    # Create visualizations
    print("\n[5] Creating visualizations...")
    viz = PerformanceVisualizer()
    
    fig1 = viz.plot_metrics_comparison(comparison_df)
    fig2 = viz.plot_feature_importance(importance_df, top_n=15, highlight_new_features=new_features)
    fig3 = viz.plot_improvement_heatmap(improvements_df)
    
    # Convert to base64
    visualizations = {
        'metrics_comparison': viz.figure_to_base64(fig1),
        'feature_importance': viz.figure_to_base64(fig2),
        'improvement_heatmap': viz.figure_to_base64(fig3)
    }
    print("✓ Created 3 visualizations")
    
    # Generate report
    print("\n[6] Generating HTML report...")
    reporter = PerformanceReporter()
    reporter.generate_report(
        comparison_df,
        improvements_df,
        importance_df,
        summary,
        original_features,
        new_features,
        visualizations
    )
    
    report_path = reporter.generate_html_report('reports/test_report.html')
    print(f"✓ Report saved: {report_path}")
    
    # Cleanup
    viz.close_all()
    
    # Assertions
    assert os.path.exists(report_path), "Report file not created"
    assert len(visualizations) == 3, "Expected 3 visualizations"
    
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE TEST PASSED!")
    print("="*70)
    print(f"\n📄 View your report: {os.path.abspath(report_path)}")


if __name__ == '__main__':
    test_complete_pipeline_with_report()