"""
Demo script showing the ML pipeline in action
Run: python demo.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.model_trainer import ModelTrainer
from src.importance_analyzer import FeatureImportanceAnalyzer
from src.comparison_analyzer import ComparisonAnalyzer
from src.visualizations import PerformanceVisualizer
from src.performance_reporter import PerformanceReporter


def main():
    print("\n" + "="*70)
    print("🚀 GenAI-Assisted Feature Engineering Demo")
    print("="*70)
    
    # 1. Generate sample data
    print("\n[1/7] Generating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        random_state=42
    )
    
    original_features = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=original_features)
    df['target'] = y
    
    print(f"    ✓ Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"    ✓ Target balance: {dict(df['target'].value_counts())}")
    
    # 2. Simulate LLM-suggested features
    print("\n[2/7] Simulating LLM-suggested features...")
    df['power_feature'] = df['feature_0'] ** 2
    df['ratio_feature'] = df['feature_1'] / (df['feature_2'] + 1)
    df['interaction'] = df['feature_0'] * df['feature_3']
    df['log_feature'] = np.log1p(np.abs(df['feature_4']))
    df['combined'] = df['feature_5'] + df['feature_6']
    
    new_features = ['power_feature', 'ratio_feature', 'interaction', 'log_feature', 'combined']
    print(f"    ✓ Added {len(new_features)} engineered features")
    
    # 3. Prepare datasets
    print("\n[3/7] Preparing train/test splits...")
    X_original = df[original_features]
    X_enriched = df[original_features + new_features]
    y = df['target']
    
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42, stratify=y
    )
    X_enrich_train, X_enrich_test, _, _ = train_test_split(
        X_enriched, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"    ✓ Train: {len(X_orig_train)} | Test: {len(X_orig_test)}")
    print(f"    ✓ Original features: {X_orig_train.shape[1]}")
    print(f"    ✓ Enriched features: {X_enrich_train.shape[1]}")
    
    # 4. Train models
    print("\n[4/7] Training models (this may take a minute)...")
    trainer = ModelTrainer(task_type='classification', random_state=42)
    
    print("    Training on ORIGINAL features...")
    trainer.train_and_evaluate(
        X_orig_train, X_orig_test, y_train, y_test, 
        dataset_name='original'
    )
    
    print("\n    Training on ENRICHED features...")
    trainer.train_and_evaluate(
        X_enrich_train, X_enrich_test, y_train, y_test, 
        dataset_name='enriched'
    )
    
    print(f"\n    ✓ Trained {len(trainer.results)} model configurations")
    
    # 5. Analyze results
    print("\n[5/7] Analyzing results...")
    comparator = ComparisonAnalyzer(trainer.results)
    comparison_df = comparator.create_comparison_table()
    improvements_df = comparator.calculate_improvements('f1')
    summary = comparator.generate_summary()
    
    analyzer = FeatureImportanceAnalyzer()
    enriched_models = {
        k.replace('enriched_', ''): v['model']
        for k, v in trainer.results.items() if 'enriched' in k
    }
    analyzer.analyze_all_models(
        enriched_models, 
        X_enrich_test.columns.tolist(), 
        X_enrich_test,
        use_shap=False
    )
    importance_df = analyzer.get_top_features(n=15)
    
    print("    ✓ Performance comparison completed")
    print("    ✓ Feature importance analyzed")
    
    # Display key results
    print("\n" + "="*70)
    print("📊 KEY RESULTS")
    print("="*70)
    
    print(f"\n🏆 Best Models:")
    print(f"   Original: {summary['best_original_model']}")
    print(f"   Enriched: {summary['best_enriched_model']}")
    
    print(f"\n📈 Improvements:")
    print(f"   Models improved: {summary['models_improved']}/{summary['total_models_tested']}")
    print(f"   Average improvement: {summary['average_improvement']:.2f}%")
    print(f"   Max improvement: {summary['max_improvement']:.2f}%")
    
    print("\n📋 Performance by Model:")
    for _, row in improvements_df.iterrows():
        status = "✓" if row['improved'] else "✗"
        print(f"   {status} {row['model']:20s} | Original: {row['original']:.4f} | Enriched: {row['enriched']:.4f} | Δ: {row['relative_improvement_%']:+.2f}%")
    
    print("\n🌟 Top 5 Important Features:")
    for idx, row in importance_df.head(5).iterrows():
        feature_type = "NEW" if row['feature'] in new_features else "ORIG"
        importance_col = 'importance' if 'importance' in row.index else row.index[1]
        print(f"   {idx+1}. [{feature_type:4s}] {row['feature']:20s} | Importance: {row[importance_col]:.4f}")
    
    # 6. Create visualizations
    print("\n[6/7] Creating visualizations...")
    viz = PerformanceVisualizer()
    
    fig1 = viz.plot_metrics_comparison(comparison_df, figsize=(16, 8))
    fig2 = viz.plot_feature_importance(
        importance_df, 
        top_n=15, 
        highlight_new_features=new_features,
        figsize=(12, 10)
    )
    fig3 = viz.plot_improvement_heatmap(improvements_df, figsize=(10, 6))
    
    # Save figures
    viz.save_all_figures('reports', dpi=150)
    
    # Convert to base64 for HTML
    visualizations = {
        'metrics_comparison': viz.figure_to_base64(fig1),
        'feature_importance': viz.figure_to_base64(fig2),
        'improvement_chart': viz.figure_to_base64(fig3)
    }
    
    print("    ✓ Created 3 visualizations")
    
    # 7. Generate report
    print("\n[7/7] Generating HTML report...")
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
    
    report_path = reporter.generate_html_report('reports/demo_report.html')
    
    # Cleanup
    viz.close_all()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📄 View your report:")
    print(f"   {report_path}")
    print(f"\n📊 View saved visualizations:")
    print(f"   reports/metrics_comparison.png")
    print(f"   reports/feature_importance.png")
    print(f"   reports/improvement_heatmap.png")
    print("\n💡 Next steps:")
    print("   - Open the HTML report in your browser")
    print("   - Review the model improvements")
    print("   - Analyze which features contributed most")
    print("   - Ready to integrate with Person A's LLM feature suggester!")
    print()


if __name__ == '__main__':
    main()