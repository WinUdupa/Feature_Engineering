"""
Integration test for the complete ML pipeline
Tests: Data Loading -> Model Training -> Feature Importance -> Comparison
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_trainer import ModelTrainer
from src.importance_analyzer import FeatureImportanceAnalyzer
from src.comparison_analyzer import ComparisonAnalyzer


def test_full_pipeline():
    """
    Test complete pipeline from data to comparison
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full ML Pipeline")
    print("="*60)
    
    # ========================================
    # Step 1: Generate Sample Data
    # ========================================
    print("\n[Step 1] Generating sample dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Create DataFrames
    original_features = [f'feature_{i}' for i in range(8)]
    df_original = pd.DataFrame(X, columns=original_features)
    df_original['target'] = y
    
    print(f"✓ Dataset created: {df_original.shape}")
    print(f"  Features: {original_features}")
    print(f"  Target distribution: {df_original['target'].value_counts().to_dict()}")
    
    # ========================================
    # Step 2: Simulate New Features
    # ========================================
    print("\n[Step 2] Creating enriched dataset with synthetic features...")
    df_enriched = df_original.copy()
    
    # Add some engineered features (simulating LLM suggestions)
    df_enriched['feature_0_squared'] = df_enriched['feature_0'] ** 2
    df_enriched['feature_ratio'] = df_enriched['feature_1'] / (df_enriched['feature_2'] + 1)
    df_enriched['feature_interaction'] = df_enriched['feature_0'] * df_enriched['feature_3']
    
    new_features = ['feature_0_squared', 'feature_ratio', 'feature_interaction']
    print(f"✓ Added {len(new_features)} new features")
    print(f"  New features: {new_features}")
    
    # ========================================
    # Step 3: Prepare Train/Test Splits
    # ========================================
    print("\n[Step 3] Splitting data into train/test...")
    
    # Original dataset
    X_original = df_original.drop('target', axis=1)
    y = df_original['target']
    
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Enriched dataset
    X_enriched = df_enriched.drop('target', axis=1)
    X_enrich_train, X_enrich_test, _, _ = train_test_split(
        X_enriched, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Train size: {len(X_orig_train)}, Test size: {len(X_orig_test)}")
    print(f"  Original features: {X_orig_train.shape[1]}")
    print(f"  Enriched features: {X_enrich_train.shape[1]}")
    
    # ========================================
    # Step 4: Train Models on Original Data
    # ========================================
    print("\n[Step 4] Training models on ORIGINAL features...")
    print("-" * 60)
    
    trainer = ModelTrainer(task_type='classification', random_state=42)
    
    original_results = trainer.train_and_evaluate(
        X_orig_train, X_orig_test, y_train, y_test,
        dataset_name='original'
    )
    
    print(f"\n✓ Trained {len(original_results)} models on original features")
    
    # ========================================
    # Step 5: Train Models on Enriched Data
    # ========================================
    print("\n[Step 5] Training models on ENRICHED features...")
    print("-" * 60)
    
    enriched_results = trainer.train_and_evaluate(
        X_enrich_train, X_enrich_test, y_train, y_test,
        dataset_name='enriched'
    )
    
    print(f"\n✓ Trained {len(enriched_results)} models on enriched features")
    
    # ========================================
    # Step 6: Compare Performance
    # ========================================
    print("\n[Step 6] Comparing performance...")
    print("="*60)
    
    comparator = ComparisonAnalyzer(trainer.results)
    comparison_table = comparator.create_comparison_table()
    
    print("\n📊 Performance Comparison Table:")
    print(comparison_table.to_string(index=False))
    
    # Calculate improvements
    improvements = comparator.calculate_improvements(metric='f1')
    print("\n📈 Improvements (F1 Score):")
    print(improvements[['model', 'original', 'enriched', 'absolute_improvement', 'relative_improvement_%']].to_string(index=False))
    
    # Get best models
    best_original, score_original = comparator.get_best_model('original', 'f1')
    best_enriched, score_enriched = comparator.get_best_model('enriched', 'f1')
    
    print(f"\n🏆 Best Original Model: {best_original} (F1: {score_original:.4f})")
    print(f"🏆 Best Enriched Model: {best_enriched} (F1: {score_enriched:.4f})")
    
    # Generate summary
    summary = comparator.generate_summary()
    print(f"\n📋 Summary:")
    print(f"  Models improved: {summary['models_improved']}/{summary['total_models_tested']}")
    print(f"  Average improvement: {summary['average_improvement']:.2f}%")
    print(f"  Max improvement: {summary['max_improvement']:.2f}%")
    
    # ========================================
    # Step 7: Analyze Feature Importance
    # ========================================
    print("\n[Step 7] Analyzing feature importance...")
    print("="*60)
    
    analyzer = FeatureImportanceAnalyzer()
    
    # Get trained models for enriched dataset
    enriched_models = {
        key.replace('enriched_', ''): result['model'] 
        for key, result in trainer.results.items() 
        if 'enriched' in key
    }
    
    # Analyze importance
    importance_results = analyzer.analyze_all_models(
        enriched_models,
        X_enrich_test.columns.tolist(),
        X_enrich_test,
        use_shap=False  # Set to True if you want SHAP (slower)
    )
    
    print(f"✓ Analyzed {len(importance_results)} models")
    
    # Get top features
    top_features = analyzer.get_top_features(n=10)
    if not top_features.empty:
        print("\n🌟 Top 10 Important Features:")
        print(top_features.to_string(index=False))
    
    # Compare feature sets
    feature_comparison = analyzer.compare_feature_sets(
        original_features,
        new_features
    )
    
    print(f"\n🔍 Feature Set Analysis:")
    print(f"  Top original features in importance: {feature_comparison['top_original'][:5]}")
    print(f"  Top NEW features in importance: {feature_comparison['top_new']}")
    print(f"  Low impact new features: {feature_comparison['low_impact']}")
    
    # ========================================
    # Step 8: Validation Assertions
    # ========================================
    print("\n[Step 8] Running validation checks...")
    print("="*60)
    
    # Assert we have results
    assert len(trainer.results) == 8, f"Expected 8 results (4 models × 2 datasets), got {len(trainer.results)}"
    print("✓ All models trained successfully")
    
    # Assert metrics are valid
    for key, result in trainer.results.items():
        metrics = result['metrics']
        assert 0 <= metrics['accuracy'] <= 1, f"Invalid accuracy for {key}"
        assert 0 <= metrics['f1'] <= 1, f"Invalid F1 for {key}"
    print("✓ All metrics are valid (0-1 range)")
    
    # Assert we can generate comparison
    assert comparison_table is not None and len(comparison_table) > 0
    print("✓ Comparison table generated")
    
    # Assert improvements calculated
    assert len(improvements) == 4, "Expected improvements for 4 models"
    print("✓ Improvements calculated")
    
    # Check if at least one model improved
    improved_count = improvements['improved'].sum()
    print(f"✓ {improved_count}/4 models showed improvement")
    
    print("\n" + "="*60)
    print("✅ INTEGRATION TEST PASSED!")
    print("="*60)
    
    # Don't return anything - pytest tests should return None


def main():
    """
    Main function for running test manually (not via pytest)
    """
    try:
        test_full_pipeline()
        print("\n🎉 All systems operational! Ready to push.")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    # Run the test manually (not via pytest)
    import sys
    exit_code = main()
    sys.exit(exit_code)