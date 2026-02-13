"""Offline integration tests (no API calls)."""

import pandas as pd
import pytest
from src.data_pipeline import DataPipeline
from src.feature_suggester import FeatureSuggestion
from pathlib import Path


@pytest.fixture
def sample_csv(tmp_path):
    """Create sample CSV file."""
    df = pd.DataFrame({
        'age': range(20, 70, 5),
        'income': [30000 + i*5000 for i in range(10)],
        'experience': range(0, 100, 10),
        'target': [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
    })
    csv_file = tmp_path / "test_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


class TestDataPipelineOffline:
    """Test pipeline without API calls."""
    
    def test_load_data(self, sample_csv):
        """Test data loading."""
        pipeline = DataPipeline(sample_csv, 'target')
        df = pipeline.load_data()
        
        assert df.shape[0] == 10
        assert 'target' in df.columns
    
    def test_analyze_dataset(self, sample_csv):
        """Test dataset analysis."""
        pipeline = DataPipeline(sample_csv, 'target')
        pipeline.load_data()
        analyzer = pipeline.analyze_dataset()
        
        assert analyzer.metadata is not None
        assert 'target_info' in analyzer.metadata
    
    def test_split_data(self, sample_csv):
        """Test train/test split."""
        pipeline = DataPipeline(sample_csv, 'target')
        pipeline.load_data()
        pipeline.analyze_dataset()
        pipeline.enriched_df = pipeline.raw_df.copy()
        
        X_train, X_test, y_train, y_test = pipeline.split_data(test_size=0.2)
        
        assert len(X_train) + len(X_test) == 10
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestFeatureGenerator:
    """Test feature generation."""
    
    def test_generate_single_feature(self):
        """Test single feature generation."""
        from src.feature_generator import FeatureGenerator
        
        df = pd.DataFrame({'age': [20, 30, 40, 50, 60]})
        generator = FeatureGenerator(df)
        
        suggestion = FeatureSuggestion(
            name='age_squared',
            formula='age ** 2',
            rationale='Test',
            feature_type='numerical',
            python_code="df['age_squared'] = df['age'] ** 2"
        )
        
        success, error = generator.generate_feature(suggestion)
        
        assert success is True
        assert error is None
        assert 'age_squared' in generator.generated_features
    
    def test_validate_features(self):
        """Test feature validation."""
        from src.feature_generator import FeatureGenerator
        
        df = pd.DataFrame({'age': [20, 30, 40, 50, 60]})
        generator = FeatureGenerator(df)
        
        suggestion = FeatureSuggestion(
            name='age_squared',
            formula='age ** 2',
            rationale='Test',
            feature_type='numerical',
            python_code="df['age_squared'] = df['age'] ** 2"
        )
        
        generator.generate_feature(suggestion)
        validation = generator.validate_features()
        
        assert 'total_features' in validation
        assert validation['valid_features'] >= 0


class TestDatasetAnalyzer:
    """Test dataset analyzer in detail."""
    
    def test_analyze_numeric_features(self):
        """Test numeric feature analysis."""
        df = pd.DataFrame({
            'age': [20, 30, 40, 50, 60],
            'income': [30000, 50000, 70000, 90000, 100000]
        })
        
        from src.dataset_analyzer import DatasetAnalyzer
        analyzer = DatasetAnalyzer(df, 'age')
        analyzer.analyze()
        
        numeric = analyzer.metadata['numeric_features']
        assert 'income' in numeric
        # Mean of [30000, 50000, 70000, 90000, 100000] = 68000
        assert numeric['income']['mean'] == 68000.0
    
    def test_analyze_numeric_features_min_max(self):
        """Test numeric feature min/max."""
        df = pd.DataFrame({
            'age': [20, 30, 40, 50, 60],
            'income': [30000, 50000, 70000, 90000, 100000]
        })
        
        from src.dataset_analyzer import DatasetAnalyzer
        analyzer = DatasetAnalyzer(df, 'age')
        analyzer.analyze()
        
        numeric = analyzer.metadata['numeric_features']
        assert numeric['income']['min'] == 30000.0
        assert numeric['income']['max'] == 100000.0
    
    def test_analyze_target_regression(self):
        """Test target analysis for regression."""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'target': [10, 20, 30, 40, 50]
        })
        
        from src.dataset_analyzer import DatasetAnalyzer
        analyzer = DatasetAnalyzer(df, 'target')
        analyzer.analyze()
        
        target_info = analyzer.metadata['target_info']
        assert target_info['type'] == 'regression'
    
    def test_analyze_target_classification(self):
        """Test target analysis for classification."""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'target': ['a', 'b', 'a', 'b', 'a']
        })
        
        from src.dataset_analyzer import DatasetAnalyzer
        analyzer = DatasetAnalyzer(df, 'target')
        analyzer.analyze()
        
        target_info = analyzer.metadata['target_info']
        assert target_info['type'] == 'classification'
    
    def test_analyze_missing_values(self):
        """Test missing value detection."""
        df = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 1, 1, 0]
        })
        
        from src.dataset_analyzer import DatasetAnalyzer
        analyzer = DatasetAnalyzer(df, 'target')
        analyzer.analyze()
        
        missing = analyzer.metadata['missing_data']
        assert 'feature1' in missing
        assert missing['feature1'] == 20.0  # 1 missing out of 5 = 20%