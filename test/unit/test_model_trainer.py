import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    """Generate sample classification data"""
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y_series = pd.Series(y, name='target')
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def test_model_trainer_initialization():
    """Test ModelTrainer initialization"""
    trainer = ModelTrainer(task_type='classification')
    assert trainer.task_type == 'classification'
    assert len(trainer.models) == 4
    assert 'logistic_regression' in trainer.models
    assert 'random_forest' in trainer.models


def test_train_single_model(sample_data):
    """Test training a single model"""
    X_train, X_test, y_train, y_test = sample_data
    trainer = ModelTrainer(task_type='classification')
    
    model = trainer.train_single_model('logistic_regression', X_train, y_train)
    assert model is not None
    assert 'logistic_regression' in trainer.trained_models


def test_evaluate_classification(sample_data):
    """Test classification evaluation"""
    X_train, X_test, y_train, y_test = sample_data
    trainer = ModelTrainer(task_type='classification')
    
    model = trainer.train_single_model('logistic_regression', X_train, y_train)
    metrics = trainer.evaluate_classification(model, X_test, y_test, 'logistic_regression')
    
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'auc_roc' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_train_and_evaluate(sample_data):
    """Test full training and evaluation pipeline"""
    X_train, X_test, y_train, y_test = sample_data
    trainer = ModelTrainer(task_type='classification')
    
    results = trainer.train_and_evaluate(
        X_train, X_test, y_train, y_test, 
        dataset_name='test'
    )
    
    assert len(results) == 4  # 4 models
    assert 'test_logistic_regression' in results
    assert results['test_logistic_regression']['metrics']['accuracy'] > 0


def test_get_results_summary(sample_data):
    """Test results summary generation"""
    X_train, X_test, y_train, y_test = sample_data
    trainer = ModelTrainer(task_type='classification')
    
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test, 'test')
    summary = trainer.get_results_summary()
    
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 4
    assert 'accuracy' in summary.columns
    assert 'f1' in summary.columns