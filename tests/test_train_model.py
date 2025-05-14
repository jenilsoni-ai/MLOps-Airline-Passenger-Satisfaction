import pytest
import pandas as pd
import numpy as np
from src.models.train_model import train_model, evaluate_model
from sklearn.model_selection import train_test_split

@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing"""
    np.random.seed(42)
    n_samples = 100
    data = {
        'Age': np.random.normal(40, 15, n_samples),
        'Flight_Distance': np.random.normal(1000, 500, n_samples),
        'Satisfaction': np.random.choice(['satisfied', 'neutral or dissatisfied'], n_samples)
    }
    return pd.DataFrame(data)

@pytest.mark.model
def test_train_model_shape(sample_data):
    """Test if model training returns correct shapes"""
    X = sample_data.drop('Satisfaction', axis=1)
    y = sample_data['Satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    assert hasattr(model, 'predict'), "Model should have predict method"
    assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"

@pytest.mark.model
def test_evaluate_model(sample_data):
    """Test if model evaluation returns valid metrics"""
    X = sample_data.drop('Satisfaction', axis=1)
    y = sample_data['Satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    
    assert 'accuracy' in metrics, "Metrics should include accuracy"
    assert 'precision' in metrics, "Metrics should include precision"
    assert 'recall' in metrics, "Metrics should include recall"
    assert 'f1' in metrics, "Metrics should include f1 score"
    
    for metric_value in metrics.values():
        assert 0 <= metric_value <= 1, f"Metric value {metric_value} should be between 0 and 1" 