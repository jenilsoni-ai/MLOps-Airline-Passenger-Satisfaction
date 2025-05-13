import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import preprocess_data, split_data

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        "id": range(100),
        "Gender": ["Male", "Female"] * 50,
        "Customer Type": ["Loyal", "Disloyal"] * 50,
        "Age": np.random.randint(18, 80, 100),
        "satisfaction": ["satisfied", "neutral"] * 50
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "data": {
            "test_size": 0.2,
            "random_state": 42
        },
        "model": {
            "target": "satisfaction",
            "features_to_drop": ["id"],
            "categorical_features": ["Gender", "Customer Type"]
        }
    }

def test_preprocess_data(sample_data, sample_config):
    """Test data preprocessing function."""
    processed_df = preprocess_data(sample_data, sample_config)
    
    # Check if specified columns are dropped
    assert "id" not in processed_df.columns
    
    # Check if categorical variables are encoded
    assert "Gender_Male" in processed_df.columns
    assert "Gender_Female" in processed_df.columns
    assert "Customer Type_Loyal" in processed_df.columns
    assert "Customer Type_Disloyal" in processed_df.columns
    
    # Check if no missing values exist
    assert processed_df.isnull().sum().sum() == 0

def test_split_data(sample_data, sample_config):
    """Test data splitting function."""
    processed_df = preprocess_data(sample_data, sample_config)
    X_train, X_test, y_train, y_test = split_data(processed_df, sample_config)
    
    # Check shapes
    assert len(X_train) == 80  # 80% of 100
    assert len(X_test) == 20   # 20% of 100
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    # Check if target is not in features
    assert sample_config["model"]["target"] not in X_train.columns
    assert sample_config["model"]["target"] not in X_test.columns 