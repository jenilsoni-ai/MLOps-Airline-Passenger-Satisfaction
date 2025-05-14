import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import preprocess_data
from src.models.train_model import train_model, evaluate_model
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def sample_raw_data():
    """Create a sample raw dataset"""
    np.random.seed(42)
    n_samples = 100
    data = {
        'Age': np.random.normal(40, 15, n_samples),
        'Flight_Distance': np.random.normal(1000, 500, n_samples),
        'Inflight_wifi_service': np.random.randint(1, 6, n_samples),
        'Satisfaction': np.random.choice(['satisfied', 'neutral or dissatisfied'], n_samples)
    }
    return pd.DataFrame(data)

@pytest.fixture
def config():
    """Create a sample configuration"""
    return {
        "model": {
            "features_to_drop": [],
            "categorical_features": ["Inflight_wifi_service"],
            "target": "Satisfaction"
        },
        "data": {
            "test_size": 0.2,
            "random_state": 42
        }
    }

@pytest.mark.integration
def test_full_pipeline(test_app, sample_raw_data, config):
    """Test the full data processing, training, and prediction pipeline"""
    # 1. Process the data
    processed_data = preprocess_data(sample_raw_data, config)
    assert isinstance(processed_data, pd.DataFrame), "Processed data should be a DataFrame"
    
    # 2. Train the model
    X = processed_data.drop('Satisfaction', axis=1)
    y = processed_data['Satisfaction']
    model = train_model(X, y)
    
    # 3. Evaluate the model
    metrics = evaluate_model(model, X, y)
    assert all(0 <= value <= 1 for value in metrics.values()), "Metrics should be between 0 and 1"
    
    # 4. Test API prediction
    test_input = {
        "Age": 30,
        "Flight_Distance": 1000,
        "Inflight_wifi_service": 3,
        "Departure_Arrival_time_convenient": 4,
        "Ease_of_Online_booking": 3,
        "Gate_location": 4,
        "Food_and_drink": 5,
        "Online_boarding": 3,
        "Seat_comfort": 4,
        "Inflight_entertainment": 3,
        "On_board_service": 4,
        "Leg_room_service": 3,
        "Baggage_handling": 4,
        "Checkin_service": 3,
        "Inflight_service": 4,
        "Cleanliness": 5
    }
    
    response = test_app.post("/predict", json=test_input)
    assert response.status_code == 200
    prediction_result = response.json()
    assert "prediction" in prediction_result
    assert "probability" in prediction_result
    assert isinstance(prediction_result["prediction"], str)
    assert isinstance(prediction_result["probability"], float)

@pytest.mark.integration
def test_monitoring_integration(test_app):
    """Test the integration of monitoring with the API"""
    # Make several predictions to generate metrics
    test_input = {
        "Age": 30,
        "Flight_Distance": 1000,
        "Inflight_wifi_service": 3,
        "Departure_Arrival_time_convenient": 4,
        "Ease_of_Online_booking": 3,
        "Gate_location": 4,
        "Food_and_drink": 5,
        "Online_boarding": 3,
        "Seat_comfort": 4,
        "Inflight_entertainment": 3,
        "On_board_service": 4,
        "Leg_room_service": 3,
        "Baggage_handling": 4,
        "Checkin_service": 3,
        "Inflight_service": 4,
        "Cleanliness": 5
    }
    
    # Make 3 predictions
    for _ in range(3):
        response = test_app.post("/predict", json=test_input)
        assert response.status_code == 200
    
    # Check metrics endpoint
    metrics_response = test_app.get("/metrics")
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    
    # Verify metrics are being recorded
    assert "prediction_requests_total" in metrics_text
    assert "prediction_latency_seconds" in metrics_text 