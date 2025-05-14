import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.mark.api
def test_health_check(test_app):
    """Test the health check endpoint"""
    response = test_app.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.api
def test_predict_endpoint(test_app):
    """Test the prediction endpoint"""
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
    assert "prediction" in response.json()
    assert "probability" in response.json()
    
@pytest.mark.api
def test_predict_endpoint_invalid_input(test_app):
    """Test the prediction endpoint with invalid input"""
    test_input = {
        "Age": "invalid",
        "Flight_Distance": 1000
    }
    
    response = test_app.post("/predict", json=test_input)
    assert response.status_code == 422  # Validation error

@pytest.mark.api
def test_metrics_endpoint(test_app):
    """Test the metrics endpoint"""
    response = test_app.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text 