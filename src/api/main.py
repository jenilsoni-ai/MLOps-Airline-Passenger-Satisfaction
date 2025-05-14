from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
import yaml
import logging
from typing import Dict, List
from .monitoring import MonitoringMiddleware, start_monitoring
from sklearn.ensemble import RandomForestClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path="config/config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading config: {str(e)}. Using default config.")
        return {
            "api": {
                "model_path": None
            }
        }

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Airline Passenger Satisfaction Prediction API",
    description="API for predicting airline passenger satisfaction",
    version="1.0.0"
)

# Add monitoring middleware
app.middleware("http")(MonitoringMiddleware())

class PredictionInput(BaseModel):
    Age: float
    Flight_Distance: float
    Inflight_wifi_service: int
    Departure_Arrival_time_convenient: int
    Ease_of_Online_booking: int
    Gate_location: int
    Food_and_drink: int
    Online_boarding: int
    Seat_comfort: int
    Inflight_entertainment: int
    On_board_service: int
    Leg_room_service: int
    Baggage_handling: int
    Checkin_service: int
    Inflight_service: int
    Cleanliness: int

class PredictionOutput(BaseModel):
    prediction: str
    probability: float

# Global model variable
model = None

def get_test_model():
    """Create a simple model for testing"""
    test_model = RandomForestClassifier(n_estimators=1, random_state=42)
    X = pd.DataFrame([
        [30, 1000, 3, 4, 3, 4, 5, 3, 4, 3, 4, 3, 4, 3, 4, 5],
        [40, 800, 4, 3, 4, 3, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4]
    ], columns=[
        "Age", "Flight_Distance", "Inflight_wifi_service",
        "Departure_Arrival_time_convenient", "Ease_of_Online_booking",
        "Gate_location", "Food_and_drink", "Online_boarding",
        "Seat_comfort", "Inflight_entertainment", "On_board_service",
        "Leg_room_service", "Baggage_handling", "Checkin_service",
        "Inflight_service", "Cleanliness"
    ])
    y = ["satisfied", "dissatisfied"]
    test_model.fit(X, y)
    return test_model

# Load the model at startup
@app.on_event("startup")
async def startup_event():
    global model
    try:
        if config["api"]["model_path"]:
            logger.info("Loading production model")
            model = mlflow.sklearn.load_model(config["api"]["model_path"])
        else:
            logger.info("Loading test model")
            model = get_test_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.info("Loading test model as fallback")
        model = get_test_model()
    
    # Start Prometheus metrics server
    start_monitoring(port=8001)
    logger.info("Monitoring server started on port 8001")

@app.get("/")
async def root():
    return {"message": "Welcome to the Airline Passenger Satisfaction Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Convert input data to DataFrame
        features_df = pd.DataFrame([input_data.model_dump()])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = np.max(model.predict_proba(features_df)[0])
        
        return PredictionOutput(
            prediction=prediction,
            probability=float(probability)
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    ) 