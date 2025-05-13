from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
import yaml
import logging
from typing import Dict, List
from .monitoring import MonitoringMiddleware, start_monitoring

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
    features: Dict[str, float]

class PredictionOutput(BaseModel):
    prediction: str
    probability: float

# Load the model at startup
@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading model")
    model = mlflow.sklearn.load_model(config["api"]["model_path"])
    logger.info("Model loaded successfully")
    
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
        features_df = pd.DataFrame([input_data.features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = np.max(model.predict_proba(features_df)[0])
        
        return PredictionOutput(
            prediction="satisfied" if prediction == 1 else "dissatisfied",
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
    return {"message": "Metrics available at :8001/metrics"} 