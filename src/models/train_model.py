import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(processed_data_path):
    """Load processed data."""
    logger.info(f"Loading processed data from {processed_data_path}")
    
    X_train = pd.read_csv(f"{processed_data_path}/X_train.csv")
    X_test = pd.read_csv(f"{processed_data_path}/X_test.csv")
    y_train = pd.read_csv(f"{processed_data_path}/y_train.csv")
    y_test = pd.read_csv(f"{processed_data_path}/y_test.csv")
    
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train_model(X_train, y_train, params=None):
    """Train the model."""
    logger.info("Training model")
    
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    logger.info("Evaluating model")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics

def save_model(model, metrics, output_path="models"):
    """Save the trained model."""
    logger.info(f"Saving model to {output_path}")
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, output_path)

def main():
    """Main model training function."""
    logger.info("Starting model training")
    
    # Load configuration
    config = load_config()
    
    # Set up MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(config["data"]["processed_data_path"])
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log parameters and metrics
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        })
        mlflow.log_metrics(metrics)
        
        # Save model
        save_model(model, metrics)
        
        logger.info(f"Model metrics: {metrics}")
    
    logger.info("Model training completed")

if __name__ == "__main__":
    main() 