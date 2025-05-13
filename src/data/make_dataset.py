import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(data_path):
    """Load raw data."""
    logger.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)

def preprocess_data(df, config):
    """Preprocess the data."""
    logger.info("Preprocessing data")
    
    # Drop specified columns
    df = df.drop(config["model"]["features_to_drop"], axis=1)
    
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables
    categorical_features = config["model"]["categorical_features"]
    df = pd.get_dummies(df, columns=categorical_features)
    
    return df

def split_data(df, config):
    """Split data into train and test sets."""
    logger.info("Splitting data into train and test sets")
    
    X = df.drop(config["model"]["target"], axis=1)
    y = df[config["model"]["target"]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test, output_path):
    """Save processed data."""
    logger.info(f"Saving processed data to {output_path}")
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(f"{output_path}/X_train.csv", index=False)
    X_test.to_csv(f"{output_path}/X_test.csv", index=False)
    y_train.to_csv(f"{output_path}/y_train.csv", index=False)
    y_test.to_csv(f"{output_path}/y_test.csv", index=False)

def main():
    """Main data processing function."""
    logger.info("Starting data processing")
    
    # Load configuration
    config = load_config()
    
    # Load data
    df = load_data("data/raw/airline_passenger_satisfaction.csv")
    
    # Preprocess data
    df_processed = preprocess_data(df, config)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_processed, config)
    
    # Save processed data
    save_data(X_train, X_test, y_train, y_test, config["data"]["processed_data_path"])
    
    logger.info("Data processing completed")

if __name__ == "__main__":
    main() 