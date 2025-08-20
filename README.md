# Airline Passenger Satisfaction Analysis - MLOps Project

This project implements MLOps practices for analyzing and predicting airline passenger satisfaction using machine learning.

## Project Structure
```
.
├── data/               # Data files (managed by DVC)
├── src/               # Source code
│   ├── data/         # Data processing scripts
│   ├── models/       # Model training and evaluation
│   ├── api/          # FastAPI application
│   └── utils/        # Utility functions
├── tests/            # Test files
├── config/           # Configuration files
├── notebooks/        # Jupyter notebooks
└── mlruns/          # MLflow tracking
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/jenilsoni-ai/MLOps-Airline-Passenger-Satisfaction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC:
```bash
dvc init
dvc add data/
```

5. Set up pre-commit hooks:
```bash
pre-commit install
```

## Usage

1. Data Processing:
```bash
python src/data/make_dataset.py
```

2. Model Training:
```bash
python src/models/train_model.py
```

3. Run API:
```bash
uvicorn src.api.main:app --reload
```

## MLflow Tracking

Access MLflow UI:
```bash
mlflow ui
```

## Testing

Run tests:
```bash
pytest tests/
```

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.
