from setuptools import find_packages, setup

setup(
    name="airline_satisfaction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "fastapi>=0.68.1",
        "uvicorn>=0.15.0",
        "prometheus-client>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "black>=21.9b0",
            "flake8>=3.9.2",
        ],
    },
) 