import pytest
from fastapi.testclient import TestClient
from src.api.main import app, startup_event
import asyncio

@pytest.fixture(scope="session")
def test_app():
    """Fixture that handles startup events for FastAPI app"""
    # Run the startup event
    asyncio.run(startup_event())
    # Create test client
    client = TestClient(app)
    return client 