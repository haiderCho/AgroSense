import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch
from backend.api.main import app

@pytest.fixture
async def client():
    # Mock the predictor to avoid loading real models/files
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {
        "consensus_crop": "Rice",
        "predictions": [
            {
                "model": "RandomForest", 
                "crop": "Rice", 
                "confidence": 0.95, 
                "explanation": {"N": 0.5}
            }
        ]
    }
    mock_predictor.models = {"mock_model": "mock_obj"}
    
    # We patch the class so when startup_event calls MultiModelPredictor(), it gets our mock
    # We also need to patch the global 'predictor' if the app has already started?
    # Actually, TestClient triggers startup. 
    # But we want to ensure startup doesn't crash on FileNotFoundError.
    
    transport = ASGITransport(app=app)
    with patch("backend.api.main.MultiModelPredictor", return_value=mock_predictor):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
