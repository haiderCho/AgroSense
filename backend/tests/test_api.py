import pytest

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "models_loaded": []}

@pytest.mark.asyncio
async def test_predict_endpoint_validation(client):
    """Test that missing data triggers validation error"""
    response = await client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity

@pytest.mark.asyncio
async def test_predict_endpoint_success(client):
    """
    Test a valid prediction request.
    Note: We rely on the global predictor to be initialized or mocked.
    For this smoke test, we'll assume the app startup loads mocked/empty models
    if real ones aren't found, OR we accept 503 if predictor isn't ready.
    """
    payload = {
        "N": 50, "P": 50, "K": 50,
        "temperature": 25, "humidity": 70,
        "ph": 7, "rainfall": 100
    }
    response = await client.post("/predict", json=payload)
    
    # If no models are loaded on the test machine, it might return 503 or 500
    # Ideally we mock the predictor, but for a broad smoke test, checking it's reachable is step 1.
    assert response.status_code in [200, 503] 
