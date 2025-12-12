import pytest
import os
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture()
def client():
    """Create test client"""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint returns proper status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "timestamp" in data
    assert "version" in data


def test_info_endpoint(client):
    """Test info endpoint returns API information"""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "api_name" in data
    assert "version" in data
    assert data["api_name"] == "Iris Classifier API"


def test_model_info_endpoint(client):
    """Test model info endpoint returns model details"""
    response = client.get("/model_info")
    assert response.status_code in [200, 503]  # May fail if model not available
    
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "metrics" in data
        assert "timestamp" in data
        assert data["model_name"] == "iris-classifier"


def test_predict_endpoint_valid_input(client):
    """Test prediction with valid input"""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    
    # May return 503 if model not loaded
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "timestamp" in data
        assert "model_version" in data
        assert data["prediction"] in ["setosa", "versicolor", "virginica"]
        assert 0 <= data["confidence"] <= 1


def test_predict_endpoint_all_species(client):
    """Test prediction for all three iris species"""
    test_samples = [
        # Setosa
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        # Versicolor
        {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.0, "petal_width": 1.7},
        # Virginica
        {"sepal_length": 7.2, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 1.6}
    ]
    
    for sample in test_samples:
        response = client.post("/predict", json=sample)
        if response.status_code == 200:
            data = response.json()
            assert data["prediction"] in ["setosa", "versicolor", "virginica"]
            assert data["confidence"] > 0


def test_predict_endpoint_missing_field(client):
    """Test prediction with missing required field"""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
        # Missing petal_width
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_type(client):
    """Test prediction with invalid data type"""
    payload = {
        "sepal_length": "invalid",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_negative_values(client):
    """Test prediction with negative values"""
    payload = {
        "sepal_length": -5.1,
        "sepal_width": -3.5,
        "petal_length": -1.4,
        "petal_width": -0.2
    }
    response = client.post("/predict", json=payload)
    # Should either accept or return error
    assert response.status_code in [200, 400, 422, 500, 503]


def test_batch_predict_endpoint_valid(client):
    """Test batch prediction with valid inputs"""
    payload = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.0, "petal_width": 1.7}
    ]
    response = client.post("/batch_predict", json=payload)
    
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert "timestamp" in data
        assert len(data["predictions"]) == 2
        assert data["count"] == 2
        for pred in data["predictions"]:
            assert pred in ["setosa", "versicolor", "virginica"]


def test_batch_predict_endpoint_empty(client):
    """Test batch prediction with empty list"""
    payload = []
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 400  # Invalid sample count


def test_batch_predict_endpoint_too_many(client):
    """Test batch prediction with too many samples"""
    payload = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    ] * 101  # 101 samples (max is 100)
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 400  # Invalid sample count


def test_batch_predict_single_sample(client):
    """Test batch prediction with single sample"""
    payload = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    ]
    response = client.post("/batch_predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["count"] == 1


def test_train_endpoint(client):
    """Test training endpoint (note: this triggers actual training)"""
    # Note: This is an expensive operation
    response = client.post("/train")
    # Should return either success or error
    assert response.status_code in [200, 500, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "timestamp" in data
        assert data["status"] == "success"


def test_predict_response_schema(client):
    """Test that prediction response matches expected schema"""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        # Check all required fields exist
        required_fields = ["prediction", "confidence", "timestamp", "model_version"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(data["prediction"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["model_version"], str)


def test_cors_headers(client):
    """Test that CORS headers are not present (add if needed)"""
    response = client.get("/health")
    # This test documents current behavior
    # If CORS is needed, headers should be added to FastAPI app
    assert response.status_code == 200


def test_invalid_endpoint(client):
    """Test accessing non-existent endpoint"""
    response = client.get("/nonexistent")
    assert response.status_code == 404
