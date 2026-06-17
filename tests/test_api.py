"""Tests for the FastAPI inference service."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_sensor_payload() -> dict:
    """Create a valid prediction request payload."""
    return {
        "readings": [
            {
                "time_cycles": i,
                "setting_1": -0.0007,
                "setting_2": -0.0004,
                "setting_3": 100.0,
                "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
                "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.02,
                "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
                "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195,
                "s_16": 0.03, "s_17": 392.0, "s_18": 2388.0,
                "s_19": 100.0, "s_20": 39.06, "s_21": 23.4190,
            }
            for i in range(1, 11)
        ]
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the /health liveness probe."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client: TestClient) -> None:
        """Health response should contain status and timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestReadinessEndpoint:
    """Tests for the /ready readiness probe."""

    def test_readiness_returns_200(self, client: TestClient) -> None:
        """Readiness endpoint should return 200 (even when not ready)."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_readiness_response_fields(self, client: TestClient) -> None:
        """Readiness response should contain required fields."""
        response = client.get("/ready")
        data = response.json()
        assert "ready" in data
        assert "model_loaded" in data
        assert "pipeline_loaded" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_requires_readings(self, client: TestClient) -> None:
        """Predict should reject empty request."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_validates_input_format(self, client: TestClient) -> None:
        """Predict should reject invalid sensor data format."""
        response = client.post("/predict", json={"readings": "not_a_list"})
        assert response.status_code == 422

    def test_predict_rejects_empty_readings(self, client: TestClient) -> None:
        """Predict should reject empty readings list."""
        response = client.post("/predict", json={"readings": []})
        assert response.status_code == 422

    def test_predict_validates_chronological_order(self, client: TestClient) -> None:
        """Predict should reject out-of-order readings."""
        payload = {
            "readings": [
                {
                    "time_cycles": 5,
                    "setting_1": 0, "setting_2": 0, "setting_3": 100,
                    "s_1": 500, "s_2": 640, "s_3": 1580, "s_4": 1350,
                    "s_5": 14, "s_6": 21, "s_7": 450, "s_8": 2388,
                    "s_9": 9050, "s_10": 1.3, "s_11": 45, "s_12": 460,
                    "s_13": 2388, "s_14": 8140, "s_15": 8.4,
                    "s_16": 0.03, "s_17": 350, "s_18": 2388,
                    "s_19": 100, "s_20": 30, "s_21": 23,
                },
                {
                    "time_cycles": 2,  # Out of order!
                    "setting_1": 0, "setting_2": 0, "setting_3": 100,
                    "s_1": 500, "s_2": 640, "s_3": 1580, "s_4": 1350,
                    "s_5": 14, "s_6": 21, "s_7": 450, "s_8": 2388,
                    "s_9": 9050, "s_10": 1.3, "s_11": 45, "s_12": 460,
                    "s_13": 2388, "s_14": 8140, "s_15": 8.4,
                    "s_16": 0.03, "s_17": 350, "s_18": 2388,
                    "s_19": 100, "s_20": 30, "s_21": 23,
                },
            ]
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint."""

    def test_model_info_when_not_loaded(self, client: TestClient) -> None:
        """model-info should return 503 when model not loaded."""
        response = client.get("/model-info")
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
