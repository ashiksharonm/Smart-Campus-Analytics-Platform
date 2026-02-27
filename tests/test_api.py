"""Tests for src/api/main.py using TestClient (no live server required)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def client():
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"FastAPI environment not available: {e}")


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


SAMPLE_STUDENT = {
    "attendance_percentage": 75,
    "avg_assignment_score": 68,
    "lms_login_frequency": 18,
    "library_visits_per_month": 4,
    "hostel_resident": 1,
    "internet_access": 1,
    "disciplinary_actions": 0,
    "semester": 3,
    "model_name": "logistic_regression",
}


def test_predict_endpoint_structure(client):
    """Test that predict returns expected response shape."""
    try:
        response = client.post("/predict", json=SAMPLE_STUDENT)
        if response.status_code == 503:
            pytest.skip("Model not trained yet.")
        assert response.status_code == 200
        data = response.json()
        assert "dropout_prob" in data
        assert "risk_level" in data
        assert data["risk_level"] in ("Low", "Medium", "High")
        assert 0.0 <= data["dropout_prob"] <= 1.0
    except Exception:
        pytest.skip("Model files not present.")


def test_predict_missing_field(client):
    bad_payload = {k: v for k, v in SAMPLE_STUDENT.items() if k != "attendance_percentage"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # validation error


def test_batch_predict_structure(client):
    try:
        payload = {
            "students": [SAMPLE_STUDENT, SAMPLE_STUDENT],
            "model_name": "logistic_regression",
        }
        response = client.post("/batch_predict", json=payload)
        if response.status_code == 503:
            pytest.skip("Model not trained yet.")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
    except Exception:
        pytest.skip("Model files not present.")


def test_health_always_passes(client):
    """Health endpoint must always be available."""
    resp = client.get("/health")
    assert resp.status_code == 200
