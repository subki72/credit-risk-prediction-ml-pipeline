"""
Integration tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from src.credit_risk.api import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service_version" in data
    assert "decision_threshold" in data


def test_predict_single_endpoint_success():
    payload = {
        "loan_id": "API-TEST-001",
        "loan_amnt": 12000.0,
        "term": "36 months",
        "int_rate": 8.9,
        "installment": 381.0,
        "grade": "A",
        "sub_grade": "A2",
        "emp_length": "4 years",
        "home_ownership": "OWN",
        "annual_inc": 95000.0,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "TX",
        "dti": 11.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["loan_id"] == "API-TEST-001"
    assert "default_probability" in data
    assert data["decision"] in ["APPROVED", "REJECTED"]
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "recommendation" in data


def test_predict_single_endpoint_invalid_payload_returns_422():
    payload = {
        "loan_amnt": -5000.0,  # Negative loan amount violates gt=0
        "term": "36 months",
        "grade": "INVALID_GRADE"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_endpoint_success():
    sample = {
        "loan_id": "BATCH-01",
        "loan_amnt": 8000.0,
        "term": "36 months",
        "int_rate": 14.5,
        "installment": 275.0,
        "grade": "C",
        "sub_grade": "C3",
        "emp_length": "1 year",
        "home_ownership": "RENT",
        "annual_inc": 45000.0,
        "verification_status": "Not Verified",
        "purpose": "credit_card",
        "addr_state": "IL",
        "dti": 22.0,
    }
    batch_payload = {"applications": [sample, sample]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
