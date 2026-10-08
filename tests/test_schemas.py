"""
Unit tests for Pydantic loan application schemas.
"""

import pytest
from pydantic import ValidationError
from src.credit_risk.schemas import LoanApplicationSchema, PredictionResponseSchema


def get_sample_valid_payload():
    return {
        "loan_id": "TEST-001",
        "loan_amnt": 10000.0,
        "term": "36 months",
        "int_rate": 10.5,
        "installment": 325.0,
        "grade": "B",
        "sub_grade": "B2",
        "emp_length": "3 years",
        "home_ownership": "RENT",
        "annual_inc": 60000.0,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "dti": 15.0,
    }


def test_valid_loan_application():
    payload = get_sample_valid_payload()
    app = LoanApplicationSchema(**payload)
    assert app.loan_amnt == 10000.0
    assert app.grade == "B"
    assert app.addr_state == "CA"


def test_invalid_grade_raises_validation_error():
    payload = get_sample_valid_payload()
    payload["grade"] = "X"  # Invalid grade
    with pytest.raises(ValidationError) as exc_info:
        LoanApplicationSchema(**payload)
    assert "Invalid grade" in str(exc_info.value)


def test_invalid_state_code_raises_validation_error():
    payload = get_sample_valid_payload()
    payload["addr_state"] = "CALIFORNIA"  # Must be 2-letter
    with pytest.raises(ValidationError) as exc_info:
        LoanApplicationSchema(**payload)
    assert "Invalid state code" in str(exc_info.value)


def test_negative_loan_amount_raises_validation_error():
    payload = get_sample_valid_payload()
    payload["loan_amnt"] = -500.0
    with pytest.raises(ValidationError):
        LoanApplicationSchema(**payload)


def test_prediction_response_schema():
    resp = PredictionResponseSchema(
        loan_id="TEST-001",
        default_probability=0.15,
        decision="APPROVED",
        risk_tier="LOW",
        threshold_applied=0.35,
        recommendation="Prime credit profile."
    )
    assert resp.decision == "APPROVED"
    assert resp.risk_tier == "LOW"
    assert resp.default_probability == 0.15
