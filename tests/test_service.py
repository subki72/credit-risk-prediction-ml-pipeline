"""
Unit tests for inference service and SHA-256 artifact verification.
"""

import os
import tempfile
import pytest
from src.credit_risk.schemas import LoanApplicationSchema
from src.credit_risk.service import (
    CreditRiskService,
    compute_file_sha256,
    save_artifact_with_checksum,
    verify_and_load_artifact,
)


def test_sha256_checksum_and_integrity_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "artifact.pkl")
        data = {"model_name": "xgboost_v1", "params": [1, 2, 3]}

        # Save with checksum
        checksum = save_artifact_with_checksum(data, test_file)
        assert os.path.exists(f"{test_file}.sha256")
        assert len(checksum) == 64

        # Load with verification
        loaded = verify_and_load_artifact(test_file, require_checksum=True)
        assert loaded == data

        # Tamper with file
        with open(test_file, "ab") as f:
            f.write(b"corrupted_bytes")

        # Must raise ValueError
        with pytest.raises(ValueError) as exc:
            verify_and_load_artifact(test_file, require_checksum=True)
        assert "integrity check failed" in str(exc.value)


def test_credit_risk_service_predict_single_and_batch():
    service = CreditRiskService()
    # Test single
    app = LoanApplicationSchema(
        loan_id="TEST-001",
        loan_amnt=10000.0,
        term="36 months",
        int_rate=9.5,
        installment=320.0,
        grade="A",
        sub_grade="A3",
        emp_length="6 years",
        home_ownership="MORTGAGE",
        annual_inc=85000.0,
        verification_status="Not Verified",
        purpose="credit_card",
        addr_state="NY",
        dti=12.0
    )
    result = service.predict_single(app)
    assert result.loan_id == "TEST-001"
    assert 0.0 <= result.default_probability <= 1.0
    assert result.decision in ["APPROVED", "REJECTED"]
    assert result.risk_tier in ["LOW", "MEDIUM", "HIGH"]

    # Test batch
    batch_results = service.predict_batch([app, app])
    assert len(batch_results) == 2
