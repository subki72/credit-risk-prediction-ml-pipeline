# Operational Runbook — Credit Risk Prediction ML Pipeline

## 1. Service Overview
- **REST Inference API**: `uvicorn src.credit_risk.api:app --host 0.0.0.0 --port 8000`
- **Health Endpoint**: `GET /health` returns service version, `model_loaded` flag, and active `decision_threshold`.
- **Batch Pipeline**: `python credit_risk_prediction.py` trains, tunes XGBoost, runs Decimal financial impact simulation (`src/credit_risk/reporting.py`), and exports SHA-256 verified artifacts.

## 2. Incident Response
| Symptom | Probable Cause | Remediation |
|---|---|---|
| `ValueError: SHA-256 integrity check failed` | Corrupted or tampered `.pkl` artifact | Quarantine file, re-export signed artifact from training pipeline or immutable artifact registry. |
| HTTP `422 Unprocessable Entity` spike | Upstream caller sending negative `loan_amnt` or invalid credit grade | Inspect Pydantic validation detail in logs and coordinate payload schema fix with upstream client. |
| `model_loaded: false` on `/health` | Missing `models/model_final.pkl` mount | Verify container volume mount and artifact path environment variables. |

## 3. Rollback Procedure
Revert container deployment to pinned release tag `v1.0.0` and restore previous SHA-256 verified `models/model_final.pkl` + `models/scaler_final.pkl`.
