"""
FastAPI REST microservice for real-time and batch credit risk predictions.
"""

from typing import Any, Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.credit_risk import __version__
from src.credit_risk.schemas import (
    BatchLoanApplicationRequest,
    BatchPredictionResponse,
    LoanApplicationSchema,
    PredictionResponseSchema,
)
from src.credit_risk.service import CreditRiskService

app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "Production-grade REST API providing credit risk scoring, default probability estimation, "
        "and approval recommendations based on Lending Club borrower data."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
service = CreditRiskService()


@app.get("/health", tags=["Monitoring"], summary="Service Health & Model Status")
def health_check() -> Dict[str, Any]:
    """Check API operational health and model readiness."""
    return {
        "status": "healthy",
        "service_version": __version__,
        "model_loaded": service.is_ready(),
        "decision_threshold": service.threshold,
    }


@app.post(
    "/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Real-time Single Loan Application Risk Scoring"
)
def predict_loan_risk(application: LoanApplicationSchema) -> PredictionResponseSchema:
    """
    Score a single loan application.

    Validates borrower inputs, cleans features, applies the ML scoring model,
    and returns default probability alongside an approval decision.
    """
    try:
        return service.predict_single(application)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(exc)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Batch Loan Applications Risk Scoring"
)
def predict_batch_loans(request: BatchLoanApplicationRequest) -> BatchPredictionResponse:
    """
    Score a batch collection of loan applications in a single API call.
    """
    try:
        results = service.predict_batch(request.applications)
        return BatchPredictionResponse(
            total_records=len(results),
            predictions=results
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failure: {str(exc)}"
        )


@app.exception_handler(ValueError)
def value_error_handler(request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )
