"""
Credit Risk Prediction Package
Enterprise-grade machine learning pipeline for loan default prediction.
"""

__version__ = "1.0.0"
__author__ = "Muhammad Syafii Assubki"

from src.credit_risk.schemas import LoanApplicationSchema, PredictionResponseSchema
from src.credit_risk.service import CreditRiskService

__all__ = [
    "LoanApplicationSchema",
    "PredictionResponseSchema",
    "CreditRiskService",
    "__version__",
]
