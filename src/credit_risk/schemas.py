"""
Pydantic data validation schemas for Loan Applications and Inference Responses.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class LoanApplicationSchema(BaseModel):
    """Schema validating raw incoming borrower loan application data."""

    loan_id: Optional[str] = Field(None, description="Optional external loan identifier")
    loan_amnt: float = Field(..., gt=0, le=100_000, description="Requested loan amount in USD")
    term: str = Field(..., description="Loan term (e.g., '36 months', '60 months', or '36')")
    int_rate: float = Field(..., ge=0.0, le=45.0, description="Interest rate percentage (e.g., 12.5 or 12.5%)")
    installment: float = Field(..., gt=0, description="Monthly payment installment")
    grade: str = Field(..., description="Lending Club assigned loan grade (A to G)")
    sub_grade: str = Field(..., description="Loan sub-grade (A1 to G5)")
    emp_length: Optional[str] = Field("1 year", description="Employment duration (e.g., '< 1 year', '10+ years')")
    home_ownership: str = Field(..., description="Home ownership status (RENT, OWN, MORTGAGE, OTHER)")
    annual_inc: float = Field(..., ge=0, description="Self-reported annual income")
    verification_status: str = Field("Not Verified", description="Income verification status")
    purpose: str = Field(..., description="Loan purpose category (e.g., debt_consolidation, credit_card)")
    addr_state: str = Field(..., description="Borrower residence state code (e.g., CA, NY, TX)")
    dti: float = Field(..., ge=0, le=100.0, description="Debt-to-income ratio percentage")
    delinq_2yrs: Optional[float] = Field(0.0, ge=0, description="Delinquencies in the past 2 years")
    earliest_cr_line: Optional[str] = Field("Jan-00", description="Month the borrower's earliest credit line was opened")
    inq_last_6mths: Optional[float] = Field(0.0, ge=0, description="Inquiries in the past 6 months")
    open_acc: Optional[float] = Field(1.0, ge=0, description="Number of open credit lines")
    pub_rec: Optional[float] = Field(0.0, ge=0, description="Number of derogatory public records")
    revol_bal: Optional[float] = Field(0.0, ge=0, description="Total credit revolving balance")
    revol_util: Optional[float] = Field(0.0, ge=0, le=150.0, description="Revolving line utilization rate (%)")
    total_acc: Optional[float] = Field(1.0, ge=0, description="Total number of credit lines")
    initial_list_status: Optional[str] = Field("f", description="Initial listing status ('f' or 'w')")
    issue_d: Optional[str] = Field("Jan-15", description="Loan issue date")

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, v: str) -> str:
        clean_v = v.strip().upper()
        if clean_v not in {"A", "B", "C", "D", "E", "F", "G"}:
            raise ValueError(f"Invalid grade '{v}'. Must be one of A, B, C, D, E, F, G.")
        return clean_v

    @field_validator("addr_state")
    @classmethod
    def validate_state(cls, v: str) -> str:
        clean_v = v.strip().upper()
        if len(clean_v) != 2:
            raise ValueError(f"Invalid state code '{v}'. Must be a 2-letter uppercase state code.")
        return clean_v

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_id": "LN-2026-001",
                "loan_amnt": 15000.0,
                "term": "36 months",
                "int_rate": 11.99,
                "installment": 498.25,
                "grade": "B",
                "sub_grade": "B3",
                "emp_length": "5 years",
                "home_ownership": "MORTGAGE",
                "annual_inc": 75000.0,
                "verification_status": "Source Verified",
                "purpose": "debt_consolidation",
                "addr_state": "CA",
                "dti": 16.5,
                "delinq_2yrs": 0.0,
                "earliest_cr_line": "May-02",
                "inq_last_6mths": 1.0,
                "open_acc": 8.0,
                "pub_rec": 0.0,
                "revol_bal": 12500.0,
                "revol_util": 45.2,
                "total_acc": 18.0,
                "initial_list_status": "f",
                "issue_d": "Mar-14"
            }
        }
    }


class PredictionResponseSchema(BaseModel):
    """Schema for individual credit risk inference response."""

    loan_id: Optional[str] = None
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of default")
    decision: str = Field(..., description="'APPROVED' (low risk) or 'REJECTED' (high risk)")
    risk_tier: str = Field(..., description="'LOW', 'MEDIUM', or 'HIGH'")
    threshold_applied: float = Field(..., description="Decision threshold applied")
    recommendation: str = Field(..., description="Actionable credit officer recommendation")


class BatchLoanApplicationRequest(BaseModel):
    """Schema for batch loan risk scoring request."""

    applications: List[LoanApplicationSchema]


class BatchPredictionResponse(BaseModel):
    """Schema for batch loan risk scoring response."""

    total_records: int
    predictions: List[PredictionResponseSchema]
