"""
Configuration settings, feature contracts, and financial parameters for Credit Risk Prediction.
"""

from decimal import Decimal
import os
from typing import List

# ==============================================================================
# Path Configurations
# ==============================================================================
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
IMAGES_DIR: str = os.path.join(BASE_DIR, "images")

RAW_DATA_PATH: str = os.path.join(DATA_DIR, "loan_data_2007_2014.csv")
PREPROCESSED_DATA_PATH: str = os.path.join(DATA_DIR, "loan_data_preprocessed.csv")

MODEL_FINAL_PATH: str = os.path.join(MODELS_DIR, "model_final.pkl")
SCALER_FINAL_PATH: str = os.path.join(MODELS_DIR, "scaler_final.pkl")
PIPELINE_FINAL_PATH: str = os.path.join(MODELS_DIR, "pipeline_final.joblib")

# ==============================================================================
# Operational & Reproducibility Parameters
# ==============================================================================
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
OPTIMAL_DECISION_THRESHOLD: float = 0.35

# ==============================================================================
# High-Precision Financial Business Parameters (Basel II/III utility calculation)
# ==============================================================================
DEFAULT_LOSS_RATE: Decimal = Decimal("0.60")       # 60% loss severity upon loan default
OPPORTUNITY_COST_RATE: Decimal = Decimal("0.15")   # 15% missed profit margin per false rejection

# ==============================================================================
# Feature Definitions & Contracts
# ==============================================================================
TARGET_COLUMN: str = "loan_status"

TARGET_MAPPING: dict = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1
}

# Free-text and identifier columns to drop
IDENTIFIER_COLUMNS: List[str] = [
    "id", "member_id", "url", "desc", "emp_title", "title", "zip_code"
]

# Post-loan information leakage columns to drop
LEAKAGE_COLUMNS: List[str] = [
    "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "pymnt_plan"
]

# Numeric continuous features expected in raw input
NUMERIC_FEATURES: List[str] = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "installment",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc"
]

# Ordinal features requiring specific integer encodings
ORDINAL_FEATURES: List[str] = [
    "grade", "sub_grade", "emp_length", "verification_status"
]

# Categorical nominal features requiring One-Hot Encoding
NOMINAL_CATEGORICAL_FEATURES: List[str] = [
    "home_ownership", "purpose", "initial_list_status", "addr_state"
]
