"""
Scikit-Learn Pipeline factory for end-to-end preprocessing and model inference.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.credit_risk.config import (
    NOMINAL_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from src.credit_risk.transformers import (
    CreditHistoryCalculator,
    EmploymentLengthEncoder,
    GradeOrdinalEncoder,
    TermCleaner,
    VerificationStatusEncoder,
)


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build unified ColumnTransformer preprocessing pipeline that handles raw inputs
    without data leakage and safely handles unseen categories.
    """
    # 1. Numeric pipeline: Median imputation -> Scaling
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical nominal pipeline: Imputation -> OHE (handle_unknown='ignore') -> Scaling
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', StandardScaler(with_mean=False))
    ])

    # 3. Term feature pipeline
    term_pipe = Pipeline([
        ('cleaner', TermCleaner(default_term=36)),
        ('scaler', StandardScaler())
    ])

    # 4. Credit history pipeline from earliest_cr_line and issue_d
    credit_hist_pipe = Pipeline([
        ('calculator', CreditHistoryCalculator()),
        ('scaler', StandardScaler())
    ])

    # 5. Employment length pipeline
    emp_length_pipe = Pipeline([
        ('encoder', EmploymentLengthEncoder(default_length=5)),
        ('scaler', StandardScaler())
    ])

    # 6. Verification status pipeline
    verif_pipe = Pipeline([
        ('encoder', VerificationStatusEncoder()),
        ('scaler', StandardScaler())
    ])

    # 7. Grade and Sub-grade ordinal pipeline
    grade_pipe = Pipeline([
        ('encoder', GradeOrdinalEncoder()),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipe, NUMERIC_FEATURES),
            ('categorical', categorical_pipe, NOMINAL_CATEGORICAL_FEATURES),
            ('term', term_pipe, ['term']),
            ('credit_history', credit_hist_pipe, ['earliest_cr_line', 'issue_d']),
            ('emp_length', emp_length_pipe, ['emp_length']),
            ('verification_status', verif_pipe, ['verification_status']),
            ('grade', grade_pipe, ['grade', 'sub_grade']),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor


def build_full_pipeline(model) -> Pipeline:
    """Combine preprocessing pipeline and trained classification estimator."""
    preprocessor = build_preprocessing_pipeline()
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
