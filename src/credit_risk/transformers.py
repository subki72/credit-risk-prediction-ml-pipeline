"""
Custom Scikit-Learn transformers for robust, leak-free feature engineering in Credit Risk.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TermCleaner(BaseEstimator, TransformerMixin):
    """Transformer converting loan term strings ('36 months', '60 months') to numeric integers."""

    def __init__(self, default_term: int = 36):
        self.default_term = default_term

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        col = X_df.columns[0]
        cleaned = X_df[col].astype(str).str.strip().str.replace(' months', '', regex=False)
        numeric_series = pd.to_numeric(cleaned, errors='coerce').fillna(self.default_term).astype(int)
        return numeric_series.to_frame()


class CreditHistoryCalculator(BaseEstimator, TransformerMixin):
    """Transformer calculating credit history duration in years from earliest_cr_line and issue_d."""

    def __init__(self):
        self.median_history_: float = 15.0

    def fit(self, X, y=None):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if 'earliest_cr_line' in X_df.columns and 'issue_d' in X_df.columns:
            calculated = self._calculate_history(X_df)
            valid = calculated.dropna()
            if not valid.empty:
                self.median_history_ = float(valid.median())
        return self

    def _calculate_history(self, X_df: pd.DataFrame) -> pd.Series:
        current_year = pd.Timestamp.now().year
        cr_dt = pd.to_datetime(X_df['earliest_cr_line'], format='%b-%y', errors='coerce')
        issue_dt = pd.to_datetime(X_df['issue_d'], format='%b-%y', errors='coerce')

        # Dynamic century correction
        mask_cr = cr_dt.dt.year > current_year
        cr_dt.loc[mask_cr] -= pd.DateOffset(years=100)

        mask_issue = issue_dt.dt.year > current_year
        issue_dt.loc[mask_issue] -= pd.DateOffset(years=100)

        years = ((issue_dt - cr_dt).dt.days / 365.25).round(1)
        return years

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if 'earliest_cr_line' in X_df.columns and 'issue_d' in X_df.columns:
            years = self._calculate_history(X_df).fillna(self.median_history_)
        else:
            years = pd.Series([self.median_history_] * len(X_df))
        return years.to_frame(name='credit_history_years')


class EmploymentLengthEncoder(BaseEstimator, TransformerMixin):
    """Transformer encoding employment length strings to ordinal integers 0-10."""

    MAPPING = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10
    }

    def __init__(self, default_length: int = 5):
        self.default_length = default_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        col = X_df.columns[0]
        mapped = X_df[col].astype(str).map(self.MAPPING).fillna(self.default_length).astype(int)
        return mapped.to_frame()


class VerificationStatusEncoder(BaseEstimator, TransformerMixin):
    """Transformer encoding verification status strings to ordinal values (0, 1, 2)."""

    MAPPING = {
        'Not Verified': 0,
        'Source Verified': 1,
        'Verified': 2
    }

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        col = X_df.columns[0]
        mapped = X_df[col].astype(str).map(self.MAPPING).fillna(0).astype(int)
        return mapped.to_frame()


class GradeOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Transformer mapping Lending Club grade (A-G) and sub_grade (A1-G5) deterministically."""

    def __init__(self):
        self.grade_map = {g: i for i, g in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'])}
        subgrades = [f"{g}{n}" for g in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] for n in range(1, 6)]
        self.sub_grade_map = {sg: i for i, sg in enumerate(subgrades)}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        result = pd.DataFrame(index=X_df.index)
        if 'grade' in X_df.columns:
            result['grade'] = X_df['grade'].astype(str).map(self.grade_map).fillna(3).astype(int)
        if 'sub_grade' in X_df.columns:
            result['sub_grade'] = X_df['sub_grade'].astype(str).map(self.sub_grade_map).fillna(15).astype(int)
        return result
