"""
Unit tests for custom Scikit-Learn transformers.
"""

import pandas as pd
import pytest
from src.credit_risk.transformers import (
    CreditHistoryCalculator,
    EmploymentLengthEncoder,
    GradeOrdinalEncoder,
    TermCleaner,
    VerificationStatusEncoder,
)


def test_term_cleaner():
    cleaner = TermCleaner()
    df = pd.DataFrame({'term': [' 36 months', '60 months ', '36', None, 'invalid']})
    transformed = cleaner.fit_transform(df)

    assert transformed.iloc[0, 0] == 36
    assert transformed.iloc[1, 0] == 60
    assert transformed.iloc[2, 0] == 36
    assert transformed.iloc[3, 0] == 36  # Default fallback


def test_credit_history_calculator():
    calc = CreditHistoryCalculator()
    df = pd.DataFrame({
        'earliest_cr_line': ['Jan-00', 'Dec-98', 'May-10'],
        'issue_d': ['Jan-15', 'Dec-14', 'May-15']
    })
    transformed = calc.fit_transform(df)

    assert len(transformed) == 3
    # Jan-00 to Jan-15 is ~15 years
    assert 14.5 <= transformed.iloc[0, 0] <= 15.5
    # Dec-98 (1998) to Dec-14 (2014) is ~16 years
    assert 15.5 <= transformed.iloc[1, 0] <= 16.5


def test_employment_length_encoder():
    encoder = EmploymentLengthEncoder()
    df = pd.DataFrame({'emp_length': ['< 1 year', '1 year', '5 years', '10+ years', 'unknown']})
    transformed = encoder.fit_transform(df)

    assert transformed.iloc[0, 0] == 0
    assert transformed.iloc[1, 0] == 1
    assert transformed.iloc[2, 0] == 5
    assert transformed.iloc[3, 0] == 10
    assert transformed.iloc[4, 0] == 5  # Default fallback for unknown


def test_verification_status_encoder():
    encoder = VerificationStatusEncoder()
    df = pd.DataFrame({'verification_status': ['Not Verified', 'Source Verified', 'Verified', 'Other']})
    transformed = encoder.fit_transform(df)

    assert transformed.iloc[0, 0] == 0
    assert transformed.iloc[1, 0] == 1
    assert transformed.iloc[2, 0] == 2
    assert transformed.iloc[3, 0] == 0  # Fallback


def test_grade_ordinal_encoder():
    encoder = GradeOrdinalEncoder()
    df = pd.DataFrame({
        'grade': ['A', 'B', 'G', 'Z'],
        'sub_grade': ['A1', 'B3', 'G5', 'Z9']
    })
    transformed = encoder.fit_transform(df)

    assert transformed.loc[0, 'grade'] == 0
    assert transformed.loc[1, 'grade'] == 1
    assert transformed.loc[2, 'grade'] == 6
    assert transformed.loc[0, 'sub_grade'] == 0
    assert transformed.loc[2, 'sub_grade'] == 34
