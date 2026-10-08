"""
Integration tests for the preprocessing and inference pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.credit_risk.pipeline import build_full_pipeline, build_preprocessing_pipeline


def create_sample_raw_dataframe(n_samples: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "loan_amnt": np.random.uniform(5000, 35000, n_samples),
        "term": np.random.choice(["36 months", "60 months"], n_samples),
        "int_rate": np.random.uniform(6.0, 25.0, n_samples),
        "installment": np.random.uniform(150, 900, n_samples),
        "grade": np.random.choice(["A", "B", "C", "D"], n_samples),
        "sub_grade": np.random.choice(["A1", "B2", "C3", "D4"], n_samples),
        "emp_length": np.random.choice(["< 1 year", "3 years", "10+ years"], n_samples),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n_samples),
        "annual_inc": np.random.uniform(30000, 120000, n_samples),
        "verification_status": np.random.choice(["Not Verified", "Verified"], n_samples),
        "purpose": np.random.choice(["debt_consolidation", "credit_card"], n_samples),
        "addr_state": np.random.choice(["CA", "NY", "TX", "FL"], n_samples),
        "dti": np.random.uniform(5.0, 35.0, n_samples),
        "delinq_2yrs": np.zeros(n_samples),
        "earliest_cr_line": ["Jan-02"] * n_samples,
        "inq_last_6mths": np.ones(n_samples),
        "open_acc": np.full(n_samples, 8.0),
        "pub_rec": np.zeros(n_samples),
        "revol_bal": np.random.uniform(1000, 20000, n_samples),
        "revol_util": np.random.uniform(10.0, 80.0, n_samples),
        "total_acc": np.full(n_samples, 15.0),
        "initial_list_status": ["f"] * n_samples,
        "issue_d": ["Jan-15"] * n_samples,
    })


def test_preprocessing_pipeline_fit_transform():
    df_train = create_sample_raw_dataframe(30)
    preprocessor = build_preprocessing_pipeline()

    # Fit and transform
    X_transformed = preprocessor.fit_transform(df_train)
    assert X_transformed.shape[0] == 30
    assert X_transformed.shape[1] > 20  # Expanded after OneHotEncoding


def test_pipeline_handles_unseen_categories_gracefully():
    df_train = create_sample_raw_dataframe(30)
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(df_train)

    # Test with unseen state and unknown category
    df_test_unseen = create_sample_raw_dataframe(5)
    df_test_unseen.loc[0, "addr_state"] = "ZZ"  # Unknown state
    df_test_unseen.loc[0, "purpose"] = "unseen_purpose"
    df_test_unseen.loc[0, "home_ownership"] = "UNSEEN_HOME"

    # Should transform without crashing due to handle_unknown='ignore'
    X_test_transformed = preprocessor.transform(df_test_unseen)
    assert X_test_transformed.shape[0] == 5
    assert X_test_transformed.shape[1] == preprocessor.transform(df_train).shape[1]


def test_full_pipeline_end_to_end():
    df_train = create_sample_raw_dataframe(30)
    y_train = np.random.choice([0, 1], 30)

    clf = LogisticRegression()
    pipeline = build_full_pipeline(clf)

    pipeline.fit(df_train, y_train)

    df_new = create_sample_raw_dataframe(3)
    preds = pipeline.predict(df_new)
    probs = pipeline.predict_proba(df_new)

    assert len(preds) == 3
    assert probs.shape == (3, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
