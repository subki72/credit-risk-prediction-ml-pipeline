"""
Threshold optimization, visualization, and financial business impact reporting
for the Credit Risk Prediction ML Pipeline.
"""

from decimal import Decimal
import logging
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.credit_risk.config import (
    DEFAULT_LOSS_RATE,
    IMAGES_DIR,
    MODELS_DIR,
    OPPORTUNITY_COST_RATE,
)
from src.credit_risk.service import save_artifact_with_checksum

logger = logging.getLogger(__name__)


def optimize_threshold_and_business_impact(
    best_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_loan_amounts: pd.Series,
) -> float:
    """Analyze precision-recall curves to find optimal decision threshold and simulate business impact."""
    logger.info("Starting Stage 7: Optimal Threshold Analysis & Financial Business Impact")

    y_prob = best_model.predict_proba(X_test)[:, 1]
    threshold_range = np.arange(0.10, 0.90, 0.05)
    tradeoff_rows = []

    for thr in threshold_range:
        y_pred_thr = (y_prob >= thr).astype(int)
        tradeoff_rows.append({
            "Threshold": round(thr, 2),
            "Precision": precision_score(y_test, y_pred_thr, zero_division=0),
            "Recall": recall_score(y_test, y_pred_thr, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred_thr, zero_division=0),
            "Bad Detected": int(y_pred_thr[y_test == 1].sum()),
            "Good Rejected": int(y_pred_thr[y_test == 0].sum()),
        })

    thr_df = pd.DataFrame(tradeoff_rows)
    best_f1_idx = thr_df["F1-Score"].idxmax()
    optimal_threshold = float(thr_df.loc[best_f1_idx, "Threshold"])
    logger.info("Optimal Decision Threshold (Maximized F1): %.2f", optimal_threshold)

    # Threshold Trade-off Curve Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thr_df["Threshold"], thr_df["Precision"], "b-o", label="Precision", markersize=4)
    ax.plot(thr_df["Threshold"], thr_df["Recall"], "r-s", label="Recall", markersize=4)
    ax.plot(thr_df["Threshold"], thr_df["F1-Score"], "g-^", label="F1-Score", linewidth=2, markersize=5)
    ax.axvline(
        optimal_threshold,
        color="purple",
        linestyle="--",
        linewidth=2,
        label=f"Optimal Threshold = {optimal_threshold:.2f}",
    )
    ax.set_xlabel("Probability Threshold", fontsize=13)
    ax.set_ylabel("Metric Score", fontsize=13)
    ax.set_title("Precision-Recall Trade-off per Threshold — Tuned XGBoost", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.10, 0.85])
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/16_threshold_analysis.png")
    plt.close()
    logger.info("Saved threshold analysis plot to %s/16_threshold_analysis.png", IMAGES_DIR)

    # Confusion matrix at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_optimal,
        annot=True,
        fmt=",d",
        cmap="YlOrRd",
        ax=ax,
        xticklabels=["Good Loan", "Bad Loan"],
        yticklabels=["Good Loan", "Bad Loan"],
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — Optimal Threshold ({optimal_threshold:.2f})\n"
        f"Recall={recall_score(y_test, y_pred_optimal):.3f} | "
        f"Precision={precision_score(y_test, y_pred_optimal):.3f}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/17_cm_optimal_threshold.png")
    plt.close()
    logger.info("Saved confusion matrix plot to %s/17_cm_optimal_threshold.png", IMAGES_DIR)

    # High-Precision Financial Impact Simulation using Decimal
    avg_loan_val = float(raw_loan_amounts.mean()) if not raw_loan_amounts.empty else 15000.0
    avg_loan_dec = Decimal(str(round(avg_loan_val, 2)))
    tn, fp, fn, tp = cm_optimal.ravel()

    loss_per_bad_loan = avg_loan_dec * DEFAULT_LOSS_RATE
    opportunity_cost_per_good_reject = avg_loan_dec * OPPORTUNITY_COST_RATE

    loss_prevented = Decimal(int(tp)) * loss_per_bad_loan
    missed_opportunity = Decimal(int(fp)) * opportunity_cost_per_good_reject
    net_financial_saving = loss_prevented - missed_opportunity

    logger.info(
        "Financial impact summary | avg_loan=$%s | tp=%d | fn=%d | fp=%d | tn=%d | net_benefit=$%s",
        f"{avg_loan_dec:,.2f}",
        tp,
        fn,
        fp,
        tn,
        f"{net_financial_saving:,.2f}",
    )

    # Export Final Production Artifacts with SHA-256 Checksums
    save_artifact_with_checksum(best_model, f"{MODELS_DIR}/model_final.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    save_artifact_with_checksum(scaler, f"{MODELS_DIR}/scaler_final.pkl")
    logger.info("Final production artifacts exported with SHA-256 checksums to %s", MODELS_DIR)
    return optimal_threshold
