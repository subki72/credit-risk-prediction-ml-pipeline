"""
Credit Risk Prediction — End-to-End Machine Learning Pipeline
Dataset : loan_data_2007_2014.csv (Lending Club Loan Data)
Project : Credit Risk Prediction ML Pipeline
Author  : Muhammad Syafii Assubki (Production-Grade Architecture)
"""

from decimal import Decimal
import os
import time
import warnings
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.credit_risk.config import (
    CLEAN_DATA_PATH,
    DEFAULT_LOSS_RATE,
    IMAGES_DIR,
    MODELS_DIR,
    OPPORTUNITY_COST_RATE,
    OPTIMAL_DECISION_THRESHOLD,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_SIZE,
)
from src.credit_risk.pipeline import build_full_pipeline
from src.credit_risk.service import save_artifact_with_checksum


def setup_environment() -> None:
    """Configure warning filters, plot styles, and ensure output directories exist."""
    warnings.filterwarnings('ignore')
    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.bbox'] = 'tight'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    for directory in ['data', IMAGES_DIR, MODELS_DIR]:
        os.makedirs(directory, exist_ok=True)


# ==============================================================================
# STAGE 1 — Initial Exploratory Data Analysis & Inspection
# ==============================================================================
def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset and output baseline data properties."""
    print("=" * 70)
    print("STAGE 1 — Initial Exploratory Data Analysis (EDA)")
    print("=" * 70)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please download 'loan_data_2007_2014.csv' and place it in the 'data/' directory."
        )

    print(f"Loading raw dataset from '{filepath}'...")
    df = pd.read_csv(filepath, index_col=0, low_memory=False)

    print(f"\n[Dataset Dimensions] Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
    print("\n--- First 5 Rows ---")
    print(df.head())

    missing_series = df.isnull().sum()
    missing_pct = (missing_series / len(df) * 100).round(2)
    info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Count': missing_series,
        'Missing Percentage (%)': missing_pct
    })
    print("\n--- Missing Values Overview (First 20 Columns) ---")
    print(info_df.head(20).to_string())

    print("\n--- Target Variable Distribution ('loan_status') ---")
    target_counts = df['loan_status'].value_counts()
    target_pct = (df['loan_status'].value_counts(normalize=True) * 100).round(2)
    target_dist = pd.DataFrame({'Count': target_counts, 'Percentage (%)': target_pct})
    print(target_dist.to_string())

    # Identify candidate columns for removal
    cols_all_missing = df.columns[df.isnull().sum() == len(df)].tolist()
    print(f"\n[Audit] Columns with 100% missing values ({len(cols_all_missing)}):")
    print(", ".join(cols_all_missing) if cols_all_missing else "None")

    cols_irrelevant = ['id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code']
    print(f"\n[Audit] Free-text and identifier columns identified for dropping: {cols_irrelevant}")

    cols_single_val = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    print(f"\n[Audit] Zero-variance columns identified: {cols_single_val}")

    print("\nStage 1 EDA completed successfully.\n")
    return df


# ==============================================================================
# STAGE 2 — Data Visualization
# ==============================================================================
def generate_eda_visualizations(df: pd.DataFrame) -> None:
    """Generate exploratory visualization charts and save to images/."""
    print("=" * 70)
    print("STAGE 2 — Exploratory Visualizations")
    print("=" * 70)

    # 1. Target distribution chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    order = df['loan_status'].value_counts().index
    colors = sns.color_palette('Set2', n_colors=len(order))

    sns.countplot(y='loan_status', data=df, order=order, palette=colors, ax=axes[0])
    axes[0].set_title('Loan Status Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Loan Status')
    for i, v in enumerate(df['loan_status'].value_counts()):
        axes[0].text(v + 1000, i, f'{v:,}', va='center', fontsize=9)

    pct = df['loan_status'].value_counts()
    axes[1].pie(
        pct.values,
        labels=pct.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        textprops={'fontsize': 8}
    )
    axes[1].set_title('Loan Status Proportions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/01_target_distribution.png')
    plt.close()
    print("[Plot Saved] 01_target_distribution.png")

    # 2. Key numeric feature histograms
    num_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'installment']
    available_num_cols = [c for c in num_cols if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, col in enumerate(available_num_cols):
        ax = axes_flat[i]
        data = df[col].dropna()
        if col == 'annual_inc':
            data = data[data <= data.quantile(0.99)]
        ax.hist(data, bins=50, color=sns.color_palette('muted')[i], edgecolor='white', alpha=0.85)
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.2, label=f'Mean: {data.mean():,.1f}')
        ax.axvline(data.median(), color='green', linestyle='-.', linewidth=1.2, label=f'Median: {data.median():,.1f}')
        ax.legend(fontsize=8)

    # Hide extra subplot if any
    for j in range(len(available_num_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/02_numeric_histograms.png')
    plt.close()
    print("[Plot Saved] 02_numeric_histograms.png")

    # 3. Correlation heatmap of numeric features
    if len(available_num_cols) > 1:
        corr_matrix = df[available_num_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap (Key Numeric Features)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{IMAGES_DIR}/03_correlation_heatmap.png')
        plt.close()
        print("[Plot Saved] 03_correlation_heatmap.png")

    print("Stage 2 visual charts saved successfully.\n")


# ==============================================================================
# STAGE 3 — Data Cleaning & Feature Engineering
# ==============================================================================
def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean data, binarize target, engineer features, encode categories, and save clean dataset."""
    print("=" * 70)
    print("STAGE 3 — Data Preprocessing & Feature Engineering")
    print("=" * 70)

    shape_initial = df.shape
    print(f"Initial shape: {shape_initial}")

    # 1. Target Binarization (0: Fully Paid / Good Loan, 1: Charged Off / Default / Bad Loan)
    target_mapping = {
        'Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1
    }
    df_clean = df[df['loan_status'].isin(target_mapping.keys())].copy()
    df_clean['loan_status'] = df_clean['loan_status'].map(target_mapping).astype(int)

    bad_loan_pct = df_clean['loan_status'].mean() * 100
    print(f"Filtered rows (Good & Bad loans only): {len(df_clean):,}")
    print(f"Bad Loan proportion: {bad_loan_pct:.2f}%")

    # 2. Drop irrelevant, high-missing, and post-loan leakage columns
    id_cols = ['id', 'member_id', 'url', 'desc', 'title', 'zip_code', 'emp_title']
    df_clean.drop(columns=[c for c in id_cols if c in df_clean.columns], inplace=True)

    missing_pct = df_clean.isnull().sum() / len(df_clean) * 100
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    df_clean.drop(columns=high_missing, inplace=True)

    post_loan_leakage = [
        'funded_amnt', 'funded_amnt_inv', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
        'pymnt_plan'
    ]
    df_clean.drop(columns=[c for c in post_loan_leakage if c in df_clean.columns], inplace=True)

    # Drop zero variance columns
    for c in list(df_clean.columns):
        if df_clean[c].nunique(dropna=True) <= 1:
            df_clean.drop(columns=[c], inplace=True)

    print(f"Columns remaining after dropping irrelevant/leakage features: {df_clean.shape[1]}")

    # 3. Missing Value Imputation
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'loan_status']
    for c in num_cols:
        if df_clean[c].isnull().sum() > 0:
            median_val = df_clean[c].median()
            df_clean[c] = df_clean[c].fillna(median_val)

    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in cat_cols:
        if df_clean[c].isnull().sum() > 0:
            mode_val = df_clean[c].mode()[0]
            df_clean[c] = df_clean[c].fillna(mode_val)

    # 4. Feature Engineering
    # 4a. term -> integer months
    if 'term' in df_clean.columns:
        df_clean['term'] = df_clean['term'].astype(str).str.strip().str.replace(' months', '', regex=False)
        df_clean['term'] = pd.to_numeric(df_clean['term'], errors='coerce').fillna(36).astype(int)

    # 4b. int_rate & revol_util -> float
    for col in ['int_rate', 'revol_util']:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.replace('%', '', regex=False)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)

    # 4c. credit_history_years from earliest_cr_line and issue_d
    if 'earliest_cr_line' in df_clean.columns and 'issue_d' in df_clean.columns:
        df_clean['earliest_cr_line_dt'] = pd.to_datetime(df_clean['earliest_cr_line'], format='%b-%y', errors='coerce')
        df_clean['issue_d_dt'] = pd.to_datetime(df_clean['issue_d'], format='%b-%y', errors='coerce')

        # Dynamic century correction
        current_year = pd.Timestamp.now().year
        mask_cr = df_clean['earliest_cr_line_dt'].dt.year > current_year
        df_clean.loc[mask_cr, 'earliest_cr_line_dt'] -= pd.DateOffset(years=100)

        mask_issue = df_clean['issue_d_dt'].dt.year > current_year
        df_clean.loc[mask_issue, 'issue_d_dt'] -= pd.DateOffset(years=100)

        df_clean['credit_history_years'] = (
            (df_clean['issue_d_dt'] - df_clean['earliest_cr_line_dt']).dt.days / 365.25
        ).round(1)
        df_clean['credit_history_years'] = df_clean['credit_history_years'].fillna(df_clean['credit_history_years'].median())

        df_clean.drop(columns=['earliest_cr_line', 'issue_d', 'earliest_cr_line_dt', 'issue_d_dt'], inplace=True)

    # 4d. emp_length -> numeric 0 to 10
    if 'emp_length' in df_clean.columns:
        emp_mapping = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years': 8, '9 years': 9, '10+ years': 10
        }
        df_clean['emp_length'] = df_clean['emp_length'].map(emp_mapping)
        df_clean['emp_length'] = df_clean['emp_length'].fillna(df_clean['emp_length'].median()).astype(int)

    # 4e. verification_status -> ordinal integer
    if 'verification_status' in df_clean.columns:
        verif_mapping = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
        df_clean['verification_status'] = df_clean['verification_status'].map(verif_mapping).fillna(0).astype(int)

    # 5. Encodings
    # Label encoding for ordinal features
    for col in ['grade', 'sub_grade']:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # One-hot encoding for nominal categorical features
    ohe_candidates = ['home_ownership', 'purpose', 'initial_list_status', 'application_type', 'addr_state']
    ohe_cols = [c for c in ohe_candidates if c in df_clean.columns]
    if ohe_cols:
        df_clean = pd.get_dummies(df_clean, columns=ohe_cols, drop_first=True, dtype=int)

    # Drop any leftover non-numeric columns
    remaining_non_num = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    if remaining_non_num:
        df_clean.drop(columns=remaining_non_num, inplace=True)

    # Save cleaned preprocessed dataset
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned dataset saved: {CLEAN_DATA_PATH}")
    print(f"Final preprocessed dimensions: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")

    X = df_clean.drop(columns=['loan_status'])
    y = df_clean['loan_status']
    return X, y


# ==============================================================================
# STAGE 4 — Train-Test Split, Resampling (SMOTE), & Scaling
# ==============================================================================
def prepare_training_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Execute stratified split, SMOTE oversampling on training data, and fit StandardScaler."""
    print("=" * 70)
    print("STAGE 4 — Stratified Split, SMOTE Oversampling, & Feature Scaling")
    print("=" * 70)

    # Class proportion visualization
    class_counts = y.value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"Class 0 (Good Loan): {class_counts[0]:,} ({class_counts[0]/len(y)*100:.2f}%)")
    print(f"Class 1 (Bad Loan) : {class_counts[1]:,} ({class_counts[1]/len(y)*100:.2f}%)")
    print(f"Imbalance Ratio    : {imbalance_ratio:.2f} : 1")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Good Loan (0)', 'Bad Loan (1)']
    wedges, texts, autotexts = ax.pie(
        class_counts.values,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0, 0.05],
        textprops={'fontsize': 12},
        pctdistance=0.85
    )
    autotexts[1].set_fontweight('bold')
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title('Target Class Proportions (loan_status)', fontsize=14, fontweight='bold')
    ax.text(0, 0, f'Ratio\n{imbalance_ratio:.1f}:1', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/07_target_proportion.png')
    plt.close()
    print("[Plot Saved] 07_target_proportion.png")

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # SMOTE Oversampling on training set only
    print(f"Training set prior to SMOTE: Class 0 = {(y_train == 0).sum():,}, Class 1 = {(y_train == 1).sum():,}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Training set after SMOTE   : Class 0 = {(y_train_res == 0).sum():,}, Class 1 = {(y_train_res == 1).sum():,}")

    # SMOTE Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    before_counts = y_train.value_counts()
    axes[0].bar(['Good Loan (0)', 'Bad Loan (1)'], before_counts.values, color=colors, edgecolor='white')
    axes[0].set_title('Training Set Before SMOTE', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(before_counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=11)

    after_counts = pd.Series(y_train_res).value_counts()
    axes[1].bar(['Good Loan (0)', 'Bad Loan (1)'], after_counts.values, color=colors, edgecolor='white')
    axes[1].set_title('Training Set After SMOTE', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(after_counts.values):
        axes[1].text(i, v + 500, f'{v:,}', ha='center', fontsize=11)

    plt.suptitle('Class Distribution Before & After SMOTE (Train Set)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/08_smote_comparison.png')
    plt.close()
    print("[Plot Saved] 08_smote_comparison.png")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Save baseline scaler with SHA-256 checksum
    save_artifact_with_checksum(scaler, f'{MODELS_DIR}/scaler.pkl')
    print(f"StandardScaler saved to '{MODELS_DIR}/scaler.pkl' with checksum verification.")

    return X_train_scaled, X_test_scaled, y_train_res, y_test


# ==============================================================================
# STAGE 5 — Baseline Model Training & Evaluation
# ==============================================================================
def train_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train 4 classification models, generate confusion matrices, and evaluate metrics."""
    print("=" * 70)
    print("STAGE 5 — Baseline Model Training (4 Algorithms)")
    print("=" * 70)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
    }

    results = {}
    roc_data = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time
        fitted_models[name] = model

        # Evaluate predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Train Time (s)': round(duration, 2)
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

        print(f"Training Time : {duration:.2f}s | ROC-AUC: {auc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Good Loan (0)', 'Bad Loan (1)']))

        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=',d',
            cmap='Blues',
            ax=ax,
            xticklabels=['Good Loan', 'Bad Loan'],
            yticklabels=['Good Loan', 'Bad Loan'],
            annot_kws={'size': 14}
        )
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(
            f'Confusion Matrix — {name}\nAccuracy={acc:.3f} | ROC-AUC={auc:.3f}',
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()

        safe_name = name.lower().replace(' ', '_')
        plt.savefig(f'{IMAGES_DIR}/09_cm_{safe_name}.png')
        plt.close()

        # Save model checkpoint with SHA-256 checksum
        model_save_path = f'{MODELS_DIR}/model_{safe_name}.pkl'
        save_artifact_with_checksum(model, model_save_path)
        print(f"Checkpoint saved: {model_save_path}")

    # Plot ROC curves comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_roc = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, color=colors_roc[i], linewidth=2.5, label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Guess (AUC = 0.5000)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves — Baseline Model Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/10_roc_curve_comparison.png')
    plt.close()
    print("[Plot Saved] 10_roc_curve_comparison.png")

    # Model metrics comparison chart
    results_df = pd.DataFrame(results).T.round(4).sort_values('ROC-AUC', ascending=False)
    print("\n--- Baseline Models Performance Table ---")
    print(results_df.to_string())

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results_df))
    width = 0.15
    colors_bar = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, results_df[metric], width, label=metric, color=colors_bar[i], edgecolor='white')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Evaluation Metrics Comparison Across Baseline Models', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results_df.index, fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/11_model_comparison.png')
    plt.close()
    print("[Plot Saved] 11_model_comparison.png")

    return fitted_models, results


# ==============================================================================
# STAGE 6 — Hyperparameter Tuning & Feature Importance
# ==============================================================================
def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Any, Dict[str, float]]:
    """Perform RandomizedSearchCV on XGBoost and compute feature importance."""
    print("=" * 70)
    print("STAGE 6 — Hyperparameter Tuning & Feature Importance (XGBoost)")
    print("=" * 70)

    # Baseline XGBoost for delta comparison
    baseline_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    baseline_xgb.fit(X_train, y_train)
    y_pred_base = baseline_xgb.predict(X_test)
    y_prob_base = baseline_xgb.predict_proba(X_test)[:, 1]

    base_auc = roc_auc_score(y_test, y_prob_base)
    print(f"XGBoost Baseline ROC-AUC: {base_auc:.4f}")

    param_distributions = {
        'n_estimators': randint(100, 400),
        'max_depth': randint(4, 10),
        'learning_rate': uniform(0.02, 0.20),
        'min_child_weight': randint(1, 8),
        'subsample': uniform(0.65, 0.35),
        'colsample_bytree': uniform(0.65, 0.35),
        'gamma': uniform(0, 0.4),
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(0.5, 1.5),
        'scale_pos_weight': uniform(0.8, 1.3),
    }

    xgb_search_estimator = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_search_estimator,
        param_distributions=param_distributions,
        n_iter=15,
        cv=3,
        scoring='roc_auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    print("Executing RandomizedSearchCV tuning...")
    start_tune = time.time()
    random_search.fit(X_train, y_train)
    tune_duration = time.time() - start_tune
    print(f"Hyperparameter tuning completed in {tune_duration:.1f}s ({tune_duration/60:.2f} mins)")
    print(f"Best CV ROC-AUC Score: {random_search.best_score_:.4f}")

    best_xgb = random_search.best_estimator_
    save_artifact_with_checksum(best_xgb, f'{MODELS_DIR}/model_xgboost_tuned.pkl')
    print(f"Tuned model saved: {MODELS_DIR}/model_xgboost_tuned.pkl")

    # Evaluate Tuned Model
    y_pred_tuned = best_xgb.predict(X_test)
    y_prob_tuned = best_xgb.predict_proba(X_test)[:, 1]

    tuned_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_tuned),
        'Precision': precision_score(y_test, y_pred_tuned, zero_division=0),
        'Recall': recall_score(y_test, y_pred_tuned, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred_tuned, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob_tuned)
    }

    print("\n--- Tuned XGBoost Classification Report ---")
    print(classification_report(y_test, y_pred_tuned, target_names=['Good Loan (0)', 'Bad Loan (1)']))
    print(f"Tuned ROC-AUC: {tuned_metrics['ROC-AUC']:.4f} (Delta vs Baseline: {tuned_metrics['ROC-AUC'] - base_auc:+.4f})")

    # Tuned Confusion Matrix Plot
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_tuned,
        annot=True,
        fmt=',d',
        cmap='RdYlGn_r',
        ax=ax,
        xticklabels=['Good Loan', 'Bad Loan'],
        yticklabels=['Good Loan', 'Bad Loan'],
        annot_kws={'size': 14}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — XGBoost (Tuned)\nROC-AUC = {tuned_metrics["ROC-AUC"]:.4f}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/12_cm_xgboost_tuned.png')
    plt.close()
    print("[Plot Saved] 12_cm_xgboost_tuned.png")

    # Baseline vs Tuned ROC Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_prob_tuned)
    ax.plot(fpr_base, tpr_base, color='#3498db', linewidth=2.5, label=f'Baseline (AUC = {base_auc:.4f})')
    ax.plot(fpr_tuned, tpr_tuned, color='#e74c3c', linewidth=2.5, label=f'Tuned (AUC = {tuned_metrics["ROC-AUC"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves — XGBoost Baseline vs. Tuned', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/13_roc_baseline_vs_tuned.png')
    plt.close()
    print("[Plot Saved] 13_roc_baseline_vs_tuned.png")

    # Feature Importance Top 20 Plot
    feat_imp = pd.Series(best_xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top20 = feat_imp.head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 20))
    bars = ax.barh(range(19, -1, -1), top20.values, color=colors, edgecolor='white')
    ax.set_yticks(range(19, -1, -1))
    ax.set_yticklabels(top20.index, fontsize=10)
    ax.set_xlabel('Feature Importance Score', fontsize=13)
    ax.set_title('Top 20 Feature Importance — XGBoost (Tuned)', fontsize=15, fontweight='bold')
    for bar, val in zip(bars, top20.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/14_feature_importance_top20.png')
    plt.close()
    print("[Plot Saved] 14_feature_importance_top20.png")

    return best_xgb, tuned_metrics


# ==============================================================================
# STAGE 7 — Final Evaluation, Decision Threshold, & Business Simulation
# ==============================================================================
def optimize_threshold_and_business_impact(
    best_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_loan_amounts: pd.Series
) -> None:
    """Analyze precision-recall curves to find optimal decision threshold and simulate business impact."""
    print("=" * 70)
    print("STAGE 7 — Optimal Threshold Analysis & Financial Business Impact")
    print("=" * 70)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    threshold_range = np.arange(0.10, 0.90, 0.05)
    tradeoff_rows = []

    for thr in threshold_range:
        y_pred_thr = (y_prob >= thr).astype(int)
        tradeoff_rows.append({
            'Threshold': round(thr, 2),
            'Precision': precision_score(y_test, y_pred_thr, zero_division=0),
            'Recall': recall_score(y_test, y_pred_thr, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_thr, zero_division=0),
            'Bad Detected': int(y_pred_thr[y_test == 1].sum()),
            'Good Rejected': int(y_pred_thr[y_test == 0].sum())
        })

    thr_df = pd.DataFrame(tradeoff_rows)
    best_f1_idx = thr_df['F1-Score'].idxmax()
    optimal_threshold = float(thr_df.loc[best_f1_idx, 'Threshold'])

    print(thr_df.round(4).to_string(index=False))
    print(f"\nOptimal Decision Threshold (Maximized F1): {optimal_threshold:.2f}")

    # Threshold Trade-off Curve Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thr_df['Threshold'], thr_df['Precision'], 'b-o', label='Precision', markersize=4)
    ax.plot(thr_df['Threshold'], thr_df['Recall'], 'r-s', label='Recall', markersize=4)
    ax.plot(thr_df['Threshold'], thr_df['F1-Score'], 'g-^', label='F1-Score', linewidth=2, markersize=5)
    ax.axvline(optimal_threshold, color='purple', linestyle='--', linewidth=2, label=f'Optimal Threshold = {optimal_threshold:.2f}')
    ax.set_xlabel('Probability Threshold', fontsize=13)
    ax.set_ylabel('Metric Score', fontsize=13)
    ax.set_title('Precision-Recall Trade-off per Threshold — Tuned XGBoost', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.10, 0.85])
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/16_threshold_analysis.png')
    plt.close()
    print("[Plot Saved] 16_threshold_analysis.png")

    # Confusion matrix at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_optimal,
        annot=True,
        fmt=',d',
        cmap='YlOrRd',
        ax=ax,
        xticklabels=['Good Loan', 'Bad Loan'],
        yticklabels=['Good Loan', 'Bad Loan'],
        annot_kws={'size': 14}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(
        f'Confusion Matrix — Optimal Threshold ({optimal_threshold:.2f})\n'
        f'Recall={recall_score(y_test, y_pred_optimal):.3f} | Precision={precision_score(y_test, y_pred_optimal):.3f}',
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/17_cm_optimal_threshold.png')
    plt.close()
    print("[Plot Saved] 17_cm_optimal_threshold.png")

    # High-Precision Financial Impact Simulation using Decimal
    avg_loan_val = float(raw_loan_amounts.mean()) if not raw_loan_amounts.empty else 15000.0
    avg_loan_dec = Decimal(str(round(avg_loan_val, 2)))
    tn, fp, fn, tp = cm_optimal.ravel()

    loss_per_bad_loan = avg_loan_dec * DEFAULT_LOSS_RATE
    opportunity_cost_per_good_reject = avg_loan_dec * OPPORTUNITY_COST_RATE

    total_loss_no_model = Decimal(int(fn + tp)) * loss_per_bad_loan
    loss_with_model = Decimal(int(fn)) * loss_per_bad_loan
    loss_prevented = Decimal(int(tp)) * loss_per_bad_loan
    missed_opportunity = Decimal(int(fp)) * opportunity_cost_per_good_reject
    net_financial_saving = loss_prevented - missed_opportunity

    print("\n--- Financial Business Simulation Results (High-Precision Decimal) ---")
    print(f"Average Loan Principal       : ${avg_loan_dec:,.2f}")
    print(f"Default Loss Severity (60%)  : ${loss_per_bad_loan:,.2f} per default")
    print(f"Opportunity Cost (15% margin): ${opportunity_cost_per_good_reject:,.2f} per false rejection")
    print(f"\nTest Set Metrics ({len(y_test):,} applications):")
    print(f"  True Positives  (Defaults Caught)   : {tp:,}")
    print(f"  False Negatives (Defaults Leaked)   : {fn:,}")
    print(f"  False Positives (Good Loans Denied) : {fp:,}")
    print(f"  True Negatives  (Good Loans Funded) : {tn:,}")
    print(f"\nEstimated Financial Totals:")
    print(f"  Gross Losses Without Model : ${total_loss_no_model:,.2f}")
    print(f"  Gross Losses With Model    : ${loss_with_model:,.2f}")
    print(f"  Default Losses Prevented   : ${loss_prevented:,.2f}")
    print(f"  Opportunity Cost Incurred  : ${missed_opportunity:,.2f}")
    print(f"  NET FINANCIAL BENEFIT      : ${net_financial_saving:,.2f}")

    # Export Final Production Artifacts with SHA-256 Checksums
    save_artifact_with_checksum(best_model, f'{MODELS_DIR}/model_final.pkl')
    scaler = joblib.load(f'{MODELS_DIR}/scaler.pkl')
    save_artifact_with_checksum(scaler, f'{MODELS_DIR}/scaler_final.pkl')

    print(f"\nFinal Production Artifacts Saved with SHA-256 Integrity Verification:")
    print(f"  - Model  : {MODELS_DIR}/model_final.pkl (Tuned XGBoost)")
    print(f"  - Scaler : {MODELS_DIR}/scaler_final.pkl (StandardScaler)")
    print(f"  - Operational Decision Threshold: {optimal_threshold:.2f}")


# ==============================================================================
# Main Orchestrator
# ==============================================================================
def main() -> None:
    """Execute end-to-end pipeline."""
    setup_environment()
    print("=" * 70)
    print("STARTING CREDIT RISK PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    df = load_and_inspect_data(RAW_DATA_PATH)

    # 2. EDA Visualizations
    generate_eda_visualizations(df)

    # Keep loan amount series for business calculation
    raw_loan_amounts = df['loan_amnt'].copy() if 'loan_amnt' in df.columns else pd.Series()

    # 3. Clean & Preprocess
    X, y = preprocess_features(df)

    # 4. Split, Resample & Scale (executed once!)
    X_train_scaled, X_test_scaled, y_train_res, y_test = prepare_training_data(X, y)

    # 5. Baseline Models
    fitted_models, baseline_results = train_baseline_models(
        X_train_scaled, y_train_res, X_test_scaled, y_test
    )

    # 6. Hyperparameter Tuning on XGBoost
    best_xgb, tuned_metrics = tune_xgboost(
        X_train_scaled, y_train_res, X_test_scaled, y_test
    )

    # 7. Final Threshold Analysis & Financial Business Impact
    optimize_threshold_and_business_impact(
        best_xgb, X_test_scaled, y_test, raw_loan_amounts
    )

    print("\n" + "=" * 70)
    print("CREDIT RISK PREDICTION PIPELINE SUCCESSFULLY COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()
