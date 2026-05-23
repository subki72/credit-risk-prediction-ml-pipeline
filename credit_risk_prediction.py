# ============================================================
# Credit Risk Prediction — End-to-End Pipeline
# Dataset : loan_data_2007_2014.csv
# Project : VIX IDX Partners — Data Scientist Intern
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings, time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

SAVE_DIR = 'images'
for d in ['data', 'images', 'models']:
    os.makedirs(d, exist_ok=True)


# ######################################################################
# STAGE 1 — EDA Awal — Load & Inspect Data
# ######################################################################


# ---- STEP 1: Load dataset & tampilkan 5 baris pertama ----
df = pd.read_csv('data/loan_data_2007_2014.csv', index_col=0, low_memory=False)
print("=" * 70)
print("STEP 1 — 5 Baris Pertama Dataset")
print("=" * 70)
print(df.head())
print()

# ---- STEP 2: Info dataset — shape, dtypes, missing values ----
print("=" * 70)
print("STEP 2 — Informasi Dataset")
print("=" * 70)
print(f"Jumlah baris  : {df.shape[0]:,}")
print(f"Jumlah kolom  : {df.shape[1]}")
print()

info_df = pd.DataFrame({
    'Tipe Data': df.dtypes,
    'Jumlah Missing': df.isnull().sum(),
    'Persentase Missing (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
print(info_df.to_string())
print()

# ---- STEP 3: Statistik deskriptif kolom numerik ----
print("=" * 70)
print("STEP 3 — Statistik Deskriptif (Kolom Numerik)")
print("=" * 70)
print(df.describe().T.to_string())
print()

# ---- STEP 4: Distribusi target variable (loan_status) ----
print("=" * 70)
print("STEP 4 — Distribusi Kolom Target: loan_status")
print("=" * 70)
target_dist = pd.DataFrame({
    'Jumlah': df['loan_status'].value_counts(),
    'Persentase (%)': (df['loan_status'].value_counts(normalize=True) * 100).round(2)
})
print(target_dist.to_string())
print()

# ---- STEP 5: Ringkasan awal & rekomendasi ----
print("=" * 70)
print("STEP 5 — Ringkasan Awal & Rekomendasi")
print("=" * 70)

# 5a. Kolom 100% missing — pasti di-drop
cols_all_missing = df.columns[df.isnull().sum() == len(df)].tolist()
print(f"\n[DROP] Kolom 100% missing ({len(cols_all_missing)} kolom):")
for c in cols_all_missing:
    print(f"  - {c}")

# 5b. Kolom identitas / URL / teks bebas — tidak relevan untuk model
cols_irrelevant = ['id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code']
print(f"\n[DROP] Kolom identitas/teks bebas (tidak relevan untuk model):")
for c in cols_irrelevant:
    miss = df[c].isnull().sum()
    pct = round(miss / len(df) * 100, 2)
    print(f"  - {c:25s} | Missing: {miss:>7,} ({pct}%)")

# 5c. Kolom dengan nilai tunggal (zero variance)
cols_single = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
print(f"\n[DROP] Kolom dengan hanya 1 nilai unik (zero variance):")
for c in cols_single:
    unique_val = df[c].dropna().unique()
    print(f"  - {c:25s} | Nilai: {unique_val}")

# 5d. Kolom missing tinggi (>50%) — perlu perhatian
cols_high_missing = [c for c in df.columns
                     if 0 < df[c].isnull().sum() / len(df) < 1
                     and df[c].isnull().sum() / len(df) > 0.50]
print(f"\n[PERHATIAN] Kolom missing >50% (pertimbangkan drop/imputasi):")
for c in cols_high_missing:
    pct = round(df[c].isnull().sum() / len(df) * 100, 2)
    print(f"  - {c:35s} | Missing: {pct}%")

# 5e. Kolom leakage (informasi setelah pinjaman diberikan)
cols_leakage = [
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
    'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
    'funded_amnt', 'funded_amnt_inv'
]
print(f"\n[PERHATIAN] Kolom potensial data leakage (informasi post-loan):")
for c in cols_leakage:
    if c in df.columns:
        print(f"  - {c}")

# 5f. Ringkasan target variable
print(f"\n[INFO] Target variable 'loan_status' memiliki {df['loan_status'].nunique()} kategori.")
print("  Untuk binary classification, perlu di-mapping menjadi 2 kelas:")
print("    - GOOD LOAN  : 'Fully Paid', 'Current'")
print("    - BAD LOAN   : 'Charged Off', 'Default', 'Late (31-120 days)',")
print("                   'Late (16-30 days)', 'In Grace Period'")
print("    - POLICY     : 'Does not meet...' — pertimbangkan dimasukkan/dibuang")

print("\n" + "=" * 70)
print("EDA Stage 1 selesai.")
print("=" * 70)


# ######################################################################
# STAGE 2 — EDA Visualisasi — Charts & Plots
# ######################################################################


# ---- Load dataset ----

# ================================================================
# STEP 1 — Distribusi Target Variable (loan_status)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Countplot
order = df['loan_status'].value_counts().index
colors = sns.color_palette('Set2', n_colors=len(order))
ax1 = axes[0]
sns.countplot(y='loan_status', data=df, order=order, palette=colors, ax=ax1)
ax1.set_title('Distribusi Loan Status', fontsize=14, fontweight='bold')
ax1.set_xlabel('Jumlah Pinjaman')
ax1.set_ylabel('Loan Status')
for i, v in enumerate(df['loan_status'].value_counts()):
    ax1.text(v + 1000, i, f'{v:,}', va='center', fontsize=9)

# Pie chart persentase
ax2 = axes[1]
pct = df['loan_status'].value_counts()
ax2.pie(pct.values, labels=pct.index, autopct='%1.1f%%',
        colors=colors, startangle=140, textprops={'fontsize': 8})
ax2.set_title('Persentase Loan Status', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/01_target_distribution.png')
plt.close()
print("[1/6] Target distribution — saved")

# ================================================================
# STEP 2 — Histogram Kolom Numerik Penting
# ================================================================
num_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'installment']
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    data = df[col].dropna()
    # Clip annual_inc agar histogram bisa terbaca (banyak outlier)
    if col == 'annual_inc':
        data = data[data <= data.quantile(0.99)]
    ax.hist(data, bins=50, color=sns.color_palette('muted')[i],
            edgecolor='white', alpha=0.85)
    ax.set_title(f'Distribusi {col}', fontsize=13, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frekuensi')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.2, label=f'Mean: {data.mean():,.1f}')
    ax.axvline(data.median(), color='green', linestyle='-.', linewidth=1.2, label=f'Median: {data.median():,.1f}')
    ax.legend(fontsize=8)

# Kosongkan subplot terakhir
axes[-1].axis('off')

plt.suptitle('Distribusi Kolom Numerik Penting', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/02_numeric_histograms.png')
plt.close()
print("[2/6] Numeric histograms — saved")

# ================================================================
# STEP 3 — Heatmap Korelasi Kolom Numerik
# ================================================================
# Pilih kolom numerik yang relevan (bukan id, bukan leakage)
exclude_cols = [
    'id', 'member_id', 'policy_code',
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt'
]
num_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
# Drop kolom yang 100% NaN
num_df = num_df.dropna(axis=1, how='all')

corr = num_df.corr()

fig, ax = plt.subplots(figsize=(20, 16))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=False,
            fmt='.2f', linewidths=0.3, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Korelasi'})
ax.set_title('Heatmap Korelasi Antar Kolom Numerik\n(Tanpa kolom ID & Leakage)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/03_correlation_heatmap.png')
plt.close()

# Top korelasi (positif & negatif)
corr_pairs = corr.unstack().reset_index()
corr_pairs.columns = ['Kolom_1', 'Kolom_2', 'Korelasi']
corr_pairs = corr_pairs[corr_pairs['Kolom_1'] != corr_pairs['Kolom_2']]
corr_pairs['abs_corr'] = corr_pairs['Korelasi'].abs()
corr_pairs = corr_pairs.drop_duplicates(subset='abs_corr').sort_values('abs_corr', ascending=False)

print("[3/6] Correlation heatmap — saved")
print("      Top 10 pasangan fitur berkorelasi tinggi:")
for _, row in corr_pairs.head(10).iterrows():
    print(f"      {row['Kolom_1']:30s} — {row['Kolom_2']:30s} | r = {row['Korelasi']:.3f}")

# ================================================================
# STEP 4 — Analisis Kolom Kategorikal vs loan_status
# ================================================================
cat_cols = ['grade', 'sub_grade', 'home_ownership', 'purpose', 'term']

for col in cat_cols:
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # 4a. Countplot distribusi keseluruhan
    order_vals = df[col].value_counts().index
    if col == 'sub_grade':
        order_vals = sorted(df[col].dropna().unique())
    ax1 = axes[0]
    sns.countplot(y=col, data=df, order=order_vals, palette='viridis', ax=ax1)
    ax1.set_title(f'Distribusi {col}', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Jumlah')
    ax1.set_ylabel(col)

    # 4b. Stacked proportion per loan_status (simplified to Good/Bad)
    ax2 = axes[1]
    df_temp = df.copy()
    good = ['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid']
    df_temp['risk_label'] = df_temp['loan_status'].apply(lambda x: 'Good Loan' if x in good else 'Bad Loan')

    ct = pd.crosstab(df_temp[col], df_temp['risk_label'], normalize='index') * 100
    if col == 'sub_grade':
        ct = ct.reindex(sorted(ct.index))
    ct.plot(kind='barh', stacked=True, ax=ax2, color=['#2ecc71', '#e74c3c'], width=0.8)
    ax2.set_title(f'Proporsi Good vs Bad Loan per {col}', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Persentase (%)')
    ax2.set_ylabel(col)
    ax2.legend(title='Risk', loc='lower right')

    plt.suptitle(f'Analisis Kategorikal: {col}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/04_categorical_{col}.png')
    plt.close()
    print(f"[4/6] Categorical {col} — saved")

# ================================================================
# STEP 5 — Distribusi emp_length & addr_state (Top 10)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# 5a. emp_length
ax1 = axes[0]
emp_order = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
             '6 years', '7 years', '8 years', '9 years', '10+ years']
emp_counts = df['emp_length'].value_counts().reindex(emp_order).dropna()
bars1 = ax1.barh(emp_counts.index, emp_counts.values, color=sns.color_palette('coolwarm', len(emp_counts)))
ax1.set_title('Distribusi Employment Length', fontsize=13, fontweight='bold')
ax1.set_xlabel('Jumlah Peminjam')
ax1.set_ylabel('Lama Bekerja')
for bar, val in zip(bars1, emp_counts.values):
    ax1.text(val + 500, bar.get_y() + bar.get_height()/2, f'{int(val):,}', va='center', fontsize=8)

# 5b. addr_state top 10
ax2 = axes[1]
top_states = df['addr_state'].value_counts().head(10)
bars2 = ax2.barh(top_states.index[::-1], top_states.values[::-1],
                 color=sns.color_palette('Spectral', 10))
ax2.set_title('Top 10 State Berdasarkan Jumlah Pinjaman', fontsize=13, fontweight='bold')
ax2.set_xlabel('Jumlah Pinjaman')
ax2.set_ylabel('State')
for bar, val in zip(bars2, top_states.values[::-1]):
    ax2.text(val + 500, bar.get_y() + bar.get_height()/2, f'{int(val):,}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/05_emp_length_addr_state.png')
plt.close()
print("[5/6] emp_length & addr_state — saved")

# ================================================================
# STEP 6 — Boxplot untuk Deteksi Outlier
# ================================================================
outlier_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti',
                'installment', 'revol_bal', 'revol_util', 'total_acc']

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for i, col in enumerate(outlier_cols):
    ax = axes[i]
    data = df[col].dropna()
    bp = ax.boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=sns.color_palette('pastel')[i], alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_ylabel('Nilai')

    # Hitung statistik outlier
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_outlier = ((data < lower) | (data > upper)).sum()
    pct_outlier = n_outlier / len(data) * 100
    ax.text(0.5, 0.02, f'Outlier: {n_outlier:,} ({pct_outlier:.1f}%)',
            transform=ax.transAxes, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Boxplot Deteksi Outlier — Kolom Numerik', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/06_outlier_boxplots.png')
plt.close()
print("[6/6] Outlier boxplots — saved")

# ================================================================
# RINGKASAN
# ================================================================
print("\n" + "=" * 70)
print("EDA Stage 2 Selesai — Semua plot tersimpan di folder: images/")
print("=" * 70)
print(f"  01_target_distribution.png")
print(f"  02_numeric_histograms.png")
print(f"  03_correlation_heatmap.png")
print(f"  04_categorical_grade.png")
print(f"  04_categorical_sub_grade.png")
print(f"  04_categorical_home_ownership.png")
print(f"  04_categorical_purpose.png")
print(f"  04_categorical_term.png")
print(f"  05_emp_length_addr_state.png")
print(f"  06_outlier_boxplots.png")


# ######################################################################
# STAGE 3 — Data Preprocessing
# ######################################################################


# ---- Load dataset ----
df = pd.read_csv('data/loan_data_2007_2014.csv', index_col=0, low_memory=False)
print(f"Shape awal: {df.shape}")
shape_before = df.shape

# ================================================================
# STEP 1 — Binarisasi Target Variable
# ================================================================
print("\n" + "=" * 60)
print("STEP 1 — Binarisasi Target Variable")
print("=" * 60)

print(f"Distribusi awal loan_status:\n{df['loan_status'].value_counts()}\n")

# Mapping: 0 = Fully Paid, 1 = Charged Off / Default
target_map = {
    'Fully Paid': 0,
    'Charged Off': 1,
    'Default': 1
}

# Filter hanya baris dengan status yang di-mapping
df = df[df['loan_status'].isin(target_map.keys())].copy()
df['loan_status'] = df['loan_status'].map(target_map)

print(f"Baris setelah filter: {len(df):,}")
print(f"Distribusi target baru:\n{df['loan_status'].value_counts()}")
print(f"Persentase Bad Loan: {df['loan_status'].mean()*100:.2f}%")

# ================================================================
# STEP 2 — Drop Kolom Tidak Relevan
# ================================================================
print("\n" + "=" * 60)
print("STEP 2 — Drop Kolom Tidak Relevan")
print("=" * 60)

cols_before = df.shape[1]

# 2a. Kolom identitas & teks bebas
id_cols = ['id', 'member_id', 'url', 'desc', 'title', 'zip_code']
df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)
print(f"[2a] Drop kolom identitas: {id_cols}")

# 2b. Kolom missing > 50%
missing_pct = df.isnull().sum() / len(df) * 100
high_missing = missing_pct[missing_pct > 50].index.tolist()
df.drop(columns=high_missing, inplace=True)
print(f"[2b] Drop kolom missing >50%: {high_missing}")

# 2c. Kolom post-loan (data leakage)
post_loan_cols = [
    'funded_amnt', 'funded_amnt_inv',
    'out_prncp', 'out_prncp_inv',
    'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt',
    'next_pymnt_d', 'last_credit_pull_d',
    'collection_recovery_fee', 'pymnt_plan'
]
df.drop(columns=[c for c in post_loan_cols if c in df.columns], inplace=True)
print(f"[2c] Drop kolom post-loan leakage: {len(post_loan_cols)} kolom")

# 2d. Kolom zero-variance (policy_code, application_type jika tunggal)
for c in df.columns:
    if df[c].nunique(dropna=True) <= 1:
        print(f"[2d] Drop kolom zero-variance: {c}")
        df.drop(columns=[c], inplace=True)

print(f"\nKolom: {cols_before} -> {df.shape[1]} (dropped {cols_before - df.shape[1]})")
print(f"Kolom tersisa: {list(df.columns)}")

# ================================================================
# STEP 3 — Handle Missing Values
# ================================================================
print("\n" + "=" * 60)
print("STEP 3 — Handle Missing Values")
print("=" * 60)

missing_before = df.isnull().sum().sum()
print(f"Total missing values sebelum: {missing_before:,}")

# 3a. Numerik -> median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'loan_status']  # exclude target
for c in num_cols:
    n_miss = df[c].isnull().sum()
    if n_miss > 0:
        median_val = df[c].median()
        df[c] = df[c].fillna(median_val)
        print(f"  [Numerik] {c:30s} | {n_miss:>6,} missing -> median ({median_val:.2f})")

# 3b. Kategorikal -> modus
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
for c in cat_cols:
    n_miss = df[c].isnull().sum()
    if n_miss > 0:
        mode_val = df[c].mode()[0]
        df[c] = df[c].fillna(mode_val)
        print(f"  [Kategorik] {c:28s} | {n_miss:>6,} missing -> modus ({mode_val})")

missing_after = df.isnull().sum().sum()
print(f"\nTotal missing values sesudah: {missing_after}")

# ================================================================
# STEP 4 — Feature Engineering
# ================================================================
print("\n" + "=" * 60)
print("STEP 4 — Feature Engineering")
print("=" * 60)

# 4a. term -> numerik (hapus " months")
if 'term' in df.columns:
    df['term'] = df['term'].str.strip().str.replace(' months', '', regex=False).astype(int)
    print(f"[4a] term -> numerik: {df['term'].unique()}")

# 4b. int_rate & revol_util — pastikan float
# (Dari EDA: sudah float64, tapi cek jika ada % string)
for col in ['int_rate', 'revol_util']:
    if col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('%', '', regex=False).astype(float)
            print(f"[4b] {col} -> hapus '%', convert float")
        else:
            print(f"[4b] {col} -> sudah float ({df[col].dtype})")

# 4c. credit_history_years dari earliest_cr_line & issue_d
if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
    df['earliest_cr_line_dt'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y', errors='coerce')
    df['issue_d_dt'] = pd.to_datetime(df['issue_d'], format='%b-%y', errors='coerce')

    # Fix century: dates parsed as future (>2025) should be 19xx
    mask_cr = df['earliest_cr_line_dt'].dt.year > 2025
    df.loc[mask_cr, 'earliest_cr_line_dt'] -= pd.DateOffset(years=100)

    mask_issue = df['issue_d_dt'].dt.year > 2025
    df.loc[mask_issue, 'issue_d_dt'] -= pd.DateOffset(years=100)

    df['credit_history_years'] = (
        (df['issue_d_dt'] - df['earliest_cr_line_dt']).dt.days / 365.25
    ).round(1)

    # Hapus kolom string asli dan kolom datetime temp
    df.drop(columns=['earliest_cr_line', 'issue_d',
                      'earliest_cr_line_dt', 'issue_d_dt'], inplace=True)
    print(f"[4c] credit_history_years -> mean: {df['credit_history_years'].mean():.1f}, "
          f"range: [{df['credit_history_years'].min()}, {df['credit_history_years'].max()}]")

# 4d. emp_length -> numerik 0-10
if 'emp_length' in df.columns:
    emp_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10
    }
    df['emp_length'] = df['emp_length'].astype(str).map(emp_map)
    # NaN (unmapped values) diisi median
    med_emp = df['emp_length'].median()
    df['emp_length'] = df['emp_length'].fillna(med_emp).astype(int)
    print(f"[4d] emp_length -> numerik: {sorted(df['emp_length'].unique())}")

# 4e. verification_status -> numerik ordinal
if 'verification_status' in df.columns:
    verif_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
    df['verification_status'] = df['verification_status'].map(verif_map)
    df['verification_status'] = df['verification_status'].fillna(0).astype(int)
    print(f"[4e] verification_status -> ordinal: {sorted(df['verification_status'].unique())}")

# ================================================================
# STEP 5 — Encoding Kolom Kategorikal
# ================================================================
print("\n" + "=" * 60)
print("STEP 5 — Encoding Kolom Kategorikal")
print("=" * 60)

# 5a. Label Encoding — grade, sub_grade (ordinal)
le_cols = ['grade', 'sub_grade']
for col in le_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"[Label Encoding] {col}: {len(le.classes_)} kelas -> {list(le.classes_[:5])}...")

# 5b. One-Hot Encoding — home_ownership, purpose, initial_list_status, etc.
ohe_cols = ['home_ownership', 'purpose', 'initial_list_status']
# Tambahkan application_type jika masih ada
if 'application_type' in df.columns:
    ohe_cols.append('application_type')
# Tambahkan addr_state jika masih ada
if 'addr_state' in df.columns:
    ohe_cols.append('addr_state')

existing_ohe = [c for c in ohe_cols if c in df.columns]
print(f"\n[One-Hot Encoding] Kolom: {existing_ohe}")
for col in existing_ohe:
    print(f"  {col}: {df[col].nunique()} kategori unik")

df = pd.get_dummies(df, columns=existing_ohe, drop_first=True, dtype=int)
print(f"Kolom setelah OHE: {df.shape[1]}")

# ================================================================
# STEP 6 — Validasi & Drop kolom non-numerik tersisa
# ================================================================
print("\n" + "=" * 60)
print("STEP 6 — Validasi Final")
print("=" * 60)

# Drop kolom non-numerik yang mungkin tersisa
remaining_obj = df.select_dtypes(include=['object']).columns.tolist()
if remaining_obj:
    print(f"Kolom non-numerik tersisa (drop): {remaining_obj}")
    df.drop(columns=remaining_obj, inplace=True)
else:
    print("Semua kolom sudah numerik.")

# ================================================================
# STEP 7 — Ringkasan & Simpan
# ================================================================
print("\n" + "=" * 60)
print("STEP 7 — Ringkasan Preprocessing")
print("=" * 60)

df_clean = df.copy()
shape_after = df_clean.shape

print(f"Shape SEBELUM : {shape_before}")
print(f"Shape SESUDAH : {shape_after}")
print(f"Baris dihapus : {shape_before[0] - shape_after[0]:,}")
print(f"Missing values: {df_clean.isnull().sum().sum()}")
print(f"Target dist   :\n{df_clean['loan_status'].value_counts()}")
print(f"\n5 baris pertama:")
print(df_clean.head())

# Simpan ke CSV
df_clean.to_csv('data/loan_data_preprocessed.csv', index=False)
print(f"\nDataset bersih disimpan: loan_data_preprocessed.csv")
print(f"Ukuran file: {df_clean.shape[0]:,} baris x {df_clean.shape[1]} kolom")
print("\n" + "=" * 60)
print("Preprocessing selesai.")
print("=" * 60)


# ######################################################################
# STAGE 4 — Split, SMOTE, & Feature Scaling
# ######################################################################


# ---- Load preprocessed dataset ----

# ================================================================
# STEP 1 — Pisahkan Fitur (X) dan Target (y)
# ================================================================
print("\n" + "=" * 60)
print("STEP 1 — Pisahkan Fitur (X) dan Target (y)")
print("=" * 60)

X = df_clean.drop(columns=['loan_status'])
y = df_clean['loan_status']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Fitur: {list(X.columns[:10])}... ({X.shape[1]} total)")

# ================================================================
# STEP 2 — Proporsi Kelas & Visualisasi
# ================================================================
print("\n" + "=" * 60)
print("STEP 2 — Proporsi Kelas Target")
print("=" * 60)

class_counts = y.value_counts()
class_pct = y.value_counts(normalize=True) * 100

print(f"Kelas 0 (Good Loan) : {class_counts[0]:,} ({class_pct[0]:.2f}%)")
print(f"Kelas 1 (Bad Loan)  : {class_counts[1]:,} ({class_pct[1]:.2f}%)")
print(f"Rasio Good:Bad      : {class_counts[0]/class_counts[1]:.1f} : 1")

imbalance_ratio = class_counts[0] / class_counts[1]
if imbalance_ratio > 3:
    print(f">> Dataset IMBALANCED (rasio {imbalance_ratio:.1f}:1, threshold > 3:1)")
else:
    print(f">> Dataset relatif balanced (rasio {imbalance_ratio:.1f}:1)")

# Pie chart
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2ecc71', '#e74c3c']
labels = ['Good Loan (0)', 'Bad Loan (1)']
wedges, texts, autotexts = ax.pie(
    class_counts.values, labels=labels, autopct='%1.1f%%',
    colors=colors, startangle=90, explode=[0, 0.05],
    textprops={'fontsize': 12}, pctdistance=0.85
)
autotexts[1].set_fontweight('bold')
centre_circle = plt.Circle((0, 0), 0.55, fc='white')
ax.add_artist(centre_circle)
ax.set_title('Proporsi Kelas Target (loan_status)', fontsize=14, fontweight='bold')
ax.text(0, 0, f'Rasio\n{imbalance_ratio:.1f}:1', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#333')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/07_target_proportion.png')
plt.close()
print("Pie chart saved: images/07_target_proportion.png")

# ================================================================
# STEP 3 — Train-Test Split (80:20, stratified)
# ================================================================
print("\n" + "=" * 60)
print("STEP 3 — Train-Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape} — distribusi: {dict(y_train.value_counts())}")
print(f"y_test  : {y_test.shape} — distribusi: {dict(y_test.value_counts())}")
print(f"Proporsi Bad Loan — train: {y_train.mean()*100:.2f}%, test: {y_test.mean()*100:.2f}%")

# ================================================================
# STEP 4 — SMOTE (hanya pada train set)
# ================================================================
print("\n" + "=" * 60)
print("STEP 4 — SMOTE pada Train Set")
print("=" * 60)

print(f"SEBELUM SMOTE:")
print(f"  Kelas 0: {(y_train == 0).sum():,}")
print(f"  Kelas 1: {(y_train == 1).sum():,}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\nSESUDAH SMOTE:")
print(f"  Kelas 0: {(y_train_res == 0).sum():,}")
print(f"  Kelas 1: {(y_train_res == 1).sum():,}")
print(f"  Total  : {len(y_train_res):,}")

# Visualisasi before/after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before
before_counts = y_train.value_counts()
axes[0].bar(['Good Loan (0)', 'Bad Loan (1)'], before_counts.values,
            color=colors, edgecolor='white')
axes[0].set_title('Sebelum SMOTE', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Jumlah')
for i, v in enumerate(before_counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=11)

# After
after_counts = pd.Series(y_train_res).value_counts()
axes[1].bar(['Good Loan (0)', 'Bad Loan (1)'], after_counts.values,
            color=colors, edgecolor='white')
axes[1].set_title('Sesudah SMOTE', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Jumlah')
for i, v in enumerate(after_counts.values):
    axes[1].text(i, v + 500, f'{v:,}', ha='center', fontsize=11)

plt.suptitle('Distribusi Kelas Sebelum & Sesudah SMOTE (Train Set)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/08_smote_comparison.png')
plt.close()
print("Plot saved: images/08_smote_comparison.png")

# ================================================================
# STEP 5 — Feature Scaling (StandardScaler)
# ================================================================
print("\n" + "=" * 60)
print("STEP 5 — Feature Scaling (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_res = pd.DataFrame(
    scaler.fit_transform(X_train_res),
    columns=X.columns
)
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns
)

# Simpan scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("Scaler di-fit pada X_train_res, transform X_train_res & X_test")
print("Scaler disimpan: scaler.pkl")
print(f"\nStatistik X_train_res setelah scaling:")
print(f"  Mean (sample): {X_train_res.iloc[:, :5].mean().values.round(4)}")
print(f"  Std  (sample): {X_train_res.iloc[:, :5].std().values.round(4)}")

# ================================================================
# STEP 6 — Ringkasan Shape
# ================================================================
print("\n" + "=" * 60)
print("STEP 6 — Ringkasan Shape Final")
print("=" * 60)

print(f"X_train_res : {X_train_res.shape}")
print(f"y_train_res : {y_train_res.shape}")
print(f"X_test      : {X_test.shape}")
print(f"y_test      : {y_test.shape}")

print("\n" + "=" * 60)
print("Stage 4 selesai — Data siap untuk modeling.")
print("=" * 60)


# ######################################################################
# STAGE 5 — Modeling & Evaluation (4 Models)
# ######################################################################



# ================================================================
# STEP 0 — Rebuild data pipeline (split, SMOTE, scale)
# ================================================================
print("=" * 60)
print("STEP 0 — Mempersiapkan Data")
print("=" * 60)

X = df_clean.drop(columns=['loan_status'])
y = df_clean['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)


# ================================================================
# STEP 1-4 — Train & Evaluate Models
# ================================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42,
                                            max_depth=15, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=42, use_label_encoder=False,
                              eval_metric='logloss', n_jobs=-1)
}

results = {}
roc_data = {}

for name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print("=" * 60)

    # Train
    start = time.time()
    model.fit(X_train_res, y_train_res)
    train_time = time.time() - start

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'ROC-AUC': auc, 'Train Time (s)': round(train_time, 2)
    }

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)

    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Good Loan (0)', 'Bad Loan (1)']))
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Good Loan', 'Bad Loan'],
                yticklabels=['Good Loan', 'Bad Loan'],
                annot_kws={'size': 14})
    ax.set_xlabel('Prediksi', fontsize=12)
    ax.set_ylabel('Aktual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {name}\n'
                 f'Accuracy={acc:.3f} | ROC-AUC={auc:.3f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe_name = name.lower().replace(' ', '_')
    plt.savefig(f'{SAVE_DIR}/09_cm_{safe_name}.png')
    plt.close()

    # Simpan model
    joblib.dump(model, f'models/models/model_{safe_name}.pkl')
    print(f"Model disimpan: models/model_{safe_name}.pkl")

# ================================================================
# STEP 5 — ROC Curve Comparison (semua model dalam 1 plot)
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 5 — ROC Curve Comparison")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 8))
colors_roc = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']

for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
    ax.plot(fpr, tpr, color=colors_roc[i], linewidth=2.5,
            label=f'{name} (AUC = {auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve — Perbandingan Model', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/10_roc_curve_comparison.png')
plt.close()
print("ROC curve saved: images/10_roc_curve_comparison.png")

# ================================================================
# STEP 6 — Tabel Perbandingan Model
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 6 — Tabel Perbandingan Model")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
results_df = results_df.sort_values('ROC-AUC', ascending=False)
print(results_df.to_string())

# Visualisasi perbandingan
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(results_df))
width = 0.15
colors_bar = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']

for i, metric in enumerate(metrics):
    bars = ax.bar(x + i * width, results_df[metric], width,
                  label=metric, color=colors_bar[i], edgecolor='white')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Metrik Evaluasi Antar Model', fontsize=15, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df.index, fontsize=10)
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/11_model_comparison.png')
plt.close()
print("Comparison chart saved: images/11_model_comparison.png")

# ================================================================
# STEP 7 — Model Terbaik
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 7 — Model Terbaik")
print("=" * 60)

best_model_name = results_df['ROC-AUC'].idxmax()
best_auc = results_df.loc[best_model_name, 'ROC-AUC']
best_recall = results_df.loc[best_model_name, 'Recall']

print(f"Model terbaik berdasarkan ROC-AUC: {best_model_name}")
print(f"  ROC-AUC  : {best_auc:.4f}")
print(f"  Recall   : {best_recall:.4f}")
print(f"  F1-Score : {results_df.loc[best_model_name, 'F1-Score']:.4f}")

print(f"\nAlasan ROC-AUC dipilih sebagai metrik utama:")
print(f"  - Credit risk: biaya salah prediksi Bad Loan (False Negative) >> False Positive")
print(f"  - ROC-AUC mengukur kemampuan model membedakan Good vs Bad Loan di berbagai threshold")
print(f"  - Accuracy bisa menyesatkan pada data imbalanced")

print(f"\n{'=' * 60}")
print("Stage 5 — Modeling selesai.")
print("=" * 60)


# ######################################################################
# STAGE 6 — Hyperparameter Tuning & Feature Importance
# ######################################################################


# ================================================================
# STEP 0 — Rebuild data pipeline
# ================================================================
print("=" * 60)
print("STEP 0 — Mempersiapkan Data")
print("=" * 60)

X = df_clean.drop(columns=['loan_status'])
y = df_clean['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)


# ================================================================
# STEP 1 — Baseline: XGBoost sebelum tuning
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 1 — Baseline XGBoost (sebelum tuning)")
print("=" * 60)

xgb_baseline = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, use_label_encoder=False,
    eval_metric='logloss', n_jobs=-1)
xgb_baseline.fit(X_train_res, y_train_res)

y_pred_base = xgb_baseline.predict(X_test)
y_prob_base = xgb_baseline.predict_proba(X_test)[:, 1]

base_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_base),
    'Precision': precision_score(y_test, y_pred_base),
    'Recall': recall_score(y_test, y_pred_base),
    'F1-Score': f1_score(y_test, y_pred_base),
    'ROC-AUC': roc_auc_score(y_test, y_prob_base)
}
print(f"Baseline ROC-AUC: {base_metrics['ROC-AUC']:.4f}")

# ================================================================
# STEP 2 — Definisi Parameter Grid
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 2 — Parameter Grid untuk RandomizedSearchCV")
print("=" * 60)

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.29),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1.0),
    'reg_lambda': uniform(0.5, 1.5),
    'scale_pos_weight': uniform(0.8, 1.4),
}

print("Parameter grid:")
for k, v in param_distributions.items():
    print(f"  {k:25s}: {v}")

# ================================================================
# STEP 3 — RandomizedSearchCV
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 3 — RandomizedSearchCV (n_iter=20, cv=5)")
print("=" * 60)

xgb_search = XGBClassifier(
    use_label_encoder=False, eval_metric='logloss',
    random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=xgb_search,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print("Mulai tuning... (ini memerlukan waktu beberapa menit)")
start = time.time()
random_search.fit(X_train_res, y_train_res)
tuning_time = time.time() - start
print(f"Tuning selesai dalam {tuning_time:.1f} detik ({tuning_time/60:.1f} menit)")

# ================================================================
# STEP 4 — Best Parameters & Score
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 4 — Best Parameters & CV Score")
print("=" * 60)

print(f"\nBest ROC-AUC (CV): {random_search.best_score_:.4f}")
print(f"\nBest Parameters:")
for k, v in random_search.best_params_.items():
    if isinstance(v, float):
        print(f"  {k:25s}: {v:.4f}")
    else:
        print(f"  {k:25s}: {v}")

# CV Results summary
cv_results = pd.DataFrame(random_search.cv_results_)
cv_results = cv_results.sort_values('rank_test_score')
print(f"\nTop 5 Kombinasi Parameter:")
for i, row in cv_results.head(5).iterrows():
    print(f"  Rank {int(row['rank_test_score'])}: "
          f"AUC={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")

# ================================================================
# STEP 5 — Re-train dengan Best Parameters
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 5 — Re-train Model dengan Best Parameters")
print("=" * 60)

best_xgb = random_search.best_estimator_
print("Model terbaik sudah di-retrain (best_estimator_)")

# Simpan model tuned
joblib.dump(best_xgb, 'models/model_xgboost_tuned.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model disimpan: model_xgboost_tuned.pkl")

# ================================================================
# STEP 6 — Evaluasi Model Tuned pada X_test
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 6 — Evaluasi Model Tuned")
print("=" * 60)

y_pred_tuned = best_xgb.predict(X_test)
y_prob_tuned = best_xgb.predict_proba(X_test)[:, 1]

tuned_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_tuned),
    'Precision': precision_score(y_test, y_pred_tuned),
    'Recall': recall_score(y_test, y_pred_tuned),
    'F1-Score': f1_score(y_test, y_pred_tuned),
    'ROC-AUC': roc_auc_score(y_test, y_prob_tuned)
}

print("\nClassification Report (Tuned):")
print(classification_report(y_test, y_pred_tuned,
      target_names=['Good Loan (0)', 'Bad Loan (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt=',d', cmap='RdYlGn_r', ax=ax,
            xticklabels=['Good Loan', 'Bad Loan'],
            yticklabels=['Good Loan', 'Bad Loan'],
            annot_kws={'size': 14})
ax.set_xlabel('Prediksi', fontsize=12)
ax.set_ylabel('Aktual', fontsize=12)
ax.set_title(f'Confusion Matrix — XGBoost (Tuned)\n'
             f'ROC-AUC = {tuned_metrics["ROC-AUC"]:.4f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/12_cm_xgboost_tuned.png')
plt.close()

# ROC Curve: baseline vs tuned
fig, ax = plt.subplots(figsize=(9, 7))
fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_base)
fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_tuned)
ax.plot(fpr_b, tpr_b, color='#3498db', linewidth=2.5,
        label=f'Baseline (AUC = {base_metrics["ROC-AUC"]:.4f})')
ax.plot(fpr_t, tpr_t, color='#e74c3c', linewidth=2.5,
        label=f'Tuned (AUC = {tuned_metrics["ROC-AUC"]:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve — XGBoost Baseline vs Tuned', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/13_roc_baseline_vs_tuned.png')
plt.close()

# Tabel perbandingan
print("\nPerbandingan Sebelum vs Sesudah Tuning:")
compare_df = pd.DataFrame({
    'Baseline': base_metrics,
    'Tuned': tuned_metrics,
    'Selisih': {k: tuned_metrics[k] - base_metrics[k] for k in base_metrics}
}).round(4)
print(compare_df.to_string())

improvement = tuned_metrics['ROC-AUC'] - base_metrics['ROC-AUC']
print(f"\nROC-AUC improvement: {improvement:+.4f}")

# ================================================================
# STEP 7 — Feature Importance Top 20
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 7 — Feature Importance Top 20")
print("=" * 60)

feat_imp = pd.Series(best_xgb.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False)
top20 = feat_imp.head(20)

print("Top 20 Fitur Terpenting:")
for i, (feat, imp) in enumerate(top20.items(), 1):
    print(f"  {i:2d}. {feat:35s} | Importance: {imp:.4f}")

# Bar chart
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 20))
bars = ax.barh(range(19, -1, -1), top20.values, color=colors, edgecolor='white')
ax.set_yticks(range(19, -1, -1))
ax.set_yticklabels(top20.index, fontsize=10)
ax.set_xlabel('Feature Importance', fontsize=13)
ax.set_title('Top 20 Feature Importance — XGBoost (Tuned)',
             fontsize=15, fontweight='bold')
for bar, val in zip(bars, top20.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/14_feature_importance_top20.png')
plt.close()

print("\nPlot saved:")
print("  12_cm_xgboost_tuned.png")
print("  13_roc_baseline_vs_tuned.png")
print("  14_feature_importance_top20.png")

print(f"\n{'=' * 60}")
print("Stage 6 — Hyperparameter Tuning selesai.")
print("=" * 60)


# ######################################################################
# STAGE 7 — Kesimpulan, Rekomendasi Bisnis & Export Final
# ######################################################################

# ================================================================
# STEP 1 — Ringkasan End-to-End Pipeline
# ================================================================
print("\n" + "#" * 70)
print("# STAGE 7 — Kesimpulan & Rekomendasi Bisnis")
print("#" * 70)

print(f"\n{'=' * 60}")
print("STEP 1 — Ringkasan Pipeline")
print("=" * 60)

print(f"""
Dataset Awal     : 466,285 baris x 74 kolom
Setelah Filter   : {len(y):,} baris (hanya Fully Paid & Charged Off/Default)
Setelah Preproc  : {len(y):,} baris x {X.shape[1]} fitur
Train (SMOTE)    : {X_train_res.shape[0]:,} baris (balanced 1:1)
Test             : {X_test.shape[0]:,} baris (distribusi asli ~19% Bad Loan)
""")

# ================================================================
# STEP 2 — Perbandingan Semua Model (Rangkuman Final)
# ================================================================
print(f"{'=' * 60}")
print("STEP 2 — Perbandingan Semua Model")
print("=" * 60)

# Rebuild results from Stage 5 predictions (re-predict with saved models)
all_models = {
    'Logistic Regression': 'models/model_logistic_regression.pkl',
    'Decision Tree': 'models/model_decision_tree.pkl',
    'Random Forest': 'models/model_random_forest.pkl',
    'XGBoost (Baseline)': 'models/model_xgboost.pkl',
    'XGBoost (Tuned)': 'models/model_xgboost_tuned.pkl',
}

final_results = {}
for name, pkl_file in all_models.items():
    try:
        mdl = joblib.load(pkl_file)
        yp = mdl.predict(X_test)
        yprob = mdl.predict_proba(X_test)[:, 1]
        final_results[name] = {
            'Accuracy': accuracy_score(y_test, yp),
            'Precision': precision_score(y_test, yp),
            'Recall': recall_score(y_test, yp),
            'F1-Score': f1_score(y_test, yp),
            'ROC-AUC': roc_auc_score(y_test, yprob)
        }
    except FileNotFoundError:
        pass

final_df = pd.DataFrame(final_results).T.round(4)
final_df = final_df.sort_values('ROC-AUC', ascending=False)
print(final_df.to_string())

# Visualisasi perbandingan final
fig, ax = plt.subplots(figsize=(14, 7))
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(final_df))
width = 0.15
colors_bar = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']

for i, m in enumerate(metrics_list):
    bars = ax.bar(x + i * width, final_df[m], width,
                  label=m, color=colors_bar[i], edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=6)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Final Semua Model (Termasuk Tuned)',
             fontsize=15, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(final_df.index, fontsize=9, rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/15_final_model_comparison.png')
plt.close()
print("Plot saved: 15_final_model_comparison.png")

# ================================================================
# STEP 3 — Analisis Threshold Optimal
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 3 — Analisis Threshold Optimal")
print("=" * 60)

best_model = joblib.load('models/model_xgboost_tuned.pkl')
y_prob_final = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for thr in thresholds:
    y_pred_thr = (y_prob_final >= thr).astype(int)
    threshold_results.append({
        'Threshold': round(thr, 2),
        'Precision': precision_score(y_test, y_pred_thr, zero_division=0),
        'Recall': recall_score(y_test, y_pred_thr, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred_thr, zero_division=0),
        'FPR': 1 - accuracy_score(y_test[y_test == 0], y_pred_thr[y_test == 0]),
        'Bad Detected': y_pred_thr[y_test == 1].sum(),
        'Good Rejected': y_pred_thr[y_test == 0].sum(),
    })

thr_df = pd.DataFrame(threshold_results)
best_f1_idx = thr_df['F1-Score'].idxmax()
optimal_thr = thr_df.loc[best_f1_idx, 'Threshold']

print(thr_df.round(4).to_string(index=False))
print(f"\nThreshold optimal (max F1): {optimal_thr}")

# Visualisasi threshold analysis
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(thr_df['Threshold'], thr_df['Precision'], 'b-o', label='Precision', markersize=4)
ax.plot(thr_df['Threshold'], thr_df['Recall'], 'r-s', label='Recall', markersize=4)
ax.plot(thr_df['Threshold'], thr_df['F1-Score'], 'g-^', label='F1-Score', linewidth=2, markersize=5)
ax.axvline(optimal_thr, color='purple', linestyle='--', linewidth=2,
           label=f'Optimal Threshold = {optimal_thr}')
ax.set_xlabel('Threshold', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Precision-Recall Trade-off per Threshold — XGBoost Tuned',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.1, 0.85])
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/16_threshold_analysis.png')
plt.close()
print("Plot saved: 16_threshold_analysis.png")

# Evaluasi pada threshold optimal
y_pred_optimal = (y_prob_final >= optimal_thr).astype(int)
print(f"\nClassification Report pada Threshold = {optimal_thr}:")
print(classification_report(y_test, y_pred_optimal,
      target_names=['Good Loan (0)', 'Bad Loan (1)']))

cm_opt = confusion_matrix(y_test, y_pred_optimal)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_opt, annot=True, fmt=',d', cmap='YlOrRd', ax=ax,
            xticklabels=['Good Loan', 'Bad Loan'],
            yticklabels=['Good Loan', 'Bad Loan'],
            annot_kws={'size': 14})
ax.set_xlabel('Prediksi', fontsize=12)
ax.set_ylabel('Aktual', fontsize=12)
ax.set_title(f'Confusion Matrix — Threshold Optimal ({optimal_thr})\n'
             f'Recall={recall_score(y_test, y_pred_optimal):.3f} | '
             f'Precision={precision_score(y_test, y_pred_optimal):.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/17_cm_optimal_threshold.png')
plt.close()

# ================================================================
# STEP 4 — Simulasi Dampak Bisnis
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 4 — Simulasi Dampak Bisnis")
print("=" * 60)

avg_loan = df_clean['loan_amnt'].mean()
tn, fp, fn, tp = cm_opt.ravel()

# Estimasi kerugian
loss_per_bad_loan = avg_loan * 0.60  # asumsi 60% dari pinjaman menjadi kerugian
opportunity_cost = avg_loan * 0.15    # asumsi 15% profit margin dari good loan

total_loss_no_model = (fn + tp) * loss_per_bad_loan  # semua bad loan lolos
loss_with_model = fn * loss_per_bad_loan               # hanya FN yang lolos
loss_prevented = tp * loss_per_bad_loan                 # bad loan tertangkap
missed_opportunity = fp * opportunity_cost              # good loan ditolak

print(f"\nAsumsi:")
print(f"  Rata-rata pinjaman          : ${avg_loan:,.0f}")
print(f"  Kerugian per bad loan       : ${loss_per_bad_loan:,.0f} (60% dari pinjaman)")
print(f"  Opportunity cost per reject : ${opportunity_cost:,.0f} (15% profit margin)")

print(f"\nDampak pada Test Set ({len(y_test):,} aplikasi):")
print(f"  True Positive  (Bad Loan tertangkap)  : {tp:,}")
print(f"  False Negative (Bad Loan lolos)        : {fn:,}")
print(f"  False Positive (Good Loan ditolak)     : {fp:,}")
print(f"  True Negative  (Good Loan disetujui)   : {tn:,}")

print(f"\nEstimasi Finansial:")
print(f"  Kerugian TANPA model       : ${total_loss_no_model:,.0f}")
print(f"  Kerugian DENGAN model      : ${loss_with_model:,.0f}")
print(f"  Kerugian DICEGAH           : ${loss_prevented:,.0f}")
print(f"  Opportunity cost (FP)      : ${missed_opportunity:,.0f}")
print(f"  NET SAVING                 : ${loss_prevented - missed_opportunity:,.0f}")

# ================================================================
# STEP 5 — Export Model Final & Rekomendasi
# ================================================================
print(f"\n{'=' * 60}")
print("STEP 5 — Export & Rekomendasi Bisnis")
print("=" * 60)

# Simpan semua artifacts
joblib.dump(best_model, 'models/model_final.pkl')
joblib.dump(scaler, 'models/scaler_final.pkl')

print(f"\nArtifacts disimpan:")
print(f"  model_final.pkl       — XGBoost Tuned (model terbaik)")
print(f"  scaler_final.pkl      — StandardScaler")
print(f"  Threshold optimal     : {optimal_thr}")

print(f"""
{'=' * 60}
REKOMENDASI BISNIS
{'=' * 60}

1. IMPLEMENTASI MODEL:
   - Deploy XGBoost Tuned dengan threshold {optimal_thr}
   - Gunakan scaler_final.pkl untuk preprocessing input baru
   - Recall {recall_score(y_test, y_pred_optimal):.1%} — menangkap mayoritas bad loan

2. FITUR KUNCI YANG HARUS DIPERHATIKAN:
   - Term (jangka waktu): pinjaman 60 bulan lebih berisiko
   - Inquiry terakhir 6 bulan: banyak inquiry = red flag
   - Home ownership & purpose: tujuan 'other' dan 'small business' lebih berisiko
   - Grade/sub_grade: indikator risiko internal yang kuat

3. STRATEGI RISIKO BERLAPIS:
   - Threshold RENDAH ({optimal_thr}): tangkap lebih banyak bad loan, beberapa good loan tertolak
   - Threshold TINGGI (0.5): hanya tolak yang sangat berisiko, lebih banyak bad loan lolos
   - Gunakan model score sebagai salah satu input dalam keputusan, bukan satu-satunya

4. MONITORING & IMPROVEMENT:
   - Pantau performa model secara berkala (model drift)
   - Kumpulkan data baru untuk retrain setiap 6 bulan
   - Pertimbangkan fitur tambahan: data behaviour, social scoring

{'=' * 60}
PIPELINE SELESAI — Credit Risk Prediction
{'=' * 60}
""")
