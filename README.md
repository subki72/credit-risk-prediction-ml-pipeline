# Credit Risk Prediction

A comprehensive machine learning project for predicting credit risk using loan data from 2007-2014. This project implements an end-to-end pipeline including exploratory data analysis, data preprocessing, feature engineering, model training with multiple algorithms, hyperparameter tuning, and threshold optimization.

## Project Overview

This project aims to predict whether a loan will be fully paid or charged off based on various borrower and loan characteristics. The solution uses multiple machine learning algorithms and addresses class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

## Dataset

The project uses the Lending Club loan data from 2007-2014, containing information about:
- Loan characteristics (amount, interest rate, term, grade)
- Borrower information (annual income, employment length, home ownership)
- Credit history (credit lines, delinquencies, inquiries)
- Loan status (target variable)

### Dataset Details

**File:** `loan_data_2007_2014.csv`
**Size:** ~228 MB (466,285 rows × 75 columns)
**Source:** Lending Club Loan Data

**Note:** Due to GitHub file size limitations, the dataset files are not included in this repository. 

### How to Get the Dataset

**Option 1: Download from Kaggle**
1. Visit [Lending Club Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Download the dataset
3. Place `loan_data_2007_2014.csv` in the `data/` folder

**Option 2: Use Alternative Sources**
- [Lending Club Historical Data](https://www.lendingclub.com/info/download-data.action) (if available)
- Contact the repository owner for dataset access

### Dataset Structure

After downloading, your project structure should look like:
```
data/
├── loan_data_2007_2014.csv          # Raw dataset (place here)
└── loan_data_preprocessed.csv       # Generated after running the pipeline
```

### Key Features in Dataset

**Numeric Features:**
- `loan_amnt`: Loan amount
- `int_rate`: Interest rate
- `annual_inc`: Annual income
- `dti`: Debt-to-income ratio
- `installment`: Monthly installment
- `revol_bal`: Revolving balance
- `revol_util`: Revolving line utilization rate

**Categorical Features:**
- `grade`: Loan grade (A-G)
- `sub_grade`: Loan sub-grade (A1-G5)
- `home_ownership`: Home ownership status
- `purpose`: Loan purpose
- `term`: Loan term (36 or 60 months)
- `addr_state`: Borrower's state

**Target Variable:**
- `loan_status`: Current status of the loan (Fully Paid, Charged Off, Default, etc.)

## Project Structure

```
.
├── src/credit_risk/                      # Core Production Package
│   ├── config.py                         # Settings, paths & feature contracts
│   ├── schemas.py                        # Pydantic data validation schemas
│   ├── transformers.py                   # Custom Scikit-Learn transformers
│   ├── pipeline.py                       # Unified ColumnTransformer pipeline
│   ├── service.py                        # Inference service & SHA-256 verification
│   └── api.py                            # FastAPI REST microservice
├── tests/                                # Automated Testing Suite (pytest)
│   ├── test_transformers.py              # Unit tests for custom transformers
│   ├── test_schemas.py                   # Schema validation unit tests
│   ├── test_pipeline.py                  # Integration tests for pipeline
│   ├── test_service.py                   # Integrity & service tests
│   └── test_api.py                       # FastAPI endpoint integration tests
├── .github/workflows/ci.yml              # GitHub Actions CI pipeline
├── Dockerfile                            # Production multi-stage Dockerfile
├── .dockerignore                         # Container build ignore rules
├── data/
│   ├── loan_data_2007_2014.csv          # Raw dataset
│   └── loan_data_preprocessed.csv       # Preprocessed dataset
├── images/                               # Visualization outputs
│   ├── 01_target_distribution.png
│   ├── 02_numeric_histograms.png
│   ├── 03_correlation_heatmap.png
│   ├── 07_target_proportion.png
│   ├── 08_smote_comparison.png
│   ├── 09_cm_*.png                      # Confusion matrices
│   ├── 10_roc_curve_comparison.png
│   ├── 11_model_comparison.png
│   ├── 12_cm_xgboost_tuned.png
│   ├── 13_roc_baseline_vs_tuned.png
│   ├── 14_feature_importance_top20.png
│   ├── 16_threshold_analysis.png
│   └── 17_cm_optimal_threshold.png
├── models/                               # Saved models & SHA-256 checksums
│   ├── model_final.pkl
│   ├── model_final.pkl.sha256
│   ├── scaler_final.pkl
│   └── scaler_final.pkl.sha256
├── credit_risk_prediction.py             # Modular training pipeline script
├── credit_risk_prediction.ipynb          # Jupyter notebook version
├── requirements.txt                      # Pinned Python dependencies
└── README.md                             # Project documentation
```

## Pipeline Stages

### Stage 1: Exploratory Data Analysis (EDA)
- Dataset inspection and summary statistics
- Missing value analysis
- Target variable distribution
- Identification of irrelevant and leakage features

### Stage 2: Data Visualization
- Target variable distribution analysis
- Numeric feature histograms
- Correlation heatmap
- Categorical feature analysis
- Outlier detection using boxplots

![Target Distribution](images/01_target_distribution.png)

![Correlation Heatmap](images/03_correlation_heatmap.png)

### Stage 3: Data Preprocessing
- Target variable binarization (Good Loan vs Bad Loan)
- Removal of irrelevant columns (IDs, URLs, text fields)
- Handling missing values (median for numeric, mode for categorical)
- Feature engineering:
  - Term conversion to numeric
  - Credit history calculation
  - Employment length encoding
  - Verification status ordinal encoding
- Label encoding for ordinal features (grade, sub_grade)
- One-hot encoding for nominal features (home_ownership, purpose, state)

### Stage 4: Data Balancing and Scaling
- Train-test split (80:20, stratified)
- SMOTE application on training set
- Feature scaling using StandardScaler

![SMOTE Comparison](images/08_smote_comparison.png)

### Stage 5: Model Training and Evaluation
Multiple algorithms trained and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

![Model Comparison](images/11_model_comparison.png)

![ROC Curve Comparison](images/10_roc_curve_comparison.png)

### Stage 6: Hyperparameter Tuning
- RandomizedSearchCV for XGBoost optimization
- Cross-validation for robust performance estimation
- Comparison of baseline vs tuned models

![Baseline vs Tuned](images/13_roc_baseline_vs_tuned.png)

### Stage 7: Feature Importance Analysis
- Top 20 most important features identified
- Feature contribution visualization

![Feature Importance](images/14_feature_importance_top20.png)

### Stage 8: Threshold Optimization
- Precision-Recall-F1 curve analysis
- Optimal threshold selection for business requirements
- Final model evaluation with optimized threshold

![Threshold Analysis](images/16_threshold_analysis.png)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd credit-risk-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

Execute the main script to run the entire pipeline:
```bash
python credit_risk_prediction.py
```

This will:
1. Perform EDA and generate visualizations
2. Preprocess the data
3. Train multiple models
4. Perform hyperparameter tuning
5. Generate evaluation reports and visualizations
6. Save trained models and scalers

### Using Jupyter Notebook

Alternatively, open and run the Jupyter notebook:
```bash
jupyter notebook credit_risk_prediction.ipynb
```

### Running the Production REST API

Start the high-performance FastAPI inference microservice with Uvicorn:
```bash
uvicorn src.credit_risk.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is automatically available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

### Making Predictions via REST API

#### Using `curl`:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_id": "APP-2026-001",
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
    "dti": 16.5
  }'
```

**Response:**
```json
{
  "loan_id": "APP-2026-001",
  "default_probability": 0.1824,
  "decision": "APPROVED",
  "risk_tier": "LOW",
  "threshold_applied": 0.35,
  "recommendation": "Prime credit profile. Approved with preferential terms."
}
```

#### Using Python Service Directly:
```python
from src.credit_risk.schemas import LoanApplicationSchema
from src.credit_risk.service import CreditRiskService

service = CreditRiskService()
application = LoanApplicationSchema(
    loan_amnt=15000.0,
    term="36 months",
    int_rate=11.99,
    installment=498.25,
    grade="B",
    sub_grade="B3",
    emp_length="5 years",
    home_ownership="MORTGAGE",
    annual_inc=75000.0,
    verification_status="Source Verified",
    purpose="debt_consolidation",
    addr_state="CA",
    dti=16.5
)

result = service.predict_single(application)
print(f"Decision: {result.decision} (Default Probability: {result.default_probability:.2%})")
```

### Running Automated Tests

Execute the comprehensive unit and integration test suite:
```bash
pytest tests/ -v
```

### Deploying with Docker

Build the production multi-stage container image:
```bash
docker build -t credit-risk-api:latest .
```

Run the containerized microservice:
```bash
docker run -d -p 8000:8000 --name credit-risk-service credit-risk-api:latest
```

Verify service health:
```bash
curl http://localhost:8000/health
```

## Key Features

- **Unified Inference Pipeline**: Custom Scikit-Learn `ColumnTransformer` handles raw categorical and string inputs directly with zero data leakage.
- **Pydantic Contract Validation**: Strict schema validation preventing invalid, corrupt, or out-of-range loan inputs.
- **SHA-256 Checksum Integrity**: Automated cryptographic hash verification protecting serialized model checkpoints against tampering.
- **High-Precision Financial Arithmetic**: Monetary simulations calculated using `Decimal` to avoid binary floating-point precision loss.
- **Production REST Microservice**: FastAPI server with async support, Swagger UI, and batch prediction endpoints.
- **Automated CI/CD & Testing**: 100% test pass rate across schemas, transformers, pipelines, services, and endpoints with GitHub Actions workflow.

- Comprehensive data preprocessing pipeline
- Handling of imbalanced datasets using SMOTE
- Multiple model comparison framework
- Hyperparameter optimization
- Feature importance analysis
- Threshold optimization for business objectives
- Extensive visualization suite
- Model persistence for deployment

## Model Performance

The final tuned XGBoost model achieves:
- High ROC-AUC score indicating strong discriminative ability
- Balanced precision and recall through threshold optimization
- Robust performance on unseen test data

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- joblib
- scipy
- pydantic
- fastapi
- uvicorn
- pytest
- httpx

See `requirements.txt` for specific versions.

## Future Improvements

- Implement additional ensemble methods
- Add cross-validation for more robust evaluation
- Develop a web interface for predictions
- Integrate real-time data pipeline
- Add model monitoring and retraining capabilities
- Implement explainability tools (SHAP, LIME)

## License

This project is part of the VIX IDX Partners Data Scientist Internship program.

## Author

Muhammad Syafii Assubki

## Acknowledgments

- Lending Club for providing the dataset
- VIX IDX Partners for the internship opportunity
