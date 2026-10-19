# 🛒 E-Commerce Customer Churn Prediction

Predict which customers are about to stop buying, explain *why* (SHAP), and suggest retention actions.
Works with **your own / Kaggle data** or the bundled synthetic data - no code changes needed.

## Quick start
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/generate_data.py      # OPTIONAL: synthetic demo data (skip if you use real data)
python src/train.py              # trains + evaluates + saves model and charts
streamlit run app.py             # opens the web app
```

## Using the real Kaggle dataset
1. Download **"E-Commerce Customer Churn Analysis and Prediction"** from Kaggle
   (file: `E Commerce Dataset.xlsx`, ~5,630 customers, ~17% churn).
2. Put the file in **`data/raw/`** (any `.xlsx` / `.csv` there is picked up automatically).
   *Or* use the helper: `pip install kagglehub` then `python src/download_data.py`
   (needs a Kaggle API token in `~/.kaggle/kaggle.json`).
3. Run `python src/train.py` then `streamlit run app.py`.

Or point to any file explicitly: `python src/train.py --data path/to/file.csv`

### What the loader does automatically (`src/data_loader.py`)
- Reads `.csv`, or `.xlsx` (finds the sheet that contains the churn column, e.g. `E Comm`, skipping `Data Dict`).
- Detects the target (`Churn`, `Exited`, `Attrition`, ...; Yes/No is converted to 1/0) and the ID column.
- Merges duplicate labels in the Kaggle file (`CC` = `Credit Card`, `COD` = `Cash on Delivery`, `Mobile Phone` = `Phone`).
- Fills missing categories with "Unknown"; missing numbers are median-imputed inside the model pipeline.
- Drops duplicate rows and constant columns, then learns a **schema** (numeric ranges, category options).
- The **app's input form and batch-scoring checks are generated from that schema**, so they adapt to whatever dataset you trained on.

### Engineered features (`src/features.py`)
Added only when the needed columns exist. For the Kaggle schema: `OrdersPerMonth`, `RecencyPerOrder`, `CashbackPerOrder`,
`CouponShare`, `ComplainAndUnhappy`, `IsNewCustomer`. Add your own there.

## Project structure
```
churn_project/
├── app.py                  # Streamlit app: dashboard, single prediction, batch scoring, model report
├── requirements.txt
├── data/
│   ├── raw/                # <- put your real dataset here
│   ├── customers.csv       # synthetic demo data
│   └── processed.csv       # cleaned copy written by train.py (used by the app)
├── models/churn_model.joblib
├── reports/                # EDA charts, ROC, confusion matrix, SHAP plots, metrics.json
└── src/
    ├── data_loader.py      # load / clean / schema detection
    ├── features.py         # feature engineering + model matrix
    ├── train.py            # CV model comparison, tuning, threshold, evaluation, SHAP
    ├── explain.py          # SHAP helpers
    ├── generate_data.py    # synthetic data generator
    └── download_data.py    # optional Kaggle downloader
```

## Pipeline
Load -> clean -> EDA -> features -> stratified 80/20 split -> 5-fold CV over Logistic Regression, Decision Tree,
Random Forest, XGBoost (+ tuned XGBoost) with imputation / scaling / **SMOTE inside the pipeline (no leakage)**
-> threshold chosen by max-F1 on out-of-fold predictions -> held-out test evaluation -> SHAP -> save.

## Notes & limitations
- Re-run `python src/train.py` whenever you change the dataset; the app reads the files it produces.
- The bundled results (ROC-AUC ~0.84) are on **synthetic** data. Real data results will differ; on the Kaggle dataset
  tree models (Random Forest / XGBoost) typically score very high, and the script picks the best model automatically.
- Check for leakage on any new dataset: columns recorded *after* churn happens must not be used as features.
- SMOTE inflates predicted probabilities: treat the score as a **risk ranking**, not a calibrated probability
  (add `CalibratedClassifierCV` if you need true probabilities).
- For production, define churn by time ("no purchase in next 60 days") and split train/test by date.
