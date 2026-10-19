"""
End-to-end training script:
  load -> EDA plots -> features -> split -> compare models (CV) -> tune best tree model
  -> choose decision threshold -> evaluate on held-out test -> SHAP -> save artifacts
Run:  python src/train.py                         (uses data/raw/* if present, else data/customers.csv)
      python src/train.py --data path/to/file.xlsx
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, classification_report)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from features import prepare
from data_loader import find_data_file, load_dataset
from explain import get_explainer, shap_values

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
SEED, USE_SMOTE = 42, True
sns.set_theme(style="whitegrid")

def make_pipe(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if USE_SMOTE:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", model))
    return Pipeline(steps)

# ------------------------------------------------------------------ 1. load
ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None); args = ap.parse_args()
data_path = find_data_file(ROOT, args.data)
df, schema = load_dataset(data_path)
TARGET = schema["target"]
print(f"Dataset: {data_path.name}  shape={df.shape}  churn rate={df[TARGET].mean():.1%}  "
      f"missing cells={int(df.isna().sum().sum())}")
print(f"Numeric: {len(schema['numeric'])} | Categorical: {len(schema['categorical'])} | id col: {schema['id_col']}")
df.to_csv(ROOT / "data" / "processed.csv", index=False)       # cleaned copy used by the app

# ------------------------------------------------------------------ 2. EDA
num, cat = schema["numeric"], schema["categorical"]
corr_t = df[num + [TARGET]].corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
top_num = list(corr_t.head(4).index)
fig, ax = plt.subplots(1, 1 + len(top_num), figsize=(4 * (1 + len(top_num)), 3.8))
df[TARGET].value_counts().sort_index().plot.bar(ax=ax[0], color=["#4C9F70", "#D1495B"])
ax[0].set_xticklabels(["Stayed", "Churned"], rotation=0); ax[0].set_title("Class balance")
for a, c in zip(ax[1:], top_num):
    sns.boxplot(data=df, x=TARGET, y=c, ax=a, hue=TARGET, palette=["#4C9F70", "#D1495B"], legend=False)
    a.set_title(f"{c} vs churn")
plt.tight_layout(); plt.savefig(REPORTS / "01_eda_overview.png", dpi=130); plt.close()

cols_h = list(corr_t.head(24).index) + [TARGET]
plt.figure(figsize=(11, 8)); sns.heatmap(df[cols_h].corr(), cmap="coolwarm", center=0)
plt.title("Correlation heatmap (top features by |corr| with churn)"); plt.tight_layout()
plt.savefig(REPORTS / "02_correlation.png", dpi=130); plt.close()

spread = {c: df.groupby(c)[TARGET].mean().agg(lambda x: x.max() - x.min()) for c in cat if df[c].nunique() <= 15}
cat_plot = sorted(spread, key=spread.get, reverse=True)[:4]
if cat_plot:
    fig, axes = plt.subplots(1, len(cat_plot), figsize=(4.2 * len(cat_plot), 3.8), squeeze=False)
    for a, c in zip(axes[0], cat_plot):
        df.groupby(c)[TARGET].mean().sort_values().plot.barh(ax=a, color="#D1495B")
        a.set_title(f"Churn rate by {c}", fontsize=10); a.set_ylabel("")
    plt.tight_layout(); plt.savefig(REPORTS / "03_churn_by_category.png", dpi=130); plt.close()

# ------------------------------------------------------------------ 3. features + split
y = df[TARGET]
X = prepare(df, schema)
feature_columns = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ------------------------------------------------------------------ 4. compare models
candidates = {
    "Logistic Regression": make_pipe(LogisticRegression(max_iter=2000), scale=True),
    "Decision Tree": make_pipe(DecisionTreeClassifier(max_depth=6, random_state=SEED)),
    "Random Forest": make_pipe(RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=SEED)),
    "XGBoost": make_pipe(XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                       eval_metric="logloss", random_state=SEED, n_jobs=-1)),
}
cv_scores = {}
for name, pipe in candidates.items():
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    cv_scores[name] = (s.mean(), s.std())
    print(f"{name:20s} CV ROC-AUC = {s.mean():.4f} +/- {s.std():.4f}")

# ------------------------------------------------------------------ 5. tune XGBoost
param_dist = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [3, 4, 5, 6],
    "clf__learning_rate": [0.02, 0.05, 0.1],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
    "clf__min_child_weight": [1, 3, 5],
}
search = RandomizedSearchCV(candidates["XGBoost"], param_dist, n_iter=15, scoring="roc_auc",
                            cv=cv, random_state=SEED, n_jobs=1, refit=True)
search.fit(X_train, y_train)
print(f"Tuned XGBoost CV ROC-AUC = {search.best_score_:.4f}")
candidates["XGBoost (tuned)"] = search.best_estimator_
cv_scores["XGBoost (tuned)"] = (search.best_score_, np.nan)

# ------------------------------------------------------------------ 6. pick best by CV, fit on train
best_name = max(cv_scores, key=lambda k: cv_scores[k][0])
best = candidates[best_name]
best.fit(X_train, y_train)
print(f"\nBest model by CV ROC-AUC: {best_name}")

# threshold chosen on out-of-fold train predictions (maximise F1) -- no test leakage
oof = cross_val_predict(make_pipe(best[-1].__class__(**best[-1].get_params()),
                                  scale="scaler" in best.named_steps),
                        X_train, y_train, cv=cv, method="predict_proba")[:, 1]
ths = np.linspace(0.1, 0.9, 81)
threshold = float(ths[np.argmax([f1_score(y_train, oof >= t) for t in ths])])
print(f"Decision threshold (max F1 on OOF): {threshold:.2f}")

# ------------------------------------------------------------------ 7. evaluate on test (all models)
rows, plt_fig = [], plt.figure(figsize=(6.5, 5.5))
for name, pipe in candidates.items():
    if name != best_name:
        pipe = pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= (threshold if name == best_name else 0.5)).astype(int)
    rows.append(dict(Model=name, Accuracy=accuracy_score(y_test, pred), Precision=precision_score(y_test, pred),
                     Recall=recall_score(y_test, pred), F1=f1_score(y_test, pred), ROC_AUC=roc_auc_score(y_test, proba)))
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC {rows[-1]['ROC_AUC']:.3f})", lw=2 if name == best_name else 1)
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (test set)"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(REPORTS / "04_roc_curves.png", dpi=130); plt.close()

results = pd.DataFrame(rows).round(4).sort_values("ROC_AUC", ascending=False)
results.to_csv(REPORTS / "model_comparison.csv", index=False)
print("\nTest-set comparison:\n", results.to_string(index=False))

proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)
print(f"\n{best_name} @ threshold {threshold:.2f}\n", classification_report(y_test, pred, target_names=["Stayed", "Churned"]))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Stayed", "Churned"]).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout()
plt.savefig(REPORTS / "05_confusion_matrix.png", dpi=130); plt.close()

# ------------------------------------------------------------------ 8. SHAP
explainer = get_explainer(best, X_train.sample(500, random_state=SEED))
X_sample = X_test.sample(min(600, len(X_test)), random_state=SEED)
sv, Xt_sample = shap_values(best, explainer, X_sample)
plt.figure(); shap.summary_plot(sv, Xt_sample, show=False, max_display=15)
plt.tight_layout(); plt.savefig(REPORTS / "06_shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()
plt.figure(); shap.summary_plot(sv, Xt_sample, plot_type="bar", show=False, max_display=15)
plt.tight_layout(); plt.savefig(REPORTS / "07_shap_importance.png", dpi=130, bbox_inches="tight"); plt.close()
top = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
print("\nTop 10 churn drivers (mean |SHAP|):\n", top.head(10).round(4).to_string())

# ------------------------------------------------------------------ 9. save
joblib.dump(dict(pipeline=best, feature_columns=feature_columns, threshold=threshold, schema=schema,
                 model_name=best_name, background=X_train.sample(500, random_state=SEED)),
            MODELS / "churn_model.joblib")
with open(REPORTS / "metrics.json", "w") as f:
    json.dump(dict(dataset=schema['source'], n_rows=schema['n_rows'], churn_rate=schema['churn_rate'],
                   best_model=best_name, threshold=threshold,
                   test_roc_auc=float(roc_auc_score(y_test, proba)),
                   test_f1=float(f1_score(y_test, pred)),
                   test_recall=float(recall_score(y_test, pred)),
                   test_precision=float(precision_score(y_test, pred)),
                   top_features=top.head(10).round(4).to_dict()), f, indent=2)
print("\nSaved model -> models/churn_model.joblib ; reports -> reports/")
