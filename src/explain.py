"""SHAP helpers that work for both linear and tree-based final models."""
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

def transform_for_model(pipeline, X):
    """Run every pipeline step except the final classifier (sampler is skipped at transform time)."""
    import pandas as pd
    Xt = X
    for name, step in pipeline.steps[:-1]:
        if name == "smote":      # samplers only act during fit
            continue
        Xt = step.transform(Xt)
    return pd.DataFrame(Xt, columns=X.columns, index=X.index)

def get_explainer(pipeline, X_background):
    clf = pipeline[-1]
    Xb = transform_for_model(pipeline, X_background)
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, Xb)
    return shap.TreeExplainer(clf)

def shap_values(pipeline, explainer, X):
    Xt = transform_for_model(pipeline, X)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):          # older shap: list per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[:, :, 1]
    return sv, Xt
