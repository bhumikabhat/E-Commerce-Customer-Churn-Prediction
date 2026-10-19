"""
Feature engineering + model-matrix preparation (shared by training AND the app).

Engineered features are added only when the columns they need exist, so the same
code works for the Kaggle e-commerce schema and for the bundled synthetic data.
"""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    has = lambda *cols: all(c in d.columns for c in cols)
    one = lambda s: s.clip(lower=1)

    # ---- Kaggle "E-Commerce Customer Churn" schema ----
    if has("OrderCount", "Tenure"):
        d["OrdersPerMonth"] = d["OrderCount"] / one(d["Tenure"])
    if has("DaySinceLastOrder", "OrderCount"):
        d["RecencyPerOrder"] = d["DaySinceLastOrder"] / one(d["OrderCount"])
    if has("CashbackAmount", "OrderCount"):
        d["CashbackPerOrder"] = d["CashbackAmount"] / one(d["OrderCount"])
    if has("CouponUsed", "OrderCount"):
        d["CouponShare"] = d["CouponUsed"] / one(d["OrderCount"])
    if has("Complain", "SatisfactionScore"):
        d["ComplainAndUnhappy"] = ((d["Complain"] == 1) & (d["SatisfactionScore"] <= 2)).astype(int)
    if has("Tenure"):
        d["IsNewCustomer"] = (d["Tenure"] <= 3).astype(int)

    # ---- bundled synthetic schema ----
    if has("OrdersLastYear", "AvgOrderValue"):
        d["TotalSpend"] = d["OrdersLastYear"] * d["AvgOrderValue"]
        d["SpendPerMonth"] = d["TotalSpend"] / 12
    if has("OrdersLastYear"):
        d["OrderFrequency"] = d["OrdersLastYear"] / 12
        d["AvgGapBetweenOrders"] = 365 / one(d["OrdersLastYear"])
    if has("DaysSinceLastOrder", "AvgGapBetweenOrders"):
        d["RecencyRatio"] = d["DaysSinceLastOrder"] / d["AvgGapBetweenOrders"]
    if has("Complaints", "Tenure_Months"):
        d["ComplaintsPerYear"] = d["Complaints"] / (one(d["Tenure_Months"]) / 12).clip(lower=1)
    if has("AppVisitsPerMonth", "CartAbandonRate"):
        d["EngagementScore"] = d["AppVisitsPerMonth"] * (1 - d["CartAbandonRate"])
    if has("Tenure_Months"):
        d["IsNewCustomer"] = (d["Tenure_Months"] <= 6).astype(int)
    return d


def prepare(df: pd.DataFrame, schema: dict, feature_columns=None) -> pd.DataFrame:
    """Raw customer rows -> numeric model matrix. Pass feature_columns at inference to align columns."""
    d = df.drop(columns=[c for c in [schema.get("target"), schema.get("id_col")] if c and c in df.columns])
    d = add_features(d)
    cats = [c for c in schema["categorical"] if c in d.columns]
    d = pd.get_dummies(d, columns=cats, dtype=int)
    if feature_columns is not None:
        d = d.reindex(columns=feature_columns, fill_value=0)
    return d
