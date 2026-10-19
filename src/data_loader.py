"""
Dataset-agnostic loading + cleaning + schema detection.

Works with the Kaggle "E-Commerce Customer Churn" Excel/CSV file and with any
other churn table that has a binary target column (Churn / Exited / Attrition ...).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

TARGET_CANDIDATES = ["churn", "churned", "is_churn", "exited", "attrition", "left"]
ID_CANDIDATES = ["customerid", "customer_id", "custid", "cust_id", "userid", "user_id", "id"]

# Known messy duplicates in the Kaggle e-commerce dataset (same meaning, different labels)
COLUMN_SYNONYMS = {
    "preferredlogindevice": {"Mobile Phone": "Phone"},
    "preferredpaymentmode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "preferedordercat": {"Mobile Phone": "Mobile"},
}
YES_NO = {"yes": 1, "no": 0, "true": 1, "false": 0, "y": 1, "n": 0, "churned": 1, "stayed": 0, "1": 1, "0": 0}


def _find(columns, candidates):
    lower = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def find_data_file(root: Path, explicit: str | None = None) -> Path:
    """--data path > first file in data/raw/ > data/customers.csv (synthetic)."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p
    raw = root / "data" / "raw"
    files = sorted([f for f in raw.glob("*") if f.suffix.lower() in (".csv", ".xlsx", ".xls")]) if raw.exists() else []
    if files:
        return files[0]
    fallback = root / "data" / "customers.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No dataset found. Put your Kaggle file in data/raw/ or run src/generate_data.py")


def read_any(path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        xl = pd.ExcelFile(p)
        for sheet in xl.sheet_names:               # pick the sheet that actually has the churn column
            df = xl.parse(sheet)
            if _find(df.columns, TARGET_CANDIDATES) is not None:
                return df
        return xl.parse(xl.sheet_names[0])
    return pd.read_csv(p)


def clean(df: pd.DataFrame, target: str | None = None):
    """Return (clean_df, target_col, id_col)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    target = target or _find(df.columns, TARGET_CANDIDATES)
    if target is None:
        raise ValueError(f"No target column found. Rename your label column to 'Churn' (looked for {TARGET_CANDIDATES}).")
    id_col = _find(df.columns, ID_CANDIDATES)

    # target -> 0/1 ints
    if not is_numeric_dtype(df[target]):
        df[target] = df[target].astype(str).str.strip().str.lower().map(YES_NO)
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    if set(df[target].unique()) - {0, 1}:
        raise ValueError("Target must be binary (0/1).")

    # tidy text columns + merge duplicate labels + explicit Unknown for missing
    text_cols = [c for c in df.columns if not is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
    for c in text_cols:
        if c == id_col or c == target:
            continue
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
        syn = COLUMN_SYNONYMS.get(c.lower())
        if syn:
            df[c] = df[c].replace(syn)
        df[c] = df[c].fillna("Unknown")
    for c in [c for c in df.columns if is_bool_dtype(df[c])]:
        df[c] = df[c].astype(int)

    df = df.drop_duplicates(subset=[c for c in df.columns if c != id_col]).reset_index(drop=True)
    const = [c for c in df.columns if c not in (target, id_col) and df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=const), target, id_col


def build_schema(df: pd.DataFrame, target: str, id_col: str | None, source: str) -> dict:
    """Describe the columns so training, the app form and batch scoring all agree."""
    feats = [c for c in df.columns if c not in (target, id_col)]
    cat = [c for c in feats if not is_numeric_dtype(df[c])]
    num = [c for c in feats if c not in cat]
    info = {}
    for c in num:
        s = df[c].dropna()
        uniq = sorted(s.unique().tolist())
        info[c] = dict(min=float(s.min()), max=float(s.max()), median=float(s.median()),
                       is_int=bool(np.allclose(s, s.round())),
                       options=uniq if len(uniq) <= 6 else None)
    cats = {c: dict(options=sorted(df[c].unique().tolist()), mode=df[c].mode().iloc[0]) for c in cat}
    return dict(source=source, target=target, id_col=id_col, numeric=num, categorical=cat,
                numeric_info=info, categorical_info=cats, n_rows=int(len(df)), churn_rate=float(df[target].mean()))


def load_dataset(path):
    raw = read_any(path)
    df, target, id_col = clean(raw)
    return df, build_schema(df, target, id_col, source=Path(path).name)
