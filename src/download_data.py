"""
Optional helper: download the Kaggle dataset straight into data/raw/.
Needs a Kaggle account + API token (https://www.kaggle.com/settings -> Create New Token,
save as ~/.kaggle/kaggle.json) and:  pip install kagglehub
Manual download works just as well: save the file into data/raw/.
"""
import shutil
from pathlib import Path
import kagglehub

SLUG = "ankitverma2010/ecommerce-customer-churn-analysis-and-prediction"
dest = Path(__file__).resolve().parents[1] / "data" / "raw"
dest.mkdir(parents=True, exist_ok=True)
src = Path(kagglehub.dataset_download(SLUG))
for f in src.rglob("*"):
    if f.suffix.lower() in (".csv", ".xlsx", ".xls"):
        shutil.copy(f, dest / f.name)
        print("Copied", f.name, "->", dest)
