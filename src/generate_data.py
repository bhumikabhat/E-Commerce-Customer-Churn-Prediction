"""
Generate a realistic synthetic e-commerce customer dataset.

Churn is generated from a hidden logistic relationship with the behaviour
features (recency, complaints, satisfaction, etc.) plus noise, so models
can learn real patterns but never reach a perfect score.

To use a real dataset (e.g. Kaggle "E-Commerce Customer Churn"), place a CSV
at data/customers.csv with a 'Churn' column (0/1) and adapt src/features.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate(n=6000, seed=42):
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 61, n)
    city_tier = rng.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25])
    device = rng.choice(["Mobile", "Desktop", "Tablet"], n, p=[0.62, 0.30, 0.08])
    payment = rng.choice(["UPI", "Credit Card", "Debit Card", "COD", "Wallet"], n,
                         p=[0.32, 0.22, 0.18, 0.18, 0.10])
    category = rng.choice(["Fashion", "Electronics", "Grocery", "Home", "Beauty"], n,
                          p=[0.28, 0.22, 0.20, 0.17, 0.13])
    gender = rng.choice(["Male", "Female"], n)

    orders_last_year = np.clip(rng.poisson(8, n) + (tenure_months // 12), 1, 60)
    avg_order_value = np.round(rng.gamma(3.0, 450, n) + 200, 2)
    days_since_last_order = np.clip(rng.exponential(35, n).astype(int), 0, 365)
    app_visits_per_month = np.clip(rng.poisson(12, n), 0, 60)
    cart_abandon_rate = np.round(np.clip(rng.beta(2, 4, n), 0, 1), 2)
    discount_usage_pct = np.round(np.clip(rng.beta(2, 3, n), 0, 1), 2)
    complaints = np.clip(rng.poisson(0.7, n), 0, 8)
    return_rate = np.round(np.clip(rng.beta(1.5, 8, n), 0, 1), 2)
    satisfaction = np.clip(np.round(rng.normal(3.5, 1.1, n)), 1, 5).astype(int)
    num_addresses = rng.integers(1, 6, n)
    cashback = np.round(rng.gamma(2, 60, n), 2)
    delivery_delay_days = np.clip(rng.poisson(1.2, n), 0, 10)

    # hidden churn logic (log-odds)
    z = (
        -1.6
        + 0.020 * days_since_last_order
        - 0.060 * tenure_months / 6
        + 0.55 * complaints
        - 0.55 * (satisfaction - 3)
        + 1.2 * cart_abandon_rate
        - 0.05 * app_visits_per_month / 2
        - 0.06 * orders_last_year / 2
        + 0.8 * return_rate
        + 0.20 * delivery_delay_days
        - 0.004 * cashback / 10
        + 0.25 * (device == "Mobile")
        + 0.35 * (payment == "COD")
        + 0.15 * (city_tier == 3)
        + rng.normal(0, 0.35, n)
    )
    z = 1.7 * (z - 0.1)  # sharpen signal / re-centre so churn rate is ~35-40%
    p = 1 / (1 + np.exp(-z))
    churn = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "CustomerID": np.arange(100001, 100001 + n),
        "Tenure_Months": tenure_months,
        "CityTier": city_tier,
        "DeviceType": device,
        "PaymentMethod": payment,
        "PreferredCategory": category,
        "Gender": gender,
        "OrdersLastYear": orders_last_year,
        "AvgOrderValue": avg_order_value,
        "DaysSinceLastOrder": days_since_last_order,
        "AppVisitsPerMonth": app_visits_per_month,
        "CartAbandonRate": cart_abandon_rate,
        "DiscountUsagePct": discount_usage_pct,
        "Complaints": complaints,
        "ReturnRate": return_rate,
        "SatisfactionScore": satisfaction,
        "NumAddresses": num_addresses,
        "CashbackAmount": cashback,
        "DeliveryDelayDays": delivery_delay_days,
        "Churn": churn,
    })

    # inject some missing values like real data
    for col in ["Tenure_Months", "AvgOrderValue", "DaysSinceLastOrder", "AppVisitsPerMonth"]:
        mask = rng.random(n) < 0.03
        df[col] = df[col].astype("float64")
        df.loc[mask, col] = np.nan
    return df

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "customers.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}  |  churn rate = {df.Churn.mean():.1%}")
