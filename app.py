"""Streamlit app:  streamlit run app.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, joblib, plotly.express as px, plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import prepare
from explain import get_explainer, shap_values

st.set_page_config(page_title="E-Commerce Churn Predictor", page_icon="🛒", layout="wide")

# ------------------------------------------------------------ loading
@st.cache_resource
def load_model():
    art = joblib.load(ROOT / "models" / "churn_model.joblib")
    art["explainer"] = get_explainer(art["pipeline"], art["background"])
    return art

@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "processed.csv")   # cleaned copy written by train.py

if not (ROOT / "models" / "churn_model.joblib").exists():
    st.error("Model not found. Run `python src/generate_data.py` then `python src/train.py` first.")
    st.stop()

art, data = load_model(), load_data()
pipe, cols, thr, schema = art["pipeline"], art["feature_columns"], art["threshold"], art["schema"]
TARGET, ID, NUM, CAT = schema["target"], schema["id_col"], schema["numeric"], schema["categorical"]

def score(df_raw: pd.DataFrame) -> pd.DataFrame:
    X = prepare(df_raw, schema, cols)
    out = df_raw.copy()
    out["ChurnProbability"] = pipe.predict_proba(X)[:, 1]
    out["RiskTier"] = np.select([out.ChurnProbability >= thr, out.ChurnProbability >= thr * 0.6],
                                ["High", "Medium"], "Low")
    return out

ACTIONS = [  # (keyword in feature name, suggested action) - first match wins
    ("daysincelast", "Send a win-back email/push with a time-limited coupon - the customer has gone quiet."),
    ("recency", "Customer is overdue versus their normal buying rhythm - trigger a reminder campaign."),
    ("satisfaction", "Run a quick satisfaction follow-up and offer a service-recovery gesture (credit, priority support)."),
    ("complain", "Escalate to customer success; resolve open complaints and follow up personally."),
    ("delay", "Apologise for delivery delays; offer free/express shipping on the next order."),
    ("warehousetohome", "Long delivery distance: promote faster shipping options or pickup points."),
    ("abandon", "Send abandoned-cart nudges with a small incentive or free shipping."),
    ("cash on delivery", "Encourage prepaid payment (UPI/wallet) with cashback to cut COD friction."),
    ("cod", "Encourage prepaid payment (UPI/wallet) with cashback to cut COD friction."),
    ("cashback", "Low cashback received: offer a targeted cashback / loyalty-points boost."),
    ("coupon", "Offer personalised coupons to raise perceived value."),
    ("hourspend", "Re-engage with personalised recommendations and app notifications."),
    ("visits", "Re-engage with personalised recommendations and app notifications."),
    ("engagement", "Re-engage with personalised recommendations and app notifications."),
    ("return", "Review product quality / size guidance for the categories they return most."),
    ("tenure", "Welcome/onboarding journey and loyalty-programme enrolment for newer customers."),
    ("isnew", "Welcome/onboarding journey and loyalty-programme enrolment for newer customers."),
    ("order", "Introduce subscription or bundle offers to raise purchase frequency."),
    ("address", "Customers with many addresses may be unsettled/less loyal: verify delivery experience and offer a loyalty perk."),
]
def action_for(feature: str):
    f = feature.lower()
    return next((tip for key, tip in ACTIONS if key in f), None)

st.title("🛒 E-Commerce Customer Churn Prediction")
st.caption(f"Model: **{art['model_name']}**  |  decision threshold: **{thr:.2f}**  |  {len(data):,} customers from {schema['source']}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predict one customer", "📁 Batch scoring", "🧪 Model report"])

# ------------------------------------------------------------ tab 1
with tab1:
    scored = score(data.drop(columns=[TARGET]))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(scored):,}")
    k2.metric("Historical churn rate", f"{data[TARGET].mean():.1%}")
    k3.metric("High-risk customers", f"{(scored.RiskTier == 'High').sum():,}")
    k4.metric("High-risk share", f"{(scored.RiskTier == 'High').mean():.1%}")

    c1, c2 = st.columns(2)
    dims = CAT + [c for c in NUM if data[c].nunique() <= 10]
    dim = c1.selectbox("Churn rate by", dims)
    g = data.groupby(dim)[TARGET].mean().reset_index()
    g[dim] = g[dim].astype(str)
    c1.plotly_chart(px.bar(g, x=dim, y=TARGET, text_auto=".0%", color=TARGET, color_continuous_scale="Reds").update_layout(
        height=360, coloraxis_showscale=False, yaxis_tickformat=".0%", yaxis_title="Churn rate"), width="stretch")
    tier = scored.RiskTier.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    tier.columns = ["RiskTier", "Customers"]
    c2.plotly_chart(px.pie(tier, names="RiskTier", values="Customers", hole=0.5, color="RiskTier",
                           color_discrete_map={"High": "#D1495B", "Medium": "#EDAE49", "Low": "#4C9F70"}
                           ).update_layout(height=360, title="Risk tier distribution"), width="stretch")
    st.subheader("Top 20 customers to contact first")
    show = ([ID] if ID else []) + ["ChurnProbability", "RiskTier"] + NUM[:6]
    st.dataframe(scored.sort_values("ChurnProbability", ascending=False).head(20)[show]
                 .style.format({"ChurnProbability": "{:.1%}"}), width="stretch", hide_index=True)

# ------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Enter customer details")
    st.caption("The form is generated from your dataset's columns (defaults = typical customer).")
    vals, grid = {}, st.columns(3)
    for i, c in enumerate(NUM):
        info = schema["numeric_info"][c]
        with grid[i % 3]:
            if info["options"]:
                opts = [int(o) if info["is_int"] else o for o in info["options"]]
                med = min(opts, key=lambda o: abs(o - info["median"]))
                vals[c] = st.selectbox(c, opts, index=opts.index(med), key=f"f_{c}")
            elif info["is_int"]:
                vals[c] = st.slider(c, int(info["min"]), int(info["max"]), int(info["median"]), key=f"f_{c}")
            else:
                step = max(round((info["max"] - info["min"]) / 200, 4), 0.01)
                vals[c] = st.slider(c, float(info["min"]), float(info["max"]), float(info["median"]), step=step, key=f"f_{c}")
    for j, c in enumerate(CAT):
        info = schema["categorical_info"][c]
        with grid[(len(NUM) + j) % 3]:
            vals[c] = st.selectbox(c, info["options"], index=info["options"].index(info["mode"]), key=f"f_{c}")

    row = pd.DataFrame([vals])
    res = score(row).iloc[0]
    p = float(res.ChurnProbability)

    st.divider()
    g1, g2 = st.columns([1, 1.4])
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": f"Risk score - {res.RiskTier} risk"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                   "steps": [{"range": [0, thr * 60], "color": "#4C9F70"},
                             {"range": [thr * 60, thr * 100], "color": "#EDAE49"},
                             {"range": [thr * 100, 100], "color": "#D1495B"}]}))
        fig.update_layout(height=300, margin=dict(t=60, b=0))
        st.plotly_chart(fig, width="stretch")
    with g2:
        X1 = prepare(row, schema, cols)
        sv, _ = shap_values(pipe, art["explainer"], X1)
        contrib = pd.Series(sv[0], index=cols).sort_values(key=np.abs, ascending=False).head(8)[::-1]
        st.plotly_chart(px.bar(x=contrib.values, y=contrib.index, orientation="h",
                               color=contrib.values > 0, color_discrete_map={True: "#D1495B", False: "#4C9F70"},
                               title="Why? Top drivers (red = pushes toward churn)",
                               labels={"x": "SHAP impact (log-odds)", "y": ""}).update_layout(
                               height=300, showlegend=False, margin=dict(t=50, b=0)), width="stretch")

    st.subheader("Suggested retention actions")
    if res.RiskTier == "Low":
        st.success("Low risk - keep nurturing with regular loyalty communication.")
    else:
        shown = []
        for f in contrib[contrib > 0].sort_values(ascending=False).index:
            tip = action_for(f)
            if tip and tip not in shown:
                shown.append(tip)
        for t in shown[:4] or ["Offer a personalised discount and monitor engagement."]:
            st.markdown(f"- {t}")

# ------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Score many customers at once")
    st.write(f"Upload a CSV with the same feature columns as your training data (`{TARGET}` is optional).")
    up = st.file_uploader("CSV file", type="csv")
    if st.button("Or use 500 rows of the training data"):
        st.session_state["batch"] = data.drop(columns=[TARGET]).head(500)
    if up is not None:
        st.session_state["batch"] = pd.read_csv(up)
    if "batch" in st.session_state:
        raw = st.session_state["batch"]
        missing = [c for c in NUM + CAT if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            raw = raw.copy()
            for c in CAT:
                raw[c] = raw[c].fillna("Unknown").astype(str).str.strip()
            out = score(raw).sort_values("ChurnProbability", ascending=False)
            st.write(out.RiskTier.value_counts().to_dict())
            st.dataframe(out.style.format({"ChurnProbability": "{:.1%}"}), width="stretch", hide_index=True)
            st.download_button("⬇️ Download scored CSV", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")

# ------------------------------------------------------------ tab 4
with tab4:
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    st.caption(f"Trained on **{m['dataset']}** ({m['n_rows']:,} rows, churn rate {m['churn_rate']:.1%}) - best model: {m['best_model']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['test_roc_auc']:.3f}"); c2.metric("F1", f"{m['test_f1']:.3f}")
    c3.metric("Recall", f"{m['test_recall']:.3f}"); c4.metric("Precision", f"{m['test_precision']:.3f}")
    st.dataframe(pd.read_csv(ROOT / "reports" / "model_comparison.csv"), width="stretch", hide_index=True)
    for img, cap in [("04_roc_curves.png", "ROC curves"), ("05_confusion_matrix.png", "Confusion matrix"),
                     ("06_shap_summary.png", "SHAP summary"), ("03_churn_by_category.png", "Churn by category")]:
        if (ROOT / "reports" / img).exists():
            st.image(str(ROOT / "reports" / img), caption=cap)
