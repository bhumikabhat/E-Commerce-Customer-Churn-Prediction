"""Streamlit app:  streamlit run app.py"""
import sys, json, html
from pathlib import Path
import numpy as np, pandas as pd, joblib
import plotly.express as px, plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import prepare
from explain import get_explainer, shap_values

st.set_page_config(page_title="Churn Predictor", page_icon="🛒", layout="wide")

# ------------------------------------------------------------ look & feel
RED, AMBER, GREEN, INDIGO = "#EF4444", "#F59E0B", "#10B981", "#6366F1"
TIER_COLOR = {"High": RED, "Medium": AMBER, "Low": GREEN}

st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1280px;}
footer{visibility:hidden;}
.hero{background:linear-gradient(135deg,#4F46E5 0%,#7C3AED 55%,#DB2777 100%);border-radius:22px;
      padding:30px 34px;margin-bottom:22px;box-shadow:0 12px 30px rgba(79,70,229,.30);}
.hero-title{font-size:2.05rem;font-weight:800;color:#fff;line-height:1.2;letter-spacing:-.5px;}
.hero-sub{color:rgba(255,255,255,.9);font-size:1.02rem;margin:6px 0 16px 0;}
.chip{display:inline-block;background:rgba(255,255,255,.18);color:#fff;border:1px solid rgba(255,255,255,.30);
      padding:5px 13px;border-radius:999px;font-size:.82rem;margin:0 8px 6px 0;}
div[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(99,102,241,.12),rgba(139,92,246,.05));
      border:1px solid rgba(99,102,241,.30);border-radius:18px;padding:16px 20px;
      box-shadow:0 4px 14px rgba(0,0,0,.10);transition:transform .15s ease,box-shadow .15s ease;}
div[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 10px 22px rgba(99,102,241,.25);}
div[data-testid="stMetricLabel"] p{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;opacity:.75;font-weight:600;}
div[data-testid="stMetricValue"]{font-weight:800;}
div[data-testid="stPlotlyChart"]{border:1px solid rgba(128,128,128,.20);border-radius:18px;padding:8px;}
button[data-baseweb="tab"]{font-weight:600;font-size:1rem;padding:10px 18px;}
.sec{display:flex;align-items:center;gap:10px;margin:20px 0 4px 0;}
.sec .bar{width:5px;height:26px;border-radius:4px;background:linear-gradient(180deg,#6366F1,#EC4899);}
.sec .t{font-size:1.25rem;font-weight:700;}
.sec-sub{opacity:.7;font-size:.9rem;margin:0 0 12px 15px;}
.risk-banner{border-radius:16px;padding:16px 22px;font-size:1.05rem;margin:4px 0 14px 0;}
</style>
""", unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec"><div class="bar"></div><div class="t">{html.escape(title)}</div></div>'
                + (f'<div class="sec-sub">{html.escape(sub)}</div>' if sub else ""), unsafe_allow_html=True)


def style(fig, h=360, legend=False):
    fig.update_layout(height=h, margin=dict(t=50, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, title_font=dict(size=16))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.20)")
    return fig


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
    st.error("Model not found. Run `python src/train.py` first (and commit models/ + data/processed.csv if deploying).")
    st.stop()

art, data = load_model(), load_data()
pipe, cols, thr, schema = art["pipeline"], art["feature_columns"], art["threshold"], art["schema"]
TARGET, ID, NUM, CAT = schema["target"], schema["id_col"], schema["numeric"], schema["categorical"]
MED_THR = thr * 0.6


def score(df_raw: pd.DataFrame) -> pd.DataFrame:
    X = prepare(df_raw, schema, cols)
    out = df_raw.copy()
    out["ChurnProbability"] = pipe.predict_proba(X)[:, 1]
    out["RiskTier"] = np.select([out.ChurnProbability >= thr, out.ChurnProbability >= MED_THR],
                                ["High", "Medium"], "Low")
    return out


@st.cache_data(show_spinner=False)
def score_all():
    s = score(data.drop(columns=[TARGET]))
    s[TARGET] = data[TARGET].values
    return s


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
    ("address", "Many saved addresses can signal an unsettled customer: check delivery experience and offer a loyalty perk."),
]
def action_for(feature: str):
    f = feature.lower()
    return next((tip for key, tip in ACTIONS if key in f), None)


# ------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🛒 E-Commerce Customer Churn Prediction</div>
  <div class="hero-sub">Find the customers who are about to leave - and understand why.</div>
  <span class="chip">🤖 {html.escape(art['model_name'])}</span>
  <span class="chip">🎯 Threshold {thr:.2f}</span>
  <span class="chip">👥 {len(data):,} customers</span>
  <span class="chip">📄 {html.escape(schema['source'])}</span>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predict one customer", "📁 Batch scoring", "🧪 Model report"])

# ------------------------------------------------------------ tab 1 : dashboard
def blend(v):  # 0..1 -> amber -> orange -> red
    stops = [(252, 211, 77), (249, 115, 22), (220, 38, 38)]
    t = min(max(float(v), 0.0), 1.0) * 2
    i = int(min(t, 1.999)); f = t - i
    x, y = stops[i], stops[i + 1]
    return "rgb(%d,%d,%d)" % tuple(round(x[k] + (y[k] - x[k]) * f) for k in range(3))


with tab1:
    scored = score_all()
    section("Customer base overview", "Pick a column, then click one or more segments - every card and chart below updates instantly.")

    dims = CAT + [c for c in NUM if data[c].nunique() <= 10]
    f1, f2 = st.columns([1, 3])
    dim = f1.selectbox("Churn rate by", dims, key="dash_dim")
    opts = sorted(data[dim].unique().tolist())
    with f2:
        picked = st.pills(f"👆 Click a {dim} value to filter (select several to combine; none selected = all customers)",
                          opts, selection_mode="multi", key=f"dash_pick_{dim}", format_func=str)

    sel = list(picked) if picked else opts
    filtered = bool(picked) and len(sel) < len(opts)
    seg_label = ", ".join(map(str, sel)) if len(sel) <= 3 else f"{len(sel)} segments"
    view = scored[scored[dim].isin(sel)]
    all_churn, all_high = scored[TARGET].mean(), (scored.RiskTier == "High").mean()
    v_churn, v_high = view[TARGET].mean(), (view.RiskTier == "High").mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers in view", f"{len(view):,}")
    k2.metric("Historical churn rate", f"{v_churn:.1%}",
              delta=f"{(v_churn - all_churn) * 100:+.1f} pts vs overall" if filtered else None, delta_color="inverse")
    k3.metric("High-risk customers", f"{(view.RiskTier == 'High').sum():,}")
    k4.metric("High-risk share", f"{v_high:.1%}",
              delta=f"{(v_high - all_high) * 100:+.1f} pts vs overall" if filtered else None, delta_color="inverse")
    st.caption(f"Showing {len(view):,} of {len(scored):,} customers ({len(view) / len(scored):.0%}) "
               f"· {dim}: {seg_label if filtered else 'all customers'}")

    c1, c2 = st.columns([1.35, 1])
    g = (scored.groupby(dim).agg(Customers=(TARGET, "size"), ChurnRate=(TARGET, "mean"),
                                 AvgRisk=("ChurnProbability", "mean"),
                                 HighRiskShare=("RiskTier", lambda s: (s == "High").mean())).reset_index())
    g = g.sort_values("ChurnRate", ascending=False) if dim in CAT else g.sort_values(dim)
    g["Segment"] = g[dim].astype(str)
    lo, hi = g["ChurnRate"].min(), g["ChurnRate"].max()
    colors = [blend((v - lo) / (hi - lo + 1e-9)) if on else "rgba(128,128,128,.35)"
              for v, on in zip(g["ChurnRate"], g[dim].isin(sel))]          # selected = coloured, others = grey
    bar = go.Figure(go.Bar(x=g["Segment"], y=g["ChurnRate"], marker_color=colors, customdata=g[["Customers"]].values,
                           text=[f"{v:.1%}" for v in g["ChurnRate"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>Churn rate: %{y:.1%}<br>Customers: %{customdata[0]:,}<extra></extra>"))
    bar.add_hline(y=all_churn, line_dash="dash", line_color=INDIGO,
                  annotation_text=f"Overall {all_churn:.1%}", annotation_position="top left")
    bar.update_layout(title=f"Actual churn rate by {dim}" + (" (selected highlighted)" if filtered else ""),
                      yaxis_tickformat=".0%", yaxis_title="Churn rate",
                      yaxis_range=[0, max(g["ChurnRate"].max(), all_churn) * 1.3])
    c1.plotly_chart(style(bar, 380), width="stretch")

    counts = view.RiskTier.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
    total = max(int(counts.sum()), 1)
    donut = go.Figure(go.Pie(
        labels=[f"{t} · {n:,} ({n / total:.1%})" for t, n in counts.items()], values=counts.values, hole=0.64,
        marker=dict(colors=[TIER_COLOR[t] for t in counts.index], line=dict(color="rgba(0,0,0,0)", width=2)),
        sort=False, textinfo="none", hovertemplate="%{label}<extra></extra>"))
    donut.add_annotation(text=f"<b>{v_high:.0%}</b><br>high risk", showarrow=False, font=dict(size=20))
    donut.update_layout(title="Risk tier distribution" + (f" - {seg_label}" if filtered else ""),
                        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
    c2.plotly_chart(style(donut, 380, legend=True), width="stretch")
    c2.caption(f"High ≥ {thr:.2f} · Medium ≥ {MED_THR:.2f} · Low below that (model risk score)")

    section("Top 20 customers to contact first", "Highest risk scores within the current selection.")
    tbl = view.sort_values("ChurnProbability", ascending=False).head(20).copy()
    tbl["Risk score"] = tbl["ChurnProbability"] * 100
    tbl["Actually churned"] = tbl[TARGET].map({1: "Yes", 0: "No"})
    show = ([ID] if ID else []) + ["Risk score", "RiskTier", "Actually churned"] + NUM[:5]
    st.dataframe(tbl[show], width="stretch", hide_index=True, column_config={
        "Risk score": st.column_config.ProgressColumn("Risk score", format="%.0f%%", min_value=0, max_value=100)})

    with st.expander("📋 Segment breakdown table (all segments)"):
        st.dataframe(g.drop(columns="Segment").rename(columns={"ChurnRate": "Actual churn rate", "AvgRisk": "Avg risk score",
                     "HighRiskShare": "High-risk share"}), width="stretch", hide_index=True, column_config={
            "Actual churn rate": st.column_config.NumberColumn(format="percent"),
            "Avg risk score": st.column_config.NumberColumn(format="percent"),
            "High-risk share": st.column_config.NumberColumn(format="percent")})

# ------------------------------------------------------------ tab 2 : single customer
with tab2:
    section("Customer details", "The form is generated from your dataset's columns (defaults = a typical customer).")
    vals = {}
    with st.expander("Edit customer attributes", expanded=True):
        grid = st.columns(3)
        for i, c in enumerate(NUM):
            info = schema["numeric_info"][c]
            with grid[i % 3]:
                if info["options"]:
                    opts_ = [int(o) if info["is_int"] else o for o in info["options"]]
                    med = min(opts_, key=lambda o: abs(o - info["median"]))
                    vals[c] = st.selectbox(c, opts_, index=opts_.index(med), key=f"f_{c}")
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
    color = TIER_COLOR[res.RiskTier]
    emoji = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[res.RiskTier]

    section("Prediction")
    st.markdown(f'<div class="risk-banner" style="border-left:6px solid {color};background:{color}1F;">'
                f'{emoji} <b style="color:{color}">{res.RiskTier.upper()} RISK</b> &nbsp;·&nbsp; risk score <b>{p:.0%}</b></div>',
                unsafe_allow_html=True)
    g1, g2 = st.columns([1, 1.4])
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 44}},
            title={"text": "Risk score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.28},
                   "steps": [{"range": [0, MED_THR * 100], "color": "rgba(16,185,129,.25)"},
                             {"range": [MED_THR * 100, thr * 100], "color": "rgba(245,158,11,.25)"},
                             {"range": [thr * 100, 100], "color": "rgba(239,68,68,.25)"}],
                   "threshold": {"line": {"color": INDIGO, "width": 4}, "value": thr * 100}}))
        st.plotly_chart(style(fig, 320), width="stretch")
    with g2:
        X1 = prepare(row, schema, cols)
        sv, _ = shap_values(pipe, art["explainer"], X1)
        contrib = pd.Series(sv[0], index=cols).sort_values(key=np.abs, ascending=False).head(8)[::-1]
        drv = px.bar(x=contrib.values, y=contrib.index, orientation="h", color=contrib.values > 0,
                     color_discrete_map={True: RED, False: GREEN},
                     title="Why? Top drivers (red = pushes toward churn)",
                     labels={"x": "SHAP impact (log-odds)", "y": ""})
        st.plotly_chart(style(drv, 320), width="stretch")

    section("Suggested retention actions")
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

# ------------------------------------------------------------ tab 3 : batch
with tab3:
    section("Score many customers at once", f"Upload a CSV with the same feature columns as your training data ('{TARGET}' is optional).")
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
            out["Risk score"] = out["ChurnProbability"] * 100
            vc = out.RiskTier.value_counts()
            b1, b2, b3 = st.columns(3)
            b1.metric("🔴 High risk", f"{vc.get('High', 0):,}")
            b2.metric("🟠 Medium risk", f"{vc.get('Medium', 0):,}")
            b3.metric("🟢 Low risk", f"{vc.get('Low', 0):,}")
            st.dataframe(out.drop(columns="ChurnProbability"), width="stretch", hide_index=True, column_config={
                "Risk score": st.column_config.ProgressColumn("Risk score", format="%.0f%%", min_value=0, max_value=100)})
            st.download_button("⬇️ Download scored CSV", out.drop(columns="Risk score").to_csv(index=False).encode(),
                               "scored_customers.csv", "text/csv")

# ------------------------------------------------------------ tab 4 : report
with tab4:
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    section("Model performance", f"Trained on {m['dataset']} ({m['n_rows']:,} rows, churn rate {m['churn_rate']:.1%}) · best model: {m['best_model']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['test_roc_auc']:.3f}"); c2.metric("F1", f"{m['test_f1']:.3f}")
    c3.metric("Recall", f"{m['test_recall']:.3f}"); c4.metric("Precision", f"{m['test_precision']:.3f}")
    st.dataframe(pd.read_csv(ROOT / "reports" / "model_comparison.csv"), width="stretch", hide_index=True)
    imgs = [("04_roc_curves.png", "ROC curves"), ("05_confusion_matrix.png", "Confusion matrix"),
            ("06_shap_summary.png", "SHAP summary"), ("03_churn_by_category.png", "Churn by category")]
    imgs = [(i, c) for i, c in imgs if (ROOT / "reports" / i).exists()]
    for k in range(0, len(imgs), 2):
        for col, (img, cap) in zip(st.columns(2), imgs[k:k + 2]):
            col.image(str(ROOT / "reports" / img), caption=cap)