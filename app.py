"""
Telco Customer Churn Intelligence Dashboard
-------------------------------------------
Streamlit + Plotly + scikit-learn
Run:  streamlit run app.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "telco_churn.csv"

RED, TEAL, PURPLE, AMBER, BLUE = "#ff4b6e", "#2dd4bf", "#7c5cff", "#fbbf24", "#38bdf8"
CMAP = {"Yes": RED, "No": TEAL}
SCALE = [TEAL, AMBER, RED]

ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
PRETTY = {
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
}
CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["tenure", "MonthlyCharges"]
FEATURES = CAT_FEATURES + NUM_FEATURES

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px;}

.hero {
    background: linear-gradient(120deg, #1e1b4b 0%, #312e81 45%, #7c3aed 100%);
    border-radius: 20px; padding: 26px 34px; margin-bottom: 18px;
    box-shadow: 0 10px 40px rgba(124, 58, 237, .28);
}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; letter-spacing: -.5px;}
.hero p {margin: 6px 0 0; color: #c7d2fe; font-size: 1rem;}
.chip {display: inline-block; padding: 4px 13px; border-radius: 999px;
       background: rgba(255,255,255,.15); color: #fff; font-size: .8rem;
       margin: 14px 8px 0 0;}

.kpi {
    position: relative; background: linear-gradient(145deg, #161a2e, #1c2140);
    border: 1px solid rgba(255,255,255,.07); border-left: 4px solid var(--accent);
    border-radius: 16px; padding: 16px 18px 14px; margin-bottom: 14px;
    transition: transform .2s ease, box-shadow .2s ease;
}
.kpi:hover {transform: translateY(-4px); box-shadow: 0 14px 32px rgba(0,0,0,.5);}
.kpi-icon {position: absolute; right: 16px; top: 12px; font-size: 1.6rem; opacity: .9;}
.kpi-label {font-size: .74rem; text-transform: uppercase; letter-spacing: .09em; color: #9aa3c7;}
.kpi-value {font-size: 2rem; font-weight: 700; color: #fff; line-height: 1.2; margin-top: 4px;}
.kpi-sub {font-size: .8rem; color: #9aa3c7; margin-top: 2px;}
.up {color: #ff7a93; font-weight: 600;}
.down {color: #5eead4; font-weight: 600;}

.insight {
    background: #141830; border: 1px solid rgba(255,255,255,.07);
    border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; height: 100%;
}
.insight .t {font-weight: 700; color: #fff; margin-bottom: 4px;}
.insight .b {color: #b6bedc; font-size: .93rem; line-height: 1.45;}
.action {border-left: 4px solid #7c5cff;}

.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background: #141830; border-radius: 10px; padding: 8px 18px; height: auto;
}
.stTabs [aria-selected="true"] {background: #7c3aed !important; color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,.06);}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def show(fig):
    """Render a plotly figure full width (works on old and new Streamlit)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(data, **kwargs):
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def style(fig, h=350):
    fig.update_layout(
        template="plotly_dark",
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        title=dict(font=dict(size=16)),
        legend=dict(orientation="h", y=-0.18, title=None),
    )
    return fig


def card(label, value, sub="", accent=PURPLE, icon=""):
    return (
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def insight(title, body, action=False):
    cls = "insight action" if action else "insight"
    return f'<div class="{cls}"><div class="t">{title}</div><div class="b">{body}</div></div>'


def money(x):
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.1f}K"
    return f"${x:,.0f}"


def rate_by(d, col):
    g = (
        d.groupby(col, observed=True)
        .agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum"))
        .reset_index()
    )
    g["rate"] = g["churned"] / g["customers"] * 100
    return g


def rate_bar(d, col, title, horizontal=False, height=340):
    g = rate_by(d, col)
    g[col] = g[col].astype(str)
    if horizontal:
        g = g.sort_values("rate")
    g["label"] = g["rate"].map("{:.1f}%".format)
    top = max(50.0, g["rate"].max() * 1.2)
    kw = dict(x="rate", y=col, orientation="h") if horizontal else dict(x=col, y="rate")
    fig = px.bar(
        g, text="label", color="rate", color_continuous_scale=SCALE,
        range_color=[0, top], title=title,
        hover_data={"customers": True, "rate": ":.1f"}, **kw,
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    if horizontal:
        fig.update_xaxes(range=[0, top], title="Churn rate (%)")
        fig.update_yaxes(title=None)
    else:
        fig.update_yaxes(range=[0, top], title="Churn rate (%)")
        fig.update_xaxes(title=None)
    fig.update_layout(coloraxis_showscale=False)
    return style(fig, height)


def top_group(d, col, min_n=30):
    g = rate_by(d, col)
    g = g[g["customers"] >= min_n]
    if g.empty:
        return None
    return g.sort_values("rate", ascending=False).iloc[0]


# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["PaymentMethod"] = df["PaymentMethod"].str.replace(" (automatic)", " (auto)", regex=False)
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )
    df["ChargeTier"] = pd.cut(
        df["MonthlyCharges"], bins=[0, 35, 70, 90, 200],
        labels=["Low (<$35)", "Mid ($35-70)", "High ($70-90)", "Premium ($90+)"],
    )
    df["NumAddons"] = sum((df[a] == "Yes").astype(int) for a in ADDONS)
    return df


@st.cache_resource(show_spinner="Training churn model...")
def train_model():
    df = load_data()
    X, y = df[FEATURES], df["ChurnFlag"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, max_depth=8, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)

    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.DataFrame(
        {
            "feature": [n.split("__", 1)[1].split("_")[0] for n in names],
            "importance": pipe.named_steps["clf"].feature_importances_,
        }
    )
    imp = imp.groupby("feature", as_index=False)["importance"].sum()
    imp = imp.sort_values("importance", ascending=True).tail(12)

    metrics = {
        "auc": roc_auc_score(y_te, proba),
        "acc": accuracy_score(y_te, pred),
        "prec": precision_score(y_te, pred),
        "rec": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
    }
    return pipe, metrics, (fpr, tpr), confusion_matrix(y_te, pred), imp


df = load_data()
OVERALL_RATE = df["ChurnFlag"].mean() * 100

# ----------------------------------------------------------------------------
# Sidebar: slicers / filters
# ----------------------------------------------------------------------------
def _opts(col):
    return sorted(df[col].astype(str).unique().tolist())


MULTI_FILTERS = {
    "f_contract": ("Contract", "Contract type"),
    "f_internet": ("InternetService", "Internet service"),
    "f_payment": ("PaymentMethod", "Payment method"),
    "f_gender": ("gender", "Gender"),
    "f_senior": ("SeniorCitizen", "Senior citizen"),
    "f_partner": ("Partner", "Has partner"),
    "f_dependents": ("Dependents", "Has dependents"),
}
T_MIN, T_MAX = int(df["tenure"].min()), int(df["tenure"].max())
C_MIN, C_MAX = int(math.floor(df["MonthlyCharges"].min())), int(math.ceil(df["MonthlyCharges"].max()))

DEFAULTS = {k: _opts(col) for k, (col, _) in MULTI_FILTERS.items()}
DEFAULTS["f_tenure"] = (T_MIN, T_MAX)
DEFAULTS["f_charge"] = (C_MIN, C_MAX)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("## 🎛️ Filters")
    st.caption("Slice the data - every chart and KPI updates live.")
    st.button("↺ Reset all filters", on_click=reset_filters)

    st.markdown("**Account**")
    st.multiselect("Contract type", _opts("Contract"), key="f_contract")
    st.multiselect("Internet service", _opts("InternetService"), key="f_internet")
    st.multiselect("Payment method", _opts("PaymentMethod"), key="f_payment")
    st.slider("Tenure (months)", T_MIN, T_MAX, key="f_tenure")
    st.slider("Monthly charges ($)", C_MIN, C_MAX, key="f_charge")

    with st.expander("👥 Demographics"):
        st.multiselect("Gender", _opts("gender"), key="f_gender")
        st.multiselect("Senior citizen", _opts("SeniorCitizen"), key="f_senior")
        st.multiselect("Has partner", _opts("Partner"), key="f_partner")
        st.multiselect("Has dependents", _opts("Dependents"), key="f_dependents")

    st.markdown("---")
    st.caption("Data: IBM Telco Customer Churn sample · 7,043 customers")

ss = st.session_state
mask = (
    df["Contract"].isin(ss.f_contract)
    & df["InternetService"].isin(ss.f_internet)
    & df["PaymentMethod"].isin(ss.f_payment)
    & df["gender"].isin(ss.f_gender)
    & df["SeniorCitizen"].isin(ss.f_senior)
    & df["Partner"].isin(ss.f_partner)
    & df["Dependents"].isin(ss.f_dependents)
    & df["tenure"].between(*ss.f_tenure)
    & df["MonthlyCharges"].between(*ss.f_charge)
)
d = df[mask].copy()

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <h1>📉 Telco Customer Churn Intelligence</h1>
  <p>Find out who is leaving, why they leave, and how much revenue is at stake.</p>
  <span class="chip">👥 {len(d):,} of {len(df):,} customers in view</span>
  <span class="chip">🤖 ML churn predictor included</span>
  <span class="chip">⚡ Live filters</span>
</div>
""",
    unsafe_allow_html=True,
)

if d.empty:
    st.warning("No customers match the current filters. Click **Reset all filters** in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------
n = len(d)
churned = int(d["ChurnFlag"].sum())
rate = churned / n * 100
mrr = d["MonthlyCharges"].sum()
risk_mrr = d.loc[d["ChurnFlag"] == 1, "MonthlyCharges"].sum()
diff = rate - OVERALL_RATE
if abs(diff) < 0.05:
    delta_html = '<span class="down">= same as overall</span>'
elif diff > 0:
    delta_html = f'<span class="up">▲ {diff:+.1f} pts vs overall</span>'
else:
    delta_html = f'<span class="down">▼ {diff:+.1f} pts vs overall</span>'

r1 = st.columns(4)
r1[0].markdown(card("Total customers", f"{n:,}", f"{n / len(df) * 100:.0f}% of all customers", PURPLE, "👥"), unsafe_allow_html=True)
r1[1].markdown(card("Churned customers", f"{churned:,}", f"{n - churned:,} still active", RED, "🚪"), unsafe_allow_html=True)
r1[2].markdown(card("Churn rate", f"{rate:.1f}%", delta_html, AMBER, "📉"), unsafe_allow_html=True)
r1[3].markdown(card("Monthly revenue at risk", money(risk_mrr), f"{risk_mrr / mrr * 100:.1f}% of {money(mrr)} MRR", RED, "💸"), unsafe_allow_html=True)

r2 = st.columns(4)
r2[0].markdown(card("Avg monthly charge", f"${d['MonthlyCharges'].mean():.2f}", f"churned: ${d.loc[d.ChurnFlag == 1, 'MonthlyCharges'].mean():.2f}", BLUE, "💳"), unsafe_allow_html=True)
r2[1].markdown(card("Avg tenure", f"{d['tenure'].mean():.1f} mo", f"churned: {d.loc[d.ChurnFlag == 1, 'tenure'].mean():.1f} mo", TEAL, "⏳"), unsafe_allow_html=True)
r2[2].markdown(card("Avg lifetime value", money(d["TotalCharges"].mean()), f"total billed: {money(d['TotalCharges'].sum())}", PURPLE, "💎"), unsafe_allow_html=True)
r2[3].markdown(card("Annual revenue lost", money(risk_mrr * 12), "if churned customers stay gone", RED, "🔥"), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_over, tab_deep, tab_ml, tab_ins, tab_data = st.tabs(
    ["📊 Overview", "🔍 Deep Dive", "🤖 Churn Predictor", "💡 Insights", "🗂️ Data"]
)

# ============================== OVERVIEW =====================================
with tab_over:
    c1, c2, c3 = st.columns([1, 1.2, 1.2])
    with c1:
        fig = px.pie(d, names="Churn", hole=0.68, color="Churn", color_discrete_map=CMAP, title="Churn split")
        fig.update_traces(textinfo="percent+label", marker_line_width=0)
        fig.add_annotation(text=f"<b>{rate:.1f}%</b><br>churn", showarrow=False, font=dict(size=22))
        fig.update_layout(showlegend=False)
        show(style(fig, 360))
    with c2:
        show(rate_bar(d, "Contract", "Churn rate by contract type", height=360))
    with c3:
        show(rate_bar(d, "TenureGroup", "Churn rate by tenure", height=360))

    c4, c5 = st.columns(2)
    with c4:
        fig = px.histogram(
            d, x="tenure", color="Churn", nbins=36, barmode="overlay", opacity=0.8,
            color_discrete_map=CMAP, title="Tenure distribution: leavers churn early",
        )
        fig.update_xaxes(title="Tenure (months)")
        fig.update_yaxes(title="Customers")
        show(style(fig, 360))
    with c5:
        fig = px.violin(
            d, x="Churn", y="MonthlyCharges", color="Churn", box=True, points=False,
            color_discrete_map=CMAP, title="Monthly charges: churned vs retained",
        )
        fig.update_yaxes(title="Monthly charges ($)")
        fig.update_xaxes(title=None)
        fig.update_layout(showlegend=False)
        show(style(fig, 360))

    c6, c7, c8 = st.columns(3)
    with c6:
        show(rate_bar(d, "PaymentMethod", "Churn rate by payment method", horizontal=True))
    with c7:
        show(rate_bar(d, "InternetService", "Churn rate by internet service", horizontal=True))
    with c8:
        rev = (
            d[d["ChurnFlag"] == 1].groupby("Contract", observed=True)["MonthlyCharges"].sum().reset_index()
        )
        fig = px.bar(
            rev, x="Contract", y="MonthlyCharges", color="Contract", title="Monthly revenue lost by contract",
            color_discrete_sequence=[RED, AMBER, PURPLE], text=rev["MonthlyCharges"].map(money),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Revenue lost ($ / month)")
        fig.update_xaxes(title=None)
        show(style(fig, 340))

# ============================== DEEP DIVE ====================================
with tab_deep:
    c1, c2 = st.columns(2)
    with c1:
        piv = d.pivot_table(index="Contract", columns="InternetService", values="ChurnFlag", aggfunc="mean") * 100
        fig = px.imshow(
            piv, text_auto=".1f", aspect="auto", color_continuous_scale=SCALE, zmin=0, zmax=60,
            title="Churn % heatmap: contract × internet service",
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        show(style(fig, 360))
    with c2:
        sb = px.sunburst(
            d.assign(n=1), path=["Contract", "InternetService", "Churn"], values="n", color="Churn",
            color_discrete_map={**CMAP, "(?)": "#2a2f55"}, title="Customer journey: contract → internet → outcome",
        )
        show(style(sb, 360))

    # add-on services
    inet = d[d["InternetService"] != "No"]
    rows = []
    for s in ADDONS:
        for flag, lab in [("Yes", "With service"), ("No", "Without service")]:
            sub = inet[inet[s] == flag]
            if len(sub) >= 10:
                rows.append({"Service": PRETTY[s], "Has": lab, "rate": sub["ChurnFlag"].mean() * 100, "customers": len(sub)})
    c3, c4 = st.columns([1.5, 1])
    with c3:
        if rows:
            ad = pd.DataFrame(rows)
            fig = px.bar(
                ad, x="Service", y="rate", color="Has", barmode="group",
                color_discrete_map={"With service": TEAL, "Without service": RED},
                title="Do add-on services protect against churn? (internet customers)",
                hover_data={"customers": True, "rate": ":.1f"},
            )
            fig.update_yaxes(title="Churn rate (%)")
            fig.update_xaxes(title=None)
            show(style(fig, 380))
        else:
            st.info("Not enough internet customers in the current filter for this chart.")
    with c4:
        show(rate_bar(d, "NumAddons", "Churn rate by number of add-ons", height=380))

    # demographics
    demo_rows = []
    for col in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
        g = rate_by(d, col)
        for _, r in g.iterrows():
            demo_rows.append({"Feature": PRETTY.get(col, col), "Value": str(r[col]), "rate": r["rate"], "customers": r["customers"]})
    demo = pd.DataFrame(demo_rows)
    c5, c6 = st.columns([1.3, 1])
    with c5:
        fig = px.bar(
            demo, x="Value", y="rate", color="rate", facet_col="Feature", color_continuous_scale=SCALE,
            range_color=[0, 50], text=demo["rate"].map("{:.0f}%".format), title="Demographics: who churns?",
            hover_data={"customers": True, "rate": ":.1f"},
        )
        fig.update_xaxes(matches=None, title=None)
        fig.update_yaxes(title="Churn rate (%)", range=[0, 55])
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False)
        show(style(fig, 360))
    with c6:
        show(rate_bar(d, "ChargeTier", "Churn rate by price tier", height=360))

    sample = d.sample(min(len(d), 2500), random_state=1)
    fig = px.scatter(
        sample, x="tenure", y="MonthlyCharges", color="Churn", color_discrete_map=CMAP, opacity=0.6,
        hover_data=["Contract", "InternetService", "PaymentMethod"],
        title="Tenure vs monthly charges: the 'new + expensive' danger zone",
    )
    fig.update_xaxes(title="Tenure (months)")
    fig.update_yaxes(title="Monthly charges ($)")
    show(style(fig, 420))

# ============================== CHURN PREDICTOR ==============================
with tab_ml:
    model, m, (fpr, tpr), cm, imp = train_model()

    st.markdown("### 🤖 Model performance")
    st.caption("Random Forest trained on 80% of customers, evaluated on a held-out 20%.")
    mc = st.columns(5)
    mc[0].markdown(card("ROC-AUC", f"{m['auc']:.3f}", "ranking quality", PURPLE, "🎯"), unsafe_allow_html=True)
    mc[1].markdown(card("Accuracy", f"{m['acc'] * 100:.1f}%", "overall correct", BLUE, "✅"), unsafe_allow_html=True)
    mc[2].markdown(card("Precision", f"{m['prec'] * 100:.1f}%", "flagged who really churn", AMBER, "🔬"), unsafe_allow_html=True)
    mc[3].markdown(card("Recall", f"{m['rec'] * 100:.1f}%", "churners we catch", TEAL, "📡"), unsafe_allow_html=True)
    mc[4].markdown(card("F1 score", f"{m['f1']:.3f}", "precision / recall balance", RED, "⚖️"), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=PURPLE, width=3), fill="tozeroy", fillcolor="rgba(124,92,255,.15)"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#6b7280", dash="dash")))
        fig.update_layout(title=f"ROC curve (AUC {m['auc']:.2f})", xaxis_title="False positive rate", yaxis_title="True positive rate")
        show(style(fig, 340))
    with c2:
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale=["#161a2e", PURPLE],
            x=["Pred: stay", "Pred: churn"], y=["Actual: stay", "Actual: churn"], title="Confusion matrix",
        )
        fig.update_layout(coloraxis_showscale=False)
        show(style(fig, 340))
    with c3:
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="What drives churn? (feature importance)", color="importance", color_continuous_scale=["#2a2f55", PURPLE, RED])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="Importance")
        fig.update_yaxes(title=None)
        show(style(fig, 340))

    st.markdown("---")
    st.markdown("### 🔮 Predict churn risk for a customer")

    def pick(label, col, default=None, container=st):
        o = sorted(df[col].astype(str).unique().tolist())
        idx = o.index(default) if default in o else 0
        return container.selectbox(label, o, index=idx)

    with st.form("predict_form"):
        a, b, c = st.columns(3)
        gender = pick("Gender", "gender", container=a)
        senior = pick("Senior citizen", "SeniorCitizen", "No", a)
        partner = pick("Has partner", "Partner", "No", a)
        dependents = pick("Has dependents", "Dependents", "No", a)
        tenure_in = a.slider("Tenure (months)", 0, 72, 5)
        monthly_in = a.slider("Monthly charges ($)", 18, 120, 85)

        contract = pick("Contract", "Contract", "Month-to-month", b)
        payment = pick("Payment method", "PaymentMethod", "Electronic check", b)
        paperless = pick("Paperless billing", "PaperlessBilling", "Yes", b)
        phone = pick("Phone service", "PhoneService", "Yes", b)
        lines = pick("Multiple lines", "MultipleLines", "No", b)
        internet = pick("Internet service", "InternetService", "Fiber optic", b)

        sec = pick("Online security", "OnlineSecurity", "No", c)
        bak = pick("Online backup", "OnlineBackup", "No", c)
        dev = pick("Device protection", "DeviceProtection", "No", c)
        tech = pick("Tech support", "TechSupport", "No", c)
        stv = pick("Streaming TV", "StreamingTV", "No", c)
        smov = pick("Streaming movies", "StreamingMovies", "No", c)
        submitted = st.form_submit_button("🔮 Predict churn risk")

    if submitted:
        row = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
            "OnlineSecurity": sec, "OnlineBackup": bak, "DeviceProtection": dev, "TechSupport": tech,
            "StreamingTV": stv, "StreamingMovies": smov, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "tenure": tenure_in, "MonthlyCharges": float(monthly_in),
        }
        if internet == "No":
            for k in ADDONS:
                row[k] = "No internet service"
        if phone == "No":
            row["MultipleLines"] = "No phone service"
        p = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
        st.session_state["pred"] = (p, row)

    if "pred" in st.session_state:
        p, row = st.session_state["pred"]
        g1, g2 = st.columns([1, 1.2])
        with g1:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=p * 100,
                    number=dict(suffix="%", font=dict(size=46)),
                    title=dict(text="Churn probability"),
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color="#e5e7eb", thickness=0.25),
                        steps=[
                            dict(range=[0, 30], color="rgba(45,212,191,.55)"),
                            dict(range=[30, 60], color="rgba(251,191,36,.55)"),
                            dict(range=[60, 100], color="rgba(255,75,110,.6)"),
                        ],
                    ),
                )
            )
            show(style(fig, 330))
        with g2:
            level = "LOW" if p < 0.3 else "MEDIUM" if p < 0.6 else "HIGH"
            colr = TEAL if level == "LOW" else AMBER if level == "MEDIUM" else RED
            similar = df[(df["Contract"] == row["Contract"]) & (df["InternetService"] == row["InternetService"])]
            sim_rate = similar["ChurnFlag"].mean() * 100 if len(similar) else float("nan")
            st.markdown(
                card("Risk level", level, f"Similar customers (same contract + internet) churn at {sim_rate:.1f}%", colr, "🚨"),
                unsafe_allow_html=True,
            )
            recs = []
            if row["Contract"] == "Month-to-month":
                recs.append("Offer a discount for a 1-year contract (month-to-month is the biggest churn driver).")
            if row["InternetService"] != "No" and row["TechSupport"] == "No":
                recs.append("Bundle free Tech Support for 3 months; customers without it churn far more.")
            if row["PaymentMethod"] == "Electronic check":
                recs.append("Nudge to auto-pay (card / bank) with a small bill credit.")
            if row["InternetService"] == "Fiber optic" and row["MonthlyCharges"] >= 80:
                recs.append("Review price/quality: high-priced fiber customers are a flight risk.")
            if row["tenure"] <= 12:
                recs.append("Trigger the first-year onboarding and loyalty programme.")
            if not recs:
                recs.append("Customer looks stable. Keep service quality high and reward loyalty.")
            st.markdown("**Recommended retention actions**")
            for r in recs:
                st.markdown(f"- {r}")

    st.markdown("---")
    st.markdown("### 🎯 Retention hit-list: active customers most likely to leave")
    st.caption("Uses the filters in the sidebar, so you can target a specific segment.")
    active = d[d["Churn"] == "No"].copy()
    if active.empty:
        st.info("No active customers in the current filter.")
    else:
        active["Risk"] = model.predict_proba(active[FEATURES])[:, 1] * 100
        topn = st.slider("How many customers to show", 10, 100, 25, step=5)
        hit = active.sort_values("Risk", ascending=False).head(topn)[
            ["customerID", "Contract", "InternetService", "PaymentMethod", "tenure", "MonthlyCharges", "Risk"]
        ]
        show_df(
            hit,
            column_config={
                "Risk": st.column_config.ProgressColumn("Churn risk %", min_value=0, max_value=100, format="%.0f%%"),
                "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
                "tenure": st.column_config.NumberColumn("Tenure (mo)"),
            },
        )
        st.download_button("⬇️ Download hit-list (CSV)", hit.to_csv(index=False).encode(), "retention_hitlist.csv", "text/csv")

# ============================== INSIGHTS =====================================
with tab_ins:
    st.markdown("### 💡 Auto-generated insights (update with your filters)")
    cards = []

    r = top_group(d, "Contract")
    if r is not None and rate > 0:
        cards.append(insight(f"📄 {r['Contract']} contracts are the riskiest", f"They churn at <b>{r['rate']:.1f}%</b> - {r['rate'] / rate:.1f}x the average of {rate:.1f}% in this view ({int(r['customers']):,} customers)."))
    first = d[d["tenure"] <= 12]
    later = d[d["tenure"] > 12]
    if len(first) >= 30 and len(later) >= 30:
        cards.append(insight("⏳ The first year is critical", f"Customers in their first 12 months churn at <b>{first['ChurnFlag'].mean() * 100:.1f}%</b> versus <b>{later['ChurnFlag'].mean() * 100:.1f}%</b> after that."))
    r = top_group(d, "PaymentMethod")
    if r is not None:
        cards.append(insight(f"💳 {r['PaymentMethod']} has the highest churn", f"<b>{r['rate']:.1f}%</b> of these customers leave. Moving them to auto-pay is a cheap retention win."))
    r = top_group(d, "InternetService")
    if r is not None:
        cards.append(insight(f"🌐 {r['InternetService']} customers leave the most", f"Churn is <b>{r['rate']:.1f}%</b> across {int(r['customers']):,} customers. Investigate pricing and service quality."))
    inet2 = d[d["InternetService"] != "No"]
    w, wo = inet2[inet2["TechSupport"] == "Yes"], inet2[inet2["TechSupport"] == "No"]
    if len(w) >= 30 and len(wo) >= 30:
        cards.append(insight("🛟 Tech Support keeps customers", f"Internet customers <b>with</b> tech support churn at <b>{w['ChurnFlag'].mean() * 100:.1f}%</b> vs <b>{wo['ChurnFlag'].mean() * 100:.1f}%</b> without it."))
    cards.append(insight("💸 Revenue at stake", f"Churned customers represent <b>{money(risk_mrr)}</b> per month, about <b>{money(risk_mrr * 12)}</b> per year."))

    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, ctn in zip(cols, cards[i : i + 3]):
            col.markdown(ctn, unsafe_allow_html=True)

    st.markdown("### ✅ Recommended actions")
    acts = [
        ("1. Lock in month-to-month users", "Offer 1-2 year contract incentives (discount, free upgrade). Biggest single lever on churn."),
        ("2. Fix the first 90 days", "Onboarding calls, setup help and early check-ins for customers with tenure under 12 months."),
        ("3. Push auto-pay", "Small bill credit for switching from electronic check to card / bank auto-pay."),
        ("4. Bundle protection services", "Tech Support, Security and Backup customers stay longer. Bundle them into plans."),
        ("5. Review fiber pricing & quality", "High-priced fiber customers churn heavily; check outages, speed and price competitiveness."),
        ("6. Work the hit-list weekly", "Use the Churn Predictor hit-list so retention teams call the highest-risk customers first."),
    ]
    for i in range(0, len(acts), 3):
        cols = st.columns(3)
        for col, (t, b) in zip(cols, acts[i : i + 3]):
            col.markdown(insight(t, b, action=True), unsafe_allow_html=True)

# ============================== DATA =========================================
with tab_data:
    st.markdown("### 🗂️ Filtered data")
    st.caption(f"{len(d):,} rows · {d.shape[1]} columns")
    show_df(d.drop(columns=["ChurnFlag"]))
    st.download_button("⬇️ Download filtered data (CSV)", d.drop(columns=["ChurnFlag"]).to_csv(index=False).encode(), "telco_filtered.csv", "text/csv")
    with st.expander("Data notes"):
        st.markdown(
            "- Source: IBM Telco Customer Churn sample dataset.\n"
            "- 11 customers with blank `TotalCharges` (tenure = 0) were set to 0.\n"
            "- `SeniorCitizen` was converted from 0/1 to No/Yes.\n"
            "- Engineered columns: `TenureGroup`, `ChargeTier`, `NumAddons`, `ChurnFlag`."
        )
