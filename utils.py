"""Shared helpers for the Telco Churn Intelligence dashboard (data, styling, filters, charts, ML)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).parent

RED, TEAL, PURPLE, AMBER, BLUE, PINK = "#ff4b6e", "#2dd4bf", "#7c5cff", "#fbbf24", "#38bdf8", "#f472b6"
CMAP = {"Yes": RED, "No": TEAL}
SCALE = [TEAL, AMBER, RED]

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PRETTY = {
    "OnlineSecurity": "Online Security", "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection", "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies",
    "gender": "Gender", "SeniorCitizen": "Senior", "Partner": "Partner", "Dependents": "Dependents",
}
CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["tenure", "MonthlyCharges"]
FEATURES = CAT_FEATURES + NUM_FEATURES

# ----------------------------------------------------------------------------
# CSS (responsive: desktop full-width, mobile friendly)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.block-container, .hero, .kpi, .insight, .sec {font-family: 'Inter', system-ui, sans-serif;}
#MainMenu, footer, [data-testid="stAppDeployButton"] {display: none !important;}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(1100px 500px at 8% -8%, rgba(124,58,237,.20), transparent 60%),
                radial-gradient(900px 400px at 100% 0%, rgba(56,189,248,.10), transparent 60%), #0b0e1a;
}
.block-container {max-width: 100% !important; padding: 3.4rem 2.2rem 3rem !important;}

.hero {
    background: linear-gradient(120deg, #1e1b4b 0%, #312e81 45%, #7c3aed 100%);
    border-radius: 20px; padding: 22px 30px; margin-bottom: 14px;
    box-shadow: 0 10px 40px rgba(124,58,237,.28);
}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff; letter-spacing: -.5px; font-weight: 800;}
.hero p {margin: 4px 0 0; color: #c7d2fe; font-size: .98rem;}
.chip {display: inline-block; padding: 4px 13px; border-radius: 999px; background: rgba(255,255,255,.15);
       color: #fff; font-size: .78rem; margin: 12px 8px 0 0;}

.kpi-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 14px; margin: 6px 0 18px;}
.kpi {
    position: relative; background: linear-gradient(145deg, #161a2e, #1c2140);
    border: 1px solid rgba(255,255,255,.07); border-left: 4px solid var(--accent);
    border-radius: 16px; padding: 15px 18px 13px; transition: transform .2s ease, box-shadow .2s ease;
}
.kpi:hover {transform: translateY(-4px); box-shadow: 0 14px 32px rgba(0,0,0,.5);}
.kpi-icon {position: absolute; right: 14px; top: 10px; font-size: 1.5rem; opacity: .9;}
.kpi-label {font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; color: #9aa3c7;}
.kpi-value {font-size: 1.9rem; font-weight: 800; color: #fff; line-height: 1.2; margin-top: 4px;}
.kpi-sub {font-size: .78rem; color: #9aa3c7; margin-top: 2px;}
.up {color: #ff7a93; font-weight: 600;} .down {color: #5eead4; font-weight: 600;}

.insight-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; margin-bottom: 14px;}
.insight {background: #141830; border: 1px solid rgba(255,255,255,.07); border-radius: 14px; padding: 15px 18px;}
.insight .t {font-weight: 700; color: #fff; margin-bottom: 4px;}
.insight .b {color: #b6bedc; font-size: .92rem; line-height: 1.45;}
.action {border-left: 4px solid #7c5cff;}

.sec {font-size: 1.05rem; font-weight: 700; color: #e5e7eb; margin: 20px 0 8px; padding-left: 12px;
      border-left: 4px solid #7c5cff;}
.sec small {color: #9aa3c7; font-weight: 400; margin-left: 8px; font-size: .82rem;}

div[data-testid="stPlotlyChart"] {
    background: rgba(20,24,48,.55); border: 1px solid rgba(255,255,255,.06);
    border-radius: 16px; padding: 4px; overflow: hidden;
}
div[data-testid="stExpander"] {background: rgba(20,24,48,.55); border-radius: 14px;}

@media (max-width: 640px) {
    .block-container {padding: 3.2rem .7rem 2.5rem !important;}
    .hero {padding: 14px 16px; border-radius: 14px;}
    .hero h1 {font-size: 1.3rem;} .hero p {font-size: .85rem;}
    .chip {font-size: .7rem; padding: 3px 10px; margin: 8px 6px 0 0;}
    .kpi-grid {grid-template-columns: repeat(2, 1fr); gap: 8px;}
    .kpi {padding: 10px 12px;} .kpi-value {font-size: 1.3rem;} .kpi-icon {display: none;}
    .kpi-label {font-size: .64rem;} .kpi-sub {font-size: .68rem;}
    .insight-grid {grid-template-columns: 1fr;}
    .sec {font-size: .95rem;}
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def show(fig):
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def show_df(data, **kwargs):
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def style(fig, h=340):
    fig.update_layout(
        template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10), title=dict(font=dict(size=15)),
        legend=dict(orientation="h", y=-0.2, title=None), font=dict(size=12),
    )
    return fig


def kpi(label, value, sub="", accent=PURPLE, icon=""):
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_grid(cards):
    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def insight(title, body, action=False):
    cls = "insight action" if action else "insight"
    return f'<div class="{cls}"><div class="t">{title}</div><div class="b">{body}</div></div>'


def insight_grid(items):
    st.markdown('<div class="insight-grid">' + "".join(items) + "</div>", unsafe_allow_html=True)


def section(text, sub=""):
    small = f"<small>{sub}</small>" if sub else ""
    st.markdown(f'<div class="sec">{text}{small}</div>', unsafe_allow_html=True)


def money(x):
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.1f}K"
    return f"${x:,.0f}"


def delta_html(rate, overall):
    diff = rate - overall
    if abs(diff) < 0.05:
        return '<span class="down">= same as overall</span>'
    if diff > 0:
        return f'<span class="up">▲ {diff:+.1f} pts vs overall</span>'
    return f'<span class="down">▼ {diff:+.1f} pts vs overall</span>'


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def find_data_file():
    exact = BASE_DIR / "data" / "telco_churn.csv"
    if exact.exists():
        return exact
    for p in sorted(BASE_DIR.rglob("*.csv")):
        if "churn" in p.name.lower() or "telco" in p.name.lower():
            return p
    return None


@st.cache_data
def load_data():
    path = find_data_file()
    if path is None:
        st.error("Dataset not found. Put the churn CSV in `data/telco_churn.csv` (next to `app.py`).")
        st.stop()
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["PaymentMethod"] = df["PaymentMethod"].str.replace(" (automatic)", " (auto)", regex=False)
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                               labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
    df["ChargeTier"] = pd.cut(df["MonthlyCharges"], bins=[0, 35, 70, 90, 200],
                              labels=["Low (<$35)", "Mid ($35-70)", "High ($70-90)", "Premium ($90+)"])
    df["NumAddons"] = sum((df[a] == "Yes").astype(int) for a in ADDONS)
    df["Household"] = np.select(
        [(df.Partner == "Yes") & (df.Dependents == "Yes"), df.Partner == "Yes", df.Dependents == "Yes"],
        ["Family", "Couple", "Single parent"], default="Single")
    prot = df[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]].eq("Yes").any(axis=1)
    df["Protection"] = np.select([df.InternetService == "No", prot],
                                 ["No internet", "Has protection"], default="No protection")
    return df


def get_d():
    """Filtered dataframe shared across pages."""
    return st.session_state.get("d", load_data())


# ----------------------------------------------------------------------------
# Filters (quick segments + dropdowns + sliders). Rendered once in app.py.
# ----------------------------------------------------------------------------
PRESETS = {
    "🌐 All customers": lambda x: pd.Series(True, index=x.index),
    "🆕 New (≤12 mo)": lambda x: x.tenure <= 12,
    "🏆 Loyal (>48 mo)": lambda x: x.tenure > 48,
    "💎 High value ($90+)": lambda x: x.MonthlyCharges >= 90,
    "📄 Month-to-month": lambda x: x.Contract == "Month-to-month",
    "🚨 High-risk profile": lambda x: (x.Contract == "Month-to-month") & (x.InternetService == "Fiber optic") & (x.PaymentMethod == "Electronic check"),
    "👴 Seniors": lambda x: x.SeniorCitizen == "Yes",
}
MULTI = {
    "f_contract": ("Contract", "Contract"),
    "f_internet": ("InternetService", "Internet service"),
    "f_payment": ("PaymentMethod", "Payment method"),
    "f_gender": ("gender", "Gender"),
    "f_senior": ("SeniorCitizen", "Senior citizen"),
    "f_partner": ("Partner", "Has partner"),
    "f_dependents": ("Dependents", "Has dependents"),
    "f_paperless": ("PaperlessBilling", "Paperless billing"),
}


def _opts(df, col):
    return sorted(df[col].astype(str).unique().tolist())


def render_filters(df):
    first = list(PRESETS)[0]
    t_min, t_max = int(df.tenure.min()), int(df.tenure.max())
    c_min, c_max = int(math.floor(df.MonthlyCharges.min())), int(math.ceil(df.MonthlyCharges.max()))
    defaults = {k: _opts(df, col) for k, (col, _) in MULTI.items()}
    defaults["f_tenure"] = (t_min, t_max)
    defaults["f_charge"] = (c_min, c_max)
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    def reset():
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state["preset"] = first

    active = 0
    for k, v in defaults.items():
        cur = st.session_state.get(k)
        if isinstance(v, list):
            active += set(cur) != set(v)
        else:
            active += tuple(cur) != tuple(v)

    st.session_state.setdefault("preset", first)
    if hasattr(st, "pills"):
        preset = st.pills("Quick segment", list(PRESETS), selection_mode="single",
                          key="preset", label_visibility="collapsed") or first
    else:
        preset = st.selectbox("Quick segment", list(PRESETS), key="preset")
    active += preset != first

    label = "🎛️ Filters & slicers" + (f"  ·  {active} active" if active else "")
    with st.expander(label):
        cols = st.columns(4)
        for i, (k, (col, name)) in enumerate(MULTI.items()):
            cols[i % 4].multiselect(name, _opts(df, col), key=k)
        s1, s2, s3 = st.columns([2, 2, 1])
        s1.slider("Tenure (months)", t_min, t_max, key="f_tenure")
        s2.slider("Monthly charges ($)", c_min, c_max, key="f_charge")
        s3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        s3.button("↺ Reset all", on_click=reset)

    ss = st.session_state
    mask = PRESETS[preset](df)
    for k, (col, _) in MULTI.items():
        mask &= df[col].astype(str).isin(ss[k])
    mask &= df.tenure.between(*ss.f_tenure) & df.MonthlyCharges.between(*ss.f_charge)
    d = df[mask].copy()
    ss["d"], ss["n_active"] = d, active
    return d


def page_header(icon, title, subtitle):
    d, df = get_d(), load_data()
    rate = d.ChurnFlag.mean() * 100 if len(d) else 0
    n_act = st.session_state.get("n_active", 0)
    seg = st.session_state.get("preset") or list(PRESETS)[0]
    st.markdown(
        f'<div class="hero"><h1>{icon} {title}</h1><p>{subtitle}</p>'
        f'<span class="chip">👥 {len(d):,} / {len(df):,} customers</span>'
        f'<span class="chip">📉 {rate:.1f}% churn</span>'
        f'<span class="chip">🎯 {seg}</span>'
        f'<span class="chip">🎛️ {n_act} filters active</span></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------------------------
def rate_by(d, col):
    g = (d.groupby(col, observed=True)
         .agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum")).reset_index())
    g["rate"] = g["churned"] / g["customers"] * 100
    return g


def rate_bar(d, col, title, horizontal=False, height=330):
    g = rate_by(d, col)
    g[col] = g[col].astype(str)
    if horizontal:
        g = g.sort_values("rate")
    g["label"] = g["rate"].map("{:.1f}%".format)
    top = max(50.0, g["rate"].max() * 1.2)
    kw = dict(x="rate", y=col, orientation="h") if horizontal else dict(x=col, y="rate")
    fig = px.bar(g, text="label", color="rate", color_continuous_scale=SCALE, range_color=[0, top],
                 title=title, hover_data={"customers": True, "rate": ":.1f"}, **kw)
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    if horizontal:
        fig.update_xaxes(range=[0, top], title="Churn rate (%)")
        fig.update_yaxes(title=None)
    else:
        fig.update_yaxes(range=[0, top], title="Churn rate (%)")
        fig.update_xaxes(title=None)
    fig.update_layout(coloraxis_showscale=False)
    return style(fig, height)


def heat(d, row, col, title, h=330, zmax=60):
    piv = d.pivot_table(index=row, columns=col, values="ChurnFlag", aggfunc="mean", observed=True) * 100
    piv.index, piv.columns = piv.index.astype(str), piv.columns.astype(str)
    fig = px.imshow(piv, text_auto=".1f", aspect="auto", color_continuous_scale=SCALE, zmin=0, zmax=zmax, title=title)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    return style(fig, h)


def donut(d, title="Churn split", h=330):
    rate = d.ChurnFlag.mean() * 100
    fig = px.pie(d, names="Churn", hole=0.68, color="Churn", color_discrete_map=CMAP, title=title)
    fig.update_traces(textinfo="percent+label", marker_line_width=0)
    fig.add_annotation(text=f"<b>{rate:.1f}%</b><br>churn", showarrow=False, font=dict(size=22))
    fig.update_layout(showlegend=False)
    return style(fig, h)


def top_group(d, col, min_n=30):
    g = rate_by(d, col)
    g = g[g["customers"] >= min_n]
    return None if g.empty else g.sort_values("rate", ascending=False).iloc[0]


def km_curve(d, max_t=72):
    """Kaplan-Meier retention curve (tenure = time, churn = event)."""
    t = np.arange(0, max_t + 1)
    ev = d.groupby("tenure")["ChurnFlag"].sum().reindex(t, fill_value=0)
    cnt = d.groupby("tenure").size().reindex(t, fill_value=0)
    at_risk = cnt[::-1].cumsum()[::-1]
    h = (ev / at_risk.replace(0, np.nan)).fillna(0)
    return (1 - h).cumprod()


def auto_insights(d):
    rate = d.ChurnFlag.mean() * 100
    risk = d.loc[d.ChurnFlag == 1, "MonthlyCharges"].sum()
    cards = []
    r = top_group(d, "Contract")
    if r is not None and rate > 0:
        cards.append(insight(f"📄 {r['Contract']} is the riskiest contract",
                             f"Churns at <b>{r['rate']:.1f}%</b>, {r['rate'] / rate:.1f}x the average of {rate:.1f}% ({int(r['customers']):,} customers)."))
    first, later = d[d.tenure <= 12], d[d.tenure > 12]
    if len(first) >= 30 and len(later) >= 30:
        cards.append(insight("⏳ The first year is critical",
                             f"First-year customers churn at <b>{first.ChurnFlag.mean() * 100:.1f}%</b> vs <b>{later.ChurnFlag.mean() * 100:.1f}%</b> afterwards."))
    r = top_group(d, "PaymentMethod")
    if r is not None:
        cards.append(insight(f"💳 {r['PaymentMethod']} churns most",
                             f"<b>{r['rate']:.1f}%</b> of these customers leave. Moving them to auto-pay is a cheap retention win."))
    r = top_group(d, "InternetService")
    if r is not None:
        cards.append(insight(f"🌐 {r['InternetService']} customers leave the most",
                             f"Churn is <b>{r['rate']:.1f}%</b> across {int(r['customers']):,} customers. Check pricing and service quality."))
    net = d[d.InternetService != "No"]
    w, wo = net[net.TechSupport == "Yes"], net[net.TechSupport == "No"]
    if len(w) >= 30 and len(wo) >= 30:
        cards.append(insight("🛟 Tech Support keeps customers",
                             f"With tech support: <b>{w.ChurnFlag.mean() * 100:.1f}%</b> churn. Without: <b>{wo.ChurnFlag.mean() * 100:.1f}%</b>."))
    cards.append(insight("💸 Revenue at stake",
                         f"Churned customers represent <b>{money(risk)}</b> per month, about <b>{money(risk * 12)}</b> per year."))
    return cards


# ----------------------------------------------------------------------------
# ML model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training churn model...")
def train_model():
    df = load_data()
    X, y = df[FEATURES], df["ChurnFlag"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                             ("num", "passthrough", NUM_FEATURES)])
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=250, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.DataFrame({"feature": [n.split("__", 1)[1].split("_")[0] for n in names],
                        "importance": pipe.named_steps["clf"].feature_importances_})
    imp = imp.groupby("feature", as_index=False)["importance"].sum().sort_values("importance").tail(12)
    metrics = {"auc": roc_auc_score(y_te, proba), "acc": accuracy_score(y_te, pred),
               "prec": precision_score(y_te, pred), "rec": recall_score(y_te, pred), "f1": f1_score(y_te, pred)}
    return pipe, metrics, (fpr, tpr), confusion_matrix(y_te, pred), imp
