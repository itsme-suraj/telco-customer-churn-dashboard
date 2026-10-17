from utils import *

d, df = get_d(), load_data()
page_header("⏳", "Customer Lifecycle", "How long do customers stay, and when do they leave?")

S = km_curve(d)
def at(t):
    return S.iloc[min(t, len(S) - 1)] * 100
below = S[S <= 0.5]
median = f"{int(below.index[0])} mo" if len(below) else ">72 mo"
ch = d[d.ChurnFlag == 1]
kpi_grid([
    kpi("Median customer lifetime", median, "50% of customers still active", PURPLE, "📏"),
    kpi("Retained after 6 months", f"{at(6):.0f}%", "Kaplan-Meier estimate", TEAL, "🌱"),
    kpi("Retained after 12 months", f"{at(12):.0f}%", "end of first year", BLUE, "📆"),
    kpi("Retained after 24 months", f"{at(24):.0f}%", "two-year survival", AMBER, "🏅"),
    kpi("Avg tenure at churn", f"{ch.tenure.mean() if len(ch) else 0:.1f} mo", f"{(ch.tenure <= 12).mean() * 100 if len(ch) else 0:.0f}% leave in year 1", RED, "🚪"),
])

section("Retention curves", "share of customers still active after N months")
c1, c2 = st.columns([1.5, 1])
with c1:
    fig = go.Figure()
    colors = {"Month-to-month": RED, "One year": AMBER, "Two year": TEAL}
    for ct in sorted(d.Contract.unique()):
        sub = d[d.Contract == ct]
        if len(sub) >= 30:
            fig.add_scatter(x=np.arange(0, 73), y=km_curve(sub) * 100, mode="lines", name=ct,
                            line=dict(color=colors.get(ct, PURPLE), width=3, shape="hv"))
    fig.add_scatter(x=np.arange(0, 73), y=S * 100, mode="lines", name="All in view",
                    line=dict(color="#e5e7eb", width=2, dash="dot", shape="hv"))
    fig.add_hline(y=50, line_dash="dash", line_color="#6b7280", annotation_text="50%")
    fig.update_layout(title="Customer retention (survival) curve by contract")
    fig.update_xaxes(title="Tenure (months)")
    fig.update_yaxes(title="% still active", range=[0, 102])
    show(style(fig, 380))
with c2:
    rows = []
    for ct in sorted(d.Contract.unique()):
        sub = d[d.Contract == ct]
        if len(sub) >= 30:
            k = km_curve(sub)
            rows.append({"Contract": ct, **{f"{m} mo": f"{k.iloc[min(m, 72)] * 100:.0f}%" for m in (6, 12, 24, 36, 48, 60)}})
    st.markdown("**Retention at key milestones**")
    if rows:
        show_df(pd.DataFrame(rows))
    st.caption("Kaplan-Meier: handles customers who are still active (their final tenure isn't a churn).")

section("When do they leave?")
c3, c4 = st.columns(2)
with c3:
    w = d.assign(Window=pd.cut(d.tenure, bins=[-1, 6, 12, 18, 24, 36, 48, 60, 72],
                               labels=["0-6", "7-12", "13-18", "19-24", "25-36", "37-48", "49-60", "61-72"]))
    g = w.groupby(["Window", "Contract"], observed=True).ChurnFlag.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 15]
    g["mean"] *= 100
    g["Window"] = g.Window.astype(str)
    fig = px.line(g, x="Window", y="mean", color="Contract", markers=True, title="Churn rate by tenure window & contract",
                  color_discrete_sequence=[RED, AMBER, TEAL], hover_data={"size": True, "mean": ":.1f"})
    fig.update_traces(line=dict(width=3))
    fig.update_xaxes(title="Tenure window (months)")
    fig.update_yaxes(title="Churn rate (%)")
    show(style(fig, 350))
with c4:
    if len(ch):
        fig = px.histogram(ch, x="tenure", nbins=36, color_discrete_sequence=[RED], title="Tenure at the moment of churn")
        fig.update_xaxes(title="Tenure (months)")
        fig.update_yaxes(title="Churned customers")
        show(style(fig, 350))

section("Lifecycle heatmaps")
c5, c6 = st.columns(2)
with c5:
    show(heat(d, "TenureGroup", "Contract", "Churn %: tenure × contract", h=330))
with c6:
    show(heat(d, "TenureGroup", "InternetService", "Churn %: tenure × internet", h=330))

insight_grid([
    insight("🌱 Early-life risk", f"{(ch.tenure <= 12).mean() * 100 if len(ch) else 0:.0f}% of churners leave within their first year. Onboarding is the highest-ROI fix."),
    insight("🔒 Contracts flatten the curve", "Two-year contracts stay near 100% retention. Month-to-month customers drop fastest in the first 6 months."),
    insight("📏 Use the median", f"Median lifetime here is <b>{median}</b>. Set campaign timing before that point."),
])
