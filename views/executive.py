from utils import *

d, df = get_d(), load_data()
page_header("🏠", "Executive Summary", "The big picture: who is leaving, what it costs, and where to act first.")

n, churned = len(d), int(d.ChurnFlag.sum())
rate, overall = churned / n * 100, df.ChurnFlag.mean() * 100
mrr = d.MonthlyCharges.sum()
risk = d.loc[d.ChurnFlag == 1, "MonthlyCharges"].sum()
ch = d[d.ChurnFlag == 1]

kpi_grid([
    kpi("Total customers", f"{n:,}", f"{n / len(df) * 100:.0f}% of all customers", PURPLE, "👥"),
    kpi("Churned customers", f"{churned:,}", f"{n - churned:,} still active", RED, "🚪"),
    kpi("Churn rate", f"{rate:.1f}%", delta_html(rate, overall), AMBER, "📉"),
    kpi("Monthly revenue at risk", money(risk), f"{risk / mrr * 100:.1f}% of {money(mrr)} MRR", RED, "💸"),
    kpi("Avg monthly charge", f"${d.MonthlyCharges.mean():.2f}", f"churned: ${ch.MonthlyCharges.mean() if len(ch) else 0:.2f}", BLUE, "💳"),
    kpi("Avg tenure", f"{d.tenure.mean():.1f} mo", f"churned: {ch.tenure.mean() if len(ch) else 0:.1f} mo", TEAL, "⏳"),
    kpi("Avg lifetime value", money(d.TotalCharges.mean()), f"total billed {money(d.TotalCharges.sum())}", PURPLE, "💎"),
    kpi("Annual revenue lost", money(risk * 12), "if churned customers stay gone", RED, "🔥"),
])

section("Churn at a glance")
c1, c2, c3 = st.columns([1, 1.2, 1.2])
with c1:
    show(donut(d))
with c2:
    show(rate_bar(d, "Contract", "Churn rate by contract"))
with c3:
    show(rate_bar(d, "TenureGroup", "Churn rate by tenure"))

section("Where does it hurt?", "customers vs churn across the customer lifetime, and revenue lost by segment")
c4, c5 = st.columns([1.3, 1])
with c4:
    w = d.assign(Window=pd.cut(d.tenure, bins=[-1, 6, 12, 18, 24, 36, 48, 60, 72],
                               labels=["0-6", "7-12", "13-18", "19-24", "25-36", "37-48", "49-60", "61-72"]))
    g = rate_by(w, "Window")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=g.Window.astype(str), y=g.customers, name="Customers", marker_color="rgba(124,92,255,.5)")
    fig.add_scatter(x=g.Window.astype(str), y=g.rate, name="Churn %", mode="lines+markers",
                    line=dict(color=RED, width=3), secondary_y=True)
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="Churn rate (%)", secondary_y=True)
    fig.update_xaxes(title="Tenure window (months)")
    fig.update_layout(title="Customers & churn rate by tenure window")
    show(style(fig, 360))
with c5:
    g = d[d.ChurnFlag == 1].groupby(["Contract", "InternetService"], observed=True)["MonthlyCharges"].sum().reset_index()
    if len(g):
        fig = px.treemap(g, path=[px.Constant("All"), "Contract", "InternetService"], values="MonthlyCharges",
                         color="MonthlyCharges", color_continuous_scale=["#2a2f55", PURPLE, RED],
                         title="Monthly revenue lost by segment")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(texttemplate="%{label}<br>$%{value:,.0f}")
        show(style(fig, 360))
    else:
        st.info("No churned customers in this view.")

c6, c7, c8 = st.columns(3)
with c6:
    show(rate_bar(d, "PaymentMethod", "Churn by payment method", horizontal=True))
with c7:
    show(rate_bar(d, "InternetService", "Churn by internet service", horizontal=True))
with c8:
    show(rate_bar(d, "Household", "Churn by household type", horizontal=True))

section("Auto-generated insights", "they change with your filters")
insight_grid(auto_insights(d))

section("Recommended actions")
insight_grid([
    insight("1. Lock in month-to-month users", "Offer 1-2 year contract incentives. The biggest single lever on churn.", True),
    insight("2. Fix the first 90 days", "Onboarding calls and early check-ins for customers under 12 months of tenure.", True),
    insight("3. Push auto-pay", "A small bill credit for moving from electronic check to card / bank auto-pay.", True),
    insight("4. Bundle protection", "Tech Support, Security and Backup customers stay far longer. Bundle them into plans.", True),
    insight("5. Review fiber pricing", "High-priced fiber customers churn heavily. Check outages, speed and price.", True),
    insight("6. Run the Simulator", "Test these levers on the Simulator page and see the ROI before you spend a dollar.", True),
])
