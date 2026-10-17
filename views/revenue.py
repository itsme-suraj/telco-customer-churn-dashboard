from utils import *

d, df = get_d(), load_data()
page_header("💰", "Revenue & Billing", "How much money is at stake, and which billing choices drive churn?")

mrr = d.MonthlyCharges.sum()
ch = d[d.ChurnFlag == 1]
risk = ch.MonthlyCharges.sum()
kpi_grid([
    kpi("Monthly recurring revenue", money(mrr), f"{len(d):,} customers", PURPLE, "💵"),
    kpi("ARPU", f"${d.MonthlyCharges.mean():.2f}", "avg revenue per user / month", BLUE, "🧾"),
    kpi("Revenue at risk / month", money(risk), f"{risk / mrr * 100:.1f}% of MRR", RED, "💸"),
    kpi("Lifetime value (avg)", money(d.TotalCharges.mean()), f"churned: {money(ch.TotalCharges.mean() if len(ch) else 0)}", TEAL, "💎"),
    kpi("Month-to-month MRR share", f"{d.loc[d.Contract == 'Month-to-month', 'MonthlyCharges'].sum() / mrr * 100:.0f}%", "least protected revenue", AMBER, "📄"),
    kpi("Auto-pay customers", f"{d.PaymentMethod.str.contains('auto').mean() * 100:.0f}%", "bank or card auto-pay", TEAL, "🔁"),
])

section("Pricing & churn")
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(d, x="MonthlyCharges", color="Churn", nbins=40, barmode="overlay", opacity=0.8,
                       color_discrete_map=CMAP, title="Monthly charge distribution")
    fig.update_xaxes(title="Monthly charges ($)")
    fig.update_yaxes(title="Customers")
    show(style(fig, 340))
with c2:
    show(rate_bar(d, "ChargeTier", "Churn rate by price tier", height=340))

section("Billing choices")
c3, c4, c5 = st.columns(3)
with c3:
    show(rate_bar(d, "PaymentMethod", "Churn by payment method", horizontal=True))
with c4:
    show(rate_bar(d, "PaperlessBilling", "Churn by paperless billing"))
with c5:
    g = d.groupby("Contract", observed=True)["MonthlyCharges"].mean().reset_index()
    fig = px.bar(g, x="Contract", y="MonthlyCharges", color="Contract", text=g.MonthlyCharges.map("${:.0f}".format),
                 color_discrete_sequence=[RED, AMBER, TEAL], title="ARPU by contract")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Avg monthly charge ($)")
    fig.update_xaxes(title=None)
    show(style(fig, 330))

section("Where is the money leaking?")
c6, c7 = st.columns([1, 1.2])
with c6:
    g = ch.groupby(["Contract", "PaymentMethod"], observed=True)["MonthlyCharges"].sum().reset_index()
    if len(g):
        fig = px.treemap(g, path=[px.Constant("All"), "Contract", "PaymentMethod"], values="MonthlyCharges",
                         color="MonthlyCharges", color_continuous_scale=["#2a2f55", PURPLE, RED],
                         title="Revenue lost: contract → payment")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(texttemplate="%{label}<br>$%{value:,.0f}")
        show(style(fig, 380))
with c7:
    seg = (ch.groupby(["Contract", "InternetService"], observed=True)["MonthlyCharges"].sum()
           .reset_index().sort_values("MonthlyCharges", ascending=False))
    if len(seg):
        seg["name"] = seg.Contract.astype(str) + " · " + seg.InternetService.astype(str)
        seg["cum"] = seg.MonthlyCharges.cumsum() / seg.MonthlyCharges.sum() * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=seg.name, y=seg.MonthlyCharges, name="Revenue lost", marker_color=RED)
        fig.add_scatter(x=seg.name, y=seg.cum, name="Cumulative %", mode="lines+markers",
                        line=dict(color=AMBER, width=3), secondary_y=True)
        fig.update_yaxes(title_text="$ / month", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
        fig.update_layout(title="Pareto: which segments lose the most revenue?")
        show(style(fig, 380))

section("Lifetime value")
c8, c9 = st.columns(2)
with c8:
    g = d.groupby(["TenureGroup", "Churn"], observed=True)["TotalCharges"].mean().reset_index()
    g["TenureGroup"] = g.TenureGroup.astype(str)
    fig = px.bar(g, x="TenureGroup", y="TotalCharges", color="Churn", barmode="group", color_discrete_map=CMAP,
                 title="Avg lifetime billed by tenure group")
    fig.update_yaxes(title="Avg total charges ($)")
    fig.update_xaxes(title=None)
    show(style(fig, 350))
with c9:
    s = d.sample(min(len(d), 2500), random_state=1)
    fig = px.scatter(s, x="MonthlyCharges", y="TotalCharges", color="Churn", color_discrete_map=CMAP, opacity=0.55,
                     hover_data=["Contract", "tenure"], title="Monthly vs total charges")
    fig.update_xaxes(title="Monthly charges ($)")
    fig.update_yaxes(title="Total charges ($)")
    show(style(fig, 350))

insight_grid([
    insight("💳 Billing nudges are cheap wins", "Electronic-check payers churn about 3x more than auto-pay customers. A $5 auto-pay credit pays for itself."),
    insight("💸 Concentrated losses", "A few segments (see the Pareto chart) cause most of the lost revenue. Target those first."),
    insight("📈 High price ≠ low loyalty by itself", "Churn rises with price mainly for month-to-month and fiber customers. Bundles beat discounts."),
])
