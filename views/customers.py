from utils import *

d, df = get_d(), load_data()
page_header("👥", "Customer Profile", "Who are our customers, and which groups are most likely to leave?")

n = len(d)
pct = lambda col, val: (d[col] == val).mean() * 100
kpi_grid([
    kpi("Senior citizens", f"{pct('SeniorCitizen', 'Yes'):.0f}%", f"churn {d[d.SeniorCitizen == 'Yes'].ChurnFlag.mean() * 100 if (d.SeniorCitizen == 'Yes').any() else 0:.1f}%", AMBER, "👴"),
    kpi("Have a partner", f"{pct('Partner', 'Yes'):.0f}%", f"churn {d[d.Partner == 'Yes'].ChurnFlag.mean() * 100 if (d.Partner == 'Yes').any() else 0:.1f}%", BLUE, "💑"),
    kpi("Have dependents", f"{pct('Dependents', 'Yes'):.0f}%", f"churn {d[d.Dependents == 'Yes'].ChurnFlag.mean() * 100 if (d.Dependents == 'Yes').any() else 0:.1f}%", TEAL, "👨‍👩‍👧"),
    kpi("Male customers", f"{pct('gender', 'Male'):.0f}%", f"female {pct('gender', 'Female'):.0f}%", PURPLE, "🧑"),
])

section("Who churns?")
demo = []
for col in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
    for _, r in rate_by(d, col).iterrows():
        demo.append({"Feature": PRETTY[col], "Value": str(r[col]), "rate": r["rate"], "customers": r["customers"]})
demo = pd.DataFrame(demo)
c1, c2 = st.columns([1.6, 1])
with c1:
    fig = px.bar(demo, x="Value", y="rate", color="rate", facet_col="Feature", color_continuous_scale=SCALE,
                 range_color=[0, 50], text=demo.rate.map("{:.0f}%".format), title="Churn rate by demographic",
                 hover_data={"customers": True, "rate": ":.1f"})
    fig.update_xaxes(matches=None, title=None)
    fig.update_yaxes(title="Churn rate (%)", range=[0, 58])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    show(style(fig, 340))
with c2:
    show(rate_bar(d, "Household", "Churn by household type"))

section("Households & segments")
c3, c4, c5 = st.columns(3)
with c3:
    show(heat(d, "Household", "Contract", "Churn %: household × contract"))
with c4:
    fig = px.histogram(d, x="Household", color="Churn", barmode="stack", color_discrete_map=CMAP,
                       title="Customers by household type")
    fig.update_yaxes(title="Customers")
    fig.update_xaxes(title=None)
    show(style(fig, 330))
with c5:
    fig = px.histogram(d, x="SeniorCitizen", color="InternetService", barnorm="percent", text_auto=".0f",
                       title="Internet mix: senior vs non-senior (%)",
                       color_discrete_sequence=[BLUE, RED, "#6b7280"])
    fig.update_yaxes(title="% of customers")
    fig.update_xaxes(title="Senior citizen")
    show(style(fig, 330))

section("Tenure & spend by group")
c6, c7 = st.columns(2)
with c6:
    fig = px.box(d, x="Household", y="tenure", color="Churn", color_discrete_map=CMAP, title="Tenure by household & outcome")
    fig.update_yaxes(title="Tenure (months)")
    fig.update_xaxes(title=None)
    show(style(fig, 340))
with c7:
    g = d.groupby(["SeniorCitizen", "Churn"], observed=True)["MonthlyCharges"].mean().reset_index()
    fig = px.bar(g, x="SeniorCitizen", y="MonthlyCharges", color="Churn", barmode="group", color_discrete_map=CMAP,
                 text=g.MonthlyCharges.map("${:.0f}".format), title="Avg monthly charge: senior vs non-senior")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title="Avg monthly charge ($)")
    fig.update_xaxes(title="Senior citizen")
    show(style(fig, 340))

section("Insights")
sen, non = d[d.SeniorCitizen == "Yes"], d[d.SeniorCitizen == "No"]
cards = []
if len(sen) >= 30 and len(non) >= 30:
    cards.append(insight("👴 Seniors leave more", f"Senior citizens churn at <b>{sen.ChurnFlag.mean() * 100:.1f}%</b> vs <b>{non.ChurnFlag.mean() * 100:.1f}%</b> for the rest. They also pay more on average (${sen.MonthlyCharges.mean():.0f} vs ${non.MonthlyCharges.mean():.0f})."))
r = top_group(d, "Household")
if r is not None:
    cards.append(insight(f"🏠 {r['Household']} households are riskiest", f"Churn is <b>{r['rate']:.1f}%</b> across {int(r['customers']):,} customers. Family and couple plans stabilise retention."))
dep = d[d.Dependents == "Yes"]
if len(dep) >= 30:
    cards.append(insight("👨‍👩‍👧 Dependents add stickiness", f"Customers with dependents churn at <b>{dep.ChurnFlag.mean() * 100:.1f}%</b>. Household bundles can raise switching cost."))
if cards:
    insight_grid(cards)
