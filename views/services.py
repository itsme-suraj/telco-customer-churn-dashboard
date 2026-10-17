from utils import *

d, df = get_d(), load_data()
page_header("📦", "Services & Products", "Which products keep customers, and which ones don't?")

net = d[d.InternetService != "No"]
fiber = (d.InternetService == "Fiber optic").mean() * 100
kpi_grid([
    kpi("With internet", f"{len(net) / len(d) * 100:.0f}%", f"{len(net):,} customers", BLUE, "🌐"),
    kpi("Fiber optic share", f"{fiber:.0f}%", f"churn {d[d.InternetService == 'Fiber optic'].ChurnFlag.mean() * 100 if fiber > 0 else 0:.1f}%", RED, "⚡"),
    kpi("Avg add-ons / customer", f"{d.NumAddons.mean():.1f}", "of 6 add-on services", PURPLE, "🧩"),
    kpi("Have protection bundle", f"{(d.Protection == 'Has protection').mean() * 100:.0f}%", "security, backup, device or tech", TEAL, "🛡️"),
])

section("Core services")
c1, c2, c3 = st.columns(3)
with c1:
    show(rate_bar(d, "InternetService", "Churn by internet service"))
with c2:
    show(rate_bar(d, "MultipleLines", "Churn by phone lines"))
with c3:
    show(rate_bar(d, "Protection", "Churn by protection status"))

section("Do add-ons protect customers?", "internet customers only")
rows, pen = [], []
for s in ADDONS:
    for flag, lab in [("Yes", "With service"), ("No", "Without service")]:
        sub = net[net[s] == flag]
        if len(sub) >= 10:
            rows.append({"Service": PRETTY[s], "Has": lab, "rate": sub.ChurnFlag.mean() * 100, "customers": len(sub)})
    if len(net):
        pen.append({"Service": PRETTY[s], "share": (net[s] == "Yes").mean() * 100})
c4, c5 = st.columns([1.5, 1])
with c4:
    if rows:
        fig = px.bar(pd.DataFrame(rows), x="Service", y="rate", color="Has", barmode="group",
                     color_discrete_map={"With service": TEAL, "Without service": RED},
                     title="Churn rate with vs without each add-on", hover_data={"customers": True, "rate": ":.1f"})
        fig.update_yaxes(title="Churn rate (%)")
        fig.update_xaxes(title=None)
        show(style(fig, 360))
    else:
        st.info("Not enough internet customers in this view.")
with c5:
    if pen:
        p = pd.DataFrame(pen).sort_values("share")
        fig = px.bar(p, x="share", y="Service", orientation="h", text=p.share.map("{:.0f}%".format),
                     title="Add-on adoption (% of internet customers)", color="share",
                     color_continuous_scale=["#2a2f55", PURPLE, BLUE])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="% of internet customers", range=[0, 100])
        fig.update_yaxes(title=None)
        show(style(fig, 360))

section("Bundles & combinations")
c6, c7, c8 = st.columns(3)
with c6:
    show(rate_bar(d, "NumAddons", "Churn by number of add-ons"))
with c7:
    stream = net.assign(Streaming=np.select(
        [(net.StreamingTV == "Yes") & (net.StreamingMovies == "Yes"), (net.StreamingTV == "Yes") | (net.StreamingMovies == "Yes")],
        ["Both", "One"], default="None"))
    if len(stream):
        show(rate_bar(stream, "Streaming", "Churn by streaming bundle"))
with c8:
    show(heat(d, "InternetService", "NumAddons", "Churn %: internet × add-ons"))

section("Price vs service")
c9, c10 = st.columns(2)
with c9:
    fig = px.box(d, x="InternetService", y="MonthlyCharges", color="Churn", color_discrete_map=CMAP,
                 title="Monthly charges by internet service & outcome")
    fig.update_yaxes(title="Monthly charges ($)")
    fig.update_xaxes(title=None)
    show(style(fig, 350))
with c10:
    g = d.groupby(["InternetService", "Contract"], observed=True).agg(c=("ChurnFlag", "mean"), n=("ChurnFlag", "size")).reset_index()
    g["c"] *= 100
    fig = px.bar(g, x="InternetService", y="c", color="Contract", barmode="group",
                 color_discrete_sequence=[RED, AMBER, TEAL], title="Churn: internet × contract",
                 hover_data={"n": True, "c": ":.1f"})
    fig.update_yaxes(title="Churn rate (%)")
    fig.update_xaxes(title=None)
    show(style(fig, 350))
