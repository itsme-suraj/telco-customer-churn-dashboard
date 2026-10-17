from utils import *

d = get_d()
page_header("🎬", "What-if Scenario Simulator", "Pick retention actions, and the ML model estimates how many customers you save and the ROI.")
model = train_model()[0]

active = d[d.Churn == "No"].reset_index(drop=True)
if len(active) < 30:
    st.warning("Need at least 30 active customers in this view. Relax your filters.")
    st.stop()

SIM_PRESETS = {
    "Custom": None,
    "🟢 Conservative": (20, 10, 20, 0, 0),
    "🟡 Balanced": (40, 25, 40, 5, 30),
    "🔴 Aggressive": (60, 40, 60, 10, 60),
}
KEYS = ["sim_c", "sim_t", "sim_p", "sim_cut", "sim_f"]
for k, v in zip(KEYS, (30, 20, 30, 0, 0)):
    st.session_state.setdefault(k, v)


def load_preset():
    v = SIM_PRESETS[st.session_state["sim_preset"]]
    if v:
        for k, val in zip(KEYS, v):
            st.session_state[k] = val


RNG = np.random.default_rng(7).random((len(active), 4))
P0 = model.predict_proba(active[FEATURES])[:, 1]


def simulate(fc, ft, fp, cut, ff):
    X = active[FEATURES].copy()
    mc = (X.Contract == "Month-to-month") & (RNG[:, 0] < fc)
    X.loc[mc, "Contract"] = "One year"
    mt = (X.InternetService != "No") & (X.TechSupport == "No") & (RNG[:, 1] < ft)
    X.loc[mt, "TechSupport"] = "Yes"
    mp = (X.PaymentMethod == "Electronic check") & (RNG[:, 2] < fp)
    X.loc[mp, "PaymentMethod"] = "Bank transfer (auto)"
    mf = (X.InternetService == "Fiber optic") & (RNG[:, 3] < ff)
    X.loc[mf, "MonthlyCharges"] = (X.loc[mf, "MonthlyCharges"] - cut).clip(lower=18)
    p1 = model.predict_proba(X)[:, 1]
    return X, p1, int(mc.sum()), int(mt.sum()), int(mp.sum()), int(mf.sum())


left, right = st.columns([1, 2])
with left:
    st.markdown("#### 🎚️ Choose your actions")
    st.selectbox("Load a preset", list(SIM_PRESETS), key="sim_preset", on_change=load_preset)
    fc = st.slider("Convert month-to-month → 1-year contract (%)", 0, 100, key="sim_c")
    ft = st.slider("Give Tech Support to internet users without it (%)", 0, 100, key="sim_t")
    fp = st.slider("Move e-check payers to auto-pay (%)", 0, 100, key="sim_p")
    cut = st.slider("Fiber price cut ($ / month)", 0, 30, key="sim_cut")
    ff = st.slider("…applied to this % of fiber customers", 0, 100, key="sim_f")
    with st.expander("💵 Program costs"):
        cost_c = st.number_input("Contract incentive ($ per customer, one-time)", 0, 500, 50)
        cost_t = st.number_input("Tech Support cost ($ per customer / year)", 0, 500, 30)
        cost_p = st.number_input("Auto-pay credit ($ per customer / year)", 0, 200, 20)

X1, P1, nc, nt, np_, nf = simulate(fc / 100, ft / 100, fp / 100, cut, ff / 100)
base_churn, scn_churn = P0.sum(), P1.sum()
saved = base_churn - scn_churn
mc0 = active.MonthlyCharges.to_numpy()
mc1 = X1.MonthlyCharges.to_numpy()
gain = (((1 - P1) * mc1).sum() - ((1 - P0) * mc0).sum()) * 12
cost = nc * cost_c + nt * cost_t + np_ * cost_p
net = gain - cost
roi = net / cost * 100 if cost > 0 else float("nan")

with right:
    kpi_grid([
        kpi("Expected churners (now)", f"{base_churn:,.0f}", f"{base_churn / len(active) * 100:.1f}% of {len(active):,} active", RED, "🚪"),
        kpi("Expected churners (scenario)", f"{scn_churn:,.0f}", f"{scn_churn / len(active) * 100:.1f}% churn", TEAL, "🛡️"),
        kpi("Customers saved", f"{saved:,.0f}", f"{saved / base_churn * 100 if base_churn else 0:.0f}% fewer churners", PURPLE, "🙌"),
        kpi("Annual revenue gain", money(gain), "retained revenue after price cuts", BLUE, "📈"),
        kpi("Program cost", money(cost), f"{nc:,} contracts · {nt:,} tech · {np_:,} autopay", AMBER, "🧾"),
        kpi("Net benefit / year", money(net), "gain − program cost", TEAL if net >= 0 else RED, "💰"),
        kpi("ROI", "n/a" if math.isnan(roi) else f"{roi:,.0f}%", "net benefit ÷ cost", PURPLE, "🚀"),
        kpi("Customers touched", f"{nc + nt + np_ + nf:,}", "across all actions", BLUE, "🎯"),
    ])
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=["Today", "With scenario"], y=[base_churn, scn_churn], marker_color=[RED, TEAL],
                               text=[f"{base_churn:,.0f}", f"{scn_churn:,.0f}"], textposition="outside"))
        fig.update_layout(title="Expected churners: today vs scenario")
        fig.update_yaxes(title="Customers", range=[0, base_churn * 1.2 + 1])
        show(style(fig, 340))
    with c2:
        lev = []
        for name, args in [("Contract", (fc / 100, 0, 0, 0, 0)), ("Tech Support", (0, ft / 100, 0, 0, 0)),
                           ("Auto-pay", (0, 0, fp / 100, 0, 0)), ("Fiber price cut", (0, 0, 0, cut, ff / 100))]:
            Xl, Pl, a, b, c_, e = simulate(*args)
            g_ = (((1 - Pl) * Xl.MonthlyCharges.to_numpy()).sum() - ((1 - P0) * mc0).sum()) * 12
            k_ = a * cost_c + b * cost_t + c_ * cost_p
            lev.append({"Lever": name, "Customers saved": P0.sum() - Pl.sum(), "Net benefit ($/yr)": g_ - k_})
        lev = pd.DataFrame(lev)
        fig = px.bar(lev, x="Lever", y="Customers saved", color="Net benefit ($/yr)",
                     color_continuous_scale=[RED, AMBER, TEAL], title="Which lever saves the most customers?",
                     text=lev["Customers saved"].map("{:,.0f}".format))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(title=None)
        show(style(fig, 340))

st.caption("⚠️ Estimates come from the churn model's predicted probabilities for currently active customers in your filtered view. "
           "They show correlation, not guaranteed causal effects, so validate with a small pilot first.")
