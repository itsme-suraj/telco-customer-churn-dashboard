from utils import *

d, df = get_d(), load_data()
page_header("🤖", "Churn Predictor", "Machine-learning model, single-customer risk score, and a ranked retention hit-list.")
model, m, (fpr, tpr), cm, imp = train_model()

section("Model performance", "Random Forest · trained on 80% of customers · tested on the held-out 20%")
kpi_grid([
    kpi("ROC-AUC", f"{m['auc']:.3f}", "ranking quality", PURPLE, "🎯"),
    kpi("Accuracy", f"{m['acc'] * 100:.1f}%", "overall correct", BLUE, "✅"),
    kpi("Precision", f"{m['prec'] * 100:.1f}%", "flagged who really churn", AMBER, "🔬"),
    kpi("Recall", f"{m['rec'] * 100:.1f}%", "churners we catch", TEAL, "📡"),
    kpi("F1 score", f"{m['f1']:.3f}", "precision / recall balance", RED, "⚖️"),
])
c1, c2, c3 = st.columns(3)
with c1:
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=PURPLE, width=3), fill="tozeroy", fillcolor="rgba(124,92,255,.15)")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#6b7280", dash="dash"))
    fig.update_layout(title=f"ROC curve (AUC {m['auc']:.2f})", xaxis_title="False positive rate", yaxis_title="True positive rate")
    show(style(fig, 330))
with c2:
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=["#161a2e", PURPLE],
                    x=["Pred: stay", "Pred: churn"], y=["Actual: stay", "Actual: churn"], title="Confusion matrix")
    fig.update_layout(coloraxis_showscale=False)
    show(style(fig, 330))
with c3:
    fig = px.bar(imp, x="importance", y="feature", orientation="h", title="What drives churn?", color="importance",
                 color_continuous_scale=["#2a2f55", PURPLE, RED])
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="Importance")
    fig.update_yaxes(title=None)
    show(style(fig, 330))

section("Predict churn risk for one customer")


def pick(label, col, default=None, container=st):
    o = sorted(df[col].astype(str).unique().tolist())
    return container.selectbox(label, o, index=o.index(default) if default in o else 0, key=f"p_{col}")


with st.form("predict_form"):
    a, b, c = st.columns(3)
    gender = pick("Gender", "gender", None, a)
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
    row = {"gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
           "PhoneService": phone, "MultipleLines": lines, "InternetService": internet, "OnlineSecurity": sec,
           "OnlineBackup": bak, "DeviceProtection": dev, "TechSupport": tech, "StreamingTV": stv,
           "StreamingMovies": smov, "Contract": contract, "PaperlessBilling": paperless,
           "PaymentMethod": payment, "tenure": tenure_in, "MonthlyCharges": float(monthly_in)}
    if internet == "No":
        for k in ADDONS:
            row[k] = "No internet service"
    if phone == "No":
        row["MultipleLines"] = "No phone service"
    st.session_state["pred"] = (float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1]), row)

if "pred" in st.session_state:
    p, row = st.session_state["pred"]
    g1, g2 = st.columns([1, 1.2])
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(size=44)),
            title=dict(text="Churn probability"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#e5e7eb", thickness=0.25),
                       steps=[dict(range=[0, 30], color="rgba(45,212,191,.55)"),
                              dict(range=[30, 60], color="rgba(251,191,36,.55)"),
                              dict(range=[60, 100], color="rgba(255,75,110,.6)")])))
        show(style(fig, 320))
    with g2:
        level = "LOW" if p < 0.3 else "MEDIUM" if p < 0.6 else "HIGH"
        colr = TEAL if level == "LOW" else AMBER if level == "MEDIUM" else RED
        sim = df[(df.Contract == row["Contract"]) & (df.InternetService == row["InternetService"])]
        kpi_grid([kpi("Risk level", level, f"similar customers churn at {sim.ChurnFlag.mean() * 100:.1f}%", colr, "🚨")])
        recs = []
        if row["Contract"] == "Month-to-month":
            recs.append("Offer a discount for a 1-year contract (the biggest churn driver).")
        if row["InternetService"] != "No" and row["TechSupport"] == "No":
            recs.append("Bundle free Tech Support for 3 months.")
        if row["PaymentMethod"] == "Electronic check":
            recs.append("Nudge to auto-pay with a small bill credit.")
        if row["InternetService"] == "Fiber optic" and row["MonthlyCharges"] >= 80:
            recs.append("Review price/quality: high-priced fiber customers are a flight risk.")
        if row["tenure"] <= 12:
            recs.append("Trigger first-year onboarding and the loyalty programme.")
        st.markdown("**Recommended retention actions**")
        for r in recs or ["Customer looks stable. Keep service quality high and reward loyalty."]:
            st.markdown(f"- {r}")

section("Retention hit-list", "active customers most likely to leave · follows your filters")
active = d[d.Churn == "No"].copy()
if active.empty:
    st.info("No active customers in this view.")
else:
    active["Risk"] = model.predict_proba(active[FEATURES])[:, 1] * 100
    topn = st.slider("Customers to show", 10, 100, 25, step=5)
    hit = active.sort_values("Risk", ascending=False).head(topn)[
        ["customerID", "Contract", "InternetService", "PaymentMethod", "tenure", "MonthlyCharges", "Risk"]]
    show_df(hit, column_config={
        "Risk": st.column_config.ProgressColumn("Churn risk %", min_value=0, max_value=100, format="%.0f%%"),
        "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
        "tenure": st.column_config.NumberColumn("Tenure (mo)")})
    st.download_button("⬇️ Download hit-list (CSV)", hit.to_csv(index=False).encode(), "retention_hitlist.csv", "text/csv")
