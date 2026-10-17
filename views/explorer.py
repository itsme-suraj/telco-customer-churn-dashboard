from utils import *

d, df = get_d(), load_data()
page_header("🗂️", "Data Explorer", "Build your own cross-tab, then export the filtered data.")

DIMS = {"Contract": "Contract", "Internet service": "InternetService", "Payment method": "PaymentMethod",
        "Tenure group": "TenureGroup", "Price tier": "ChargeTier", "Household": "Household", "Gender": "gender",
        "Senior": "SeniorCitizen", "Paperless billing": "PaperlessBilling", "# Add-ons": "NumAddons",
        "Protection": "Protection"}
METRICS = ["Churn rate %", "Customers", "Avg monthly charge ($)", "Revenue at risk ($/mo)", "Avg tenure (mo)"]

section("Pivot builder")
c1, c2, c3 = st.columns(3)
row = c1.selectbox("Rows", list(DIMS), index=0)
col = c2.selectbox("Columns", list(DIMS), index=1)
met = c3.selectbox("Metric", METRICS, index=0)

x = d.assign(AtRisk=np.where(d.ChurnFlag == 1, d.MonthlyCharges, 0.0))
agg = {"Churn rate %": ("ChurnFlag", lambda s: s.mean() * 100), "Customers": ("ChurnFlag", "size"),
       "Avg monthly charge ($)": ("MonthlyCharges", "mean"), "Revenue at risk ($/mo)": ("AtRisk", "sum"),
       "Avg tenure (mo)": ("tenure", "mean")}[met]
if row == col:
    st.info("Choose different fields for rows and columns.")
else:
    piv = x.pivot_table(index=DIMS[row], columns=DIMS[col], values=agg[0], aggfunc=agg[1], observed=True)
    piv.index, piv.columns = piv.index.astype(str), piv.columns.astype(str)
    fig = px.imshow(piv, text_auto=".1f" if met != "Customers" else True, aspect="auto",
                    color_continuous_scale=SCALE if "Churn" in met or "risk" in met else ["#161a2e", PURPLE, BLUE],
                    title=f"{met}: {row} × {col}")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    show(style(fig, 420))
    with st.expander("Show as table"):
        show_df(piv.round(1).reset_index())

section("Filtered data", f"{len(d):,} rows")
q = st.text_input("Search customer ID", placeholder="e.g. 7590-VHVEG")
t = d.drop(columns=["ChurnFlag"])
if q:
    t = t[t.customerID.str.contains(q, case=False, na=False)]
show_df(t)
st.download_button("⬇️ Download filtered data (CSV)", t.to_csv(index=False).encode(), "telco_filtered.csv", "text/csv")
with st.expander("Data notes"):
    st.markdown("- Source: IBM Telco Customer Churn sample dataset (7,043 customers).\n"
                "- 11 blank `TotalCharges` (tenure 0) set to 0; `SeniorCitizen` mapped 0/1 → No/Yes.\n"
                "- Engineered: `TenureGroup`, `ChargeTier`, `NumAddons`, `Household`, `Protection`, `ChurnFlag`.")
