# 📉 Telco Customer Churn Intelligence

A **multi-page, full-width, mobile-friendly** analytics dashboard that shows who is churning, why, how much revenue is at risk, and what to do about it. Includes an ML churn predictor and a **what-if scenario simulator** with ROI.

🔗 **Live demo:** _paste your Streamlit link here_

![Dashboard preview](screenshots/executive.png)

## 📑 Pages
| Page | What you get |
|---|---|
| 🏠 Executive | 8 KPI cards, churn split, tenure windows, revenue-lost treemap, auto-insights, actions |
| 👥 Customers | Demographics, household types, senior vs non-senior behaviour |
| 📦 Services | Internet, phone, add-on protection effect, bundles, price vs service |
| 💰 Revenue | MRR, ARPU, billing choices, Pareto of revenue loss, lifetime value |
| ⏳ Lifecycle | Kaplan-Meier retention curves, churn by tenure window, lifecycle heatmaps |
| 🎬 Simulator | Pick retention actions, get customers saved, revenue gain, cost and ROI |
| 🤖 Predictor | Random Forest metrics, single-customer risk gauge, downloadable retention hit-list |
| 🗂️ Explorer | Build-your-own pivot heatmap, search and export data |

## 🎛️ Filters (work on every page)
Quick-segment pills (New, Loyal, High value, Month-to-month, High-risk profile, Seniors), dropdown multi-selects for 8 fields, tenure and charge sliders, one-click reset.

## 🔑 Key findings
- Month-to-month churn is ~43% vs ~3% for two-year contracts.
- ~47% of first-year customers churn vs ~17% afterwards.
- Electronic-check payers (~45%) and fiber customers (~42%) churn the most.
- Customers with Tech Support or Online Security churn ~15% vs ~42% without.

## 🛠️ Tech stack
Python · Streamlit (multipage navigation) · Plotly · scikit-learn · pandas

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Structure
```
app.py            # navigation + global filters
utils.py          # data, CSS, filters, chart helpers, ML model
views/            # one file per page
data/telco_churn.csv
.streamlit/config.toml
```

## 📊 Data
IBM Telco Customer Churn sample dataset (7,043 customers).
