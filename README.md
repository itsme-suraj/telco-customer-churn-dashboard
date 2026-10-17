# 📉 Telco Customer Churn Intelligence Dashboard

An interactive analytics dashboard that shows **who is churning, why, and how much revenue is at risk**, with an ML model that predicts churn for any customer.

🔗 **Live demo:** _add your Streamlit link here_

![Dashboard preview](screenshots/overview.png)

## ✨ Features
- **8 live KPI cards** (churn rate vs overall, revenue at risk, avg tenure, lifetime value...)
- **Sidebar slicers**: contract, internet service, payment method, tenure, monthly charges, demographics + one-click reset
- **Overview, Deep Dive, Churn Predictor, Insights, Data** tabs
- **Random Forest churn predictor** with ROC curve, confusion matrix, feature importance
- **Single-customer risk gauge** with recommended retention actions
- **Retention hit-list**: active customers ranked by churn risk (downloadable CSV)
- **Auto-generated insights** that change with the filters

## 🔑 Key insights
- Month-to-month contracts churn at ~43% vs ~3% for two-year contracts.
- Customers in their first 12 months are the most likely to leave.
- Electronic-check payers and fiber-optic customers churn the most.
- Tech Support and Online Security customers churn far less.

## 🛠️ Tech stack
Python · Streamlit · Plotly · scikit-learn · pandas

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Structure
```
app.py                 # dashboard
data/telco_churn.csv   # IBM Telco churn dataset (7,043 customers)
requirements.txt
.streamlit/config.toml # dark theme
```

## 📊 Data
IBM Telco Customer Churn sample dataset.
