"""
Telco Customer Churn Intelligence - multi-page Streamlit dashboard.
Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Telco Churn Intelligence", page_icon="📉", layout="wide",
                   initial_sidebar_state="collapsed")

from utils import inject_css, load_data, render_filters  # noqa: E402

inject_css()

pages = [
    st.Page("views/executive.py", title="Executive", icon="🏠", default=True),
    st.Page("views/customers.py", title="Customers", icon="👥"),
    st.Page("views/services.py", title="Services", icon="📦"),
    st.Page("views/revenue.py", title="Revenue", icon="💰"),
    st.Page("views/lifecycle.py", title="Lifecycle", icon="⏳"),
    st.Page("views/simulator.py", title="Simulator", icon="🎬"),
    st.Page("views/predictor.py", title="Predictor", icon="🤖"),
    st.Page("views/explorer.py", title="Explorer", icon="🗂️"),
]
try:
    pg = st.navigation(pages, position="top")      # top nav bar (newer Streamlit)
except TypeError:
    pg = st.navigation(pages)                      # fallback: sidebar nav

df = load_data()
d = render_filters(df)
if d.empty:
    st.warning("No customers match these filters. Open **Filters & slicers** and press **Reset all**.")
    st.stop()
pg.run()
