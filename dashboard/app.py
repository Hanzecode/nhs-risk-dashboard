"""
NHS Readmission Risk Dashboard — entry point.
Run with: streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="NHS Readmission Risk",
    page_icon=":material/ecg_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview = st.Page(
    "app_pages/overview.py", title="Overview",
    icon=":material/dashboard:", default=True,
)
conditions = st.Page(
    "app_pages/conditions.py", title="Conditions",
    icon=":material/stethoscope:",
)
risk = st.Page(
    "app_pages/risk.py", title="Risk stratification",
    icon=":material/emergency_heat:",
)
methodology = st.Page(
    "app_pages/methodology.py", title="Methodology",
    icon=":material/science:",
)

nav = st.navigation([overview, conditions, risk, methodology])

with st.sidebar:
    st.markdown("### :material/ecg_heart: NHS risk")
    st.caption("Cohort")
    st.markdown("Synthetic clinical data · 10,000 episodes")
    st.caption("Risk model")
    st.markdown("LACE index — rule-based, explainable")

nav.run()
