"""
Streamlit medical dashboard: cardiac risk assessment, ECG visualizer, admin portal.

Run:
    streamlit run app.py

The dedicated admin route is available at /admin via pages/admin.py.
"""

from __future__ import annotations

import streamlit as st

from ui import admin_view, clinician_view, inject_theme

st.set_page_config(
    page_title="Cardiac Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    inject_theme()
    with st.sidebar:
        st.markdown("### Cardiac AI workstation")
        st.write("IEEE SPIN 2024 baseline + stacking / XGBoost ensembles.")
        page = st.radio("Navigation", ["Clinical assessment", "Admin portal"], index=0)
        st.caption("Dedicated admin view: sidebar Pages → admin, or `?view=admin`.")
    query_view = st.query_params.get("view", "")
    if page == "Admin portal" or str(query_view).lower() == "admin":
        admin_view()
    else:
        clinician_view()


if __name__ == "__main__":
    main()
