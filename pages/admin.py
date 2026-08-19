"""Dedicated Streamlit admin view (URL: /admin)."""

import streamlit as st

from ui import admin_view, inject_theme

st.set_page_config(
    page_title="Cardiac Admin Portal",
    page_icon="🛡️",
    layout="wide",
)
inject_theme()
admin_view()
