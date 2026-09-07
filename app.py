"""
Streamlit landing page for RoleRadar.

This is the multipage app entrypoint. Feature pages live under pages/
and are picked up automatically by Streamlit's navigation.
"""

import streamlit as st

from backend.utils.logging import configure_logging

configure_logging()

st.set_page_config(page_title="RoleRadar", page_icon="🎯")

st.title("RoleRadar")
st.subheader("AI-powered opportunity matching and adaptive interview readiness.")
