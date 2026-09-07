"""
Discover page: browse ingested job opportunities and save the ones
you're interested in.

There is no in-app trigger for the ingestion pipeline yet
(backend.ingestion.concurrent.run_ingestion_pipeline is run outside
Streamlit today - see its demo/tests) - this page only browses whatever
roles already exist in Postgres.
"""

from __future__ import annotations

import streamlit as st

from backend.services import discovery, tracking
from ui_common import configure_page, database_not_configured_notice, empty_state, get_current_user_id, is_demo_mode

configure_page("Discover", icon="🔍")
st.title("🔍 Discover")
st.caption("Roles currently in your RoleRadar database.")
st.divider()

_DEMO_ROLES = [
    {"external_id": "demo-1", "title": "Quantitative Research Intern", "company": "Meridian Capital", "location": "New York, NY", "description": "Build and backtest trading signals using Python and C++."},
    {"external_id": "demo-2", "title": "Software Engineering Intern", "company": "Acme Corp", "location": "Remote", "description": "Build backend services in Python."},
]

user_id: str | None = None

if is_demo_mode():
    roles = _DEMO_ROLES
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()
    roles = discovery.list_available_roles(limit=50)

if not roles:
    empty_state(
        "No roles ingested yet",
        detail="Run backend.ingestion.concurrent.run_ingestion_pipeline() to populate roles - "
        "see backend/ingestion/jobs.py for the demo source adapters.",
    )
    st.stop()

for role in roles:
    with st.container(border=True):
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(f"**{role['title']}** at {role['company']}")
            if role.get("location"):
                st.caption(role["location"])
            if role.get("description"):
                with st.expander("Description"):
                    st.write(role["description"])
        with cols[1]:
            if is_demo_mode():
                st.button("🚩 Save", key=f"save_{role['external_id']}", disabled=True, help="Saving is disabled in demo mode.")
            else:
                if st.button("🚩 Save", key=f"save_{role['id']}"):
                    tracking.save_role(user_id, role["id"])
                    st.success("Saved to Favorites.")
