"""
Favorites page: saved/flagged roles, priority, and notes.

All state changes go through backend.services.tracking - this file only
renders what that service returns and forwards user actions to it.
"""

from __future__ import annotations

import streamlit as st

from backend.db.favorites import PRIORITY_LEVELS
from backend.services import tracking
from ui_common import configure_page, database_not_configured_notice, empty_state, get_current_user_id, is_demo_mode

configure_page("Favorites", icon="🚩")
st.title("🚩 Favorites")
st.caption("Roles you've flagged, with priority and notes - persisted in Postgres, not session state.")
st.divider()

_DEMO_FAVORITES = [
    {"role_id": "demo-1", "title": "Quantitative Research Intern", "company": "Meridian Capital", "priority": "dream", "notes": "Review probability before applying."},
    {"role_id": "demo-2", "title": "Software Engineering Intern", "company": "Acme Corp", "priority": "backup", "notes": ""},
]


def _priority_badge(priority: str) -> str:
    return {"dream": "💎 Dream", "high": "🔥 High", "interested": "🙂 Interested", "backup": "🧊 Backup"}.get(priority, priority)


user_id: str | None = None

if is_demo_mode():
    favorites = _DEMO_FAVORITES
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()
    raw_favorites = tracking.list_saved_roles_with_details(user_id)
    favorites = [
        {
            "role_id": row["role_id"],
            "title": (row.get("roles") or {}).get("title", "Unknown role"),
            "company": (row.get("roles") or {}).get("company", "Unknown company"),
            "priority": row["priority"],
            "notes": row.get("notes") or "",
        }
        for row in raw_favorites
    ]

if not favorites:
    empty_state("No saved roles yet", detail="Flag roles you're interested in from the Discover page to see them here.")
    st.stop()

for favorite in favorites:
    with st.container(border=True):
        cols = st.columns([3, 1])
        cols[0].markdown(f"**{favorite['title']}** at {favorite['company']}")
        cols[1].markdown(_priority_badge(favorite["priority"]))

        if st.button("🔬 View full analysis", key=f"analysis_{favorite['role_id']}"):
            st.session_state["selected_role_id"] = favorite["role_id"]
            st.switch_page("pages/7_Role_Analysis.py")

        if is_demo_mode():
            if favorite["notes"]:
                st.caption(favorite["notes"])
            continue

        with st.expander("Edit priority / notes / unsave"):
            priority_options = sorted(PRIORITY_LEVELS)
            new_priority = st.selectbox(
                "Priority", options=priority_options, index=priority_options.index(favorite["priority"]),
                key=f"priority_{favorite['role_id']}",
            )
            if new_priority != favorite["priority"]:
                tracking.set_role_priority(user_id, favorite["role_id"], new_priority)
                st.rerun()

            new_notes = st.text_area("Notes", value=favorite["notes"], key=f"notes_{favorite['role_id']}")
            if st.button("Save notes", key=f"save_notes_{favorite['role_id']}"):
                tracking.set_role_notes(user_id, favorite["role_id"], new_notes)
                st.rerun()

            if st.button("Remove from favorites", key=f"unsave_{favorite['role_id']}", type="secondary"):
                tracking.unsave_role(user_id, favorite["role_id"])
                st.rerun()
