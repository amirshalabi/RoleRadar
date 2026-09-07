"""
Applications page: pipeline tracking from discovered through
offer/rejected/withdrawn.

All state changes go through backend.services.tracking - this file only
renders what that service returns and forwards user actions to it.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from backend.db.applications import VALID_STATUSES
from backend.services import tracking
from ui_common import configure_page, database_not_configured_notice, empty_state, get_current_user_id, is_demo_mode

configure_page("Applications", icon="📋")
st.title("📋 Applications")
st.caption("Discovered → Saved → Applied → OA → Interview → Offer / Rejected / Withdrawn.")
st.divider()

STAGE_ORDER = ["discovered", "saved", "applied", "oa", "interview", "offer", "rejected", "withdrawn"]

_DEMO_APPLICATIONS = [
    {"role_id": "demo-1", "title": "Quantitative Research Intern", "company": "Meridian Capital", "status": "interview", "application_date": "2026-08-20", "deadline": None, "interview_date": "2026-09-13", "notes": "Prep probability + mental math."},
    {"role_id": "demo-2", "title": "Software Engineering Intern", "company": "Acme Corp", "status": "applied", "application_date": "2026-08-28", "deadline": "2026-09-10", "interview_date": None, "notes": ""},
]

user_id: str | None = None

if is_demo_mode():
    applications = _DEMO_APPLICATIONS
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()
    raw_applications = tracking.list_applications_with_details(user_id)
    applications = [
        {
            "role_id": row["role_id"],
            "title": (row.get("roles") or {}).get("title", "Unknown role"),
            "company": (row.get("roles") or {}).get("company", "Unknown company"),
            "status": row["status"],
            "application_date": row.get("application_date"),
            "deadline": row.get("deadline"),
            "interview_date": row.get("interview_date"),
            "notes": row.get("notes") or "",
        }
        for row in raw_applications
    ]

if not applications:
    empty_state("No applications tracked yet", detail="Saving a role from Discover automatically starts tracking it here.")
    st.stop()

status_filter = st.multiselect("Filter by stage", options=STAGE_ORDER, default=[])
visible = [a for a in applications if not status_filter or a["status"] in status_filter]

for application in visible:
    with st.container(border=True):
        cols = st.columns([3, 1])
        cols[0].markdown(f"**{application['title']}** at {application['company']}")
        cols[1].markdown(f"`{application['status']}`")

        detail_bits = []
        if application["application_date"]:
            detail_bits.append(f"Applied {application['application_date']}")
        if application["deadline"]:
            detail_bits.append(f"Deadline {application['deadline']}")
        if application["interview_date"]:
            detail_bits.append(f"Interview {application['interview_date']}")
        if detail_bits:
            st.caption(" · ".join(detail_bits))
        if application["notes"]:
            st.caption(f"📝 {application['notes']}")

        if is_demo_mode():
            continue

        with st.expander("Update stage / dates / notes"):
            new_status = st.selectbox(
                "Stage", options=sorted(VALID_STATUSES), index=sorted(VALID_STATUSES).index(application["status"]),
                key=f"status_{application['role_id']}",
            )
            new_deadline = st.date_input(
                "Deadline", value=date.fromisoformat(application["deadline"]) if application["deadline"] else None,
                key=f"deadline_{application['role_id']}",
            )
            new_interview_date = st.date_input(
                "Interview date", value=date.fromisoformat(application["interview_date"]) if application["interview_date"] else None,
                key=f"interview_{application['role_id']}",
            )
            new_notes = st.text_area("Notes", value=application["notes"], key=f"appnotes_{application['role_id']}")

            if st.button("Save", key=f"save_app_{application['role_id']}"):
                tracking.update_application_stage(
                    user_id,
                    application["role_id"],
                    status=new_status,
                    deadline=new_deadline.isoformat() if new_deadline else None,
                    interview_date=new_interview_date.isoformat() if new_interview_date else None,
                    notes=new_notes or None,
                )
                st.rerun()
