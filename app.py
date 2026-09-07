"""
RoleRadar Dashboard - the main multipage app entrypoint.

Renders backend.services.dashboard.DashboardData. Contains no business
logic of its own - every number shown here was computed by
backend.services.dashboard (which in turn calls backend.matching /
backend.planning / backend.db), never by this file. Feature pages live
under pages/ and are picked up automatically by Streamlit's native
multipage navigation.
"""

from __future__ import annotations

import streamlit as st

from backend.services.dashboard import DashboardData, build_dashboard_data, build_demo_dashboard_data
from ui_common import configure_page, database_not_configured_notice, empty_state, get_current_user_id, is_demo_mode

configure_page("Dashboard", icon="🎯")

st.title("🎯 RoleRadar")
st.caption("AI-powered opportunity matching and adaptive interview readiness.")


def _load_dashboard_data() -> DashboardData | None:
    if is_demo_mode():
        return build_demo_dashboard_data()

    user_id = get_current_user_id()
    if user_id is None:
        return None

    data = build_dashboard_data(user_id)
    if not data.database_connected:
        return None
    return data


data = _load_dashboard_data()

if data is None:
    database_not_configured_notice()
    st.stop()

st.divider()

# --- Quick stats row ---
quick_cols = st.columns(4)
with quick_cols[0]:
    st.metric("Overall Readiness", f"{data.overall_readiness:.0f}%" if data.overall_readiness is not None else "—")
with quick_cols[1]:
    st.metric("Top Match", f"{data.top_match.overall_score:.0f}%" if data.top_match else "—")
with quick_cols[2]:
    st.metric("Saved Roles", data.saved_roles_count)
with quick_cols[3]:
    st.metric("Active Applications", data.active_applications_count)

st.divider()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("👤 Profile")
    with st.container(border=True):
        if data.profile_status.has_profile:
            st.markdown(f"**Resume on file** · {data.profile_status.skill_count} skills tracked")
            if data.profile_status.updated_at:
                st.caption(f"Last updated {data.profile_status.updated_at}")
        else:
            empty_state("No resume uploaded yet", detail="Upload a resume to unlock fit scoring and readiness tracking.")

    st.subheader("🏆 Top Match")
    with st.container(border=True):
        if data.top_match:
            st.markdown(f"**{data.top_match.role_title}** at {data.top_match.company}")
            st.progress(min(data.top_match.overall_score / 100, 1.0), text=f"{data.top_match.overall_score:.1f} / 100 fit")
        else:
            empty_state("No fit scores yet", detail="Score a job opportunity on the Discover page to see your best match here.")

    st.subheader("📅 Upcoming Interview")
    with st.container(border=True):
        if data.upcoming_interview:
            st.markdown(f"**{data.upcoming_interview.role_title}** at {data.upcoming_interview.company}")
            day_word = "day" if data.upcoming_interview.days_away == 1 else "days"
            st.caption(f"{data.upcoming_interview.interview_date} · {data.upcoming_interview.days_away} {day_word} away")
        else:
            empty_state(
                "No interview scheduled",
                detail="Set an interview date on the Applications page to track your countdown here.",
            )

with right:
    st.subheader("🧠 Interview Readiness")
    with st.container(border=True):
        if data.overall_readiness is not None:
            st.progress(
                min(data.overall_readiness / 100, 1.0),
                text=f"{data.overall_readiness:.1f} / 100 ({data.readiness_role_family or 'general'})",
            )
        else:
            empty_state(
                "Not enough data for a readiness estimate yet",
                detail="Track candidate skills and a top match to compute this.",
            )

    st.subheader("⚡ Highest-Leverage Skill")
    with st.container(border=True):
        if data.highest_leverage_skill:
            st.markdown(f"**{data.highest_leverage_skill}**")
            roi = data.highest_leverage_skill_roi
            st.caption(f"ROI score {roi:.0f} / 100 across your saved roles" if roi is not None else "")
        else:
            empty_state(
                "No skill ROI available yet",
                detail="Skill ROI needs extracted requirements for your saved roles - see the Skill Gaps page.",
            )

    st.subheader("📝 Today's Study Tasks")
    with st.container(border=True):
        empty_state(
            "No active study plan",
            detail="Build a deadline-aware prep plan on the Interview Prep page once you have an interview date.",
        )

st.divider()

st.subheader("🚨 Urgent Applications")
if data.urgent_applications:
    for application in data.urgent_applications:
        with st.container(border=True):
            cols = st.columns([3, 1, 1])
            cols[0].markdown(f"**{application.role_title}** at {application.company}")
            cols[1].caption(application.status.replace("_", " ").title())
            if application.days_until_deadline is not None:
                cols[2].caption(f"{application.days_until_deadline}d left")
            else:
                cols[2].caption("Interview stage")
else:
    empty_state("Nothing urgent right now", icon="✅")

st.divider()

st.subheader("📊 Pipeline Snapshot")
if data.pipeline_stats:
    stats = data.pipeline_stats
    stat_cols = st.columns(5)
    stat_cols[0].metric("Ingested", stats.ingested)
    stat_cols[1].metric("Deduplicated", stats.deduplicated)
    stat_cols[2].metric("Hard-filtered", stats.hard_filtered)
    stat_cols[3].metric("Keyword-filtered", stats.keyword_filtered)
    stat_cols[4].metric("Reached LLM", stats.llm_analyzed)
    if stats.llm_avoidance_rate is not None:
        st.success(
            f"**{stats.llm_avoidance_rate * 100:.0f}%** of candidate postings were eliminated before LLM analysis "
            f"in the most recent run."
        )
else:
    empty_state("No ingestion runs recorded yet", detail="Run the ingestion pipeline to see pipeline stats here.")
