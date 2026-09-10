"""
RoleRadar Dashboard - the main multipage app entrypoint.

Renders backend.services.dashboard.DashboardData. Contains no business
logic of its own - every number shown here was computed by
backend.services.dashboard (which in turn calls backend.matching /
backend.planning / backend.db), never by this file. The one exception
is the compact application-pipeline strip, which calls the existing
backend.services.analytics.get_applications_by_stage() - a real,
already-used computation, just consumed here too. Feature pages live
under pages/ and are picked up automatically by Streamlit's native
multipage navigation.
"""

from __future__ import annotations

import streamlit as st

from backend.services.analytics import get_applications_by_stage
from backend.services.dashboard import DashboardData, build_dashboard_data, build_demo_dashboard_data
from ui import components
from ui_common import configure_page, database_not_configured_notice, get_current_user_id, is_demo_mode

configure_page("Dashboard", icon="🎯")

components.render_page_header(
    "Signal Brief",
    "Dashboard",
    "Your strongest opportunities, readiness signals, and next actions.",
)


def _load_dashboard_data() -> tuple[DashboardData | None, str | None]:
    if is_demo_mode():
        return build_demo_dashboard_data(), None

    user_id = get_current_user_id()
    if user_id is None:
        return None, None

    data = build_dashboard_data(user_id)
    if not data.database_connected:
        return None, None
    return data, user_id


data, user_id = _load_dashboard_data()

if data is None:
    database_not_configured_notice()
    st.stop()

if not data.profile_status.has_profile:
    components.render_onboarding_banner(
        "Start Here",
        "Start with your resume",
        "RoleRadar needs your background before it can rank opportunities, estimate readiness, or find skill gaps.",
    )
    if st.button("Upload Resume", key="dashboard-onboard-upload", type="primary"):
        st.switch_page("pages/0_Profile.py")

_STAGE_ORDER = ["discovered", "saved", "applied", "oa", "interview", "offer", "rejected", "withdrawn"]
_STAGE_LABEL = {
    "discovered": "Discovered", "saved": "Saved", "applied": "Applied", "oa": "OA",
    "interview": "Interview", "offer": "Offer", "rejected": "Rejected", "withdrawn": "Withdrawn",
}
_DEMO_STAGE_COUNTS = {"discovered": 3, "saved": 2, "applied": 1, "interview": 1}

# ---------------------------------------------------------------------
# Top metric strip
# ---------------------------------------------------------------------

components.render_metric_strip(
    [
        {
            "label": "Overall Readiness",
            "value": f"{data.overall_readiness:.0f}" if data.overall_readiness is not None else "—",
            "tone": "gold" if data.overall_readiness else "default",
        },
        {
            "label": "Top Match",
            "value": f"{data.top_match.overall_score:.0f}" if data.top_match else "—",
            "sublabel": data.top_match.company if data.top_match else None,
            "tone": "gold" if data.top_match else "default",
        },
        {"label": "Saved Roles", "value": str(data.saved_roles_count)},
        {"label": "Active Applications", "value": str(data.active_applications_count)},
        {
            "label": "Interviews",
            "value": "1" if data.upcoming_interview else "0",
        },
        {
            "label": "Skill Gaps",
            "value": "1" if data.highest_leverage_skill else "0",
        },
    ]
)

left, right = st.columns([65, 35], gap="large")

# ---------------------------------------------------------------------
# LEFT column
# ---------------------------------------------------------------------

with left:
    components.render_section_header("Top Opportunity")
    with components.card("dashboard-top-match", high_signal=bool(data.top_match)):
        if data.top_match:
            header_cols = st.columns([1, 6])
            with header_cols[0]:
                components.render_score_badge(data.top_match.overall_score, size="lg")
            with header_cols[1]:
                st.markdown(f"**{data.top_match.role_title}**")
                components.render_meta_line([data.top_match.company])
        else:
            st.markdown(
                '<p style="color:var(--rr-text-secondary);font-size:0.9rem;margin:0;">No fit scores yet</p>',
                unsafe_allow_html=True,
            )
            st.caption("Score a job opportunity on the Discover page to see your best match here.")

    components.render_section_header("Application Pipeline")
    if is_demo_mode():
        stage_counts = _DEMO_STAGE_COUNTS
    else:
        stage_counts = get_applications_by_stage(user_id) if user_id else {}
    if stage_counts:
        ordered = [s for s in _STAGE_ORDER if s in stage_counts]
        pipeline_metrics = [{"label": _STAGE_LABEL[s], "value": str(stage_counts[s])} for s in ordered]
        components.render_metric_strip(pipeline_metrics)
    else:
        components.render_empty_state(
            "No applications tracked yet",
            detail="Saving a role from Discover automatically starts tracking it here.",
        )

    components.render_section_header("Profile")
    with components.panel("dashboard-profile"):
        if data.profile_status.has_profile:
            st.markdown("**Resume analyzed ✓**")
            components.render_ranked_list(
                [
                    {"rank": "01", "label": "Skills identified", "value": str(data.profile_status.skill_count)},
                    {"rank": "02", "label": "Projects", "value": str(data.profile_status.project_count)},
                    {"rank": "03", "label": "Work experiences", "value": str(data.profile_status.experience_count)},
                ]
            )
            if data.profile_status.top_skills:
                st.markdown('<p class="rr-eyebrow">Top Skills</p>', unsafe_allow_html=True)
                components.render_tags(data.profile_status.top_skills, kind="gold")
            profile_cols = st.columns(2)
            with profile_cols[0]:
                if st.button("View Profile", key="dashboard-view-profile", use_container_width=True):
                    st.switch_page("pages/0_Profile.py")
            with profile_cols[1]:
                if st.button("Replace Resume", key="dashboard-replace-resume", use_container_width=True):
                    st.switch_page("pages/0_Profile.py")
        else:
            st.markdown(
                '<p style="color:var(--rr-text-secondary);font-size:0.9rem;margin:0;">No resume uploaded</p>',
                unsafe_allow_html=True,
            )
            st.caption("Upload your resume to unlock personalized fit scoring, skill-gap analysis, and interview readiness.")
            if st.button("Upload Resume", key="dashboard-profile-upload", type="primary", use_container_width=True):
                st.switch_page("pages/0_Profile.py")

    components.render_section_header("Urgent Applications")
    if data.urgent_applications:
        for index, application in enumerate(data.urgent_applications):
            with components.card(f"urgent-{index}"):
                cols = st.columns([3, 1, 1])
                cols[0].markdown(f"**{application.role_title}**")
                components.render_meta_line([application.company])
                cols[1].markdown(components.status_badge_html(application.status.replace("_", " ").upper()), unsafe_allow_html=True)
                if application.days_until_deadline is not None:
                    cols[2].markdown(f'<span class="rr-meta">{application.days_until_deadline}d left</span>', unsafe_allow_html=True)
                else:
                    cols[2].markdown('<span class="rr-meta">Interview stage</span>', unsafe_allow_html=True)
    else:
        components.render_empty_state("Nothing urgent right now")

# ---------------------------------------------------------------------
# RIGHT column
# ---------------------------------------------------------------------

with right:
    components.render_section_header("Interview Readiness")
    with components.panel("dashboard-readiness"):
        if data.overall_readiness is not None:
            components.render_bar(
                (data.readiness_role_family or "general").upper(),
                data.overall_readiness,
                value_text=f"{data.overall_readiness:.1f}",
            )
        else:
            st.markdown(
                '<p style="color:var(--rr-text-secondary);font-size:0.9rem;margin:0;">Not enough data for a readiness estimate yet</p>',
                unsafe_allow_html=True,
            )
            st.caption("Track candidate skills and a top match to compute this.")

    components.render_section_header("Highest-Leverage Skill")
    with components.panel("dashboard-leverage-skill"):
        if data.highest_leverage_skill:
            roi = data.highest_leverage_skill_roi
            components.render_ranked_list(
                [
                    {
                        "rank": "01",
                        "label": data.highest_leverage_skill,
                        "value": f"ROI {roi:.0f}" if roi is not None else "—",
                    }
                ]
            )
        else:
            st.markdown(
                '<p style="color:var(--rr-text-secondary);font-size:0.9rem;margin:0;">No skill ROI available yet</p>',
                unsafe_allow_html=True,
            )
            st.caption("Skill ROI needs extracted requirements for your saved roles - see the Skill Gaps page.")

    components.render_section_header("Upcoming Interview")
    with components.panel("dashboard-interview"):
        if data.upcoming_interview:
            st.markdown(f"**{data.upcoming_interview.role_title}**")
            components.render_meta_line([data.upcoming_interview.company])
            day_word = "day" if data.upcoming_interview.days_away == 1 else "days"
            components.render_meta_line([f"{data.upcoming_interview.interview_date}", f"{data.upcoming_interview.days_away} {day_word} away"])
            if st.button("Open Prep Plan", key="dashboard-open-prep", type="primary"):
                st.switch_page("pages/5_Interview_Prep.py")
        else:
            st.markdown(
                '<p style="color:var(--rr-text-secondary);font-size:0.9rem;margin:0;">No interview scheduled</p>',
                unsafe_allow_html=True,
            )
            st.caption("Set an interview date on the Applications page to track your countdown here.")

    components.render_section_header("Pipeline Snapshot")
    with components.panel("dashboard-pipeline"):
        if data.pipeline_stats:
            stats = data.pipeline_stats
            components.render_ranked_list(
                [
                    {"rank": "01", "label": "Ingested", "value": str(stats.ingested)},
                    {"rank": "02", "label": "Deduplicated", "value": str(stats.deduplicated)},
                    {"rank": "03", "label": "Hard-filtered", "value": str(stats.hard_filtered)},
                    {"rank": "04", "label": "Keyword-filtered", "value": str(stats.keyword_filtered)},
                    {"rank": "05", "label": "Reached LLM", "value": str(stats.llm_analyzed)},
                ]
            )
            if stats.llm_avoidance_rate is not None:
                components.render_status_badge(f"{stats.llm_avoidance_rate * 100:.0f}% LLM AVOIDANCE", tone="positive")
        else:
            st.markdown(
                '<p style="color:var(--rr-text-secondary);font-size:0.9rem;margin:0;">No ingestion runs recorded yet</p>',
                unsafe_allow_html=True,
            )
            st.caption("Run the ingestion pipeline to see pipeline stats here.")
