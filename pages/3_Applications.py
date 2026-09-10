"""
Applications page: pipeline tracking from discovered through
offer/rejected/withdrawn, with fit/readiness/priority indicators and an
on-demand interview prep plan.

All state changes go through backend.services.tracking - this file only
renders what that service (and backend.services.discovery, for
fit/readiness/priority and the prep plan) returns and forwards user
actions to it. Applications are read fresh from Postgres on every rerun
(never solely from st.session_state), so a status/date/notes edit is
immediately reflected the next time this script runs - the only
domain-shaped thing kept in session_state is which role's prep plan is
currently expanded, which is transient UI state, not application data.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from backend.db.applications import VALID_STATUSES
from backend.services import discovery, tracking
from ui import components
from ui_common import configure_page, database_not_configured_notice, get_current_user_id, is_demo_mode

configure_page("Applications", icon="📋")
components.render_page_header(
    "Application Pipeline",
    "Applications",
    "Discovered → Saved → Applied → OA → Interview → Offer / Rejected / Withdrawn.",
)

STAGE_ORDER = ["discovered", "saved", "applied", "oa", "interview", "offer", "rejected", "withdrawn"]
STAGE_LABEL = {
    "discovered": "Discovered", "saved": "Saved", "applied": "Applied", "oa": "OA",
    "interview": "Interview", "offer": "Offer", "rejected": "Rejected", "withdrawn": "Withdrawn",
}
_TERMINAL_TONE = {"offer": "positive", "rejected": "negative", "interview": "gold"}

_DEMO_APPLICATIONS = [
    {
        "role_id": "demo-1", "title": "Quantitative Research Intern", "company": "Meridian Capital",
        "status": "interview", "application_date": "2026-08-20", "deadline": None, "interview_date": "2026-09-13",
        "notes": "Prep probability + mental math.", "fit_score": 57.4, "readiness_score": 31.5, "priority": "dream",
    },
    {
        "role_id": "demo-2", "title": "Software Engineering Intern", "company": "Acme Corp",
        "status": "applied", "application_date": "2026-08-28", "deadline": "2026-09-10", "interview_date": None,
        "notes": "", "fit_score": None, "readiness_score": 62.0, "priority": "backup",
    },
]


def _days_until(iso_date: str | None) -> int | None:
    """Calendar days from today until `iso_date` (negative if in the past). None if `iso_date` is None."""
    if not iso_date:
        return None
    return (date.fromisoformat(iso_date) - date.today()).days


def _days_delta_text(days: int | None) -> str:
    if days is None:
        return "—"
    if days < 0:
        return f"{-days}d ago"
    if days == 0:
        return "today"
    return f"in {days}d"


# ---------------------------------------------------------------------
# Load applications (Supabase is the source of truth)
# ---------------------------------------------------------------------

user_id: str | None = None
demo = is_demo_mode()

if demo:
    applications = _DEMO_APPLICATIONS
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()

    raw_applications = tracking.list_applications_with_details(user_id)
    card_by_role_id = {card.role_id: card for card in discovery.list_role_cards(user_id)}
    applications = []
    for row in raw_applications:
        role = row.get("roles") or {}
        card = card_by_role_id.get(row["role_id"])
        applications.append(
            {
                "role_id": row["role_id"],
                "title": role.get("title", "Unknown role"),
                "company": role.get("company", "Unknown company"),
                "status": row["status"],
                "application_date": row.get("application_date"),
                "deadline": row.get("deadline"),
                "interview_date": row.get("interview_date"),
                "notes": row.get("notes") or "",
                "hours_available_per_day": row.get("hours_available_per_day"),
                "fit_score": card.fit_score if card else None,
                "readiness_score": card.readiness_score if card else None,
                "priority": card.priority if card else None,
            }
        )

if not applications:
    components.render_empty_state("No applications tracked yet", detail="Saving a role from Discover automatically starts tracking it here.")
    st.stop()


# ---------------------------------------------------------------------
# Pipeline metric strip
# ---------------------------------------------------------------------

_INACTIVE = {"rejected", "withdrawn", "offer"}
total_count = len(applications)
active_count = sum(1 for a in applications if a["status"] not in _INACTIVE)
interview_count = sum(1 for a in applications if a["status"] == "interview")
offer_count = sum(1 for a in applications if a["status"] == "offer")
rejected_count = sum(1 for a in applications if a["status"] == "rejected")

components.render_metric_strip(
    [
        {"label": "Total", "value": str(total_count)},
        {"label": "Active", "value": str(active_count), "tone": "gold"},
        {"label": "Interviews", "value": str(interview_count)},
        {"label": "Offers", "value": str(offer_count), "tone": "positive"},
        {"label": "Rejected", "value": str(rejected_count), "tone": "negative"},
    ]
)


# ---------------------------------------------------------------------
# Render one application
# ---------------------------------------------------------------------


def _render_prep_plan_action(application: dict) -> None:
    """
    Surfaced whenever an application has an interview date - lets the
    user generate (or recalculate, if skills/requirements/date changed
    since last time) a fresh deterministic study plan
    (backend.planning.scheduler via backend.services.discovery). Nothing
    about the plan is persisted or cached across reruns - it is always
    computed fresh on click from current data, so there is no stale copy
    to keep in sync.
    """
    if not application["interview_date"]:
        return

    role_id = application["role_id"]
    plan_key = f"show_prep_plan_{role_id}"

    if st.button("Generate / Recalculate Prep Plan", key=f"prep_btn_{role_id}"):
        st.session_state[plan_key] = True

    if not st.session_state.get(plan_key):
        return

    with st.spinner("Building prep plan from current skills, requirements, and interview date..."):
        try:
            if demo:
                plan = _demo_study_plan(application)
            else:
                plan = discovery.build_study_plan_for_application(user_id, role_id)
        except ValueError as exc:
            st.warning(str(exc))
            return

    components.render_meta_line(
        [f"{plan.scheduling_days} day(s) remaining", f"{plan.hours_available_per_day:g}h/day", f"{plan.total_available_minutes:.0f} min available"]
    )
    if plan.notes:
        for note in plan.notes:
            st.caption(f"ℹ️ {note}")
    if not plan.tasks:
        st.caption("No prep tasks scheduled yet.")
    for task in plan.tasks[:8]:
        st.markdown(f"- **{task.display_name}** — {task.allocated_minutes:.0f} min ({task.description})")


def _demo_study_plan(application: dict):
    """Demo mode still runs the REAL scheduler - only the profile/requirements feeding it are synthetic."""
    from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
    from backend.ingestion.normalize import normalize_role
    from backend.llm.extract_requirements import RoleRequirement
    from backend.matching.cross_role import FAVORITE_PRIORITY_MULTIPLIERS
    from backend.matching.gaps import calculate_skill_gaps
    from backend.planning.readiness import calculate_readiness
    from backend.planning.scheduler import generate_study_plan

    profile = CandidateProfile(
        skills=[
            CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.5, confidence=0.7),
            CandidateSkillEstimate(normalized_skill_name="algorithms", display_name="Algorithms", estimated_level=4.0, confidence=0.4),
            CandidateSkillEstimate(normalized_skill_name="probability", display_name="Probability", estimated_level=3.0, confidence=0.3),
        ]
    )
    if application["role_id"] == "demo-1":
        role_family = "quant"
        requirements = [
            RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8, importance=9, required=True, evidence=["x"]),
            RoleRequirement(skill="Python", normalized_skill="python", target_level=7, importance=7, required=True, evidence=["x"]),
        ]
    else:
        role_family = "swe"
        requirements = [
            RoleRequirement(skill="Algorithms", normalized_skill="algorithms", target_level=7, importance=8, required=True, evidence=["x"]),
            RoleRequirement(skill="Python", normalized_skill="python", target_level=6, importance=6, required=True, evidence=["x"]),
        ]
    normalize_role({"title": application["title"], "company": application["company"], "role_family": role_family})

    gaps = calculate_skill_gaps(profile.skills, requirements)
    readiness = calculate_readiness(profile, role_family)
    priority_multiplier = FAVORITE_PRIORITY_MULTIPLIERS.get(application["priority"], 1.0)

    return generate_study_plan(
        gaps,
        interview_date=date.fromisoformat(application["interview_date"]),
        current_date=date.today(),
        hours_available_per_day=2.0,
        readiness=readiness,
        favorite_priority_multiplier=priority_multiplier,
    )


def _render_application(application: dict) -> None:
    with components.card(f"app-{application['role_id']}"):
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown(f"**{application['title']}**")
            components.render_meta_line([application["company"]])
        with header_cols[1]:
            components.render_status_badge(
                STAGE_LABEL.get(application["status"], application["status"]).upper(),
                tone=_TERMINAL_TONE.get(application["status"], "default"),
            )

        indicator_cols = st.columns(4)
        with indicator_cols[0]:
            st.markdown('<p class="rr-metric-label">Fit</p>', unsafe_allow_html=True)
            components.render_score_badge(application["fit_score"], size="sm")
        with indicator_cols[1]:
            st.markdown('<p class="rr-metric-label">Readiness</p>', unsafe_allow_html=True)
            components.render_score_badge(application["readiness_score"], size="sm")
        with indicator_cols[2]:
            deadline_days = _days_until(application["deadline"])
            st.markdown('<p class="rr-metric-label">Deadline</p>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="rr-meta">{application["deadline"] or "—"} ({_days_delta_text(deadline_days)})</span>',
                unsafe_allow_html=True,
            )
        with indicator_cols[3]:
            interview_days = _days_until(application["interview_date"])
            st.markdown('<p class="rr-metric-label">Interview</p>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="rr-meta">{application["interview_date"] or "—"} ({_days_delta_text(interview_days)})</span>',
                unsafe_allow_html=True,
            )

        if application["priority"]:
            components.render_priority_badge(application["priority"])

        if application["notes"]:
            st.caption(f"📝 {application['notes']}")

        _render_prep_plan_action(application)

        if demo:
            return

        with st.expander("Update stage / dates / notes"):
            new_status = st.selectbox(
                "Stage", options=sorted(VALID_STATUSES), index=sorted(VALID_STATUSES).index(application["status"]),
                key=f"status_{application['role_id']}",
            )
            date_cols = st.columns(3)
            new_application_date = date_cols[0].date_input(
                "Application date",
                value=date.fromisoformat(application["application_date"]) if application["application_date"] else None,
                key=f"appdate_{application['role_id']}",
            )
            new_deadline = date_cols[1].date_input(
                "Deadline", value=date.fromisoformat(application["deadline"]) if application["deadline"] else None,
                key=f"deadline_{application['role_id']}",
            )
            new_interview_date = date_cols[2].date_input(
                "Interview date", value=date.fromisoformat(application["interview_date"]) if application["interview_date"] else None,
                key=f"interview_{application['role_id']}",
            )
            new_hours = st.number_input(
                "Hours available per day (for the prep plan)",
                min_value=0.0, max_value=24.0, step=0.5,
                value=float(application.get("hours_available_per_day") or discovery.DEFAULT_PREP_HOURS_PER_DAY),
                key=f"hours_{application['role_id']}",
            )
            new_notes = st.text_area("Notes", value=application["notes"], key=f"appnotes_{application['role_id']}")

            if st.button("Save", key=f"save_app_{application['role_id']}", type="primary"):
                old_interview_date = application["interview_date"]
                new_interview_date_iso = new_interview_date.isoformat() if new_interview_date else None

                tracking.update_application_stage(
                    user_id,
                    application["role_id"],
                    status=new_status,
                    application_date=new_application_date.isoformat() if new_application_date else None,
                    deadline=new_deadline.isoformat() if new_deadline else None,
                    interview_date=new_interview_date_iso,
                    notes=new_notes or None,
                    hours_available_per_day=new_hours,
                )

                if new_interview_date_iso and new_interview_date_iso != old_interview_date:
                    # Interview date was just added/changed - drop any
                    # stale prep plan display so the next view is a fresh
                    # "Generate" prompt against the new date, not last
                    # run's schedule.
                    st.session_state.pop(f"show_prep_plan_{application['role_id']}", None)
                    st.success("Saved. Interview date updated - generate a prep plan below.")

                st.rerun()


# ---------------------------------------------------------------------
# Grouped by default; a stage filter switches to a flat filtered list
# ---------------------------------------------------------------------

status_filter = st.multiselect(
    "Filter by stage (leave empty to group all applications by stage)", options=STAGE_ORDER,
)

if status_filter:
    visible = [a for a in applications if a["status"] in status_filter]
    if not visible:
        st.caption("No applications at the selected stage(s).")
    for application in visible:
        _render_application(application)
else:
    for stage in STAGE_ORDER:
        stage_applications = [a for a in applications if a["status"] == stage]
        if not stage_applications:
            continue
        components.render_section_header(f"{STAGE_LABEL[stage]} ({len(stage_applications)})")
        for application in stage_applications:
            _render_application(application)
