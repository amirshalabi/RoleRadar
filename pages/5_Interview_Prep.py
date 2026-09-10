"""
Interview Prep page: a persisted, diagnostic-adaptive study plan for one
tracked application with an interview date.

This file renders only. All planning arithmetic comes from
backend.services.prep, which wraps backend.planning.scheduler (the
initial plan) and backend.planning.adaptive (diagnostic-driven
revisions) with real Postgres persistence
(backend.db.study_plans, backend.db.assessment_results). A plan and its
completed-task history are read fresh from the database on every rerun
- marking a task complete or submitting a diagnostic immediately
persists through backend.services.prep, never held only in
st.session_state. The one thing kept in session_state is the last
diagnostic's before/after readiness numbers for display - transient UI
feedback, not the plan's source of truth.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import streamlit as st

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import calculate_skill_gaps
from backend.planning.adaptive import revise_study_plan
from backend.planning.readiness import DiagnosticResult, calculate_readiness, get_readiness_weights
from backend.planning.scheduler import StudyPlan, generate_study_plan
from backend.services import discovery, prep, tracking
from ui import components
from ui_common import configure_page, database_not_configured_notice, get_current_user_id, is_demo_mode

configure_page("Interview Prep", icon="🧠")
components.render_page_header(
    "Prep Terminal",
    "Interview Readiness",
    "A persisted, diagnostic-adaptive study plan for one tracked application.",
)


# ---------------------------------------------------------------------
# Demo data - the same candidate/role used elsewhere in demo mode. The
# demo plan and its diagnostics live only in st.session_state (there is
# no real application/database to persist against in demo mode), but
# every number is still produced by the REAL scheduler/adaptive modules.
# ---------------------------------------------------------------------

_DEMO_ROLE_FAMILY = "quant"
_DEMO_INTERVIEW_DATE = date.today() + timedelta(days=6)
_DEMO_REQUIREMENTS = [
    RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8, importance=9, required=True, evidence=["x"]),
    RoleRequirement(skill="Python", normalized_skill="python", target_level=7, importance=7, required=True, evidence=["x"]),
]


def _demo_profile() -> CandidateProfile:
    return CandidateProfile(
        skills=[
            CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.5, confidence=0.7),
            CandidateSkillEstimate(normalized_skill_name="algorithms", display_name="Algorithms", estimated_level=4.0, confidence=0.4),
            CandidateSkillEstimate(normalized_skill_name="probability", display_name="Probability", estimated_level=3.0, confidence=0.3),
        ]
    )


def _init_demo_state() -> None:
    if "demo_prep_tasks" in st.session_state:
        return
    profile = _demo_profile()
    gaps = calculate_skill_gaps(profile.skills, _DEMO_REQUIREMENTS)
    readiness = calculate_readiness(profile, _DEMO_ROLE_FAMILY)
    plan = generate_study_plan(
        gaps, interview_date=_DEMO_INTERVIEW_DATE, current_date=date.today(), hours_available_per_day=2.0, readiness=readiness
    )
    st.session_state["demo_prep_meta"] = {
        "interview_date": plan.interview_date.isoformat(),
        "hours_available_per_day": plan.hours_available_per_day,
        "days_remaining": plan.days_remaining,
        "scheduling_days": plan.scheduling_days,
        "total_available_minutes": plan.total_available_minutes,
        "version": 1,
    }
    st.session_state["demo_prep_tasks"] = [
        {**prep.task_to_row(t), "id": f"demo-task-{i}"} for i, t in enumerate(plan.tasks)
    ]
    st.session_state["demo_prep_diagnostics"] = []


def _demo_plain_plan() -> StudyPlan:
    meta = st.session_state["demo_prep_meta"]
    tasks = [prep.row_to_task(t) for t in st.session_state["demo_prep_tasks"]]
    return StudyPlan(
        interview_date=date.fromisoformat(meta["interview_date"]),
        current_date=date.today(),
        days_remaining=meta["days_remaining"],
        scheduling_days=meta["scheduling_days"],
        hours_available_per_day=meta["hours_available_per_day"],
        total_available_minutes=meta["total_available_minutes"],
        prep_items=[],
        allocations=[],
        tasks=tasks,
        notes=[],
    )


def _demo_mark_task_complete(task_id: str, is_complete: bool) -> None:
    for task in st.session_state["demo_prep_tasks"]:
        if task["id"] == task_id:
            task["is_complete"] = is_complete


def _demo_current_readiness() -> Any:
    diagnostics = st.session_state["demo_prep_diagnostics"]
    return calculate_readiness(_demo_profile(), _DEMO_ROLE_FAMILY, diagnostics)


def _demo_submit_diagnostic(topic: str, observed_level: float, confidence: float) -> dict:
    prev_diagnostics = st.session_state["demo_prep_diagnostics"]
    profile = _demo_profile()
    readiness_before = calculate_readiness(profile, _DEMO_ROLE_FAMILY, prev_diagnostics)

    previous_plan = _demo_plain_plan()
    new_diagnostic = DiagnosticResult(topic=topic, observed_level=observed_level, confidence=confidence)
    revised = revise_study_plan(
        previous_plan, new_diagnostic, profile, _DEMO_ROLE_FAMILY, _DEMO_REQUIREMENTS, date.today(), prev_diagnostics
    )

    st.session_state["demo_prep_diagnostics"] = [*prev_diagnostics, new_diagnostic]
    st.session_state["demo_prep_meta"] = {
        "interview_date": revised.interview_date.isoformat(),
        "hours_available_per_day": revised.hours_available_per_day,
        "days_remaining": revised.days_remaining,
        "scheduling_days": revised.scheduling_days,
        "total_available_minutes": revised.remaining_available_minutes,
        "version": st.session_state["demo_prep_meta"]["version"] + 1,
    }
    all_tasks = [*revised.completed_tasks, *revised.new_tasks]
    st.session_state["demo_prep_tasks"] = [
        {**prep.task_to_row(t), "id": f"demo-task-{i}"} for i, t in enumerate(all_tasks)
    ]

    return {
        "readiness_before": readiness_before,
        "readiness_after": revised.readiness,
        "message": prep.build_readiness_change_message(readiness_before, revised.readiness),
    }


# ---------------------------------------------------------------------
# Shared rendering
# ---------------------------------------------------------------------


def _render_header(interview_date: str | None, days_remaining: int, hours_per_day: float, total_minutes: float, readiness) -> None:
    components.render_metric_strip(
        [
            {"label": "Interview Date", "value": interview_date or "—"},
            {"label": "Days Remaining", "value": str(days_remaining)},
            {"label": "Hours / Day", "value": f"{hours_per_day:g}"},
            {"label": "Total Prep Time", "value": f"{total_minutes / 60.0:.1f}h"},
            {"label": "Readiness", "value": f"{readiness.overall_readiness:.1f}", "tone": "gold"},
        ]
    )
    st.caption(
        "Projected readiness after finishing the remaining plan isn't modeled yet - our deterministic model "
        "only reports readiness as-measured. Submit a diagnostic below to see a real before/after comparison instead."
    )


def _render_topic_allocation(tasks: list[dict]) -> None:
    components.render_section_header("Topic Allocation")
    if not tasks:
        st.caption("No tasks scheduled.")
        return
    totals: dict[str, float] = {}
    labels: dict[str, str] = {}
    for task in tasks:
        skill = task["normalized_skill_name"]
        totals[skill] = totals.get(skill, 0.0) + (task.get("allocated_minutes") or 0.0)
        labels[skill] = task.get("display_name") or skill
    max_minutes = max(totals.values()) if totals else 1.0
    with components.panel("topic-allocation"):
        for skill, minutes in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
            components.render_bar(labels[skill].upper(), minutes, max_minutes, value_text=f"{minutes / 60.0:.1f}h")


def _render_tasks(tasks: list[dict], on_toggle) -> None:
    components.render_section_header("Daily Study Tasks")
    if not tasks:
        st.caption("No tasks scheduled yet.")
        return
    by_day: dict[int, list[dict]] = {}
    for task in tasks:
        by_day.setdefault(task.get("day_index") or 0, []).append(task)

    for day_index in sorted(by_day):
        day_tasks = by_day[day_index]
        day_label = day_tasks[0].get("scheduled_date") or f"Day {day_index + 1}"
        with components.panel(f"day-{day_index}"):
            st.markdown(f'<p class="rr-eyebrow">{day_label}</p>', unsafe_allow_html=True)
            for task_index, task in enumerate(day_tasks):
                label = f"{task.get('display_name') or task['normalized_skill_name']} — {task.get('allocated_minutes', 0):.0f} min ({task.get('task_description') or ''})"
                checked = st.checkbox(label, value=bool(task["is_complete"]), key=f"task_{task['id']}")
                if checked != bool(task["is_complete"]):
                    on_toggle(task["id"], checked)
                    st.rerun()


def _render_diagnostic_form(role_family: str | None, on_submit) -> None:
    components.render_section_header("Enter a Diagnostic Result", "Recording a real diagnostic replans your remaining (incomplete) schedule - completed tasks are never touched.")
    topics = sorted(get_readiness_weights(role_family).keys())
    with st.form("diagnostic_form", clear_on_submit=True):
        topic = st.selectbox("Topic", options=topics, format_func=lambda t: t.replace("_", " ").title())
        observed_level = st.slider("Observed level (0-10)", 0.0, 10.0, 5.0, 0.5)
        confidence = st.slider("Confidence in this result", 0.0, 1.0, 0.9, 0.05)
        submitted = st.form_submit_button("Submit Diagnostic & Replan", type="primary")
    if submitted:
        on_submit(topic, observed_level, confidence)


def _render_diagnostic_result(result: dict) -> None:
    st.info(result["message"])
    st.caption("This reflects the deterministic scoring model's before/after numbers only - not a claim about what specifically caused them.")
    before, after = result["before"], result["after"]
    components.render_metric_strip(
        [
            {"label": "Readiness Before", "value": f"{before:.1f}"},
            {"label": "Readiness After", "value": f"{after:.1f}", "tone": "positive" if after >= before else "negative"},
            {"label": "Change", "value": f"{after - before:+.1f}", "tone": "positive" if after >= before else "negative"},
        ]
    )


# ---------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------

if is_demo_mode():
    _init_demo_state()
    st.caption("Demo scenario: Quantitative Research Intern at Meridian Capital.")

    meta = st.session_state["demo_prep_meta"]
    readiness = _demo_current_readiness()
    _render_header(meta["interview_date"], meta["days_remaining"], meta["hours_available_per_day"], meta["total_available_minutes"], readiness)

    components.divider()
    _render_topic_allocation(st.session_state["demo_prep_tasks"])

    components.divider()
    _render_tasks(st.session_state["demo_prep_tasks"], _demo_mark_task_complete)

    components.divider()

    def _on_demo_submit(topic: str, observed_level: float, confidence: float) -> None:
        result = _demo_submit_diagnostic(topic, observed_level, confidence)
        st.session_state["demo_last_diagnostic"] = {
            "message": result["message"],
            "before": result["readiness_before"].overall_readiness,
            "after": result["readiness_after"].overall_readiness,
        }
        st.rerun()

    _render_diagnostic_form(_DEMO_ROLE_FAMILY, _on_demo_submit)

    if "demo_last_diagnostic" in st.session_state:
        _render_diagnostic_result(st.session_state["demo_last_diagnostic"])

else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()

    applications = tracking.list_applications_with_details(user_id)
    with_interview = [a for a in applications if a.get("interview_date")]
    if not with_interview:
        components.render_empty_state("No applications with an interview date yet", detail="Set an interview date on the Applications page first.")
        st.stop()
    with_interview.sort(key=lambda a: a["interview_date"])

    options = {a["role_id"]: a for a in with_interview}
    selected_role_id = st.selectbox(
        "Select an application",
        options=list(options.keys()),
        format_func=lambda rid: (
            f"{(options[rid].get('roles') or {}).get('title', 'Unknown role')} at "
            f"{(options[rid].get('roles') or {}).get('company', 'Unknown company')} — interview {options[rid]['interview_date']}"
        ),
    )

    role_row = discovery.get_role(selected_role_id)
    role_family = role_row.get("role_family") if role_row else None

    try:
        plan = prep.get_or_create_plan_view(user_id, selected_role_id)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()

    readiness = prep.get_current_readiness(user_id, selected_role_id)

    _render_header(plan["interview_date"], plan["days_remaining"], plan["hours_available_per_day"], plan["total_available_minutes"], readiness)

    components.divider()
    _render_topic_allocation(plan["tasks"])

    components.divider()
    _render_tasks(plan["tasks"], lambda task_id, checked: prep.mark_task_complete(task_id, checked))

    components.divider()

    result_key = f"last_diagnostic_{selected_role_id}"

    def _on_submit(topic: str, observed_level: float, confidence: float) -> None:
        result = prep.submit_diagnostic(user_id, selected_role_id, topic, observed_level, confidence)
        st.session_state[result_key] = {
            "message": result["message"],
            "before": result["readiness_before"].overall_readiness,
            "after": result["readiness_after"].overall_readiness,
        }
        st.rerun()

    _render_diagnostic_form(role_family, _on_submit)

    if result_key in st.session_state:
        _render_diagnostic_result(st.session_state[result_key])
