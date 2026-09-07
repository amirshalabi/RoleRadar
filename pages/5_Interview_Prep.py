"""
Interview Prep page: interview readiness and a deadline-aware prep plan.

Readiness (backend.planning.readiness) only needs candidate skills and
a role family, both of which ARE persisted - so this page computes a
real readiness breakdown for your nearest upcoming interview. A full
minute-by-minute study plan (backend.planning.scheduler) additionally
needs extracted RoleRequirement data, which isn't persisted yet (see
Skill Gaps for the same limitation) - that section shows an honest
empty state in production and a real computation against sample data in
Demo mode.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.db import candidates as candidates_db
from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import calculate_skill_gaps
from backend.planning.readiness import calculate_readiness
from backend.planning.scheduler import generate_study_plan
from backend.services import tracking
from ui_common import configure_page, database_not_configured_notice, empty_state, get_current_user_id, is_demo_mode

configure_page("Interview Prep", icon="🧠")
st.title("🧠 Interview Prep")
st.divider()


def _skills_from_rows(rows: list[dict]) -> list[CandidateSkillEstimate]:
    return [
        CandidateSkillEstimate(
            normalized_skill_name=row["normalized_skill_name"],
            display_name=row.get("display_name") or row["normalized_skill_name"],
            estimated_level=row["estimated_level"],
            confidence=row["confidence"],
        )
        for row in rows
    ]


def _render_readiness(readiness) -> None:
    st.progress(min(readiness.overall_readiness / 100, 1.0), text=f"Overall readiness: {readiness.overall_readiness:.1f} / 100")
    cols = st.columns(len(readiness.topic_readiness))
    for col, topic in zip(cols, readiness.topic_readiness):
        col.metric(topic.topic.replace("_", " ").title(), f"{topic.readiness_score:.0f}")
    if readiness.major_gaps:
        st.markdown("**Major gaps**")
        for gap in readiness.major_gaps:
            st.caption(f"{gap.topic.replace('_', ' ').title()} · {gap.readiness_score:.0f}/100")


if is_demo_mode():
    profile = CandidateProfile(
        skills=[
            CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.5, confidence=0.7),
            CandidateSkillEstimate(normalized_skill_name="algorithms", display_name="Algorithms", estimated_level=4.0, confidence=0.4),
            CandidateSkillEstimate(normalized_skill_name="probability", display_name="Probability", estimated_level=3.0, confidence=0.3),
        ]
    )
    role_family = "quant"
    role = normalize_role({"title": "Quantitative Research Intern", "company": "Meridian Capital", "role_family": role_family})
    requirements = [
        RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8, importance=9, required=True, evidence=["x"]),
        RoleRequirement(skill="Python", normalized_skill="python", target_level=7, importance=7, required=True, evidence=["x"]),
    ]
    interview_date = date.today().replace(day=min(date.today().day + 6, 28))

    st.subheader("🧠 Readiness")
    with st.container(border=True):
        _render_readiness(calculate_readiness(profile, role_family))

    st.subheader("📝 Prep plan")
    with st.container(border=True):
        gaps = calculate_skill_gaps(profile.skills, requirements)
        plan = generate_study_plan(gaps, interview_date=interview_date, current_date=date.today(), hours_available_per_day=2.0)
        st.caption(f"{plan.scheduling_days} day(s) remaining · {plan.total_available_minutes:.0f} minutes available")
        for task in plan.tasks[:8]:
            st.markdown(f"- **{task.display_name}** — {task.allocated_minutes:.0f} min ({task.description})")
    st.stop()

user_id = get_current_user_id()
if user_id is None:
    database_not_configured_notice()
    st.stop()

applications = tracking.list_applications_with_details(user_id)
upcoming = sorted(
    (a for a in applications if a.get("interview_date") and date.fromisoformat(a["interview_date"]) >= date.today()),
    key=lambda a: a["interview_date"],
)
skill_rows = candidates_db.list_candidate_skills(user_id)

if not upcoming:
    empty_state("No upcoming interview", detail="Set an interview date on the Applications page to see readiness here.")
    st.stop()

if not skill_rows:
    empty_state("No candidate skills tracked yet", detail="Upload a resume to unlock readiness tracking.")
    st.stop()

next_interview = upcoming[0]
role = next_interview.get("roles") or {}
st.subheader(f"Next up: {role.get('title', 'Unknown role')} at {role.get('company', 'Unknown company')}")
st.caption(f"Interview on {next_interview['interview_date']}")

profile = CandidateProfile(skills=_skills_from_rows(skill_rows))
readiness = calculate_readiness(profile, role.get("role_family"))

with st.container(border=True):
    _render_readiness(readiness)

st.subheader("📝 Prep plan")
empty_state(
    "Not enough data for a full prep plan yet",
    detail="A minute-by-minute study plan needs extracted job requirements, which aren't persisted yet - see Skill Gaps.",
)
