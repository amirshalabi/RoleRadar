"""
Skill Gaps page: per-role and cross-role skill gap analysis.

Skill gap analysis (backend.matching.gaps) and cross-role skill ROI
(backend.matching.cross_role) both need extracted RoleRequirement data
for the candidate's saved roles. That extraction has no persistence
layer yet (see backend/services/dashboard.py's docstring for the same
limitation) - real requirement data lives only in-memory for the
duration of one extract_role_requirements() call today. Until that's
wired up, this page shows an honest empty state in production and
demonstrates the real computation against sample data in Demo mode.
"""

from __future__ import annotations

import streamlit as st

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.cross_role import FavoriteRoleContext, calculate_skill_roi
from backend.matching.gaps import calculate_skill_gaps
from ui_common import configure_page, empty_state, is_demo_mode

configure_page("Skill Gaps", icon="📉")
st.title("📉 Skill Gaps")
st.caption("Per-role gaps and cross-role skill ROI across your saved roles.")
st.divider()

if not is_demo_mode():
    empty_state(
        "Skill gap analysis isn't available yet",
        detail=(
            "This needs extracted job requirements for your saved roles, which RoleRadar doesn't "
            "persist yet. Turn on Demo mode in the sidebar to see this page with sample data, computed "
            "through the real backend.matching.gaps / backend.matching.cross_role logic."
        ),
    )
    st.stop()

# --- Demo mode: real computation, sample inputs ---
profile = CandidateProfile(
    skills=[
        CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.5, confidence=0.7),
        CandidateSkillEstimate(normalized_skill_name="algorithms", display_name="Algorithms", estimated_level=4.0, confidence=0.4),
        CandidateSkillEstimate(normalized_skill_name="probability", display_name="Probability", estimated_level=3.0, confidence=0.3),
    ]
)
dream_role = normalize_role({"title": "Quantitative Research Intern", "company": "Meridian Capital", "role_family": "quant"})
backup_role = normalize_role({"title": "Software Engineering Intern", "company": "Acme Corp", "role_family": "swe"})
dream_requirements = [
    RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8, importance=9, required=True, evidence=["Strong foundation in probability."]),
    RoleRequirement(skill="Python", normalized_skill="python", target_level=7, importance=7, required=True, evidence=["Proficiency in Python."]),
]
backup_requirements = [
    RoleRequirement(skill="Algorithms", normalized_skill="algorithms", target_level=7, importance=8, required=True, evidence=["Comfortable with algorithms."]),
    RoleRequirement(skill="Python", normalized_skill="python", target_level=6, importance=6, required=True, evidence=["Proficiency in Python."]),
]

st.subheader("Per-role gaps")
for role, requirements in ((dream_role, dream_requirements), (backup_role, backup_requirements)):
    with st.container(border=True):
        st.markdown(f"**{role.title}** at {role.company}")
        gaps = calculate_skill_gaps(profile.skills, requirements)
        for gap in gaps:
            cols = st.columns([2, 1, 1, 1])
            cols[0].markdown(gap.display_skill)
            cols[1].caption(f"target {gap.target_level:g}")
            cols[2].caption(f"you: {gap.candidate_level:g}")
            cols[3].progress(min(gap.satisfaction_ratio, 1.0))

st.subheader("⚡ Cross-role skill ROI")
st.caption("Which skill would most improve your position across ALL saved roles, per backend.matching.cross_role.")
roi_results = calculate_skill_roi(
    profile,
    [
        FavoriteRoleContext(role=dream_role, requirements=dream_requirements, priority="dream"),
        FavoriteRoleContext(role=backup_role, requirements=backup_requirements, priority="backup"),
    ],
)
for result in roi_results:
    with st.container(border=True):
        cols = st.columns([2, 1, 2])
        cols[0].markdown(f"**{result.display_skill}**")
        cols[1].metric("ROI", f"{result.roi_score:.0f}")
        cols[2].caption(f"Needed by {result.roles_requiring_it} saved role(s) · avg gap {result.average_gap:.1f}/10")
