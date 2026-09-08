"""
Skill Gaps page: the candidate's own skill estimates, combined with
every favorite/target role's requirements, into one cross-role picture.

This file renders only. The cross-role arithmetic (frequency, average
gap/importance/target level, skill ROI) always comes from
backend.matching.cross_role.calculate_skill_roi() via
backend.services.discovery.get_skill_roi_for_favorites() - nothing here
recomputes that math. Per-skill evidence shown in the expandable detail
view is either a real persisted resume quote
(candidate_skills.evidence_snippets) or a real persisted job-posting
quote (role_requirements.evidence, via
backend.services.discovery.get_requirement_evidence_for_skill()) -
never fabricated text.
"""

from __future__ import annotations

import streamlit as st

from backend.db import candidates as candidates_db
from backend.matching.confidence import confidence_label
from backend.services import discovery
from ui_common import configure_page, database_not_configured_notice, empty_state, get_current_user_id, is_demo_mode

configure_page("Skill Gaps", icon="📉")
st.title("📉 Skill Gaps")
st.caption("Your skills combined with every favorite role's requirements - one cross-role picture, computed by the real backend.")
st.divider()


# ---------------------------------------------------------------------
# Demo data - the same candidate/roles used elsewhere in demo mode
# (Discover, Favorites, Applications), so numbers stay consistent
# across pages.
# ---------------------------------------------------------------------

_DEMO_SKILL_ROWS = [
    {"normalized_skill_name": "python", "display_name": "Python", "estimated_level": 7.5, "confidence": 0.7,
     "evidence_snippets": ["Built internal Python ETL pipelines", "Skills: Python, C++, React, Docker"]},
    {"normalized_skill_name": "algorithms", "display_name": "Algorithms", "estimated_level": 4.0, "confidence": 0.4,
     "evidence_snippets": ["Coursework: Data Structures, Algorithms"]},
    {"normalized_skill_name": "probability", "display_name": "Probability", "estimated_level": 3.0, "confidence": 0.25,
     "evidence_snippets": []},
]
_DEMO_REQUIREMENT_EVIDENCE = {
    ("demo-1", "probability"): ["Strong foundation in probability and statistics required."],
    ("demo-1", "python"): ["Proficiency in Python for backtesting is required."],
    ("demo-2", "algorithms"): ["Comfortable with core algorithms and data structures."],
    ("demo-2", "python"): ["Proficiency in Python is required."],
}


def _demo_roi_results():
    from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
    from backend.ingestion.normalize import normalize_role
    from backend.llm.extract_requirements import RoleRequirement
    from backend.matching.cross_role import FavoriteRoleContext, calculate_skill_roi

    profile = CandidateProfile(
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name=row["normalized_skill_name"], display_name=row["display_name"],
                estimated_level=row["estimated_level"], confidence=row["confidence"],
                evidence_snippets=row["evidence_snippets"],
            )
            for row in _DEMO_SKILL_ROWS
        ]
    )
    contexts = [
        FavoriteRoleContext(
            role=normalize_role({"external_id": "demo-1", "title": "Quantitative Research Intern", "company": "Meridian Capital", "role_family": "quant"}),
            requirements=[
                RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8, importance=9, required=True, evidence=_DEMO_REQUIREMENT_EVIDENCE[("demo-1", "probability")]),
                RoleRequirement(skill="Python", normalized_skill="python", target_level=7, importance=7, required=True, evidence=_DEMO_REQUIREMENT_EVIDENCE[("demo-1", "python")]),
            ],
            priority="dream", role_id="demo-1",
        ),
        FavoriteRoleContext(
            role=normalize_role({"external_id": "demo-2", "title": "Software Engineering Intern", "company": "Acme Corp", "role_family": "swe"}),
            requirements=[
                RoleRequirement(skill="Algorithms", normalized_skill="algorithms", target_level=7, importance=8, required=True, evidence=_DEMO_REQUIREMENT_EVIDENCE[("demo-2", "algorithms")]),
                RoleRequirement(skill="Python", normalized_skill="python", target_level=6, importance=6, required=True, evidence=_DEMO_REQUIREMENT_EVIDENCE[("demo-2", "python")]),
            ],
            priority="backup", role_id="demo-2",
        ),
    ]
    return profile, calculate_skill_roi(profile, contexts)


# ---------------------------------------------------------------------
# Load candidate skills + cross-role ROI
# ---------------------------------------------------------------------

user_id: str | None = None
demo = is_demo_mode()

if demo:
    skill_rows = _DEMO_SKILL_ROWS
    _, roi_results = _demo_roi_results()
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()
    skill_rows = candidates_db.list_candidate_skills(user_id)
    roi_results = discovery.get_skill_roi_for_favorites(user_id)

if not skill_rows:
    empty_state("No candidate skills tracked yet", detail="Upload a resume to unlock skill gap analysis.")
    st.stop()


# ---------------------------------------------------------------------
# 1. Candidate Skills
# ---------------------------------------------------------------------

st.subheader("🧬 Candidate Skills")
skill_table = [
    {
        "Skill": row.get("display_name") or row["normalized_skill_name"],
        "Estimated level": f"{row['estimated_level']:g}",
        "Confidence": f"{row['confidence']:.2f} ({confidence_label(row['confidence'])})",
        "Evidence count": len(row.get("evidence_snippets") or []),
    }
    for row in skill_rows
]
st.dataframe(skill_table, hide_index=True, use_container_width=True)

if not roi_results:
    st.divider()
    empty_state(
        "No cross-role skill gaps yet",
        detail="Analyze at least one favorite role (Discover → View full analysis) to see cross-role gaps here.",
    )
    st.stop()


# ---------------------------------------------------------------------
# 2. Cross-Role Skill Gaps
# ---------------------------------------------------------------------

st.divider()
st.subheader("🔀 Cross-Role Skill Gaps")
st.caption("Every skill required by at least one favorite role, computed by backend.matching.cross_role.calculate_skill_roi().")

roi_table = [
    {
        "Skill": result.display_skill,
        "Roles requiring it": result.roles_requiring_it,
        "Avg. target level": f"{result.average_target_level:.1f}",
        "Candidate level": f"{result.candidate_level:g}",
        "Avg. gap": f"{result.average_gap:.1f}",
        "Avg. importance": f"{result.average_importance:.1f}",
        "Skill ROI": f"{result.roi_score:.0f}",
    }
    for result in roi_results
]
st.dataframe(roi_table, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------
# 3. Highlights
# ---------------------------------------------------------------------

highest_leverage = roi_results[0]  # already sorted by roi_score descending
most_common = max(roi_results, key=lambda r: r.roles_requiring_it)
largest_gap = max(roi_results, key=lambda r: r.average_gap)
skills_by_confidence = sorted(skill_rows, key=lambda row: row["confidence"])
lowest_confidence = skills_by_confidence[0]

highlight_cols = st.columns(4)
with highlight_cols[0]:
    with st.container(border=True):
        st.markdown("**⚡ Highest-Leverage Skill**")
        st.write(f"{highest_leverage.display_skill} (ROI {highest_leverage.roi_score:.0f})")
with highlight_cols[1]:
    with st.container(border=True):
        st.markdown("**📋 Most Common Requirement**")
        st.write(f"{most_common.display_skill} ({most_common.roles_requiring_it} role(s))")
with highlight_cols[2]:
    with st.container(border=True):
        st.markdown("**📐 Largest Gap**")
        st.write(f"{largest_gap.display_skill} (gap {largest_gap.average_gap:.1f}/10)")
with highlight_cols[3]:
    with st.container(border=True):
        st.markdown("**❓ Lowest-Confidence Estimate**")
        low_conf_name = lowest_confidence.get("display_name") or lowest_confidence["normalized_skill_name"]
        st.write(f"{low_conf_name} ({lowest_confidence['confidence']:.2f})")


# ---------------------------------------------------------------------
# 4. Expandable per-skill detail
# ---------------------------------------------------------------------

st.divider()
st.subheader("🔎 Skill Detail")

skill_row_by_normalized = {row["normalized_skill_name"]: row for row in skill_rows}

for result in roi_results:
    with st.expander(f"{result.display_skill} — ROI {result.roi_score:.0f}, needed by {result.roles_requiring_it} role(s)"):
        skill_row = skill_row_by_normalized.get(result.normalized_skill)
        candidate_evidence = (skill_row.get("evidence_snippets") if skill_row else None) or []

        st.markdown("**Candidate evidence**")
        if candidate_evidence:
            for snippet in candidate_evidence:
                st.caption(f"“{snippet}”")
        else:
            st.caption("No stored resume evidence for this skill yet.")

        st.markdown(f"**Roles requiring it / affected favorite roles ({result.roles_requiring_it})**")
        for role in result.affected_roles:
            st.markdown(f"- {role.title} at {role.company}")
            if demo:
                role_evidence = _DEMO_REQUIREMENT_EVIDENCE.get((role.role_id, result.normalized_skill), [])
            else:
                role_evidence = discovery.get_requirement_evidence_for_skill(role.role_id, result.normalized_skill) if role.role_id else []
            if role_evidence:
                for snippet in role_evidence:
                    st.caption(f"  “{snippet}”")
            else:
                st.caption("  No persisted requirement evidence for this role/skill yet.")

        st.markdown("**Why it matters**")
        st.write(
            f"Required by {result.roles_requiring_it} of your favorite role(s), with average importance "
            f"{result.average_importance:.1f}/10 and an average gap of {result.average_gap:.1f}/10 against a "
            f"target level of {result.average_target_level:.1f}/10. Estimated cost to close this gap: "
            f"~{result.estimated_learning_cost_hours * (result.average_gap / 10.0):.0f} hours. "
            f"Skill ROI: {result.roi_score:.0f}/100 relative to your other current gaps."
        )
