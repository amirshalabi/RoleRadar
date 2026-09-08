"""
Favorites page: saved roles with priority/notes/fit/readiness/deadline/
application stage, a 2-4 way "Compare Favorites" view, and a "Common
Skill Gaps" cross-role ROI view.

This file renders only - every number comes from backend.services.discovery
(list_role_cards, compare_roles, get_skill_roi_for_favorites), which in
turn calls the deterministic backend.matching / backend.planning
functions. Favorites are always read fresh from Postgres via
backend.services.tracking/discovery on every rerun - never solely from
st.session_state - so priority/notes/remove actions immediately reflect
the real, persisted state.
"""

from __future__ import annotations

import streamlit as st

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.db.favorites import PRIORITY_LEVELS
from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.cross_role import (
    FavoriteRoleContext,
    calculate_skill_roi,
    compare_favorite_roles,
    estimate_prep_hours_for_role,
    summarize_comparison,
)
from backend.matching.gaps import calculate_skill_gaps
from backend.planning.readiness import calculate_readiness
from backend.services import discovery, tracking
from backend.services.discovery import RoleCard
from ui_common import configure_page, database_not_configured_notice, empty_state, get_current_user_id, is_demo_mode

configure_page("Favorites", icon="🚩")
st.title("🚩 Favorites")
st.caption("Saved roles, priority, notes, fit, readiness, and application stage - persisted in Postgres, not session state.")
st.divider()

_PRIORITY_BADGE = {"dream": "💎 Dream", "high": "🔥 High", "interested": "🙂 Interested", "backup": "🧊 Backup"}


def _priority_badge(priority: str | None) -> str:
    return _PRIORITY_BADGE.get(priority, "—") if priority else "—"


# ---------------------------------------------------------------------
# Demo data - the same two roles used elsewhere in demo mode (Discover,
# Role Analysis), plus their requirements so Compare Favorites / Common
# Skill Gaps below can run the REAL backend.matching.cross_role
# functions against synthetic input, not fabricated output.
# ---------------------------------------------------------------------

_DEMO_CARDS = [
    RoleCard(
        role_id="demo-1", external_id="demo-1", title="Quantitative Research Intern", company="Meridian Capital",
        location="New York, NY", role_family="quant",
        is_saved=True, priority="dream", notes="Review probability before applying.",
        application_status="interview", deadline=None, interview_date="2026-09-13",
        analyzed=True, fit_score=57.4, readiness_score=31.5, top_strengths=["Python"], top_gap="Probability",
    ),
    RoleCard(
        role_id="demo-2", external_id="demo-2", title="Software Engineering Intern", company="Acme Corp",
        location="Remote", role_family="swe",
        is_saved=True, priority="backup", notes="",
        application_status="applied", deadline="2026-09-14", interview_date=None,
        analyzed=False, fit_score=None, readiness_score=62.0, top_strengths=[], top_gap=None,
    ),
]
_DEMO_PROFILE = CandidateProfile(
    coursework=["Algorithms"],
    skills=[
        CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.5, confidence=0.7),
        CandidateSkillEstimate(normalized_skill_name="algorithms", display_name="Algorithms", estimated_level=4.0, confidence=0.4),
        CandidateSkillEstimate(normalized_skill_name="probability", display_name="Probability", estimated_level=3.0, confidence=0.3),
    ],
)
_DEMO_CONTEXTS_BY_ROLE_ID = {
    "demo-1": FavoriteRoleContext(
        role=normalize_role({"external_id": "demo-1", "title": "Quantitative Research Intern", "company": "Meridian Capital", "role_family": "quant"}),
        requirements=[
            RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8, importance=9, required=True, evidence=["x"]),
            RoleRequirement(skill="Python", normalized_skill="python", target_level=7, importance=7, required=True, evidence=["x"]),
        ],
        priority="dream",
    ),
    "demo-2": FavoriteRoleContext(
        role=normalize_role({"external_id": "demo-2", "title": "Software Engineering Intern", "company": "Acme Corp", "role_family": "swe"}),
        requirements=[
            RoleRequirement(skill="Algorithms", normalized_skill="algorithms", target_level=7, importance=8, required=True, evidence=["x"]),
            RoleRequirement(skill="Python", normalized_skill="python", target_level=6, importance=6, required=True, evidence=["x"]),
        ],
        priority="backup",
    ),
}


# ---------------------------------------------------------------------
# Load favorites (Supabase is the source of truth)
# ---------------------------------------------------------------------

user_id: str | None = None
demo = is_demo_mode()

if demo:
    cards = _DEMO_CARDS
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()
    cards = [card for card in discovery.list_role_cards(user_id) if card.is_saved]

if not cards:
    empty_state("No saved roles yet", detail="Flag roles you're interested in from the Discover page to see them here.")
    st.stop()


# ---------------------------------------------------------------------
# 1. All saved roles
# ---------------------------------------------------------------------

for card in cards:
    with st.container(border=True):
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown(f"**{card.title}** at {card.company}")
            meta_bits = []
            if card.location:
                meta_bits.append(card.location)
            if card.deadline:
                meta_bits.append(f"Deadline {card.deadline}")
            meta_bits.append(f"Stage: `{card.application_status}`" if card.application_status else "Stage: not tracked")
            st.caption(" · ".join(meta_bits))
        with header_cols[1]:
            st.markdown(_priority_badge(card.priority))

        metric_cols = st.columns(3)
        metric_cols[0].metric("Fit", f"{card.fit_score:.0f}" if card.fit_score is not None else "—")
        metric_cols[1].metric("Readiness", f"{card.readiness_score:.0f}" if card.readiness_score is not None else "—")
        metric_cols[2].metric("Deadline", card.deadline or "—")

        if st.button("🔬 View full analysis", key=f"analysis_{card.role_id}"):
            st.session_state["selected_role_id"] = card.role_id
            st.switch_page("pages/7_Role_Analysis.py")

        if demo:
            if card.notes:
                st.caption(f"📝 {card.notes}")
            continue

        with st.expander("Edit priority / notes / unsave"):
            priority_options = sorted(PRIORITY_LEVELS)
            new_priority = st.selectbox(
                "Priority", options=priority_options, index=priority_options.index(card.priority),
                key=f"priority_{card.role_id}",
            )
            if new_priority != card.priority:
                tracking.set_role_priority(user_id, card.role_id, new_priority)
                st.rerun()

            new_notes = st.text_area("Notes", value=card.notes or "", key=f"notes_{card.role_id}")
            if st.button("Save notes", key=f"save_notes_{card.role_id}"):
                tracking.set_role_notes(user_id, card.role_id, new_notes)
                st.rerun()

            if st.button("Remove from favorites", key=f"unsave_{card.role_id}", type="secondary"):
                tracking.unsave_role(user_id, card.role_id)
                st.rerun()


# ---------------------------------------------------------------------
# 2. Compare Favorites
# ---------------------------------------------------------------------

st.divider()
st.header("⚖️ Compare Favorites")
st.caption("Select 2-4 saved roles to compare fit, readiness, top skill gaps, and estimated prep burden side by side.")

card_by_id = {card.role_id: card for card in cards}
selected_ids = st.multiselect(
    "Roles to compare",
    options=list(card_by_id.keys()),
    format_func=lambda rid: f"{card_by_id[rid].title} at {card_by_id[rid].company}",
    max_selections=4,
)

if len(selected_ids) < 2:
    st.caption("Select at least 2 roles above to see a comparison.")
else:
    if demo:
        contexts = [_DEMO_CONTEXTS_BY_ROLE_ID[rid] for rid in selected_ids]
        readiness_scores: dict[str, float] = {}
        prep_hours_by_role: dict[str, float] = {}
        for context in contexts:
            gaps = calculate_skill_gaps(_DEMO_PROFILE.skills, context.requirements)
            prep_hours_by_role[context.role.external_id] = estimate_prep_hours_for_role(gaps)
            if context.role.role_family:
                readiness_scores[context.role.external_id] = calculate_readiness(
                    _DEMO_PROFILE, context.role.role_family
                ).overall_readiness
        comparisons = compare_favorite_roles(
            _DEMO_PROFILE, contexts, readiness_scores=readiness_scores, prep_hours_by_role=prep_hours_by_role
        )
        summary = summarize_comparison(comparisons)
    else:
        comparisons, summary = discovery.compare_roles(user_id, selected_ids)

    # compare_roles()/compare_favorite_roles() preserve the order of the
    # role ids they were given, so zipping with selected_ids here is safe.
    compare_cols = st.columns(len(comparisons))
    for col, comparison, role_id in zip(compare_cols, comparisons, selected_ids):
        source_card = card_by_id[role_id]
        with col:
            st.markdown(f"**{comparison.role.title}**")
            st.caption(comparison.role.company)
            st.metric("Fit", f"{comparison.fit_score:.0f}")
            st.metric("Readiness", f"{comparison.readiness_score:.0f}" if comparison.readiness_score is not None else "—")
            st.caption(f"Technical: {comparison.fit_components['technical']:.0f}")
            st.caption(f"Experience: {comparison.fit_components['experience']:.0f}")
            st.caption(f"Domain: {comparison.fit_components['domain']:.0f}")
            st.markdown("**Top gaps**")
            if comparison.top_gaps:
                for gap in comparison.top_gaps:
                    st.caption(f"{gap.skill} (gap {gap.gap:g})" + (" *" if gap.required else ""))
            else:
                st.caption("None.")
            st.metric(
                "Est. prep hours",
                f"{comparison.prep_hours_allocated:.0f}h" if comparison.prep_hours_allocated is not None else "—",
            )
            st.caption(f"Deadline: {source_card.deadline}" if source_card.deadline else "Deadline: —")

    st.markdown("**Summary**")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        with st.container(border=True):
            st.markdown("**🏆 Best current match**")
            st.write(summary.best_current_match.title if summary.best_current_match else "—")
    with summary_cols[1]:
        with st.container(border=True):
            st.markdown("**📈 Highest potential upside**")
            st.write(summary.highest_potential_upside.title if summary.highest_potential_upside else "—")
    with summary_cols[2]:
        with st.container(border=True):
            st.markdown("**⏳ Largest prep burden**")
            st.write(summary.largest_prep_burden.title if summary.largest_prep_burden else "—")


# ---------------------------------------------------------------------
# 3. Common Skill Gaps
# ---------------------------------------------------------------------

st.divider()
st.header("🧩 Common Skill Gaps")
st.caption("Skills needed across your saved roles, ranked by return on investment (backend.matching.cross_role).")

if demo:
    roi_results = calculate_skill_roi(_DEMO_PROFILE, list(_DEMO_CONTEXTS_BY_ROLE_ID.values()))
else:
    roi_results = discovery.get_skill_roi_for_favorites(user_id)

if not roi_results:
    st.caption("No skill requirements found across your saved roles yet.")
else:
    top_skill = roi_results[0]
    st.success(
        f"**Highest-leverage skill: {top_skill.display_skill}** — needed by {top_skill.roles_requiring_it} "
        f"saved role(s), ROI {top_skill.roi_score:.0f}/100."
    )

    table_rows = [
        {
            "Skill": result.display_skill,
            "Roles requiring it": result.roles_requiring_it,
            "Avg. gap": f"{result.average_gap:.1f}",
            "Avg. importance": f"{result.average_importance:.1f}",
            "Skill ROI": f"{result.roi_score:.0f}",
            "Affected roles": ", ".join(f"{r.title} ({r.company})" for r in result.affected_roles),
        }
        for result in roi_results
    ]
    st.dataframe(table_rows, hide_index=True, use_container_width=True)
