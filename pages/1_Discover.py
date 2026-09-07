"""
Discover page: browse ingested job opportunities, filter/sort them, flag
favorites, and view a role's detailed fit/readiness analysis.

This file renders only - it never computes a fit score, skill gap, or
readiness number itself. All of that comes from backend.services.discovery
(which in turn calls backend.matching / backend.planning / backend.llm).
Favorites/application state always round-trips through
backend.services.tracking, so Supabase - never st.session_state - is
the source of truth: session state here only caches which filter/sort
options are selected and which role's detail dialog is open, not the
saved-state of any role.

There is still no in-app trigger for the ingestion pipeline itself
(backend.ingestion.concurrent.run_ingestion_pipeline runs outside
Streamlit today) - this page only browses whatever roles already exist
in Postgres, and lets you request deep analysis for any of them.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from backend.db.applications import VALID_STATUSES
from backend.services import discovery, tracking
from backend.services.discovery import RoleCard
from ui_common import configure_page, database_not_configured_notice, empty_state, get_current_user_id, is_demo_mode

configure_page("Discover", icon="🔍")
st.title("🔍 Discover")
st.caption("Roles currently in your RoleRadar database.")
st.divider()

_PRIORITY_RANK = {"dream": 4, "high": 3, "interested": 2, "backup": 1}
_PRIORITY_BADGE = {"dream": "💎 Dream", "high": "🔥 High", "interested": "🙂 Interested", "backup": "🧊 Backup"}
_SORT_OPTIONS = ["Best Fit", "Highest Priority", "Soonest Deadline", "Highest Readiness", "Largest Potential Improvement"]

_DEMO_CARDS = [
    RoleCard(
        role_id="demo-1", external_id="demo-1", title="Quantitative Research Intern", company="Meridian Capital",
        location="New York, NY", role_family="quant",
        description="Build and backtest trading signals using Python and C++.",
        is_saved=True, priority="dream", application_status="interview", deadline=None, interview_date="2026-09-13",
        analyzed=True, fit_score=57.4, readiness_score=31.5, top_strengths=["Python"], top_gap="Probability",
    ),
    RoleCard(
        role_id="demo-2", external_id="demo-2", title="Software Engineering Intern", company="Acme Corp",
        location="Remote", role_family="swe",
        description="Build backend services in Python.",
        is_saved=False, priority=None, application_status=None, deadline="2026-09-14", interview_date=None,
        analyzed=False, fit_score=None, readiness_score=62.0, top_strengths=[], top_gap=None,
    ),
]


# ---------------------------------------------------------------------
# Load cards (Supabase is the source of truth - re-read fresh on every run)
# ---------------------------------------------------------------------

user_id: str | None = None

if is_demo_mode():
    cards = _DEMO_CARDS
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()
    cards = discovery.list_role_cards(user_id)

if not cards:
    empty_state(
        "No roles ingested yet",
        detail="Run backend.ingestion.concurrent.run_ingestion_pipeline() to populate roles - "
        "see backend/ingestion/jobs.py for the demo source adapters.",
    )
    st.stop()


# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------

with st.expander("Filters", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        search_text = st.text_input("Role / company search")
        location_text = st.text_input("Location")
        missing_skill_text = st.text_input("Missing skill", help="Show only analyzed roles with an unmet gap in this skill.")
    with col2:
        min_fit = st.slider("Minimum fit", 0, 100, 0)
        min_readiness = st.slider("Minimum readiness", 0, 100, 0)
        deadline_on_or_before = st.date_input("Deadline on or before", value=None)

    status_options = sorted(VALID_STATUSES) + ["not_tracked"]
    status_filter = st.multiselect("Application status", options=status_options)

sort_by = st.selectbox("Sort by", options=_SORT_OPTIONS)


def _matches_filters(card: RoleCard) -> bool:
    if search_text and search_text.lower() not in f"{card.title} {card.company}".lower():
        return False
    if location_text and location_text.lower() not in (card.location or "").lower():
        return False
    if min_fit > 0 and (card.fit_score is None or card.fit_score < min_fit):
        return False
    if min_readiness > 0 and (card.readiness_score is None or card.readiness_score < min_readiness):
        return False
    if deadline_on_or_before:
        if not card.deadline or date.fromisoformat(card.deadline) > deadline_on_or_before:
            return False
    if missing_skill_text:
        needle = missing_skill_text.lower()
        if not card.analyzed or card.top_gap is None or needle not in card.top_gap.lower():
            return False
    if status_filter:
        card_status = card.application_status or "not_tracked"
        if card_status not in status_filter:
            return False
    return True


filtered = [card for card in cards if _matches_filters(card)]


def _sort_key(card: RoleCard):
    if sort_by == "Best Fit":
        return (card.fit_score is None, -(card.fit_score or 0))
    if sort_by == "Highest Priority":
        return (-_PRIORITY_RANK.get(card.priority, 0),)
    if sort_by == "Soonest Deadline":
        return (card.deadline is None, card.deadline or "9999-99-99")
    if sort_by == "Highest Readiness":
        return (card.readiness_score is None, -(card.readiness_score or 0))
    if sort_by == "Largest Potential Improvement":
        # Headroom on the already-computed overall fit score - see
        # backend/services/discovery.py's docstring for why this stays
        # a simple, documented metric rather than isolating one
        # component of the six that make up fit_score.
        return (card.fit_score is None, -(100 - (card.fit_score or 0)))
    return (0,)


filtered.sort(key=_sort_key)

st.caption(f"Showing {len(filtered)} of {len(cards)} role(s).")


# ---------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------

for card in filtered:
    with st.container(border=True):
        header_cols = st.columns([4, 1, 1])
        with header_cols[0]:
            st.markdown(f"**{card.title}** at {card.company}")
            meta_bits = []
            if card.location:
                meta_bits.append(card.location)
            if card.deadline:
                meta_bits.append(f"Deadline {card.deadline}")
            if card.interview_date:
                meta_bits.append(f"Interview {card.interview_date}")
            if meta_bits:
                st.caption(" · ".join(meta_bits))
            if card.priority:
                st.markdown(_PRIORITY_BADGE.get(card.priority, card.priority))
            elif card.application_status:
                st.caption(f"Status: `{card.application_status}`")

        with header_cols[1]:
            if card.fit_score is not None:
                st.metric("Fit", f"{card.fit_score:.0f}")
            else:
                st.caption("Not yet analyzed")
            if card.readiness_score is not None:
                st.caption(f"Readiness: {card.readiness_score:.0f}")

        with header_cols[2]:
            if is_demo_mode():
                st.button("🚩 Save" if not card.is_saved else "🚩 Saved", key=f"save_{card.role_id}", disabled=True)
            elif card.is_saved:
                if st.button("🚩 Saved", key=f"unsave_{card.role_id}", help="Click to remove from favorites"):
                    tracking.unsave_role(user_id, card.role_id)
                    st.rerun()
            else:
                if st.button("🏳️ Save", key=f"save_{card.role_id}"):
                    tracking.save_role(user_id, card.role_id)
                    st.rerun()

        if card.top_strengths or card.top_gap:
            strengths_text = ", ".join(card.top_strengths) if card.top_strengths else "—"
            st.caption(f"✅ Strengths: {strengths_text}" + (f"  ·  ⚠️ Top gap: {card.top_gap}" if card.top_gap else ""))

        if st.button("🔬 View full analysis", key=f"detail_{card.role_id}"):
            st.session_state["selected_role_id"] = card.role_id
            st.switch_page("pages/7_Role_Analysis.py")
