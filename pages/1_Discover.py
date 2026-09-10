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
Streamlit today, or via scripts/run_ingestion.py) - this page only
browses whatever roles already exist in Postgres, and lets you request
deep analysis for any of them.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from backend.db.applications import VALID_STATUSES
from backend.services import discovery, tracking
from backend.services.discovery import RoleCard
from ui import components
from ui_common import configure_page, database_not_configured_notice, get_current_user_id, is_demo_mode

configure_page("Discover", icon="🔍")
components.render_page_header(
    "Role Intelligence",
    "Discover",
    "High-signal roles ranked against your experience, skills, and preferences.",
)

_PRIORITY_RANK = {"dream": 4, "high": 3, "interested": 2, "backup": 1}
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
    components.render_empty_state(
        "No roles ingested yet",
        detail="Run scripts/run_ingestion.py (or backend.ingestion.concurrent.run_ingestion_pipeline) to populate roles.",
    )
    st.stop()


# ---------------------------------------------------------------------
# Filter bar
# ---------------------------------------------------------------------

with components.panel("filters"):
    row1 = st.columns(3)
    with row1[0]:
        search_text = st.text_input("Role / Company")
    with row1[1]:
        location_text = st.text_input("Location")
    with row1[2]:
        missing_skill_text = st.text_input("Missing Skill")

    row2 = st.columns(3)
    with row2[0]:
        min_fit = st.slider("Minimum Fit", 0, 100, 0)
    with row2[1]:
        min_readiness = st.slider("Minimum Readiness", 0, 100, 0)
    with row2[2]:
        deadline_on_or_before = st.date_input("Deadline On Or Before", value=None)

    row3 = st.columns([2, 1])
    with row3[0]:
        status_options = sorted(VALID_STATUSES) + ["not_tracked"]
        status_filter = st.multiselect("Application Status", options=status_options)
    with row3[1]:
        sort_by = st.selectbox("Sort By", options=_SORT_OPTIONS)


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

st.markdown(f'<p class="rr-meta">{len(filtered)} ROLE(S) OF {len(cards)}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Role cards - two-column grid on desktop
# ---------------------------------------------------------------------


def _render_card(card: RoleCard) -> None:
    high_signal = card.fit_score is not None and card.fit_score >= 80
    with components.card(card.role_id, high_signal=high_signal):
        header_cols = st.columns([1, 5])
        with header_cols[0]:
            components.render_score_badge(card.fit_score)
        with header_cols[1]:
            components.render_role_title(card.title, card.url)
            components.render_meta_line([card.company])

        meta_bits = []
        if card.location:
            meta_bits.append(card.location)
        if card.deadline:
            meta_bits.append(f"Deadline {card.deadline}")
        if card.interview_date:
            meta_bits.append(f"Interview {card.interview_date}")
        if card.readiness_score is not None:
            meta_bits.append(f"Readiness {card.readiness_score:.0f}")
        components.render_meta_line(meta_bits)

        if card.priority:
            components.render_priority_badge(card.priority)
        elif card.application_status:
            components.render_status_badge(card.application_status.replace("_", " ").upper())
        elif not card.analyzed:
            components.render_status_badge("NOT YET ANALYZED")

        if card.top_strengths:
            components.render_tags(card.top_strengths, kind="strength")
        if card.top_gap:
            components.render_tags([card.top_gap], kind="gap")

        action_cols = st.columns(3)
        with action_cols[0]:
            if is_demo_mode():
                st.button("Saved" if card.is_saved else "Save", key=f"save_{card.role_id}", disabled=True, use_container_width=True)
            elif card.is_saved:
                if st.button("Saved ✓", key=f"unsave_{card.role_id}", help="Click to remove from favorites", use_container_width=True):
                    tracking.unsave_role(user_id, card.role_id)
                    st.rerun()
            else:
                if st.button("Save", key=f"save_{card.role_id}", use_container_width=True):
                    tracking.save_role(user_id, card.role_id)
                    st.rerun()
        with action_cols[1]:
            components.render_view_posting_button(card.url, key=f"posting_{card.role_id}")
        with action_cols[2]:
            if st.button("View Analysis", key=f"detail_{card.role_id}", type="primary", use_container_width=True):
                st.session_state["selected_role_id"] = card.role_id
                st.switch_page("pages/7_Role_Analysis.py")


grid_cols = st.columns(2, gap="medium")
for index, card in enumerate(filtered):
    with grid_cols[index % 2]:
        _render_card(card)
