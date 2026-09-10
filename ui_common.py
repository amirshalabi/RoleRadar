"""
Shared Streamlit UI helpers: page setup, the demo/dev user, the demo-
mode toggle, and small presentational helpers reused across app.py and
every page in pages/.

Deliberately thin - there is no business logic here. Every number a
page displays was computed by backend.services.* (or the
backend.matching/planning modules a service calls into); this module
only knows how to render things and resolve "who is using this
Streamlit session right now."

Visual design (colors, typography, card/table/tag styling) lives in
ui.theme and ui.components - this module wires them into the one
per-page setup call (configure_page) plus the sidebar chrome.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from backend.db.client import SupabaseNotConfiguredError
from backend.db.users import get_or_create_demo_user
from backend.utils.logging import configure_logging
from ui.components import render_empty_state
from ui.theme import inject_global_css

# RoleRadar has no real authentication yet (see backend/db/users.py) -
# every Streamlit session acts as this single demo/dev user, resolved
# once and cached in session state.
DEMO_USER_EMAIL = "demo@roleradar.local"

_LOGO_PATH = str(Path(__file__).parent / "ui" / "assets" / "logo.svg")

_DEMO_SIDEBAR_METRICS = {"tracked": "2", "saved": "2", "active": "2"}
_INACTIVE_APPLICATION_STATUSES = {"rejected", "withdrawn", "offer"}


def configure_page(title: str, icon: str = "🎯") -> None:
    """Call once at the top of every page: sets the tab title/icon, wires logging, applies the shared design system, renders sidebar chrome."""
    configure_logging()
    st.set_page_config(page_title=f"{title} · RoleRadar", page_icon=icon, layout="wide")
    inject_global_css()
    st.logo(_LOGO_PATH, size="medium")
    _render_sidebar_metrics()
    render_demo_mode_toggle()
    _render_sidebar_status_footer()
    if is_demo_mode():
        st.info("**Demo mode** — this page is showing clearly-labeled sample data, not your real account.", icon="🧪")


def is_demo_mode() -> bool:
    return bool(st.session_state.get("demo_mode", False))


def render_demo_mode_toggle() -> None:
    """
    Sidebar control every page includes.

    KNOWN LIMITATION: in this project's Streamlit native
    pages/-directory multipage setup, a page switch does not reliably
    carry widget state forward - verified by hand against three
    documented mechanisms (a bare session_state key, persist_state=
    "session", and bind="query-params"), none of which survived a
    sidebar page-switch click in this environment. Rather than add
    unverified complexity chasing that, this uses the plain, standard
    form: demo mode is toggled per-page - flip it on whichever page you
    want to preview with sample data. Revisit this if a future
    Streamlit version resolves the underlying behavior.
    """
    st.sidebar.markdown('<p class="rr-sidebar-label">Data Source</p>', unsafe_allow_html=True)
    st.sidebar.toggle(
        "Demo mode (sample data)",
        key="demo_mode",
        help="Preview this page populated with clearly-labeled sample data instead of your real account.",
    )


def _render_sidebar_metrics() -> None:
    metrics = _sidebar_metric_values()
    rows = "".join(
        f'<div class="rr-sidebar-metric-row"><span>{label}</span><span class="val">{value}</span></div>'
        for label, value in (
            ("Roles Tracked", metrics["tracked"]),
            ("Saved", metrics["saved"]),
            ("Active Apps", metrics["active"]),
        )
    )
    st.sidebar.markdown('<p class="rr-sidebar-label">Signal Overview</p>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="rr-sidebar-metrics">{rows}</div>', unsafe_allow_html=True)


def _sidebar_metric_values() -> dict[str, str]:
    if is_demo_mode():
        return _DEMO_SIDEBAR_METRICS

    user_id = get_current_user_id()
    if user_id is None:
        return {"tracked": "—", "saved": "—", "active": "—"}

    try:
        from backend.db import applications as applications_db
        from backend.db import favorites as favorites_db
        from backend.db import roles as roles_db

        tracked = len(roles_db.list_roles(limit=500))
        saved = len(favorites_db.list_favorites(user_id))
        active = sum(
            1
            for application in applications_db.list_applications(user_id)
            if application.get("status") not in _INACTIVE_APPLICATION_STATUSES
        )
    except SupabaseNotConfiguredError:
        return {"tracked": "—", "saved": "—", "active": "—"}

    return {"tracked": str(tracked), "saved": str(saved), "active": str(active)}


def _render_sidebar_status_footer() -> None:
    mode = "DEMO" if is_demo_mode() else "LIVE"
    db_connected = is_demo_mode() or get_current_user_id() is not None
    dot_class = "dot" if db_connected else "dot off"
    db_label = "CONNECTED" if db_connected else "OFFLINE"
    st.sidebar.markdown(
        '<div class="rr-sidebar-footer">'
        f'<div><span class="{dot_class}">●</span> DATABASE: {db_label}</div>'
        f"<div>MODE: {mode}</div>"
        "<div>ROLERADAR v0.1</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def get_current_user_id() -> str | None:
    """
    Resolve (or create) the single demo/dev user this session acts as -
    see backend.db.users for why this stands in for real auth. Returns
    None if Postgres isn't configured at all; callers should show
    database_not_configured_notice() in that case.
    """
    if "user_id" in st.session_state:
        return st.session_state["user_id"]
    try:
        user = get_or_create_demo_user(DEMO_USER_EMAIL, display_name="Demo User")
    except SupabaseNotConfiguredError:
        return None
    st.session_state["user_id"] = user["id"]
    return user["id"]


def empty_state(message: str, icon: str = "📭", detail: str | None = None) -> None:
    """A consistent empty-state block for any section with no real data yet."""
    render_empty_state(message, detail)


def database_not_configured_notice() -> None:
    st.warning(
        "**Database not connected.** Set `SUPABASE_URL` and `SUPABASE_KEY` in your `.env` file to see "
        "your real data here — see `sql/schema.sql` and the README for setup. Turn on **Demo mode** "
        "in the sidebar to preview this page with sample data instead.",
        icon="🔌",
    )
