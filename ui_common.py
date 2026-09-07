"""
Shared Streamlit UI helpers: page setup, the demo/dev user, the demo-
mode toggle, and small presentational helpers reused across app.py and
every page in pages/.

Deliberately thin - there is no business logic here. Every number a
page displays was computed by backend.services.* (or the
backend.matching/planning modules a service calls into); this module
only knows how to render things and resolve "who is using this
Streamlit session right now."
"""

from __future__ import annotations

import streamlit as st

from backend.db.client import SupabaseNotConfiguredError
from backend.db.users import get_or_create_demo_user
from backend.utils.logging import configure_logging

# RoleRadar has no real authentication yet (see backend/db/users.py) -
# every Streamlit session acts as this single demo/dev user, resolved
# once and cached in session state.
DEMO_USER_EMAIL = "demo@roleradar.local"

_CUSTOM_CSS = """
<style>
/* Subtle card styling for st.metric, on top of Streamlit's own theme
   variables so this still respects light/dark mode - not a copied
   third-party design, just a bit of native polish. */
div[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    border-radius: 0.75rem;
    padding: 0.9rem 1rem 0.6rem 1rem;
}
</style>
"""


def configure_page(title: str, icon: str = "🎯") -> None:
    """Call once at the top of every page: sets the tab title/icon, wires logging, applies shared styling."""
    configure_logging()
    st.set_page_config(page_title=f"{title} · RoleRadar", page_icon=icon, layout="wide")
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    render_demo_mode_toggle()
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
    st.sidebar.toggle(
        "Demo mode (sample data)",
        key="demo_mode",
        help="Preview this page populated with clearly-labeled sample data instead of your real account.",
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
    st.markdown(f"##### {icon} {message}")
    if detail:
        st.caption(detail)


def database_not_configured_notice() -> None:
    st.warning(
        "**Database not connected.** Set `SUPABASE_URL` and `SUPABASE_KEY` in your `.env` file to see "
        "your real data here — see `sql/schema.sql` and the README for setup. Turn on **Demo mode** "
        "in the sidebar to preview this page with sample data instead.",
        icon="🔌",
    )
