"""
Shared FastAPI dependencies.

RoleRadar has no real authentication yet (see backend.db.users) - every
API request acts as the same single demo/dev user, resolved via
get_or_create_demo_user(), exactly like every Streamlit page already
does (ui_common.get_current_user_id()). Swapping in real auth later
only means replacing this one function.
"""

from __future__ import annotations

from backend.db.users import get_or_create_demo_user

DEMO_USER_EMAIL = "demo@roleradar.local"


def get_user_id() -> str:
    """Resolve (creating on first use) the single demo/dev user every API request acts as."""
    user = get_or_create_demo_user(DEMO_USER_EMAIL, display_name="Demo User")
    return user["id"]
