"""
Users data access.

RoleRadar's entire "auth" system today: a single known email resolves
to a stable user_id (see the project architecture note: "Do not
overcomplicate authentication yet. It is okay to support a simple
development/demo user flow."). get_or_create_demo_user() is what every
Streamlit page uses to get a real user_id to read/write against,
without building real authentication.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.db.client import get_client

logger = logging.getLogger(__name__)

USERS_TABLE = "users"


def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Fetch a user row by email, or None if it doesn't exist."""
    client = get_client()
    response = client.table(USERS_TABLE).select("*").eq("email", email).limit(1).execute()
    return response.data[0] if response.data else None


def get_or_create_demo_user(email: str, display_name: str | None = None) -> dict[str, Any]:
    """
    Return the user row for `email`, creating it on first use. Calling
    this repeatedly with the same email always resolves to the same
    row - a real auth system would replace this, not extend it.
    """
    existing = get_user_by_email(email)
    if existing is not None:
        return existing

    client = get_client()
    values = {"email": email, "display_name": display_name}
    response = client.table(USERS_TABLE).insert(values).execute()
    if not response.data:
        raise RuntimeError(f"Insert into {USERS_TABLE} returned no data")
    return response.data[0]
