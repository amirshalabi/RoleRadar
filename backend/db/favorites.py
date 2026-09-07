"""
Favorites data access.

CRUD operations for user-flagged favorite roles, keyed on
UNIQUE(user_id, role_id). Favorites persist in PostgreSQL, not Streamlit
session state, and feed comparison, cross-role skill-gap analysis, skill
ROI, and interview prep prioritization.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.db.client import get_client
from backend.db.upserts import upsert_row

logger = logging.getLogger(__name__)

FAVORITES_TABLE = "favorites"


def save_favorite(user_id: str, role_id: str) -> dict[str, Any]:
    """
    Flag a role as a favorite for a user, keyed on
    UNIQUE(user_id, role_id). Saving an already-favorited role returns
    the existing row rather than creating a duplicate.
    """
    values = {"user_id": user_id, "role_id": role_id}
    return upsert_row(FAVORITES_TABLE, values, on_conflict="user_id,role_id")


def remove_favorite(user_id: str, role_id: str) -> None:
    """Un-flag a role as a favorite for a user, if it was favorited."""
    client = get_client()
    client.table(FAVORITES_TABLE).delete().eq("user_id", user_id).eq(
        "role_id", role_id
    ).execute()


def list_favorites(user_id: str) -> list[dict[str, Any]]:
    """Return all favorited roles for a user, most recently saved first."""
    client = get_client()
    response = (
        client.table(FAVORITES_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []
