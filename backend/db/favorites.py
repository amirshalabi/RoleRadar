"""
Favorites data access.

CRUD operations for user-flagged favorite roles, keyed on
UNIQUE(user_id, role_id). Favorites persist in PostgreSQL, not Streamlit
session state, and feed comparison, cross-role skill-gap analysis, skill
ROI, and interview prep prioritization.

save_favorite() is deliberately NOT a blind upsert: it checks whether
the favorite already exists and, if so, returns that row untouched. A
blind upsert would silently reset priority/notes back to their defaults
on every repeat call (e.g. a user re-clicking "save" on a role they
already favorited with priority="dream" would otherwise wipe that
choice back to "interested"). Changing priority/notes on an existing
favorite goes through update_favorite_priority()/update_favorite_notes()
instead, which use a targeted SQL UPDATE (only touching the column
being changed) rather than a full-row upsert.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from backend.db.client import get_client

logger = logging.getLogger(__name__)

FAVORITES_TABLE = "favorites"

PRIORITY_LEVELS = {"dream", "high", "interested", "backup"}
DEFAULT_PRIORITY = "interested"


def _validate_priority(priority: str) -> None:
    if priority not in PRIORITY_LEVELS:
        raise ValueError(f"Invalid favorite priority: {priority!r}. Must be one of {sorted(PRIORITY_LEVELS)}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_favorite(user_id: str, role_id: str) -> dict[str, Any] | None:
    """Fetch a user's favorite row for a role, or None if it isn't favorited."""
    client = get_client()
    response = (
        client.table(FAVORITES_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("role_id", role_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def save_favorite(
    user_id: str,
    role_id: str,
    priority: str = DEFAULT_PRIORITY,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Flag a role as a favorite for a user, keyed on
    UNIQUE(user_id, role_id).

    Idempotent: calling this again for an already-favorited role is a
    no-op that returns the existing row unchanged - `priority`/`notes`
    only apply on first save. Never creates a duplicate row.
    """
    existing = get_favorite(user_id, role_id)
    if existing is not None:
        logger.debug("Role %s already favorited by user %s; leaving unchanged", role_id, user_id)
        return existing

    _validate_priority(priority)
    client = get_client()
    values = {
        "user_id": user_id,
        "role_id": role_id,
        "priority": priority,
        "notes": notes,
        "updated_at": _now_iso(),
    }
    response = client.table(FAVORITES_TABLE).insert(values).execute()
    if not response.data:
        raise RuntimeError(f"Insert into {FAVORITES_TABLE} returned no data")
    return response.data[0]


def remove_favorite(user_id: str, role_id: str) -> None:
    """Un-flag a role as a favorite for a user, if it was favorited. A no-op if it wasn't."""
    client = get_client()
    client.table(FAVORITES_TABLE).delete().eq("user_id", user_id).eq("role_id", role_id).execute()


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


def list_favorites_with_roles(user_id: str) -> list[dict[str, Any]]:
    """
    Every favorite row for a user, with its role's title/company/etc
    embedded (a Postgrest foreign-table select) - what a Favorites page
    needs without a second query per row. Additive alongside
    list_favorites(), which stays a plain flat select so its existing
    callers are unaffected.
    """
    client = get_client()
    response = (
        client.table(FAVORITES_TABLE)
        .select("*, roles(*)")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


def update_favorite_priority(user_id: str, role_id: str, priority: str) -> dict[str, Any]:
    """
    Update the priority of an existing favorite. Raises ValueError if
    the role isn't currently favorited (this updates, it doesn't
    implicitly create - call save_favorite() first).
    """
    _validate_priority(priority)
    if get_favorite(user_id, role_id) is None:
        raise ValueError(f"Role {role_id!r} is not favorited by user {user_id!r}; call save_favorite() first.")

    client = get_client()
    response = (
        client.table(FAVORITES_TABLE)
        .update({"priority": priority, "updated_at": _now_iso()})
        .eq("user_id", user_id)
        .eq("role_id", role_id)
        .execute()
    )
    if not response.data:
        raise RuntimeError(f"Update on {FAVORITES_TABLE} returned no data")
    return response.data[0]


def update_favorite_notes(user_id: str, role_id: str, notes: str | None) -> dict[str, Any]:
    """
    Update the notes on an existing favorite. Raises ValueError if the
    role isn't currently favorited (this updates, it doesn't implicitly
    create - call save_favorite() first).
    """
    if get_favorite(user_id, role_id) is None:
        raise ValueError(f"Role {role_id!r} is not favorited by user {user_id!r}; call save_favorite() first.")

    client = get_client()
    response = (
        client.table(FAVORITES_TABLE)
        .update({"notes": notes, "updated_at": _now_iso()})
        .eq("user_id", user_id)
        .eq("role_id", role_id)
        .execute()
    )
    if not response.data:
        raise RuntimeError(f"Update on {FAVORITES_TABLE} returned no data")
    return response.data[0]
