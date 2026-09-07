"""
Application pipeline data access.

Tracks application state through:
Discovered -> Saved -> Applied -> OA -> Interview -> Offer / Rejected / Withdrawn,
keyed on UNIQUE(user_id, role_id).
"""

from __future__ import annotations

import logging
from typing import Any

from backend.db.client import get_client
from backend.db.upserts import upsert_row

logger = logging.getLogger(__name__)

APPLICATIONS_TABLE = "applications"

VALID_STATUSES = {
    "Discovered",
    "Saved",
    "Applied",
    "OA",
    "Interview",
    "Offer",
    "Rejected",
    "Withdrawn",
}


def upsert_application(
    user_id: str,
    role_id: str,
    status: str = "Discovered",
    interview_date: str | None = None,
    hours_available_per_day: float | None = None,
) -> dict[str, Any]:
    """
    Insert or update a user's application status for a role, keyed on
    UNIQUE(user_id, role_id). Advancing a role through the pipeline
    (Discovered -> Saved -> Applied -> ...) updates the same row rather
    than creating a new one.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid application status: {status!r}. Must be one of {sorted(VALID_STATUSES)}"
        )
    values = {
        "user_id": user_id,
        "role_id": role_id,
        "status": status,
        "interview_date": interview_date,
        "hours_available_per_day": hours_available_per_day,
    }
    return upsert_row(APPLICATIONS_TABLE, values, on_conflict="user_id,role_id")


def get_application(user_id: str, role_id: str) -> dict[str, Any] | None:
    """Fetch a user's application row for a role, or None if none exists."""
    client = get_client()
    response = (
        client.table(APPLICATIONS_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("role_id", role_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def list_applications(user_id: str) -> list[dict[str, Any]]:
    """Return all application rows for a user."""
    client = get_client()
    response = (
        client.table(APPLICATIONS_TABLE).select("*").eq("user_id", user_id).execute()
    )
    return response.data or []
