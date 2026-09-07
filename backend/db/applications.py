"""
Application pipeline data access.

Tracks application state through:
discovered -> saved -> applied -> oa -> interview -> offer / rejected / withdrawn,
keyed on UNIQUE(user_id, role_id). Status values are validated against
VALID_STATUSES but no stage ORDER is enforced here - this is
deliberately not a state machine yet (e.g. nothing stops moving
directly from 'discovered' to 'offer'). The one deliberately simple
exception to "no ordering logic" lives in backend.services.tracking,
where favoriting a role never downgrades an application past 'saved'.

upsert_application() is NOT a blind full-row upsert: every optional
field left as None is preserved from the existing row rather than
overwritten, so calling this again to set (say) interview_date does not
erase a previously-set deadline or notes. This is implemented as an
explicit "UPDATE if a row exists, else INSERT" rather than a Postgres
ON CONFLICT upsert, since a plain upsert would need every column
supplied on every call (Postgrest replaces the whole row on conflict),
which is exactly the clobbering behavior this avoids.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from backend.db.client import get_client

logger = logging.getLogger(__name__)

APPLICATIONS_TABLE = "applications"

VALID_STATUSES = {
    "discovered",
    "saved",
    "applied",
    "oa",
    "interview",
    "offer",
    "rejected",
    "withdrawn",
}
DEFAULT_STATUS = "discovered"


def _validate_status(status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid application status: {status!r}. Must be one of {sorted(VALID_STATUSES)}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def upsert_application(
    user_id: str,
    role_id: str,
    status: str | None = None,
    *,
    application_date: str | None = None,
    deadline: str | None = None,
    interview_date: str | None = None,
    hours_available_per_day: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Insert or update a user's application row for a role, keyed on
    UNIQUE(user_id, role_id).

    Every parameter besides user_id/role_id is optional. Any field left
    None is left UNCHANGED if the row already exists - only the fields
    you actually pass are written. On first creation (no existing row),
    an omitted `status` defaults to 'discovered' and every other omitted
    field is stored as NULL.
    """
    if status is not None:
        _validate_status(status)

    provided: dict[str, Any] = {
        "status": status,
        "application_date": application_date,
        "deadline": deadline,
        "interview_date": interview_date,
        "hours_available_per_day": hours_available_per_day,
        "notes": notes,
    }
    provided = {key: value for key, value in provided.items() if value is not None}
    provided["updated_at"] = _now_iso()

    client = get_client()
    existing = get_application(user_id, role_id)

    if existing is not None:
        response = (
            client.table(APPLICATIONS_TABLE)
            .update(provided)
            .eq("user_id", user_id)
            .eq("role_id", role_id)
            .execute()
        )
    else:
        values = {"user_id": user_id, "role_id": role_id, "status": DEFAULT_STATUS, **provided}
        response = client.table(APPLICATIONS_TABLE).insert(values).execute()

    if not response.data:
        raise RuntimeError(f"Upsert into {APPLICATIONS_TABLE} returned no data")
    return response.data[0]


def list_applications(user_id: str) -> list[dict[str, Any]]:
    """Return all application rows for a user."""
    client = get_client()
    response = client.table(APPLICATIONS_TABLE).select("*").eq("user_id", user_id).execute()
    return response.data or []


def list_applications_with_roles(user_id: str) -> list[dict[str, Any]]:
    """
    Every application row for a user, with its role's title/company/etc
    embedded (a Postgrest foreign-table select) - what dashboard/list
    views need without a second query per row. Additive alongside
    list_applications(), which stays a plain flat select so its existing
    callers are unaffected.
    """
    client = get_client()
    response = (
        client.table(APPLICATIONS_TABLE).select("*, roles(*)").eq("user_id", user_id).execute()
    )
    return response.data or []
