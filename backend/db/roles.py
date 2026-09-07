"""
Roles data access.

Idempotent upsert and query operations for the roles table, keyed on
UNIQUE(external_id), so repeated ingestion never creates duplicate rows.

Also owns fit_scores persistence (upsert_fit_score). A fit score is a
per-(user, role) fact, and lives here rather than in candidates.py so
everything that revolves around a role - the role row itself and each
candidate's computed fit against it - stays in one module.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.db.client import get_client
from backend.db.upserts import upsert_row
from backend.utils.hashing import generate_role_external_id

logger = logging.getLogger(__name__)

ROLES_TABLE = "roles"
FIT_SCORES_TABLE = "fit_scores"


def upsert_role(
    company: str,
    title: str,
    location: str | None = None,
    url: str | None = None,
    description: str | None = None,
    role_family: str | None = None,
    external_id: str | None = None,
    raw_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Insert or update a role keyed on external_id.

    If `external_id` is not supplied (the source API doesn't provide a
    stable id of its own), one is derived deterministically from
    normalized company + title + location + url, so re-ingesting the
    same posting always upserts onto the same row instead of creating a
    duplicate.
    """
    resolved_external_id = external_id or generate_role_external_id(
        company, title, location, url
    )
    values = {
        "external_id": resolved_external_id,
        "company": company,
        "title": title,
        "location": location,
        "url": url,
        "description": description,
        "role_family": role_family,
        "raw_source": raw_source or {},
    }
    return upsert_row(ROLES_TABLE, values, on_conflict="external_id")


def get_role_by_external_id(external_id: str) -> dict[str, Any] | None:
    """Fetch a role by its external_id, or None if it doesn't exist."""
    client = get_client()
    response = (
        client.table(ROLES_TABLE)
        .select("*")
        .eq("external_id", external_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def list_roles(limit: int = 50) -> list[dict[str, Any]]:
    """Most recently seen roles first - for a Discover page's browse list."""
    client = get_client()
    response = (
        client.table(ROLES_TABLE).select("*").order("created_at", desc=True).limit(limit).execute()
    )
    return response.data or []


def upsert_fit_score(
    user_id: str,
    role_id: str,
    overall_score: float,
    weights_used: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Insert or update the deterministic overall fit score for a
    (user, role) pair, keyed on UNIQUE(user_id, role_id). Recomputing a
    score (e.g. after a skill estimate changes) updates the same row.
    """
    values = {
        "user_id": user_id,
        "role_id": role_id,
        "overall_score": overall_score,
        "weights_used": weights_used or {},
    }
    return upsert_row(FIT_SCORES_TABLE, values, on_conflict="user_id,role_id")


def list_fit_scores_for_user(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    A user's fit scores, highest first, with each matching role embedded
    (a Postgrest foreign-table select) - what a "top matches" view needs
    without a second query per role.
    """
    client = get_client()
    response = (
        client.table(FIT_SCORES_TABLE)
        .select("*, roles(*)")
        .eq("user_id", user_id)
        .order("overall_score", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []
