"""
Candidate preferences data access.

A separate table (not columns on candidate_profiles) for user-STATED
target role families / preferred locations / employment types /
interests - see sql/schema.sql's candidate_preferences table docstring
for why these must never be mixed with resume-INFERRED facts.
"""

from __future__ import annotations

from typing import Any

from backend.db.client import get_client
from backend.db.upserts import upsert_row

CANDIDATE_PREFERENCES_TABLE = "candidate_preferences"


def upsert_candidate_preferences(
    user_id: str,
    target_role_families: list[str] | None = None,
    preferred_locations: list[str] | None = None,
    employment_types: list[str] | None = None,
    interests: list[str] | None = None,
) -> dict[str, Any]:
    """Insert or update a candidate's single preferences row, keyed on UNIQUE(user_id)."""
    values = {
        "user_id": user_id,
        "target_role_families": target_role_families or [],
        "preferred_locations": preferred_locations or [],
        "employment_types": employment_types or [],
        "interests": interests or [],
    }
    return upsert_row(CANDIDATE_PREFERENCES_TABLE, values, on_conflict="user_id")


def get_candidate_preferences(user_id: str) -> dict[str, Any] | None:
    """Fetch a candidate's preferences row, or None if they haven't set any yet."""
    client = get_client()
    response = (
        client.table(CANDIDATE_PREFERENCES_TABLE).select("*").eq("user_id", user_id).limit(1).execute()
    )
    return response.data[0] if response.data else None
