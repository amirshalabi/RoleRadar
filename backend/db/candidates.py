"""
Candidate data access.

CRUD/upsert operations for candidate profiles (UNIQUE(user_id)) and the
candidate_skills table (UNIQUE(user_id, normalized_skill_name)).
"""

from __future__ import annotations

import logging
from typing import Any

from backend.db.client import get_client
from backend.db.upserts import upsert_row

logger = logging.getLogger(__name__)

CANDIDATE_PROFILES_TABLE = "candidate_profiles"
CANDIDATE_SKILLS_TABLE = "candidate_skills"


def upsert_candidate_profile(
    user_id: str,
    raw_resume_text: str | None = None,
    education: list[dict[str, Any]] | None = None,
    experience: list[dict[str, Any]] | None = None,
    projects: list[dict[str, Any]] | None = None,
    interests: list[dict[str, Any]] | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Insert or update a candidate's single profile row, keyed on
    UNIQUE(user_id). Re-uploading a resume updates the existing profile
    instead of creating a second one for the same user.
    """
    values = {
        "user_id": user_id,
        "raw_resume_text": raw_resume_text,
        "education": education or [],
        "experience": experience or [],
        "projects": projects or [],
        "interests": interests or [],
        "constraints": constraints or {},
    }
    return upsert_row(CANDIDATE_PROFILES_TABLE, values, on_conflict="user_id")


def get_candidate_profile(user_id: str) -> dict[str, Any] | None:
    """Fetch a candidate's profile row, or None if they haven't uploaded a resume yet."""
    client = get_client()
    response = (
        client.table(CANDIDATE_PROFILES_TABLE).select("*").eq("user_id", user_id).limit(1).execute()
    )
    return response.data[0] if response.data else None


def upsert_candidate_skill(
    user_id: str,
    normalized_skill_name: str,
    estimated_level: float,
    confidence: float,
    display_name: str | None = None,
    evidence_source: str | None = None,
) -> dict[str, Any]:
    """
    Insert or update a candidate's estimate for a single skill, keyed on
    UNIQUE(user_id, normalized_skill_name). Re-estimating a skill (e.g.
    after a diagnostic result) updates the same row rather than creating
    a new one.
    """
    values = {
        "user_id": user_id,
        "normalized_skill_name": normalized_skill_name,
        "display_name": display_name,
        "estimated_level": estimated_level,
        "confidence": confidence,
        "evidence_source": evidence_source,
    }
    return upsert_row(
        CANDIDATE_SKILLS_TABLE,
        values,
        on_conflict="user_id,normalized_skill_name",
    )


def list_candidate_skills(user_id: str) -> list[dict[str, Any]]:
    """Return all skill estimates for a user."""
    client = get_client()
    response = (
        client.table(CANDIDATE_SKILLS_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    return response.data or []
