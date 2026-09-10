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
    coursework: list[str] | None = None,
    programming_languages: list[str] | None = None,
    frameworks: list[str] | None = None,
    tools: list[str] | None = None,
    research: list[dict[str, Any]] | None = None,
    domain_experience: list[str] | None = None,
    resume_filename: str | None = None,
    resume_content_hash: str | None = None,
    resume_parsed_at: str | None = None,
) -> dict[str, Any]:
    """
    Insert or update a candidate's single profile row, keyed on
    UNIQUE(user_id). Re-uploading a resume updates the existing profile
    instead of creating a second one for the same user - see
    backend.services.candidate.save_candidate_profile for how a full
    backend.candidate.profile.CandidateProfile maps onto these columns.

    resume_filename/resume_content_hash/resume_parsed_at are only
    included in the upsert payload when explicitly supplied (not simply
    passed as None) - PostgREST's upsert only SETs the columns present
    in the payload, so omitting the key entirely leaves an existing
    value on the row untouched on conflict, rather than nulling it out
    on every unrelated profile update.
    """
    values: dict[str, Any] = {
        "user_id": user_id,
        "raw_resume_text": raw_resume_text,
        "education": education or [],
        "experience": experience or [],
        "projects": projects or [],
        "interests": interests or [],
        "constraints": constraints or {},
        "coursework": coursework or [],
        "programming_languages": programming_languages or [],
        "frameworks": frameworks or [],
        "tools": tools or [],
        "research": research or [],
        "domain_experience": domain_experience or [],
    }
    if resume_filename is not None:
        values["resume_filename"] = resume_filename
    if resume_content_hash is not None:
        values["resume_content_hash"] = resume_content_hash
    if resume_parsed_at is not None:
        values["resume_parsed_at"] = resume_parsed_at
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
    evidence_snippets: list[str] | None = None,
) -> dict[str, Any]:
    """
    Insert or update a candidate's estimate for a single skill, keyed on
    UNIQUE(user_id, normalized_skill_name). Re-estimating a skill (e.g.
    after a diagnostic result) updates the same row rather than creating
    a new one. `evidence_snippets` persists the real resume quotes
    behind this estimate (backend.candidate.profile.CandidateSkillEstimate
    .evidence_snippets) so an "evidence count" display never has to
    fabricate a number - it stays empty until real snippets are saved.
    """
    values = {
        "user_id": user_id,
        "normalized_skill_name": normalized_skill_name,
        "display_name": display_name,
        "estimated_level": estimated_level,
        "confidence": confidence,
        "evidence_source": evidence_source,
        "evidence_snippets": evidence_snippets or [],
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


def delete_candidate_skill(user_id: str, normalized_skill_name: str) -> None:
    """
    Remove one candidate_skills row. Used when a resume is replaced and
    a previously-estimated skill no longer appears anywhere in the new
    resume - see backend.services.candidate.save_candidate_profile,
    which computes the stale set and calls this once per stale skill
    rather than ever leaving a resume-derived skill behind after its
    source evidence is gone.
    """
    client = get_client()
    client.table(CANDIDATE_SKILLS_TABLE).delete().eq("user_id", user_id).eq(
        "normalized_skill_name", normalized_skill_name
    ).execute()
