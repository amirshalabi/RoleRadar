"""
Candidate preferences service: thin pass-through to
backend.db.candidate_preferences, kept as its own module (not folded
into backend.services.candidate) so it's structurally obvious that
user-STATED preferences never touch the resume-parsing/persistence path
- see sql/schema.sql's candidate_preferences table docstring.
"""

from __future__ import annotations

from backend.db import candidate_preferences as preferences_db

TARGET_ROLE_FAMILY_OPTIONS = ["SWE", "Quant", "AI / ML", "Research"]
EMPLOYMENT_TYPE_OPTIONS = ["Internship", "Full-time"]
INTEREST_OPTIONS = ["Systems", "Trading", "Machine Learning", "Data", "Infrastructure", "Research"]


def save_candidate_preferences(
    user_id: str,
    target_role_families: list[str],
    preferred_locations: list[str],
    employment_types: list[str],
    interests: list[str],
) -> dict:
    return preferences_db.upsert_candidate_preferences(
        user_id,
        target_role_families=target_role_families,
        preferred_locations=preferred_locations,
        employment_types=employment_types,
        interests=interests,
    )


def get_candidate_preferences(user_id: str) -> dict | None:
    return preferences_db.get_candidate_preferences(user_id)
