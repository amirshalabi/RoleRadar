"""
Candidate onboarding: raw resume text -> structured profile
(backend.candidate.profile.extract_candidate_profile, the only LLM call
here) -> Python-side dedup (backend.candidate.skills.skills_from_profile)
-> persisted profile + skills.

This is RoleRadar's resume-save entry point - no Streamlit page calls it
yet (see backend.services.discovery's and backend.services.dashboard's
docstrings for that same pre-existing gap), so the FastAPI
POST /candidate/parse route (backend.api.routes.candidate) is the first
real caller.
"""

from __future__ import annotations

from backend.candidate.profile import CandidateProfile, extract_candidate_profile
from backend.candidate.skills import skills_from_profile
from backend.db import candidates as candidates_db


def save_candidate_profile(user_id: str, profile: CandidateProfile, resume_text: str | None = None) -> CandidateProfile:
    """
    Persist an already-extracted CandidateProfile - the persistence half
    of parse_and_save_resume(), split out so a caller that already has a
    profile in hand (e.g. scripts/demo_end_to_end.py building one
    without an LLM call, or a future "edit my profile" flow) doesn't
    need to duplicate this logic:

    - One candidate_profiles row. Only education/experience/projects are
      written - candidate_profiles' interests/constraints columns have
      no corresponding CandidateProfile field yet (a pre-existing
      data-model gap, not something this function papers over), and
      "experience" is populated from `profile.internships`, the closest
      existing field.
    - One candidate_skills row per DEDUPLICATED skill estimate
      (skills_from_profile - never a possibly-repeated raw list),
      including its real evidence_snippets.

    Returns the profile actually persisted (post-dedup), so a caller
    doesn't need a second read to see what was saved.
    """
    deduped_skills = skills_from_profile(profile)
    profile = profile.model_copy(update={"skills": deduped_skills})

    candidates_db.upsert_candidate_profile(
        user_id,
        raw_resume_text=resume_text,
        education=[entry.model_dump() for entry in profile.education],
        experience=[entry.model_dump() for entry in profile.internships],
        projects=[entry.model_dump() for entry in profile.projects],
    )
    for skill in deduped_skills:
        candidates_db.upsert_candidate_skill(
            user_id,
            skill.normalized_skill_name,
            estimated_level=skill.estimated_level,
            confidence=skill.confidence,
            display_name=skill.display_name,
            evidence_snippets=skill.evidence_snippets,
        )
    return profile


def parse_and_save_resume(user_id: str, resume_text: str, model: str | None = None) -> CandidateProfile:
    """Extract a CandidateProfile from `resume_text` (the only LLM call here) and persist it - see save_candidate_profile()."""
    profile = extract_candidate_profile(resume_text, model=model)
    return save_candidate_profile(user_id, profile, resume_text=resume_text)
