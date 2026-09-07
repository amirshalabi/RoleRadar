"""
Candidate skill tracking.

Deterministic post-processing of LLM-extracted skill estimates: name
normalization and de-duplication happen here in Python, not in the LLM
prompt, so the same skill named differently across resumes/requirements
always maps to one identity (matching candidate_skills'
UNIQUE(user_id, normalized_skill_name) constraint) and re-extraction
never silently keeps a lower-confidence duplicate.
"""

from __future__ import annotations

import re

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_skill_name(name: str) -> str:
    """Lowercase, strip, and collapse whitespace for a stable skill identity."""
    return _WHITESPACE_RE.sub(" ", name.strip().lower())


def skills_from_profile(profile: CandidateProfile) -> list[CandidateSkillEstimate]:
    """
    Return the profile's skill estimates with normalized_skill_name
    guaranteed to be in normalized form, de-duplicated by that
    normalized name. If the LLM reports the same skill twice (e.g. once
    from a skills list and once inferred from a project), the
    higher-confidence estimate wins.
    """
    best_by_name: dict[str, CandidateSkillEstimate] = {}
    for skill in profile.skills:
        key = normalize_skill_name(skill.normalized_skill_name or skill.display_name)
        candidate = skill.model_copy(update={"normalized_skill_name": key})
        existing = best_by_name.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            best_by_name[key] = candidate
    return list(best_by_name.values())
