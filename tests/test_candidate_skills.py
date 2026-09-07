"""Tests for backend.candidate.skills (deterministic normalization/dedup)."""

from __future__ import annotations

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.candidate.skills import normalize_skill_name, skills_from_profile


def test_normalize_skill_name_lowercases_and_collapses_whitespace() -> None:
    assert normalize_skill_name("  C++   Programming ") == "c++ programming"


def test_skills_from_profile_normalizes_names() -> None:
    profile = CandidateProfile(
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="  Python ",
                display_name="Python",
                estimated_level=5,
                confidence=0.4,
            )
        ]
    )

    result = skills_from_profile(profile)

    assert result[0].normalized_skill_name == "python"


def test_skills_from_profile_deduplicates_keeping_higher_confidence() -> None:
    profile = CandidateProfile(
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="python",
                display_name="Python",
                estimated_level=4,
                confidence=0.3,
                evidence_snippets=["Skills: Python"],
            ),
            CandidateSkillEstimate(
                normalized_skill_name="Python",
                display_name="Python",
                estimated_level=7,
                confidence=0.8,
                evidence_snippets=["Built a Python ETL pipeline processing 1M rows/day"],
            ),
        ]
    )

    result = skills_from_profile(profile)

    assert len(result) == 1
    assert result[0].confidence == 0.8
    assert result[0].estimated_level == 7


def test_skills_from_profile_handles_no_skills() -> None:
    assert skills_from_profile(CandidateProfile()) == []
