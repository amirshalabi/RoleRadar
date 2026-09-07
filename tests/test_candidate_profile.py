"""
Tests for backend.candidate.profile (LLM-backed structured extraction).

parse_structured is monkeypatched so no real OpenAI call is made; these
tests verify (a) the schema's own validation rules, and (b) that
extract_candidate_profile wires resume text into the prompt and returns
whatever Pydantic-validated structure the LLM layer produces, unmodified.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.candidate import profile as profile_module
from backend.candidate.profile import (
    CandidateProfile,
    CandidateSkillEstimate,
    extract_candidate_profile,
)


def test_extract_candidate_profile_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        extract_candidate_profile("   ")


def test_extract_candidate_profile_returns_mocked_llm_result(monkeypatch: pytest.MonkeyPatch) -> None:
    canned_profile = CandidateProfile(
        programming_languages=["Python", "C++"],
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="python",
                display_name="Python",
                estimated_level=6.0,
                confidence=0.4,
                evidence_snippets=["Skills: Python, C++, React, Docker"],
            )
        ],
    )
    captured: dict = {}

    def fake_parse_structured(system_prompt, user_prompt, response_model, model=None):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["response_model"] = response_model
        captured["model"] = model
        return canned_profile

    monkeypatch.setattr(profile_module, "parse_structured", fake_parse_structured)

    result = extract_candidate_profile("Jane Doe resume text...", model="gpt-4o")

    assert result is canned_profile
    assert captured["response_model"] is CandidateProfile
    assert captured["model"] == "gpt-4o"
    assert "Jane Doe resume text..." in captured["user_prompt"]


def test_candidate_skill_estimate_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        CandidateSkillEstimate(
            normalized_skill_name="python",
            display_name="Python",
            estimated_level=6.0,
            confidence=1.5,
        )


def test_candidate_skill_estimate_rejects_out_of_range_level() -> None:
    with pytest.raises(ValidationError):
        CandidateSkillEstimate(
            normalized_skill_name="python",
            display_name="Python",
            estimated_level=15.0,
            confidence=0.5,
        )


def test_candidate_profile_defaults_are_empty_lists() -> None:
    profile = CandidateProfile()

    assert profile.education == []
    assert profile.skills == []
    assert profile.internships == []
