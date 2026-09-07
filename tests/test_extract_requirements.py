"""
Tests for backend.llm.extract_requirements (LLM-backed requirement
extraction). parse_structured is monkeypatched so no real OpenAI call is
made.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.ingestion.normalize import normalize_role
from backend.llm import extract_requirements as extract_requirements_module
from backend.llm.extract_requirements import (
    RoleRequirement,
    _RoleRequirementExtractionResult,
    extract_role_requirements,
)

SAMPLE_RAW_JOB = {
    "jobTitle": "Quantitative Research Intern",
    "companyName": "Meridian Capital",
    "jobDescription": (
        "Requirements: Strong foundation in probability and statistics. "
        "Proficiency in Python; C++ experience is a plus. Comfortable "
        "with algorithms."
    ),
}


def test_extract_role_requirements_rejects_missing_description() -> None:
    role = normalize_role({"title": "SWE", "company": "Acme"})

    with pytest.raises(ValueError):
        extract_role_requirements(role)


def test_extract_role_requirements_returns_mocked_llm_result(monkeypatch: pytest.MonkeyPatch) -> None:
    role = normalize_role(SAMPLE_RAW_JOB)
    canned_result = _RoleRequirementExtractionResult(
        requirements=[
            RoleRequirement(
                skill="Python",
                normalized_skill="python",
                target_level=6.0,
                importance=8.0,
                required=True,
                evidence=["Proficiency in Python"],
            ),
            RoleRequirement(
                skill="C++",
                normalized_skill="c++",
                target_level=4.0,
                importance=3.0,
                required=False,
                evidence=["C++ experience is a plus"],
            ),
            RoleRequirement(
                skill="Probability & Statistics",
                normalized_skill="probability and statistics",
                target_level=7.0,
                importance=9.0,
                required=True,
                evidence=["Strong foundation in probability and statistics"],
            ),
        ]
    )
    captured: dict = {}

    def fake_parse_structured(system_prompt, user_prompt, response_model, model=None):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["response_model"] = response_model
        captured["model"] = model
        return canned_result

    monkeypatch.setattr(extract_requirements_module, "parse_structured", fake_parse_structured)

    result = extract_role_requirements(role, model="gpt-4o")

    assert result == canned_result.requirements
    assert len(result) == 3
    assert captured["response_model"] is _RoleRequirementExtractionResult
    assert captured["model"] == "gpt-4o"
    assert "Meridian Capital" in captured["user_prompt"]
    assert "Quantitative Research Intern" in captured["user_prompt"]
    assert role.description in captured["user_prompt"]


def test_extract_role_requirements_returns_empty_list_when_llm_finds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role = normalize_role(SAMPLE_RAW_JOB)
    monkeypatch.setattr(
        extract_requirements_module,
        "parse_structured",
        lambda **kwargs: _RoleRequirementExtractionResult(requirements=[]),
    )

    result = extract_role_requirements(role)

    assert result == []


def test_role_requirement_rejects_out_of_range_target_level() -> None:
    with pytest.raises(ValidationError):
        RoleRequirement(
            skill="Python",
            normalized_skill="python",
            target_level=15.0,
            importance=5.0,
            required=True,
        )


def test_role_requirement_rejects_out_of_range_importance() -> None:
    with pytest.raises(ValidationError):
        RoleRequirement(
            skill="Python",
            normalized_skill="python",
            target_level=5.0,
            importance=-1.0,
            required=True,
        )
