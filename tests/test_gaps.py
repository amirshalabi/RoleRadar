"""Tests for backend.matching.gaps (deterministic skill gap calculation)."""

from __future__ import annotations

from backend.candidate.profile import CandidateSkillEstimate
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import calculate_skill_gaps


def _requirement(skill="Python", normalized="python", target=8.0, importance=9.0, required=True) -> RoleRequirement:
    return RoleRequirement(
        skill=skill,
        normalized_skill=normalized,
        target_level=target,
        importance=importance,
        required=required,
        evidence=["evidence text"],
    )


def _candidate_skill(name="python", level=5.0, confidence=0.5) -> CandidateSkillEstimate:
    return CandidateSkillEstimate(
        normalized_skill_name=name,
        display_name=name,
        estimated_level=level,
        confidence=confidence,
        evidence_snippets=["snippet"],
    )


def test_raw_gap_formula_matches_spec() -> None:
    requirement = _requirement(target=8.0)
    candidate_skill = _candidate_skill(level=5.0)

    [gap] = calculate_skill_gaps([candidate_skill], [requirement])

    assert gap.raw_gap == 3.0  # max(8 - 5, 0)


def test_raw_gap_never_negative_when_candidate_exceeds_target() -> None:
    requirement = _requirement(target=5.0)
    candidate_skill = _candidate_skill(level=9.0)

    [gap] = calculate_skill_gaps([candidate_skill], [requirement])

    assert gap.raw_gap == 0.0
    assert gap.satisfaction_ratio == 1.0  # capped, no bonus for exceeding


def test_weighted_gap_incorporates_importance() -> None:
    requirement = _requirement(target=8.0, importance=5.0)
    candidate_skill = _candidate_skill(level=4.0)

    [gap] = calculate_skill_gaps([candidate_skill], [requirement])

    assert gap.raw_gap == 4.0
    assert gap.weighted_gap == 4.0 * (5.0 / 10.0)


def test_missing_skill_treated_as_zero_level_zero_confidence() -> None:
    requirement = _requirement(normalized="rust", target=6.0)

    [gap] = calculate_skill_gaps([], [requirement])

    assert gap.matched is False
    assert gap.candidate_level == 0.0
    assert gap.confidence == 0.0
    assert gap.raw_gap == 6.0
    assert gap.satisfaction_ratio == 0.0


def test_skill_matching_is_case_and_whitespace_insensitive() -> None:
    requirement = _requirement(normalized="  Python ", target=5.0)
    candidate_skill = _candidate_skill(name="PYTHON", level=5.0)

    [gap] = calculate_skill_gaps([candidate_skill], [requirement])

    assert gap.matched is True
    assert gap.raw_gap == 0.0


def test_zero_target_level_is_fully_satisfied() -> None:
    requirement = _requirement(target=0.0)

    [gap] = calculate_skill_gaps([], [requirement])

    assert gap.satisfaction_ratio == 1.0
    assert gap.raw_gap == 0.0


def test_calculate_skill_gaps_returns_empty_list_for_no_requirements() -> None:
    assert calculate_skill_gaps([_candidate_skill()], []) == []


def test_calculate_skill_gaps_preserves_requirement_order() -> None:
    requirements = [
        _requirement(skill="Python", normalized="python"),
        _requirement(skill="C++", normalized="c++"),
        _requirement(skill="Statistics", normalized="statistics"),
    ]

    results = calculate_skill_gaps([], requirements)

    assert [r.display_skill for r in results] == ["Python", "C++", "Statistics"]
