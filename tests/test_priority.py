"""Tests for backend.planning.priority (deterministic prep priority ranking)."""

from __future__ import annotations

import pytest

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import calculate_skill_gaps
from backend.planning.priority import (
    DEFAULT_REQUIREMENT_IMPORTANCE,
    MIN_CONFIDENCE_ADJUSTMENT,
    UNMAPPED_TOPIC_WEIGHT,
    calculate_prep_priorities,
    confidence_adjustment,
)
from backend.planning.readiness import calculate_readiness


def _requirement(skill="Python", normalized="python", target=8.0, importance=8.0):
    return RoleRequirement(
        skill=skill, normalized_skill=normalized, target_level=target, importance=importance,
        required=True, evidence=["x"],
    )


def _skill(name="python", level=6.0, confidence=0.6):
    return CandidateSkillEstimate(normalized_skill_name=name, display_name=name, estimated_level=level, confidence=confidence)


# ---------------------------------------------------------------------
# confidence_adjustment
# ---------------------------------------------------------------------


def test_confidence_adjustment_floors_at_minimum_for_zero_confidence() -> None:
    assert confidence_adjustment(0.0) == MIN_CONFIDENCE_ADJUSTMENT


def test_confidence_adjustment_reaches_full_weight_at_full_confidence() -> None:
    assert confidence_adjustment(1.0) == 1.0


def test_confidence_adjustment_is_monotonic() -> None:
    assert confidence_adjustment(0.2) < confidence_adjustment(0.8)


# ---------------------------------------------------------------------
# calculate_prep_priorities: skill-only input
# ---------------------------------------------------------------------


def test_no_gaps_produces_no_prep_items() -> None:
    gaps = calculate_skill_gaps([_skill(level=10.0)], [_requirement(target=5.0)])
    assert calculate_prep_priorities(gaps) == []


def test_missing_skill_still_gets_meaningful_priority_despite_zero_confidence() -> None:
    """A totally unclaimed skill (confidence=0) must not be zeroed out - see MIN_CONFIDENCE_ADJUSTMENT."""
    gaps = calculate_skill_gaps([], [_requirement(target=8.0, importance=8.0)])

    [item] = calculate_prep_priorities(gaps)

    assert item.confidence == 0.0
    assert item.confidence_adjustment == MIN_CONFIDENCE_ADJUSTMENT
    assert item.priority_score > 0


def test_unmapped_requirement_gets_baseline_topic_weight() -> None:
    gaps = calculate_skill_gaps([], [_requirement(normalized="some-obscure-tool", skill="Obscure Tool")])

    [item] = calculate_prep_priorities(gaps)

    assert item.interview_topic_weight == UNMAPPED_TOPIC_WEIGHT
    assert item.source == "role_requirement"


def test_priority_score_formula_matches_spec_for_unmapped_requirement() -> None:
    gaps = calculate_skill_gaps([_skill(name="python", level=2.0)], [_requirement(target=8.0, importance=7.0)])

    [item] = calculate_prep_priorities(gaps)

    expected = item.gap * item.requirement_importance * item.interview_topic_weight * item.confidence_adjustment
    assert item.priority_score == pytest.approx(expected, rel=1e-6)


def test_higher_gap_yields_higher_priority_all_else_equal() -> None:
    small_gap = calculate_skill_gaps([_skill(level=7.0)], [_requirement(target=8.0)])
    large_gap = calculate_skill_gaps([_skill(level=1.0)], [_requirement(target=8.0)])

    [small_item] = calculate_prep_priorities(small_gap)
    [large_item] = calculate_prep_priorities(large_gap)

    assert large_item.priority_score > small_item.priority_score


def test_items_sorted_descending_by_priority_score() -> None:
    gaps = calculate_skill_gaps(
        [],
        [
            _requirement(skill="Python", normalized="python", importance=9.0),
            _requirement(skill="SQL", normalized="sql", importance=2.0),
        ],
    )

    items = calculate_prep_priorities(gaps)

    scores = [item.priority_score for item in items]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------
# calculate_prep_priorities: readiness-only input (topics without a role requirement)
# ---------------------------------------------------------------------


def test_readiness_topic_with_no_requirement_uses_default_importance() -> None:
    readiness = calculate_readiness(CandidateProfile(), role_family="swe")

    items = calculate_prep_priorities([], readiness)

    behavioral = next(item for item in items if item.normalized_skill == "behavioral")
    assert behavioral.requirement_importance == DEFAULT_REQUIREMENT_IMPORTANCE
    assert behavioral.source == "readiness_topic"


def test_fully_ready_topics_produce_no_prep_items() -> None:
    profile = CandidateProfile(
        skills=[_skill(name, 10.0, 1.0) for name in ("algorithms", "python", "systems")]
    )
    from backend.planning.readiness import DiagnosticResult

    diagnostics = [DiagnosticResult(topic=t, observed_level=10.0, confidence=1.0) for t in
                   ("dsa", "coding", "language_knowledge", "systems", "projects", "behavioral")]
    readiness = calculate_readiness(profile, role_family="swe", diagnostics=diagnostics)

    assert calculate_prep_priorities([], readiness) == []


# ---------------------------------------------------------------------
# calculate_prep_priorities: merged skill + readiness input
# ---------------------------------------------------------------------


def test_matching_skill_and_topic_are_merged_not_duplicated() -> None:
    gaps = calculate_skill_gaps([], [_requirement(skill="Algorithms", normalized="algorithms", importance=9.0)])
    readiness = calculate_readiness(CandidateProfile(), role_family="swe")

    items = calculate_prep_priorities(gaps, readiness)

    algorithm_items = [item for item in items if item.normalized_skill in ("algorithms", "dsa")]
    assert len(algorithm_items) == 1
    assert algorithm_items[0].source == "role_requirement_and_readiness_topic"


def test_merged_item_uses_role_specific_importance_not_default() -> None:
    gaps = calculate_skill_gaps([], [_requirement(skill="Algorithms", normalized="algorithms", importance=9.5)])
    readiness = calculate_readiness(CandidateProfile(), role_family="swe")

    [merged] = [item for item in calculate_prep_priorities(gaps, readiness) if item.source == "role_requirement_and_readiness_topic"]

    assert merged.requirement_importance == 9.5  # role-specific, not DEFAULT_REQUIREMENT_IMPORTANCE


def test_merged_item_uses_readiness_topic_weight() -> None:
    gaps = calculate_skill_gaps([], [_requirement(skill="Algorithms", normalized="algorithms")])
    readiness = calculate_readiness(CandidateProfile(), role_family="swe")

    [merged] = [item for item in calculate_prep_priorities(gaps, readiness) if item.source == "role_requirement_and_readiness_topic"]

    assert merged.interview_topic_weight == readiness.weights_used["dsa"]


def test_diagnostic_evidence_flows_into_merged_confidence() -> None:
    from backend.planning.readiness import DiagnosticResult

    gaps = calculate_skill_gaps([_skill(name="algorithms", level=2.0, confidence=0.3)], [_requirement(skill="Algorithms", normalized="algorithms")])
    readiness = calculate_readiness(
        CandidateProfile(skills=[_skill(name="algorithms", level=2.0, confidence=0.3)]),
        role_family="swe",
        diagnostics=[DiagnosticResult(topic="dsa", observed_level=9.0, confidence=0.9)],
    )

    [merged] = [item for item in calculate_prep_priorities(gaps, readiness) if item.source == "role_requirement_and_readiness_topic"]

    assert merged.confidence == pytest.approx(0.9)  # diagnostic confidence, not the low resume confidence


# ---------------------------------------------------------------------
# favorite_priority_multiplier
# ---------------------------------------------------------------------


def test_favorite_priority_multiplier_scales_all_priority_scores() -> None:
    gaps = calculate_skill_gaps([], [_requirement()])

    baseline = calculate_prep_priorities(gaps)[0]
    boosted = calculate_prep_priorities(gaps, favorite_priority_multiplier=3.0)[0]

    assert boosted.priority_score == pytest.approx(baseline.priority_score * 3.0)


def test_default_favorite_priority_multiplier_is_neutral() -> None:
    gaps = calculate_skill_gaps([], [_requirement()])

    explicit = calculate_prep_priorities(gaps, favorite_priority_multiplier=1.0)[0]
    implicit = calculate_prep_priorities(gaps)[0]

    assert explicit.priority_score == implicit.priority_score
