"""
Tests for backend.planning.readiness (interview readiness, deterministic
and separate from fit). No LLM calls, no database - pure Python over
hand-built Pydantic objects.
"""

from __future__ import annotations

import pytest

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate, ProjectEntry
from backend.planning.readiness import (
    DIAGNOSTIC_EVIDENCE_MULTIPLIER,
    GENERAL_TECHNICAL_READINESS_WEIGHTS,
    MAJOR_GAP_READINESS_THRESHOLD,
    QUANT_READINESS_WEIGHTS,
    SWE_READINESS_WEIGHTS,
    DiagnosticResult,
    calculate_readiness,
    get_readiness_weights,
)


def _skill(name, level, confidence):
    return CandidateSkillEstimate(normalized_skill_name=name, display_name=name, estimated_level=level, confidence=confidence)


# ---------------------------------------------------------------------
# Weight table invariants
# ---------------------------------------------------------------------


def test_quant_weights_match_spec_and_sum_to_one() -> None:
    assert QUANT_READINESS_WEIGHTS == {
        "probability": 0.25,
        "coding": 0.20,
        "algorithms": 0.15,
        "mental_math": 0.15,
        "market_knowledge": 0.10,
        "game_ev_reasoning": 0.10,
        "behavioral": 0.05,
    }
    assert abs(sum(QUANT_READINESS_WEIGHTS.values()) - 1.0) < 1e-9


def test_swe_weights_match_spec_and_sum_to_one() -> None:
    assert SWE_READINESS_WEIGHTS == {
        "dsa": 0.30,
        "coding": 0.25,
        "language_knowledge": 0.15,
        "systems": 0.15,
        "projects": 0.10,
        "behavioral": 0.05,
    }
    assert abs(sum(SWE_READINESS_WEIGHTS.values()) - 1.0) < 1e-9


def test_general_fallback_weights_sum_to_one() -> None:
    assert abs(sum(GENERAL_TECHNICAL_READINESS_WEIGHTS.values()) - 1.0) < 1e-9


def test_get_readiness_weights_falls_back_for_unknown_family() -> None:
    assert get_readiness_weights("some-made-up-family") == GENERAL_TECHNICAL_READINESS_WEIGHTS
    assert get_readiness_weights(None) == GENERAL_TECHNICAL_READINESS_WEIGHTS


def test_get_readiness_weights_is_case_insensitive() -> None:
    assert get_readiness_weights("QUANT") == QUANT_READINESS_WEIGHTS
    assert get_readiness_weights("  Swe ") == SWE_READINESS_WEIGHTS


# ---------------------------------------------------------------------
# Quant candidate
# ---------------------------------------------------------------------


def test_quant_candidate_uses_quant_weights() -> None:
    profile = CandidateProfile(skills=[_skill("probability", 7.0, 0.6)])

    result = calculate_readiness(profile, role_family="quant")

    assert result.weights_used == QUANT_READINESS_WEIGHTS
    assert {t.topic for t in result.topic_readiness} == set(QUANT_READINESS_WEIGHTS)


def test_quant_candidate_probability_skill_drives_probability_and_ev_topics() -> None:
    """'probability' is aliased into both the probability topic and game_ev_reasoning."""
    profile = CandidateProfile(skills=[_skill("probability", 8.0, 0.5)])

    result = calculate_readiness(profile, role_family="quant")

    by_topic = {t.topic: t for t in result.topic_readiness}
    assert by_topic["probability"].readiness_score == pytest.approx(80.0)
    assert by_topic["game_ev_reasoning"].readiness_score == pytest.approx(80.0)


def test_quant_candidate_with_no_evidence_scores_zero_overall() -> None:
    result = calculate_readiness(CandidateProfile(), role_family="quant")

    assert result.overall_readiness == 0.0
    assert all(t.readiness_score == 0.0 for t in result.topic_readiness)


# ---------------------------------------------------------------------
# SWE candidate
# ---------------------------------------------------------------------


def test_swe_candidate_uses_swe_weights() -> None:
    profile = CandidateProfile(skills=[_skill("algorithms", 6.0, 0.5)])

    result = calculate_readiness(profile, role_family="swe")

    assert result.weights_used == SWE_READINESS_WEIGHTS
    assert {t.topic for t in result.topic_readiness} == set(SWE_READINESS_WEIGHTS)


def test_swe_candidate_algorithms_skill_feeds_dsa_topic() -> None:
    profile = CandidateProfile(skills=[_skill("algorithms", 9.0, 0.7)])

    result = calculate_readiness(profile, role_family="swe")

    dsa = next(t for t in result.topic_readiness if t.topic == "dsa")
    assert dsa.readiness_score == pytest.approx(90.0)
    assert dsa.evidence_source == "resume_only"


def test_swe_candidate_projects_topic_scales_with_project_count() -> None:
    profile_no_projects = CandidateProfile(projects=[])
    profile_many_projects = CandidateProfile(
        projects=[ProjectEntry(name=f"p{i}", description="d") for i in range(3)]
    )

    none_result = calculate_readiness(profile_no_projects, role_family="swe")
    full_result = calculate_readiness(profile_many_projects, role_family="swe")

    none_projects = next(t for t in none_result.topic_readiness if t.topic == "projects")
    full_projects = next(t for t in full_result.topic_readiness if t.topic == "projects")
    assert none_projects.readiness_score == 0.0
    assert full_projects.readiness_score == pytest.approx(100.0)


def test_swe_candidate_language_knowledge_uses_best_matching_language() -> None:
    profile = CandidateProfile(
        skills=[_skill("python", 4.0, 0.4), _skill("java", 9.0, 0.6)]
    )

    result = calculate_readiness(profile, role_family="swe")

    language_knowledge = next(t for t in result.topic_readiness if t.topic == "language_knowledge")
    assert language_knowledge.readiness_score == pytest.approx(90.0)  # strongest evidence (java) wins, not averaged down


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_diagnostic_improves_readiness_over_no_evidence() -> None:
    profile = CandidateProfile(skills=[])

    before = calculate_readiness(profile, role_family="swe")
    after = calculate_readiness(
        profile, role_family="swe", diagnostics=[DiagnosticResult(topic="dsa", observed_level=8.0, confidence=0.9)]
    )

    dsa_before = next(t for t in before.topic_readiness if t.topic == "dsa")
    dsa_after = next(t for t in after.topic_readiness if t.topic == "dsa")
    assert dsa_before.readiness_score == 0.0
    assert dsa_after.readiness_score > dsa_before.readiness_score
    assert dsa_after.evidence_source == "diagnostic_only"
    assert after.overall_readiness > before.overall_readiness


def test_diagnostic_dominates_resume_estimate_at_equal_confidence() -> None:
    """The DIAGNOSTIC_EVIDENCE_MULTIPLIER guarantee: diagnostics outweigh resume evidence even at equal confidence."""
    profile = CandidateProfile(skills=[_skill("algorithms", 2.0, 0.6)])  # low resume estimate

    result = calculate_readiness(
        profile,
        role_family="swe",
        diagnostics=[DiagnosticResult(topic="dsa", observed_level=9.0, confidence=0.6)],  # same confidence, high score
    )

    dsa = next(t for t in result.topic_readiness if t.topic == "dsa")
    # combined score should sit much closer to the diagnostic's 9.0 than the resume's 2.0
    assert dsa.combined_score > 7.0
    assert dsa.evidence_source == "diagnostic_and_resume"


def test_diagnostic_combination_uses_max_confidence() -> None:
    profile = CandidateProfile(skills=[_skill("algorithms", 5.0, 0.3)])

    result = calculate_readiness(
        profile, role_family="swe", diagnostics=[DiagnosticResult(topic="dsa", observed_level=7.0, confidence=0.9)]
    )

    dsa = next(t for t in result.topic_readiness if t.topic == "dsa")
    assert dsa.combined_confidence == 0.9


def test_missing_diagnostic_falls_back_to_resume_only() -> None:
    profile = CandidateProfile(skills=[_skill("algorithms", 6.0, 0.5)])

    result = calculate_readiness(profile, role_family="swe", diagnostics=None)

    dsa = next(t for t in result.topic_readiness if t.topic == "dsa")
    assert dsa.evidence_source == "resume_only"
    assert dsa.diagnostic_score is None
    assert dsa.diagnostic_confidence is None
    assert dsa.combined_score == pytest.approx(6.0)


def test_diagnostic_for_unrelated_topic_does_not_affect_other_topics() -> None:
    profile = CandidateProfile(skills=[_skill("algorithms", 6.0, 0.5)])

    result = calculate_readiness(
        profile, role_family="swe", diagnostics=[DiagnosticResult(topic="behavioral", observed_level=9.0)]
    )

    dsa = next(t for t in result.topic_readiness if t.topic == "dsa")
    assert dsa.evidence_source == "resume_only"
    assert dsa.combined_score == pytest.approx(6.0)


# ---------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------


def test_overall_readiness_within_bounds_for_maximal_evidence() -> None:
    profile = CandidateProfile(
        skills=[_skill(name, 10.0, 1.0) for name in ("probability", "python", "algorithms")]
    )
    diagnostics = [DiagnosticResult(topic=topic, observed_level=10.0, confidence=1.0) for topic in QUANT_READINESS_WEIGHTS]

    result = calculate_readiness(profile, role_family="quant", diagnostics=diagnostics)

    assert 0.0 <= result.overall_readiness <= 100.0
    assert result.overall_readiness == pytest.approx(100.0)
    for topic in result.topic_readiness:
        assert 0.0 <= topic.readiness_score <= 100.0
        assert 0.0 <= topic.combined_confidence <= 1.0


def test_overall_readiness_within_bounds_for_zero_evidence() -> None:
    result = calculate_readiness(CandidateProfile(), role_family="swe")

    assert 0.0 <= result.overall_readiness <= 100.0
    assert result.overall_readiness == 0.0


def test_major_gaps_flag_topics_below_threshold() -> None:
    profile = CandidateProfile(skills=[_skill("algorithms", 10.0, 1.0)])  # only dsa well-covered

    result = calculate_readiness(profile, role_family="swe")

    gap_topics = {gap.topic for gap in result.major_gaps}
    assert "behavioral" in gap_topics
    assert "dsa" not in gap_topics
    for gap in result.major_gaps:
        assert gap.readiness_score < MAJOR_GAP_READINESS_THRESHOLD


def test_major_gaps_sorted_by_weighted_deficit_descending() -> None:
    result = calculate_readiness(CandidateProfile(), role_family="swe")  # everything is a gap

    deficits = [gap.weighted_deficit for gap in result.major_gaps]
    assert deficits == sorted(deficits, reverse=True)


def test_invalid_diagnostic_level_rejected_by_pydantic() -> None:
    with pytest.raises(Exception):
        DiagnosticResult(topic="dsa", observed_level=15.0)


def test_readiness_result_role_family_is_preserved() -> None:
    result = calculate_readiness(CandidateProfile(), role_family="quant")
    assert result.role_family == "quant"
