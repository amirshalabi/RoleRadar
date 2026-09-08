"""
Comprehensive tests for backend.matching.scorer (deterministic fit
scoring). No LLM calls occur anywhere in this module or its tests -
every input here is a hand-built Pydantic object.
"""

from __future__ import annotations

from backend.candidate.profile import (
    CandidateProfile,
    CandidateSkillEstimate,
    ExperienceEntry,
    ProjectEntry,
    ResearchEntry,
)
from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.scorer import (
    DEFAULT_WEIGHTS,
    _clamp_score,
    calculate_constraint_fit,
    calculate_coursework_fit,
    calculate_domain_fit,
    calculate_experience_fit,
    calculate_fit_score,
    calculate_interest_fit,
    calculate_technical_fit,
    get_weights_for_role_family,
)


def _requirement(skill="Python", normalized="python", target=8.0, importance=9.0, required=True) -> RoleRequirement:
    return RoleRequirement(
        skill=skill,
        normalized_skill=normalized,
        target_level=target,
        importance=importance,
        required=required,
        evidence=["evidence text"],
    )


def _candidate_skill(name="python", display=None, level=5.0, confidence=0.5) -> CandidateSkillEstimate:
    return CandidateSkillEstimate(
        normalized_skill_name=name,
        display_name=display or name,
        estimated_level=level,
        confidence=confidence,
        evidence_snippets=["snippet"],
    )


def _role(**overrides):
    raw = {"title": "Software Engineer Intern", "company": "Acme Corp", "location": "Remote"}
    raw.update(overrides)
    return normalize_role(raw)


# ---------------------------------------------------------------------
# DEFAULT_WEIGHTS invariants
# ---------------------------------------------------------------------


def test_default_weights_match_spec_and_sum_to_one() -> None:
    assert DEFAULT_WEIGHTS == {
        "technical": 0.35,
        "experience": 0.20,
        "coursework": 0.10,
        "domain": 0.15,
        "interest": 0.10,
        "constraints": 0.10,
    }
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_get_weights_for_role_family_falls_back_to_default_when_unregistered() -> None:
    assert get_weights_for_role_family("quant") == DEFAULT_WEIGHTS
    assert get_weights_for_role_family(None) == DEFAULT_WEIGHTS


# ---------------------------------------------------------------------
# Technical fit: the core scenarios requested
# ---------------------------------------------------------------------


def test_technical_fit_perfect_match_scores_100() -> None:
    requirements = [
        _requirement(skill="Python", normalized="python", target=7.0, importance=9.0, required=True),
        _requirement(skill="Algorithms", normalized="algorithms", target=6.0, importance=7.0, required=True),
    ]
    candidate_skills = [
        _candidate_skill(name="python", level=8.0, confidence=0.8),
        _candidate_skill(name="algorithms", level=9.0, confidence=0.9),
    ]

    result = calculate_technical_fit(candidate_skills, requirements)

    assert result.score == 100.0
    assert all(gap.satisfaction_ratio == 1.0 for gap in result.gaps)


def test_technical_fit_missing_required_skill_scores_low() -> None:
    requirements = [_requirement(skill="Python", normalized="python", target=8.0, importance=9.0, required=True)]

    result = calculate_technical_fit([], requirements)

    assert result.score == 0.0
    assert result.gaps[0].matched is False


def test_technical_fit_missing_required_hurts_more_than_missing_optional() -> None:
    requirements_required = [
        _requirement(skill="Python", normalized="python", target=8.0, importance=8.0, required=True)
    ]
    requirements_optional = [
        _requirement(skill="Python", normalized="python", target=8.0, importance=8.0, required=False)
    ]

    required_missing = calculate_technical_fit([], requirements_required)
    optional_missing = calculate_technical_fit([], requirements_optional)

    # Missing the only requirement scores 0 either way (100% of the weight
    # is unmet) - the required multiplier's effect shows up once there is
    # at least one *other* satisfied requirement to weigh it against.
    assert required_missing.score == 0.0
    assert optional_missing.score == 0.0

    other_met = _requirement(skill="C++", normalized="c++", target=5.0, importance=8.0, required=False)
    candidate_meeting_other = [_candidate_skill(name="c++", level=5.0, confidence=0.9)]

    mixed_required_missing = calculate_technical_fit(
        candidate_meeting_other, [requirements_required[0], other_met]
    )
    mixed_optional_missing = calculate_technical_fit(
        candidate_meeting_other, [requirements_optional[0], other_met]
    )

    assert mixed_required_missing.score < mixed_optional_missing.score


def test_technical_fit_candidate_exceeding_requirements_caps_at_100() -> None:
    requirements = [_requirement(skill="Python", normalized="python", target=4.0, importance=10.0, required=True)]
    candidate_skills = [_candidate_skill(name="python", level=10.0, confidence=0.9)]

    result = calculate_technical_fit(candidate_skills, requirements)

    assert result.score == 100.0
    assert result.gaps[0].satisfaction_ratio == 1.0
    assert result.gaps[0].raw_gap == 0.0


def test_technical_fit_optional_skill_mismatch_only_partially_hurts_score() -> None:
    requirements = [
        _requirement(skill="Python", normalized="python", target=7.0, importance=9.0, required=True),
        _requirement(skill="Rust", normalized="rust", target=6.0, importance=5.0, required=False),
    ]
    candidate_skills = [_candidate_skill(name="python", level=7.0, confidence=0.8)]  # no rust

    result = calculate_technical_fit(candidate_skills, requirements)

    assert 0.0 < result.score < 100.0


def test_technical_fit_no_skill_overlap_at_all_scores_zero() -> None:
    """Candidate's claimed skills and the role's requirements share nothing in common - the realistic 'wrong candidate' case."""
    requirements = [
        _requirement(skill="Python", normalized="python", target=7.0, importance=8.0, required=True),
        _requirement(skill="Probability", normalized="probability", target=6.0, importance=7.0, required=True),
    ]
    candidate_skills = [
        _candidate_skill(name="photoshop", display="Photoshop", level=9.0, confidence=0.9),
        _candidate_skill(name="figma", display="Figma", level=8.0, confidence=0.8),
    ]

    result = calculate_technical_fit(candidate_skills, requirements)

    assert result.score == 0.0
    assert all(not gap.matched for gap in result.gaps)
    overall = calculate_fit_score(CandidateProfile(skills=candidate_skills), _role(), requirements)
    assert 0.0 <= overall.overall_score <= 100.0


def test_technical_fit_optional_requirements_only_all_unmet() -> None:
    """Every requirement is optional and none are met - should score low, but the required-multiplier never kicks in since nothing here is required."""
    requirements = [
        _requirement(skill="Rust", normalized="rust", target=6.0, importance=7.0, required=False),
        _requirement(skill="Go", normalized="go", target=5.0, importance=5.0, required=False),
    ]

    result = calculate_technical_fit([], requirements)

    assert result.score == 0.0  # 0% of the (unweighted-by-multiplier) target met either way


def test_technical_fit_optional_requirements_only_all_met() -> None:
    requirements = [
        _requirement(skill="Rust", normalized="rust", target=6.0, importance=7.0, required=False),
        _requirement(skill="Go", normalized="go", target=5.0, importance=5.0, required=False),
    ]
    candidate_skills = [
        _candidate_skill(name="rust", level=7.0, confidence=0.7),
        _candidate_skill(name="go", level=6.0, confidence=0.6),
    ]

    result = calculate_technical_fit(candidate_skills, requirements)

    assert result.score == 100.0


def test_technical_fit_very_low_confidence_skills_still_score_on_level_alone() -> None:
    """Confidence is deliberately NOT folded into the score (see calculate_technical_fit's docstring) - only aggregate_confidence should reflect it."""
    requirements = [_requirement(skill="Python", normalized="python", target=7.0, importance=8.0, required=True)]
    high_confidence = calculate_technical_fit([_candidate_skill(name="python", level=7.0, confidence=0.95)], requirements)
    low_confidence = calculate_technical_fit([_candidate_skill(name="python", level=7.0, confidence=0.05)], requirements)

    assert high_confidence.score == low_confidence.score == 100.0
    assert high_confidence.aggregate_confidence > low_confidence.aggregate_confidence


def test_technical_fit_zero_requirements_defaults_to_full_score_with_note() -> None:
    result = calculate_technical_fit([_candidate_skill()], [])

    assert result.score == 100.0
    assert result.gaps == []
    assert result.note is not None


def test_technical_fit_confidence_does_not_change_score_but_changes_aggregate_confidence() -> None:
    requirements = [_requirement(skill="Python", normalized="python", target=6.0, importance=8.0, required=True)]
    low_confidence_result = calculate_technical_fit(
        [_candidate_skill(name="python", level=6.0, confidence=0.1)], requirements
    )
    high_confidence_result = calculate_technical_fit(
        [_candidate_skill(name="python", level=6.0, confidence=0.9)], requirements
    )

    assert low_confidence_result.score == high_confidence_result.score == 100.0
    assert low_confidence_result.aggregate_confidence < high_confidence_result.aggregate_confidence
    assert low_confidence_result.confidence_label != high_confidence_result.confidence_label


def test_technical_fit_missing_skill_yields_zero_confidence_not_score_destruction_alone() -> None:
    """
    A missing skill legitimately drags the score down (no evidence means
    no credit) - but a *present, low-confidence* estimate must not be
    punished the same way. This test pins that distinction.
    """
    requirements = [_requirement(skill="Python", normalized="python", target=6.0, importance=8.0, required=True)]

    missing = calculate_technical_fit([], requirements)
    present_low_confidence = calculate_technical_fit(
        [_candidate_skill(name="python", level=6.0, confidence=0.05)], requirements
    )

    assert missing.score == 0.0
    assert present_low_confidence.score == 100.0  # level fully met, regardless of confidence


# ---------------------------------------------------------------------
# Scores never leave [0, 100]
# ---------------------------------------------------------------------


def test_clamp_score_caps_high_and_low_values() -> None:
    assert _clamp_score(150.0) == 100.0
    assert _clamp_score(-25.0) == 0.0
    assert _clamp_score(42.0) == 42.0


def test_all_component_scores_stay_within_bounds_under_extreme_inputs() -> None:
    requirements = [
        _requirement(skill=f"Skill{i}", normalized=f"skill{i}", target=10.0, importance=10.0, required=True)
        for i in range(20)
    ]
    profile = CandidateProfile(skills=[])  # matches nothing - maximal gap scenario
    role = _role()

    result = calculate_fit_score(profile, role, requirements)

    assert 0.0 <= result.overall_score <= 100.0
    for value in result.components.model_dump().values():
        assert 0.0 <= value <= 100.0


def test_all_component_scores_stay_within_bounds_under_ideal_inputs() -> None:
    requirements = [
        _requirement(skill="Python", normalized="python", target=1.0, importance=10.0, required=True)
    ]
    profile = CandidateProfile(
        coursework=["Python"],
        skills=[_candidate_skill(name="python", level=10.0, confidence=1.0)],
        internships=[ExperienceEntry(organization="Acme", role="Intern", description="x") for _ in range(5)],
        projects=[ProjectEntry(name="p", description="d") for _ in range(5)],
        research=[ResearchEntry(title="t", description="d")],
        domain_experience=["software engineer"],
    )
    role = _role()

    result = calculate_fit_score(
        profile, role, requirements, declared_interests=["software engineer"]
    )

    assert 0.0 <= result.overall_score <= 100.0
    for value in result.components.model_dump().values():
        assert 0.0 <= value <= 100.0


# ---------------------------------------------------------------------
# Experience fit
# ---------------------------------------------------------------------


def test_experience_fit_no_evidence_scores_zero() -> None:
    result = calculate_experience_fit(CandidateProfile())

    assert result.score == 0.0


def test_experience_fit_caps_internship_points_at_two() -> None:
    profile = CandidateProfile(
        internships=[ExperienceEntry(organization="A", role="R", description="d") for _ in range(5)]
    )

    result = calculate_experience_fit(profile)

    assert result.score == 60.0  # 2 * 30, capped


def test_experience_fit_combines_all_signals_up_to_100() -> None:
    profile = CandidateProfile(
        internships=[ExperienceEntry(organization="A", role="R", description="d") for _ in range(2)],
        projects=[ProjectEntry(name="p", description="d") for _ in range(3)],
        research=[ResearchEntry(title="t", description="d")],
    )

    result = calculate_experience_fit(profile)

    assert result.score == 100.0


# ---------------------------------------------------------------------
# Coursework fit
# ---------------------------------------------------------------------


def test_coursework_fit_zero_requirements_defaults_to_full_score() -> None:
    result = calculate_coursework_fit(CandidateProfile(), [])

    assert result.score == 100.0


def test_coursework_fit_rewards_covered_requirements() -> None:
    requirements = [_requirement(skill="Probability", normalized="probability", importance=8.0)]
    profile = CandidateProfile(coursework=["Probability"])

    result = calculate_coursework_fit(profile, requirements)

    assert result.score == 100.0


def test_coursework_fit_penalizes_uncovered_requirements() -> None:
    requirements = [_requirement(skill="Probability", normalized="probability", importance=8.0)]
    profile = CandidateProfile(coursework=["Art History"])

    result = calculate_coursework_fit(profile, requirements)

    assert result.score == 0.0


# ---------------------------------------------------------------------
# Domain fit
# ---------------------------------------------------------------------


def test_domain_fit_defaults_to_neutral_without_domain_experience() -> None:
    result = calculate_domain_fit(CandidateProfile(), _role())

    assert result.score == 50.0


def test_domain_fit_floors_at_30_rather_than_zero_on_mismatch() -> None:
    profile = CandidateProfile(domain_experience=["healthcare"])
    role = _role(title="Quant Research Intern", role_family="quant")

    result = calculate_domain_fit(profile, role)

    assert result.score == 30.0


def test_domain_fit_rewards_word_overlap() -> None:
    profile = CandidateProfile(domain_experience=["quant trading"])
    role = _role(title="Quant Research Intern", role_family="quant")

    result = calculate_domain_fit(profile, role)

    assert result.score == 100.0


# ---------------------------------------------------------------------
# Interest fit
# ---------------------------------------------------------------------


def test_interest_fit_defaults_to_neutral_without_declared_interests() -> None:
    result = calculate_interest_fit(_role())

    assert result.score == 50.0


def test_interest_fit_rewards_overlap_with_role() -> None:
    role = _role(title="Machine Learning Engineer")

    result = calculate_interest_fit(role, declared_interests=["machine learning"])

    assert result.score == 100.0


# ---------------------------------------------------------------------
# Constraint fit
# ---------------------------------------------------------------------


def test_constraint_fit_defaults_to_full_score_without_constraints() -> None:
    result = calculate_constraint_fit(_role())

    assert result.score == 100.0


def test_constraint_fit_penalizes_remote_only_mismatch() -> None:
    role = _role(location="New York, NY")

    result = calculate_constraint_fit(role, candidate_constraints={"remote_only": True})

    assert result.score == 20.0
    assert result.details


def test_constraint_fit_passes_when_role_matches_preferred_location() -> None:
    role = _role(location="Remote")

    result = calculate_constraint_fit(role, candidate_constraints={"remote_only": True})

    assert result.score == 100.0


# ---------------------------------------------------------------------
# Full calculate_fit_score integration
# ---------------------------------------------------------------------


def test_calculate_fit_score_overall_is_weighted_sum_of_components() -> None:
    requirements = [_requirement(skill="Python", normalized="python", target=5.0, importance=8.0, required=True)]
    profile = CandidateProfile(skills=[_candidate_skill(name="python", level=5.0, confidence=0.7)])
    role = _role()

    result = calculate_fit_score(profile, role, requirements)

    expected = (
        result.components.technical * DEFAULT_WEIGHTS["technical"]
        + result.components.experience * DEFAULT_WEIGHTS["experience"]
        + result.components.coursework * DEFAULT_WEIGHTS["coursework"]
        + result.components.domain * DEFAULT_WEIGHTS["domain"]
        + result.components.interest * DEFAULT_WEIGHTS["interest"]
        + result.components.constraints * DEFAULT_WEIGHTS["constraints"]
    )
    assert abs(result.overall_score - expected) < 1e-9


def test_calculate_fit_score_accepts_explicit_weight_override() -> None:
    requirements = [_requirement()]
    profile = CandidateProfile(skills=[_candidate_skill(level=8.0)])
    role = _role()
    override = {
        "technical": 1.0,
        "experience": 0.0,
        "coursework": 0.0,
        "domain": 0.0,
        "interest": 0.0,
        "constraints": 0.0,
    }

    result = calculate_fit_score(profile, role, requirements, weights=override)

    assert result.overall_score == result.components.technical
    assert result.weights_used == override
