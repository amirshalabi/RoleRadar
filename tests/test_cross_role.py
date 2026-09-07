"""
Tests for backend.matching.cross_role (skill ROI across favorites and
favorite-role comparison). No LLM calls, no database - everything here
is pure Python over hand-built Pydantic objects.
"""

from __future__ import annotations

import pytest

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.cross_role import (
    DEFAULT_LEARNING_COST_HOURS,
    FALLBACK_LEARNING_COST_HOURS,
    FAVORITE_PRIORITY_MULTIPLIERS,
    FavoriteRoleContext,
    calculate_skill_roi,
    compare_favorite_roles,
    estimate_learning_cost_hours,
)


def _role(title="Role", company="Acme", **overrides):
    raw = {"title": title, "company": company}
    raw.update(overrides)
    return normalize_role(raw)


def _requirement(skill="Python", normalized="python", target=8.0, importance=8.0, required=True):
    return RoleRequirement(
        skill=skill, normalized_skill=normalized, target_level=target, importance=importance,
        required=required, evidence=["evidence"],
    )


def _skill(name="python", level=6.0, confidence=0.6):
    return CandidateSkillEstimate(normalized_skill_name=name, display_name=name, estimated_level=level, confidence=confidence)


def _context(role=None, requirements=None, priority="interested"):
    return FavoriteRoleContext(role=role or _role(), requirements=requirements or [], priority=priority)


# ---------------------------------------------------------------------
# estimate_learning_cost_hours
# ---------------------------------------------------------------------


def test_learning_cost_uses_default_table_entry() -> None:
    assert estimate_learning_cost_hours("python") == DEFAULT_LEARNING_COST_HOURS["python"]


def test_learning_cost_falls_back_for_unknown_skill() -> None:
    assert estimate_learning_cost_hours("some very niche skill") == FALLBACK_LEARNING_COST_HOURS


def test_learning_cost_override_takes_priority_over_default_table() -> None:
    assert estimate_learning_cost_hours("python", overrides={"python": 5.0}) == 5.0


def test_learning_cost_override_applies_to_unknown_skill_too() -> None:
    assert estimate_learning_cost_hours("bespoke-tool", overrides={"bespoke-tool": 12.0}) == 12.0


# ---------------------------------------------------------------------
# calculate_skill_roi
# ---------------------------------------------------------------------


def test_calculate_skill_roi_empty_favorites_returns_empty_list() -> None:
    assert calculate_skill_roi(CandidateProfile(), []) == []


def test_calculate_skill_roi_counts_roles_requiring_each_skill() -> None:
    profile = CandidateProfile(skills=[])
    ctx1 = _context(role=_role("A"), requirements=[_requirement(normalized="python")])
    ctx2 = _context(role=_role("B"), requirements=[_requirement(normalized="python")])
    ctx3 = _context(role=_role("C"), requirements=[_requirement(normalized="c++", skill="C++")])

    results = calculate_skill_roi(profile, [ctx1, ctx2, ctx3])

    by_skill = {r.normalized_skill: r for r in results}
    assert by_skill["python"].roles_requiring_it == 2
    assert by_skill["c++"].roles_requiring_it == 1


def test_calculate_skill_roi_average_gap_is_mean_raw_gap_across_roles() -> None:
    profile = CandidateProfile(skills=[_skill(name="python", level=4.0)])
    ctx1 = _context(role=_role("A"), requirements=[_requirement(normalized="python", target=8.0)])  # gap 4
    ctx2 = _context(role=_role("B"), requirements=[_requirement(normalized="python", target=6.0)])  # gap 2

    [result] = calculate_skill_roi(profile, [ctx1, ctx2])

    assert result.average_gap == pytest.approx(3.0)  # mean(4, 2)


def test_calculate_skill_roi_weighted_priority_is_mean_multiplier_not_sum() -> None:
    profile = CandidateProfile(skills=[])
    ctx1 = _context(role=_role("A"), requirements=[_requirement(normalized="python")], priority="dream")
    ctx2 = _context(role=_role("B"), requirements=[_requirement(normalized="python")], priority="backup")

    [result] = calculate_skill_roi(profile, [ctx1, ctx2])

    expected_mean = (FAVORITE_PRIORITY_MULTIPLIERS["dream"] + FAVORITE_PRIORITY_MULTIPLIERS["backup"]) / 2
    assert result.weighted_priority == pytest.approx(expected_mean)


def test_calculate_skill_roi_dream_priority_raises_roi_over_backup() -> None:
    """Same skill, same gap/importance/frequency - only the favorite priority differs."""
    profile = CandidateProfile(skills=[])
    dream_ctx = _context(role=_role("A"), requirements=[_requirement(normalized="python")], priority="dream")
    backup_ctx = _context(role=_role("B"), requirements=[_requirement(normalized="c++", skill="C++")], priority="backup")

    results = calculate_skill_roi(profile, [dream_ctx, backup_ctx])
    by_skill = {r.normalized_skill: r for r in results}

    assert by_skill["python"].raw_roi > by_skill["c++"].raw_roi


def test_calculate_skill_roi_higher_frequency_raises_roi() -> None:
    """Same per-role numbers, but python is needed by 2 roles vs c++'s 1."""
    profile = CandidateProfile(skills=[])
    ctx1 = _context(role=_role("A"), requirements=[_requirement(normalized="python")])
    ctx2 = _context(role=_role("B"), requirements=[_requirement(normalized="python")])
    ctx3 = _context(role=_role("C"), requirements=[_requirement(normalized="c++", skill="C++")])

    results = calculate_skill_roi(profile, [ctx1, ctx2, ctx3])
    by_skill = {r.normalized_skill: r for r in results}

    assert by_skill["python"].raw_roi > by_skill["c++"].raw_roi


def test_calculate_skill_roi_score_is_normalized_0_to_100() -> None:
    profile = CandidateProfile(skills=[])
    contexts = [
        _context(role=_role("A"), requirements=[_requirement(normalized="python", importance=9.0)]),
        _context(role=_role("B"), requirements=[_requirement(normalized="c++", skill="C++", importance=2.0)]),
    ]

    results = calculate_skill_roi(profile, contexts)

    assert max(r.roi_score for r in results) == 100.0  # top skill normalized to exactly 100
    for r in results:
        assert 0.0 <= r.roi_score <= 100.0


def test_calculate_skill_roi_zero_gap_everywhere_yields_zero_scores() -> None:
    profile = CandidateProfile(skills=[_skill(name="python", level=10.0)])
    ctx = _context(role=_role(), requirements=[_requirement(normalized="python", target=5.0)])

    [result] = calculate_skill_roi(profile, [ctx])

    assert result.average_gap == 0.0
    assert result.raw_roi == 0.0
    assert result.roi_score == 0.0  # no division-by-zero crash when max_raw_roi is 0


def test_calculate_skill_roi_results_sorted_descending_by_score() -> None:
    profile = CandidateProfile(skills=[])
    contexts = [
        _context(role=_role("A"), requirements=[_requirement(normalized="python", importance=9.0)]),
        _context(role=_role("B"), requirements=[_requirement(normalized="c++", skill="C++", importance=1.0)]),
    ]

    results = calculate_skill_roi(profile, contexts)

    scores = [r.roi_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_calculate_skill_roi_includes_affected_roles() -> None:
    profile = CandidateProfile(skills=[])
    role_a = _role("Backend Intern", "Acme")
    ctx = _context(role=role_a, requirements=[_requirement(normalized="python")])

    [result] = calculate_skill_roi(profile, [ctx])

    assert len(result.affected_roles) == 1
    assert result.affected_roles[0].title == "Backend Intern"
    assert result.affected_roles[0].company == "Acme"


def test_calculate_skill_roi_respects_learning_cost_overrides() -> None:
    profile = CandidateProfile(skills=[])
    ctx = _context(role=_role(), requirements=[_requirement(normalized="python")])

    default_result = calculate_skill_roi(profile, [ctx])[0]
    overridden_result = calculate_skill_roi(profile, [ctx], learning_cost_overrides={"python": 1.0})[0]

    assert overridden_result.estimated_learning_cost_hours == 1.0
    assert overridden_result.raw_roi > default_result.raw_roi  # cheaper to learn -> higher ROI


def test_calculate_skill_roi_deduplicates_repeated_skill_within_one_role() -> None:
    """Defensive: a role's own requirements shouldn't repeat a skill, but frequency must not inflate if it does."""
    profile = CandidateProfile(skills=[])
    role = _role()
    ctx = FavoriteRoleContext(
        role=role,
        requirements=[_requirement(normalized="python", target=5.0), _requirement(normalized="python", target=9.0)],
    )

    [result] = calculate_skill_roi(profile, [ctx])

    assert result.roles_requiring_it == 1


# ---------------------------------------------------------------------
# compare_favorite_roles
# ---------------------------------------------------------------------


def test_compare_favorite_roles_rejects_fewer_than_two() -> None:
    with pytest.raises(ValueError):
        compare_favorite_roles(CandidateProfile(), [_context()])


def test_compare_favorite_roles_rejects_more_than_four() -> None:
    with pytest.raises(ValueError):
        compare_favorite_roles(CandidateProfile(), [_context() for _ in range(5)])


def test_compare_favorite_roles_accepts_two_to_four() -> None:
    for count in (2, 3, 4):
        contexts = [_context(role=_role(f"Role {i}")) for i in range(count)]
        result = compare_favorite_roles(CandidateProfile(), contexts)
        assert len(result) == count


def test_compare_favorite_roles_includes_fit_score_and_components() -> None:
    profile = CandidateProfile(skills=[_skill(name="python", level=8.0)])
    contexts = [
        _context(role=_role("A"), requirements=[_requirement(normalized="python", target=7.0)]),
        _context(role=_role("B"), requirements=[_requirement(normalized="c++", skill="C++", target=7.0)]),
    ]

    result = compare_favorite_roles(profile, contexts)

    for comparison in result:
        assert 0.0 <= comparison.fit_score <= 100.0
        assert set(comparison.fit_components.keys()) == {
            "technical", "experience", "coursework", "domain", "interest", "constraints"
        }


def test_compare_favorite_roles_top_gaps_sorted_by_weighted_gap() -> None:
    profile = CandidateProfile(skills=[])
    role = _role("A")
    ctx = _context(
        role=role,
        requirements=[
            _requirement(normalized="python", target=8.0, importance=9.0),  # weighted_gap = 8 * 0.9 = 7.2
            _requirement(normalized="c++", skill="C++", target=3.0, importance=2.0),  # weighted_gap = 3 * 0.2 = 0.6
        ],
    )

    comparisons = compare_favorite_roles(profile, [ctx, _context(role=_role("B"))])
    comparison = next(c for c in comparisons if c.role.title == "A")

    assert [g.skill for g in comparison.top_gaps] == ["Python", "C++"]


def test_compare_favorite_roles_readiness_and_prep_hours_default_to_none() -> None:
    contexts = [_context(role=_role("A")), _context(role=_role("B"))]

    result = compare_favorite_roles(CandidateProfile(), contexts)

    for comparison in result:
        assert comparison.readiness_score is None
        assert comparison.prep_hours_allocated is None


def test_compare_favorite_roles_uses_supplied_readiness_and_prep_hours() -> None:
    role_a = _role("A")
    role_b = _role("B")
    contexts = [_context(role=role_a), _context(role=role_b)]

    result = compare_favorite_roles(
        CandidateProfile(),
        contexts,
        readiness_scores={role_a.external_id: 80.0},
        prep_hours_by_role={role_a.external_id: 15.0},
    )

    by_title = {c.role.title: c for c in result}
    assert by_title["A"].readiness_score == 80.0
    assert by_title["A"].prep_hours_allocated == 15.0
    assert by_title["B"].readiness_score is None
    assert by_title["B"].prep_hours_allocated is None


def test_compare_favorite_roles_preserves_favorite_priority() -> None:
    contexts = [
        _context(role=_role("A"), priority="dream"),
        _context(role=_role("B"), priority="backup"),
    ]

    result = compare_favorite_roles(CandidateProfile(), contexts)

    priorities = {c.role.title: c.priority for c in result}
    assert priorities == {"A": "dream", "B": "backup"}
