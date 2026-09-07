"""
Tests for backend.planning.adaptive (versioned, diagnostic-driven
StudyPlan revision). No LLM calls, no database - pure Python over
hand-built Pydantic objects.
"""

from __future__ import annotations

from datetime import date

import pytest

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import calculate_skill_gaps
from backend.planning.adaptive import AdaptiveStudyPlan, revise_study_plan
from backend.planning.readiness import DiagnosticResult, calculate_readiness
from backend.planning.scheduler import generate_study_plan

CURRENT = date(2026, 9, 7)


def _skill(name, level, confidence=0.5):
    return CandidateSkillEstimate(normalized_skill_name=name, display_name=name.title(), estimated_level=level, confidence=confidence)


def _requirement(skill, normalized, target=8.0, importance=8.0):
    return RoleRequirement(skill=skill, normalized_skill=normalized, target_level=target, importance=importance, required=True, evidence=["x"])


def _two_topic_scenario(prob_level=5.0, algo_level=5.0):
    profile = CandidateProfile(skills=[_skill("probability", prob_level), _skill("algorithms", algo_level)])
    requirements = [
        _requirement("Probability", "probability", target=8.0, importance=8.0),
        _requirement("Algorithms", "algorithms", target=8.0, importance=8.0),
    ]
    readiness = calculate_readiness(profile, role_family="quant")
    gaps = calculate_skill_gaps(profile.skills, requirements)
    plan = generate_study_plan(
        gaps, interview_date=date(2026, 9, 20), current_date=CURRENT, hours_available_per_day=2.0, readiness=readiness
    )
    return profile, requirements, plan


# ---------------------------------------------------------------------
# Versioning / audit trail
# ---------------------------------------------------------------------


def test_revision_increments_version_from_implicit_v1() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=CURRENT,
    )

    assert plan_v2.version == 2
    assert plan_v2.previous_version == 1


def test_chained_revision_increments_from_prior_adaptive_plan() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()
    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=CURRENT,
    )

    plan_v3 = revise_study_plan(
        plan_v2, DiagnosticResult(topic="algorithms", observed_level=2.0, confidence=0.9),
        profile, "quant", requirements, current_date=CURRENT, previous_diagnostics=plan_v2.diagnostics_applied,
    )

    assert plan_v3.version == 3
    assert plan_v3.previous_version == 2


def test_revision_reason_names_the_topic_and_result() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=CURRENT,
    )

    assert "probability" in plan_v2.revision_reason
    assert "9" in plan_v2.revision_reason


def test_diagnostics_applied_accumulates_across_revisions() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()
    diag1 = DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9)
    plan_v2 = revise_study_plan(plan_v1, diag1, profile, "quant", requirements, current_date=CURRENT)

    diag2 = DiagnosticResult(topic="algorithms", observed_level=2.0, confidence=0.9)
    plan_v3 = revise_study_plan(
        plan_v2, diag2, profile, "quant", requirements, current_date=CURRENT, previous_diagnostics=plan_v2.diagnostics_applied
    )

    topics = {d.topic for d in plan_v3.diagnostics_applied}
    assert topics == {"probability", "algorithms"}


# ---------------------------------------------------------------------
# Completed tasks preserved unchanged
# ---------------------------------------------------------------------


def test_completed_tasks_are_preserved_unchanged() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()
    for task in plan_v1.tasks:
        if task.day_index == 0:
            task.is_complete = True
    completed_before = [t for t in plan_v1.tasks if t.is_complete]
    assert completed_before  # sanity: the scenario actually produced day-0 tasks

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=date(2026, 9, 8),
    )

    assert len(plan_v2.completed_tasks) == len(completed_before)
    for before, after in zip(completed_before, plan_v2.completed_tasks):
        assert after == before  # byte-for-byte identical, not regenerated


def test_completed_tasks_never_appear_in_new_tasks() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()
    for task in plan_v1.tasks:
        if task.day_index == 0:
            task.is_complete = True

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=date(2026, 9, 8),
    )

    # day_index is relative to each list's own reference date (completed
    # tasks to plan_v1's current_date, new tasks to plan_v2's), so it is
    # not a meaningful key for disjointness here - scheduled_date is an
    # absolute calendar date and is what actually distinguishes them.
    completed_dates = {t.scheduled_date for t in plan_v2.completed_tasks}
    new_dates = {t.scheduled_date for t in plan_v2.new_tasks}
    assert completed_dates.isdisjoint(new_dates)


def test_incomplete_tasks_from_prior_plan_are_dropped_not_carried_as_completed() -> None:
    """Only is_complete=True tasks are preserved - everything else is superseded by the fresh remaining schedule."""
    profile, requirements, plan_v1 = _two_topic_scenario()
    assert all(not t.is_complete for t in plan_v1.tasks)  # nothing marked complete yet

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=CURRENT,
    )

    assert plan_v2.completed_tasks == []


def test_second_revision_preserves_tasks_completed_across_both_versions() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()
    for task in plan_v1.tasks:
        if task.day_index == 0:
            task.is_complete = True

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=date(2026, 9, 8),
    )
    for task in plan_v2.new_tasks:
        if task.day_index == 0:
            task.is_complete = True

    plan_v3 = revise_study_plan(
        plan_v2, DiagnosticResult(topic="algorithms", observed_level=2.0, confidence=0.9),
        profile, "quant", requirements, current_date=date(2026, 9, 9), previous_diagnostics=plan_v2.diagnostics_applied,
    )

    # completed tasks from BOTH v1's day-0 and v2's day-0 should survive into v3
    assert len(plan_v3.completed_tasks) >= 2


# ---------------------------------------------------------------------
# Remaining minutes reallocated / valid totals
# ---------------------------------------------------------------------


def test_remaining_allocation_total_never_exceeds_remaining_available_minutes() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=CURRENT,
    )

    total_allocated = sum(a.allocated_minutes for a in plan_v2.remaining_allocations)
    assert total_allocated <= plan_v2.remaining_available_minutes + 1e-6


def test_new_tasks_total_never_exceeds_remaining_available_minutes() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=CURRENT,
    )

    total_scheduled = sum(t.allocated_minutes for t in plan_v2.new_tasks)
    assert total_scheduled <= plan_v2.remaining_available_minutes + 1e-6


def test_remaining_available_minutes_shrinks_as_days_pass() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=date(2026, 9, 10),  # 3 days later
    )

    assert plan_v2.remaining_available_minutes < plan_v1.total_available_minutes


def test_zero_days_remaining_produces_empty_new_tasks() -> None:
    profile, requirements, _ = _two_topic_scenario()
    gaps = calculate_skill_gaps(profile.skills, requirements)
    plan_v1 = generate_study_plan(gaps, interview_date=date(2026, 9, 8), current_date=CURRENT, hours_available_per_day=2.0)

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=date(2026, 9, 8),  # interview is today
    )

    assert plan_v2.remaining_available_minutes == 0.0
    assert plan_v2.new_tasks == []
    assert any("no prep days remain" in note.lower() for note in plan_v2.notes)


# ---------------------------------------------------------------------
# Improved skill receives less time when appropriate (the example scenario)
# ---------------------------------------------------------------------


def test_improved_topic_receives_smaller_share_after_diagnostic() -> None:
    """Probability readiness improves significantly -> its remaining allocation share should fall."""
    profile, requirements, plan_v1 = _two_topic_scenario(prob_level=3.0, algo_level=3.0)
    v1_total = plan_v1.total_available_minutes
    v1_by_skill = {a.normalized_skill: a for a in plan_v1.allocations}
    v1_probability_share = v1_by_skill["probability"].allocated_minutes / v1_total

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.5, confidence=0.95),
        profile, "quant", requirements, current_date=CURRENT,  # same day - isolates the diagnostic's effect
    )
    v2_total = plan_v2.remaining_available_minutes
    v2_by_skill = {a.normalized_skill: a for a in plan_v2.remaining_allocations}
    v2_probability_share = v2_by_skill.get("probability")
    v2_probability_share = (v2_probability_share.allocated_minutes / v2_total) if v2_probability_share else 0.0

    assert v2_probability_share < v1_probability_share


def test_poorly_performing_topic_receives_larger_share_after_diagnostic() -> None:
    """Algorithms performs poorly -> its remaining allocation share should rise."""
    profile, requirements, plan_v1 = _two_topic_scenario(prob_level=5.0, algo_level=5.0)
    v1_total = plan_v1.total_available_minutes
    v1_by_skill = {a.normalized_skill: a for a in plan_v1.allocations}
    v1_algorithms_share = v1_by_skill["algorithms"].allocated_minutes / v1_total

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="algorithms", observed_level=1.0, confidence=0.95),
        profile, "quant", requirements, current_date=CURRENT,  # same day - isolates the diagnostic's effect
    )
    v2_total = plan_v2.remaining_available_minutes
    v2_by_skill = {a.normalized_skill: a for a in plan_v2.remaining_allocations}
    v2_algorithms_share = v2_by_skill["algorithms"].allocated_minutes / v2_total

    assert v2_algorithms_share > v1_algorithms_share


def test_a_fully_mastered_topic_can_drop_out_of_remaining_allocation_entirely() -> None:
    # No resume-claimed "probability" skill at all (resume confidence=0
    # for it), so a confident diagnostic can fully determine the
    # combined score - see readiness._combine_topic_evidence: a
    # nonzero-confidence resume estimate always retains some pull in
    # the weighted average, so a partial resume claim could never
    # reach an exact 10 even after a perfect diagnostic.
    profile = CandidateProfile(skills=[_skill("algorithms", 5.0)])
    requirements = [
        _requirement("Probability", "probability", target=8.0, importance=8.0),
        _requirement("Algorithms", "algorithms", target=8.0, importance=8.0),
    ]
    readiness = calculate_readiness(profile, role_family="quant")
    gaps = calculate_skill_gaps(profile.skills, requirements)
    plan_v1 = generate_study_plan(
        gaps, interview_date=date(2026, 9, 20), current_date=CURRENT, hours_available_per_day=2.0, readiness=readiness
    )

    plan_v2 = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=10.0, confidence=1.0),
        profile, "quant", requirements, current_date=CURRENT,
    )

    remaining_skills = {a.normalized_skill for a in plan_v2.remaining_allocations}
    assert "probability" not in remaining_skills
    assert "algorithms" in remaining_skills


# ---------------------------------------------------------------------
# Type/shape sanity
# ---------------------------------------------------------------------


def test_revise_study_plan_returns_adaptive_study_plan() -> None:
    profile, requirements, plan_v1 = _two_topic_scenario()

    result = revise_study_plan(
        plan_v1, DiagnosticResult(topic="probability", observed_level=9.0, confidence=0.9),
        profile, "quant", requirements, current_date=CURRENT,
    )

    assert isinstance(result, AdaptiveStudyPlan)
    assert result.readiness is not None
