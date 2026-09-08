"""
Tests for backend.planning.scheduler (deadline-aware deterministic
allocation and daily scheduling). No LLM calls anywhere in this module
or these tests.
"""

from __future__ import annotations

from datetime import date

import pytest

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import calculate_skill_gaps
from backend.planning.priority import PrepItem
from backend.planning.scheduler import (
    allocate_minutes,
    calculate_days_remaining,
    calculate_total_available_minutes,
    generate_daily_schedule,
    generate_study_plan,
)

CURRENT = date(2026, 9, 7)


def _requirement(skill="Python", normalized="python", target=8.0, importance=8.0):
    return RoleRequirement(
        skill=skill, normalized_skill=normalized, target_level=target, importance=importance,
        required=True, evidence=["x"],
    )


def _skill(name="python", level=6.0, confidence=0.6):
    return CandidateSkillEstimate(normalized_skill_name=name, display_name=name, estimated_level=level, confidence=confidence)


def _prep_item(name="python", priority=10.0):
    return PrepItem(
        normalized_skill=name, display_name=name.title(), gap=5.0, requirement_importance=8.0,
        interview_topic_weight=0.5, confidence=0.5, confidence_adjustment=0.75, priority_score=priority,
        source="role_requirement",
    )


# ---------------------------------------------------------------------
# calculate_days_remaining / calculate_total_available_minutes
# ---------------------------------------------------------------------


def test_days_remaining_positive_for_future_interview() -> None:
    assert calculate_days_remaining(date(2026, 9, 14), CURRENT) == 7


def test_days_remaining_zero_for_interview_today() -> None:
    assert calculate_days_remaining(CURRENT, CURRENT) == 0


def test_days_remaining_negative_for_passed_interview() -> None:
    assert calculate_days_remaining(date(2026, 9, 1), CURRENT) == -6


def test_days_remaining_zero_when_no_interview_date() -> None:
    assert calculate_days_remaining(None, CURRENT) == 0


def test_total_available_minutes_basic_multiplication() -> None:
    assert calculate_total_available_minutes(5, 2.0) == 600.0


def test_total_available_minutes_clamps_negative_days_to_zero() -> None:
    assert calculate_total_available_minutes(-3, 2.0) == 0.0


def test_total_available_minutes_clamps_negative_hours_to_zero() -> None:
    assert calculate_total_available_minutes(5, -1.0) == 0.0


def test_total_available_minutes_supports_fractional_hours() -> None:
    assert calculate_total_available_minutes(2, 1.5) == 180.0


# ---------------------------------------------------------------------
# allocate_minutes
# ---------------------------------------------------------------------


def test_allocate_minutes_empty_items_returns_empty() -> None:
    assert allocate_minutes([], 100.0) == []


def test_allocate_minutes_zero_total_returns_empty() -> None:
    assert allocate_minutes([_prep_item()], 0.0) == []


def test_allocate_minutes_single_item_gets_everything() -> None:
    [allocation] = allocate_minutes([_prep_item(priority=10.0)], 120.0)
    assert allocation.allocated_minutes == 120.0


def test_allocate_minutes_splits_proportionally_to_priority() -> None:
    items = [_prep_item("a", priority=30.0), _prep_item("b", priority=10.0)]

    allocations = allocate_minutes(items, 400.0)

    by_skill = {a.normalized_skill: a for a in allocations}
    assert by_skill["a"].allocated_minutes == pytest.approx(300.0)
    assert by_skill["b"].allocated_minutes == pytest.approx(100.0)


def test_allocate_minutes_sum_never_exceeds_total() -> None:
    items = [_prep_item("a", priority=7.0), _prep_item("b", priority=3.0), _prep_item("c", priority=1.0)]

    allocations = allocate_minutes(items, 137.0)

    assert sum(a.allocated_minutes for a in allocations) <= 137.0


def test_allocate_minutes_sum_equals_total_when_priorities_positive() -> None:
    items = [_prep_item("a", priority=7.0), _prep_item("b", priority=3.0), _prep_item("c", priority=1.0)]

    allocations = allocate_minutes(items, 137.0)

    assert sum(a.allocated_minutes for a in allocations) == pytest.approx(137.0, abs=0.01)


# ---------------------------------------------------------------------
# generate_daily_schedule
# ---------------------------------------------------------------------


def test_daily_schedule_empty_when_no_scheduling_days() -> None:
    allocations = allocate_minutes([_prep_item(priority=10.0)], 100.0)
    assert generate_daily_schedule(allocations, scheduling_days=0, hours_available_per_day=2.0) == []


def test_daily_schedule_never_exceeds_daily_budget() -> None:
    items = [_prep_item("a", priority=10.0), _prep_item("b", priority=10.0), _prep_item("c", priority=10.0)]
    allocations = allocate_minutes(items, 600.0)  # 10 hours total

    tasks = generate_daily_schedule(allocations, scheduling_days=5, hours_available_per_day=2.0, current_date=CURRENT)

    by_day: dict[int, float] = {}
    for task in tasks:
        by_day[task.day_index] = by_day.get(task.day_index, 0.0) + task.allocated_minutes
    for total in by_day.values():
        assert total <= 120.0 + 1e-6  # 2 hours/day


def test_daily_schedule_total_matches_allocations_total() -> None:
    items = [_prep_item("a", priority=10.0), _prep_item("b", priority=5.0)]
    allocations = allocate_minutes(items, 300.0)

    tasks = generate_daily_schedule(allocations, scheduling_days=5, hours_available_per_day=1.0, current_date=CURRENT)

    assert sum(t.allocated_minutes for t in tasks) == pytest.approx(sum(a.allocated_minutes for a in allocations), abs=0.01)


def test_daily_schedule_splits_a_large_item_across_multiple_days() -> None:
    allocations = allocate_minutes([_prep_item("a", priority=10.0)], 300.0)  # 5 hours, 1 topic

    tasks = generate_daily_schedule(allocations, scheduling_days=5, hours_available_per_day=1.0, current_date=CURRENT)

    assert len({t.day_index for t in tasks}) > 1


def test_daily_schedule_task_description_is_generic_placeholder() -> None:
    allocations = allocate_minutes([_prep_item("probability", priority=10.0)], 60.0)

    [task] = generate_daily_schedule(allocations, scheduling_days=1, hours_available_per_day=1.0)

    assert task.description == "Probability practice block"


def test_daily_schedule_dates_offset_from_current_date() -> None:
    allocations = allocate_minutes([_prep_item(priority=10.0)], 180.0)

    tasks = generate_daily_schedule(allocations, scheduling_days=3, hours_available_per_day=1.0, current_date=CURRENT)

    dates = {t.day_index: t.scheduled_date for t in tasks}
    assert dates[0] == CURRENT
    if 1 in dates:
        assert dates[1] == date(2026, 9, 8)


# ---------------------------------------------------------------------
# generate_study_plan: the required edge cases
# ---------------------------------------------------------------------


def _gaps_with_one_requirement() -> list:
    return calculate_skill_gaps([], [_requirement(target=8.0, importance=8.0)])


def test_interview_today_produces_zero_minute_plan_with_note() -> None:
    plan = generate_study_plan(_gaps_with_one_requirement(), interview_date=CURRENT, current_date=CURRENT, hours_available_per_day=3.0)

    assert plan.scheduling_days == 0
    assert plan.total_available_minutes == 0.0
    assert plan.tasks == []
    assert any("today" in note.lower() for note in plan.notes)


def test_interview_already_passed_produces_zero_minute_plan_with_note() -> None:
    plan = generate_study_plan(
        _gaps_with_one_requirement(), interview_date=date(2026, 9, 1), current_date=CURRENT, hours_available_per_day=3.0
    )

    assert plan.days_remaining < 0
    assert plan.scheduling_days == 0
    assert plan.total_available_minutes == 0.0
    assert plan.tasks == []
    assert any("passed" in note.lower() for note in plan.notes)


def test_zero_available_hours_produces_zero_minute_plan_with_note() -> None:
    plan = generate_study_plan(
        _gaps_with_one_requirement(), interview_date=date(2026, 9, 20), current_date=CURRENT, hours_available_per_day=0.0
    )

    assert plan.total_available_minutes == 0.0
    assert plan.tasks == []
    assert any("0 hours" in note for note in plan.notes)


def test_fractional_hours_produce_correct_total_minutes() -> None:
    plan = generate_study_plan(
        _gaps_with_one_requirement(), interview_date=date(2026, 9, 9), current_date=CURRENT, hours_available_per_day=1.5
    )

    assert plan.total_available_minutes == pytest.approx(180.0)  # 2 days * 1.5h * 60


def test_one_huge_gap_receives_full_allocation() -> None:
    gaps = calculate_skill_gaps([], [_requirement(target=10.0, importance=10.0)])

    plan = generate_study_plan(gaps, interview_date=date(2026, 9, 14), current_date=CURRENT, hours_available_per_day=2.0)

    assert len(plan.allocations) == 1
    assert plan.allocations[0].allocated_minutes == pytest.approx(plan.total_available_minutes, abs=0.01)


def test_no_gaps_produces_empty_plan_with_note() -> None:
    gaps = calculate_skill_gaps([_skill(level=10.0)], [_requirement(target=5.0)])

    plan = generate_study_plan(gaps, interview_date=date(2026, 9, 14), current_date=CURRENT, hours_available_per_day=2.0)

    assert plan.prep_items == []
    assert plan.allocations == []
    assert plan.tasks == []
    assert plan.total_available_minutes > 0  # time was available, just nothing to spend it on
    assert any("already meets" in note for note in plan.notes)


def test_very_short_timeline_still_produces_a_valid_plan() -> None:
    gaps = calculate_skill_gaps([], [_requirement(target=8.0, importance=8.0)])

    plan = generate_study_plan(
        gaps, interview_date=date(2026, 9, 8), current_date=CURRENT, hours_available_per_day=0.5
    )

    assert plan.scheduling_days == 1
    assert plan.total_available_minutes == pytest.approx(30.0)
    assert sum(t.allocated_minutes for t in plan.tasks) <= plan.total_available_minutes + 1e-6


def test_exactly_15_minutes_available_still_produces_a_valid_plan() -> None:
    """A single tiny prep window (0.25h/day for 1 day = 15 minutes) - the smallest realistic non-zero budget."""
    gaps = calculate_skill_gaps([], [_requirement(target=8.0, importance=8.0)])

    plan = generate_study_plan(
        gaps, interview_date=date(2026, 9, 8), current_date=CURRENT, hours_available_per_day=0.25
    )

    assert plan.scheduling_days == 1
    assert plan.total_available_minutes == pytest.approx(15.0)
    assert sum(t.allocated_minutes for t in plan.tasks) <= plan.total_available_minutes + 1e-6
    assert plan.tasks  # 15 minutes is still enough to schedule something, not silently dropped


def test_no_interview_date_produces_zero_minute_plan_with_note() -> None:
    plan = generate_study_plan(
        _gaps_with_one_requirement(), interview_date=None, current_date=CURRENT, hours_available_per_day=3.0
    )

    assert plan.scheduling_days == 0
    assert plan.total_available_minutes == 0.0
    assert any("no interview date" in note.lower() for note in plan.notes)


# ---------------------------------------------------------------------
# Core invariant across realistic scenarios: never overspend the budget
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "days,hours_per_day",
    [(1, 1.0), (3, 2.0), (7, 1.5), (14, 4.0), (30, 0.25)],
)
def test_total_scheduled_minutes_never_exceeds_available_time(days: int, hours_per_day: float) -> None:
    profile = CandidateProfile(skills=[_skill("python", level=3.0)])
    requirements = [
        _requirement(skill="Python", normalized="python", target=9.0, importance=9.0),
        _requirement(skill="C++", normalized="c++", target=7.0, importance=5.0),
        _requirement(skill="SQL", normalized="sql", target=6.0, importance=4.0),
    ]
    gaps = calculate_skill_gaps(profile.skills, requirements)
    interview_date = date.fromordinal(CURRENT.toordinal() + days)

    plan = generate_study_plan(gaps, interview_date=interview_date, current_date=CURRENT, hours_available_per_day=hours_per_day)

    assert sum(a.allocated_minutes for a in plan.allocations) <= plan.total_available_minutes + 1e-6
    assert sum(t.allocated_minutes for t in plan.tasks) <= plan.total_available_minutes + 1e-6
