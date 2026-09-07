"""
Deadline-aware prep scheduling.

Turns the ranked PrepItems from backend.planning.priority into a
StudyPlan: how many minutes are available before the interview, how
those minutes are split across skills/topics (proportional to
priority_score), and a day-by-day block schedule.

The LLM MUST NOT allocate hours - every number in this module is
deterministic arithmetic over already-computed priorities. StudyTask
descriptions are intentionally generic placeholders ("Probability
practice block") for the same reason backend.llm.rationale never
touches a score: turning a scheduled block into a specific exercise is
an explanation/content-generation task for the LLM to do LATER, against
a time allocation that has already been finalized here.
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field

from backend.matching.gaps import SkillGapResult
from backend.planning.priority import PrepItem, calculate_prep_priorities
from backend.planning.readiness import ReadinessResult


def calculate_days_remaining(interview_date: date | None, current_date: date) -> int:
    """
    Days between `current_date` and `interview_date`. Negative if the
    interview has already passed, 0 if it's today. No interview_date
    ("not scheduled yet") is treated as 0 - there's no deadline to
    compute a countdown against.
    """
    if interview_date is None:
        return 0
    return (interview_date - current_date).days


def calculate_total_available_minutes(days_remaining: int, hours_available_per_day: float) -> float:
    """
    Total prep minutes between now and the interview.
    `days_remaining` is clamped to >= 0 (a same-day or already-passed
    interview has no scheduling days left) and `hours_available_per_day`
    is clamped to >= 0 (defends against a negative input), so this never
    returns a negative total.
    """
    scheduling_days = max(days_remaining, 0)
    hours = max(hours_available_per_day, 0.0)
    return scheduling_days * hours * 60.0


class AllocatedPrepItem(PrepItem):
    """A PrepItem plus how many of the total available minutes it was allocated."""

    allocated_minutes: float = Field(ge=0)


def allocate_minutes(items: list[PrepItem], total_available_minutes: float) -> list[AllocatedPrepItem]:
    """
    Split `total_available_minutes` across `items` proportional to each
    item's priority_score. Every item except the last gets exactly
    `total * (its share of total priority)`; the LAST item (already the
    lowest-priority, since `items` is expected pre-sorted descending)
    absorbs whatever remains - this guarantees
    sum(allocated_minutes) == total_available_minutes exactly (never
    more, and not less than intended due to floating-point rounding
    drift across many items).

    Returns [] if there's nothing to allocate (`items` is empty, all
    priorities are zero, or `total_available_minutes` <= 0) - never
    raises or divides by zero.
    """
    if not items or total_available_minutes <= 0:
        return []

    total_priority = sum(item.priority_score for item in items)
    if total_priority <= 0:
        return []

    allocations: list[AllocatedPrepItem] = []
    running_total = 0.0
    last_index = len(items) - 1

    for index, item in enumerate(items):
        if index == last_index:
            minutes = total_available_minutes - running_total
        else:
            minutes = total_available_minutes * (item.priority_score / total_priority)
        minutes = max(0.0, round(minutes, 2))
        running_total += minutes
        allocations.append(AllocatedPrepItem(**item.model_dump(), allocated_minutes=minutes))

    return allocations


class StudyTask(BaseModel):
    """One scheduled block of prep time on one day. `description` is a generic placeholder for now (see module docstring)."""

    normalized_skill: str
    display_name: str
    day_index: int = Field(ge=0, description="0-based offset from current_date.")
    scheduled_date: date | None = None
    allocated_minutes: float = Field(ge=0)
    priority_score: float
    description: str
    is_complete: bool = False


def generate_daily_schedule(
    allocations: list[AllocatedPrepItem],
    scheduling_days: int,
    hours_available_per_day: float,
    current_date: date | None = None,
) -> list[StudyTask]:
    """
    Greedily fill each day's budget (hours_available_per_day, constant
    across days) with blocks from the highest-priority items that still
    have unallocated minutes, splitting an item across multiple days'
    blocks if it doesn't fit in one day. A day with nothing left to
    schedule simply has no tasks - idle time is not padded with filler.

    Guarantees every day's total allocated_minutes never exceeds that
    day's budget, and the schedule's total never exceeds
    sum(allocations' allocated_minutes).
    """
    if scheduling_days <= 0 or hours_available_per_day <= 0 or not allocations:
        return []

    daily_budget_minutes = hours_available_per_day * 60.0
    remaining_by_skill = {item.normalized_skill: item.allocated_minutes for item in allocations}
    item_by_skill = {item.normalized_skill: item for item in allocations}
    ordered_skills = [item.normalized_skill for item in allocations]  # already priority-sorted

    tasks: list[StudyTask] = []
    for day_index in range(scheduling_days):
        remaining_today = daily_budget_minutes
        for skill in ordered_skills:
            if remaining_today <= 0:
                break
            remaining_for_skill = remaining_by_skill[skill]
            if remaining_for_skill <= 0:
                continue

            block_minutes = round(min(remaining_today, remaining_for_skill), 2)
            if block_minutes <= 0:
                continue

            item = item_by_skill[skill]
            tasks.append(
                StudyTask(
                    normalized_skill=skill,
                    display_name=item.display_name,
                    day_index=day_index,
                    scheduled_date=(current_date + timedelta(days=day_index)) if current_date else None,
                    allocated_minutes=block_minutes,
                    priority_score=item.priority_score,
                    description=f"{item.display_name} practice block",
                )
            )
            remaining_by_skill[skill] -= block_minutes
            remaining_today -= block_minutes

    return tasks


class StudyPlan(BaseModel):
    """Complete deterministic output: the countdown, the priority-ranked items, their allocations, and the day-by-day schedule."""

    interview_date: date | None
    current_date: date
    days_remaining: int = Field(description="Can be negative if the interview has already passed.")
    scheduling_days: int = Field(ge=0, description="max(days_remaining, 0) - days actually available for prep.")
    hours_available_per_day: float = Field(ge=0)
    total_available_minutes: float = Field(ge=0)
    prep_items: list[PrepItem]
    allocations: list[AllocatedPrepItem]
    tasks: list[StudyTask]
    notes: list[str] = Field(default_factory=list, description="Human-readable explanations of edge cases hit while building this plan.")


def generate_study_plan(
    skill_gaps: list[SkillGapResult],
    interview_date: date | None,
    current_date: date,
    hours_available_per_day: float,
    readiness: ReadinessResult | None = None,
    favorite_priority_multiplier: float = 1.0,
) -> StudyPlan:
    """
    The single entry point tying this module together: compute the
    countdown, rank prep priorities (backend.planning.priority), split
    available minutes across them, and lay out a day-by-day schedule.
    Handles every edge case in this module's docstring (interview today,
    already passed, 0 hours/day, fractional hours, one huge gap, no
    gaps, a very short timeline) by degrading to an empty-but-valid
    StudyPlan rather than raising - `notes` explains what happened.
    """
    notes: list[str] = []

    days_remaining = calculate_days_remaining(interview_date, current_date)
    scheduling_days = max(days_remaining, 0)

    if interview_date is None:
        notes.append("No interview date set; prep window defaults to 0 days until one is provided.")
    elif days_remaining < 0:
        notes.append(f"Interview date {interview_date.isoformat()} has already passed; no further prep time is scheduled.")
    elif days_remaining == 0:
        notes.append("Interview is today; no further full prep days remain.")

    if hours_available_per_day <= 0:
        notes.append("0 hours available per day; no prep time can be scheduled.")

    clamped_hours_per_day = max(hours_available_per_day, 0.0)
    total_available_minutes = calculate_total_available_minutes(days_remaining, clamped_hours_per_day)

    prep_items = calculate_prep_priorities(skill_gaps, readiness, favorite_priority_multiplier)
    if not prep_items:
        notes.append("No skill or readiness gaps to prioritize; candidate already meets every tracked requirement/topic.")

    allocations = allocate_minutes(prep_items, total_available_minutes)
    tasks = generate_daily_schedule(allocations, scheduling_days, clamped_hours_per_day, current_date)

    return StudyPlan(
        interview_date=interview_date,
        current_date=current_date,
        days_remaining=days_remaining,
        scheduling_days=scheduling_days,
        hours_available_per_day=clamped_hours_per_day,
        total_available_minutes=round(total_available_minutes, 2),
        prep_items=prep_items,
        allocations=allocations,
        tasks=tasks,
        notes=notes,
    )
