"""
Adaptive replanning.

When a candidate completes a diagnostic, revise_study_plan() produces a
NEW, versioned StudyPlan revision rather than mutating the old one:

  1. save (accumulate) the new diagnostic result
  2-3. update the affected topic's estimate and confidence
  4. recalculate readiness
  5. recalculate remaining skill gaps
  6. determine remaining available prep minutes
  7. preserve completed tasks
  8. redistribute ONLY the remaining time
  9. generate a revised remaining StudyPlan

No LLM involvement anywhere in this module - every recalculation is
deterministic Python over already-built modules (backend.planning.readiness,
backend.planning.priority, backend.planning.scheduler).

WHY THERE IS NO SEPARATE "update the skill estimate" MUTATION STEP:
A diagnostic in this system is administered per READINESS TOPIC (e.g.
"take the DSA diagnostic" - see backend.planning.readiness), not per raw
resume skill, and a topic often aggregates multiple skills (e.g. "dsa"
covers both "algorithms" and "data structures" - see
readiness.TOPIC_SKILL_ALIASES). There is no unambiguous way to guess
which single CandidateSkillEstimate a topic-level diagnostic should
overwrite. Instead, steps 2-3 ("update relevant skill/readiness
estimate" and "update confidence") happen entirely by re-running
calculate_readiness() with the new diagnostic included:
readiness.py's own combination rule (diagnostic evidence dominates
resume evidence - see DIAGNOSTIC_EVIDENCE_MULTIPLIER) already produces
an updated per-topic score and confidence. backend.planning.priority's
calculate_prep_priorities() then already prefers that
diagnostic-informed readiness value over the raw resume-derived skill
gap for any topic it maps to - so simply feeding a fresh
ReadinessResult back through the existing pipeline correctly propagates
the diagnostic everywhere it should apply, with no separate mutation
logic needed here.

WHY COMPLETED TASKS ARE NEVER TOUCHED: days that have already elapsed
(whether or not their scheduled tasks were actually completed) are
simply excluded from the new remaining-minutes calculation, the same
way backend.planning.scheduler treats an elapsed/same-day interview as
zero remaining days. Completed tasks are carried forward into the new
plan's `completed_tasks` list UNCHANGED - not regenerated, not
re-prioritized, not re-dated - so a user's prep history is never
rewritten by a later revision.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import calculate_skill_gaps
from backend.planning.priority import PrepItem, calculate_prep_priorities
from backend.planning.readiness import DiagnosticResult, ReadinessResult, calculate_readiness
from backend.planning.scheduler import (
    AllocatedPrepItem,
    StudyPlan,
    StudyTask,
    allocate_minutes,
    calculate_days_remaining,
    calculate_total_available_minutes,
    generate_daily_schedule,
)


class AdaptiveStudyPlan(BaseModel):
    """
    A StudyPlan revision. `version`/`previous_version`/`revision_reason`/
    `revised_at` are the audit trail: enough for the system (or a future
    UI) to explain that - and why - the plan changed, and to trace a
    chain of revisions back to the original plan (version 1).

    `completed_tasks` is carried forward unchanged from whichever plan
    (StudyPlan or an earlier AdaptiveStudyPlan) this revision was built
    from. `new_tasks` is the freshly-scheduled remaining time only - the
    two are kept as separate lists (rather than merged into one, as
    StudyPlan.tasks is) specifically so a caller can render "what you've
    done" and "what's left" as visually distinct without re-deriving
    which is which.
    """

    version: int = Field(ge=1)
    previous_version: int | None = None
    revision_reason: str
    revised_at: datetime

    interview_date: date | None
    current_date: date
    days_remaining: int = Field(description="Can be negative if the interview has already passed.")
    scheduling_days: int = Field(ge=0)
    hours_available_per_day: float = Field(ge=0)

    readiness: ReadinessResult
    diagnostics_applied: list[DiagnosticResult] = Field(
        description="The full accumulated diagnostic history used to compute `readiness`, "
        "for a caller to persist (backend.db.assessment_results, not yet implemented)."
    )

    remaining_prep_items: list[PrepItem]
    remaining_allocations: list[AllocatedPrepItem]
    remaining_available_minutes: float = Field(ge=0)

    completed_tasks: list[StudyTask] = Field(description="Preserved verbatim from the prior plan - never regenerated.")
    new_tasks: list[StudyTask] = Field(description="Freshly scheduled for the remaining time only.")

    notes: list[str] = Field(default_factory=list)


def _all_tasks(plan: StudyPlan | AdaptiveStudyPlan) -> list[StudyTask]:
    if isinstance(plan, AdaptiveStudyPlan):
        return [*plan.completed_tasks, *plan.new_tasks]
    return plan.tasks


def _plan_version(plan: StudyPlan | AdaptiveStudyPlan) -> int:
    """A plain StudyPlan (from backend.planning.scheduler) has no version field - it's implicitly version 1."""
    return plan.version if isinstance(plan, AdaptiveStudyPlan) else 1


def revise_study_plan(
    previous_plan: StudyPlan | AdaptiveStudyPlan,
    new_diagnostic: DiagnosticResult,
    profile: CandidateProfile,
    role_family: str | None,
    requirements: list[RoleRequirement],
    current_date: date,
    previous_diagnostics: list[DiagnosticResult] | None = None,
    favorite_priority_multiplier: float = 1.0,
) -> AdaptiveStudyPlan:
    """
    Produce the next versioned revision of `previous_plan` after
    `new_diagnostic` is completed. `previous_plan` may be the original
    StudyPlan or an earlier AdaptiveStudyPlan (chaining revisions).
    `interview_date` and `hours_available_per_day` are carried over from
    `previous_plan` - this call is about reacting to new evidence, not
    changing the interview logistics.

    See this module's docstring for the full 9-step mapping and why
    there is no separate skill-mutation step.
    """
    # Step 1: accumulate the new diagnostic. A later entry for the same
    # topic supersedes an earlier one inside calculate_readiness() (dict
    # construction by topic key), so this is safe to call repeatedly for
    # the same topic as a candidate retakes a diagnostic.
    diagnostics_applied = [*(previous_diagnostics or []), new_diagnostic]

    interview_date = previous_plan.interview_date
    hours_available_per_day = previous_plan.hours_available_per_day

    # Steps 2-4: recompute readiness with the new diagnostic folded in -
    # see this module's docstring for why this alone covers "update the
    # relevant skill/readiness estimate and confidence."
    readiness = calculate_readiness(profile, role_family, diagnostics_applied)

    # Step 5: recalculate remaining skill gaps (role-requirement side).
    skill_gaps = calculate_skill_gaps(profile.skills, requirements)

    # Step 6: prep time remaining from *today* forward.
    days_remaining = calculate_days_remaining(interview_date, current_date)
    scheduling_days = max(days_remaining, 0)
    remaining_available_minutes = calculate_total_available_minutes(days_remaining, hours_available_per_day)

    # Step 7: preserve completed tasks from every prior version, exactly
    # as they were.
    completed_tasks = [task for task in _all_tasks(previous_plan) if task.is_complete]

    # Step 8: redistribute ONLY remaining_available_minutes across
    # freshly-ranked priorities - completed tasks' minutes are never
    # re-entered into this pool.
    remaining_prep_items = calculate_prep_priorities(skill_gaps, readiness, favorite_priority_multiplier)
    remaining_allocations = allocate_minutes(remaining_prep_items, remaining_available_minutes)

    # Step 9: lay the remaining allocation out as a fresh day-by-day
    # schedule starting today.
    new_tasks = generate_daily_schedule(remaining_allocations, scheduling_days, hours_available_per_day, current_date)

    notes: list[str] = []
    if not remaining_prep_items:
        notes.append("No remaining skill or readiness gaps to prioritize.")
    if scheduling_days <= 0:
        notes.append("No prep days remain before the interview.")

    return AdaptiveStudyPlan(
        version=_plan_version(previous_plan) + 1,
        previous_version=_plan_version(previous_plan),
        revision_reason=(
            f"Diagnostic completed for topic '{new_diagnostic.topic}': "
            f"observed_level={new_diagnostic.observed_level:g}/10, confidence={new_diagnostic.confidence:.2f}."
        ),
        revised_at=datetime.now(timezone.utc),
        interview_date=interview_date,
        current_date=current_date,
        days_remaining=days_remaining,
        scheduling_days=scheduling_days,
        hours_available_per_day=hours_available_per_day,
        readiness=readiness,
        diagnostics_applied=diagnostics_applied,
        remaining_prep_items=remaining_prep_items,
        remaining_allocations=remaining_allocations,
        remaining_available_minutes=round(remaining_available_minutes, 2),
        completed_tasks=completed_tasks,
        new_tasks=new_tasks,
        notes=notes,
    )
