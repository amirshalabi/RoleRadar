"""
LLM-generated study task activities.

The deterministic scheduler (backend.planning.scheduler) has already
decided, for every StudyTask, its topic, date, and allocated_minutes -
those three values are FIXED inputs here and are never read as
suggestions. The LLM's only job is to propose what happens INSIDE that
already-decided block (e.g. "45 minutes of Probability" ->
"15 min conditional probability review, 20 min expected value
problems, 10 min error review").

Two safety mechanisms make sure the LLM can never violate the fixed
allocation or leave a task without a usable activity:

1. Minutes are never trusted as-is. Whatever per-activity minutes the
   LLM proposes are deterministically rescaled in Python
   (_rescale_activities_to_total) to sum EXACTLY to the task's
   allocated_minutes - the LLM's numbers only ever influence the
   relative split between activities, never the total.
2. Any failure (missing credentials, refusal, malformed response,
   network error) falls back to a single deterministic generic
   activity spanning the whole block, reusing the scheduler's own
   placeholder description - see generate_task_activities()'s
   deliberately broad exception handling.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from backend.ingestion.normalize import Role
from backend.llm.client import LLMExtractionError, parse_structured
from backend.llm.prompts import STUDY_ACTIVITY_SYSTEM_PROMPT, build_study_activity_user_prompt
from backend.matching.gaps import SkillGapResult
from backend.planning.scheduler import StudyPlan, StudyTask

logger = logging.getLogger(__name__)


class StudyActivity(BaseModel):
    """One concrete activity inside an already-fixed study block."""

    description: str
    minutes: float = Field(gt=0)


class _TaskBreakdown(BaseModel):
    """Internal: the LLM's raw proposed breakdown, before Python rescales it to the fixed total."""

    activities: list[StudyActivity] = Field(default_factory=list)


class EnrichedStudyTask(BaseModel):
    """A deterministic StudyTask - topic/date/minutes untouched - plus its LLM-generated activity breakdown."""

    task: StudyTask
    activities: list[StudyActivity]
    generation_source: str = Field(description="'llm' or 'fallback_generic'.")


def _find_matching_gap(task: StudyTask, skill_gaps: list[SkillGapResult]) -> SkillGapResult | None:
    return next((gap for gap in skill_gaps if gap.normalized_skill == task.normalized_skill), None)


def _skill_context_text(gap: SkillGapResult | None) -> str | None:
    if gap is None:
        return None
    requirement_label = "a required" if gap.required else "a preferred"
    return (
        f"target proficiency level {gap.target_level:g}/10, candidate's current estimated "
        f"level {gap.candidate_level:g}/10, {requirement_label} skill for this role."
    )


def _rescale_activities_to_total(activities: list[StudyActivity], target_total_minutes: float) -> list[StudyActivity]:
    """
    Deterministically rescale LLM-proposed activity minutes so they sum
    EXACTLY to `target_total_minutes` - the scheduler's fixed
    allocation for this block. Proportional rescale of each activity's
    share of the LLM's own (possibly wrong) proposed total, with the
    LAST activity absorbing any rounding remainder so the sum is exact
    (same pattern as backend.planning.scheduler.allocate_minutes).
    Descriptions are preserved unchanged - only minutes are corrected.
    """
    if not activities:
        return []
    raw_total = sum(activity.minutes for activity in activities)
    if raw_total <= 0:
        return []

    rescaled: list[StudyActivity] = []
    running_total = 0.0
    last_index = len(activities) - 1
    for index, activity in enumerate(activities):
        if index == last_index:
            minutes = target_total_minutes - running_total
        else:
            minutes = target_total_minutes * (activity.minutes / raw_total)
        minutes = max(0.0, round(minutes, 1))
        running_total += minutes
        rescaled.append(StudyActivity(description=activity.description, minutes=minutes))
    return rescaled


def _fallback_activities(task: StudyTask) -> list[StudyActivity]:
    """
    Deterministic fallback: one activity spanning the FULL allocated
    block, reusing the scheduler's own generic placeholder description
    - so a failed enrichment degrades to exactly what the plan already
    looked like before enrichment, never to a broken or missing task.
    """
    return [StudyActivity(description=task.description, minutes=task.allocated_minutes)]


def generate_task_activities(
    task: StudyTask,
    role: Role,
    skill_gaps: list[SkillGapResult],
    model: str | None = None,
) -> tuple[list[StudyActivity], str]:
    """
    Generate a detailed activity breakdown for one already-scheduled
    StudyTask. Returns (activities, source): source is "llm" on
    success (after rescaling - see _rescale_activities_to_total()) or
    "fallback_generic" whenever the LLM call fails or returns nothing
    usable - see this module's docstring.

    task.normalized_skill/display_name/scheduled_date/allocated_minutes
    are read-only here; nothing in this function modifies them.
    """
    if task.allocated_minutes <= 0:
        return [], "fallback_generic"

    matching_gap = _find_matching_gap(task, skill_gaps)
    try:
        breakdown = parse_structured(
            system_prompt=STUDY_ACTIVITY_SYSTEM_PROMPT,
            user_prompt=build_study_activity_user_prompt(
                topic=task.display_name,
                allocated_minutes=task.allocated_minutes,
                role_title=role.title,
                role_company=role.company,
                skill_context=_skill_context_text(matching_gap),
            ),
            response_model=_TaskBreakdown,
            model=model,
        )
        if not breakdown.activities:
            raise LLMExtractionError(f"LLM returned no activities for task {task.display_name!r}")

        activities = _rescale_activities_to_total(breakdown.activities, task.allocated_minutes)
        if not activities:
            raise LLMExtractionError(f"Rescaling produced no usable activities for task {task.display_name!r}")
        return activities, "llm"

    except Exception:
        # Deliberately broad: ANY failure mode here (missing
        # credentials, model refusal, malformed response, network
        # error) must degrade to the deterministic fallback, never
        # propagate and break the plan - see this module's docstring.
        logger.warning(
            "Falling back to a generic study activity for %r after LLM enrichment failed",
            task.display_name,
            exc_info=True,
        )
        return _fallback_activities(task), "fallback_generic"


def enrich_study_plan(
    plan: StudyPlan,
    role: Role,
    skill_gaps: list[SkillGapResult],
    model: str | None = None,
) -> list[EnrichedStudyTask]:
    """
    Enrich every task in `plan.tasks` with an LLM-generated activity
    breakdown. `plan` itself is never mutated - every task's
    topic/date/minutes were already finalized by
    backend.planning.scheduler and are carried through unchanged; this
    only adds detail INSIDE each already-fixed block.
    """
    enriched: list[EnrichedStudyTask] = []
    for task in plan.tasks:
        activities, source = generate_task_activities(task, role, skill_gaps, model=model)
        enriched.append(EnrichedStudyTask(task=task, activities=activities, generation_source=source))
    return enriched
