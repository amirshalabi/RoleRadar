"""
Interview prep service: persisted study plans, task completion, and
diagnostic-driven adaptive replanning for one tracked application.

Wraps backend.planning.scheduler (initial plan, via
backend.services.discovery.build_study_plan_for_application) and
backend.planning.adaptive (diagnostic-driven revision) with real
Postgres persistence (backend.db.study_plans, backend.db.assessment_results),
so a plan and its completed-task history survive across page loads -
nothing here is ever the only copy of state kept in Streamlit's
session_state.

Every revision is deterministic arithmetic (backend.planning.*) - no LLM
involvement anywhere in this module. submit_diagnostic()'s returned
"why did this change" message is built ONLY from the before/after
readiness NUMBERS this module itself computed; it never asserts a
causal story beyond "topic X's score moved by this many points."
"""

from __future__ import annotations

from datetime import date
from typing import Any

from backend.db import assessment_results as assessment_results_db
from backend.db import roles as roles_db
from backend.db import study_plans as study_plans_db
from backend.matching.cross_role import DEFAULT_FAVORITE_PRIORITY, FAVORITE_PRIORITY_MULTIPLIERS
from backend.planning.adaptive import revise_study_plan
from backend.planning.readiness import DiagnosticResult, ReadinessResult, calculate_readiness
from backend.planning.scheduler import StudyPlan, StudyTask
from backend.services import discovery, tracking

# A topic-readiness move smaller than this (0-100 scale) is treated as
# noise, not a reportable change - avoids a "Plan updated because X
# improved by 0.1 points" message for a move that's really just
# floating-point rounding.
_MIN_REPORTABLE_READINESS_DELTA = 0.5


def task_to_row(task: StudyTask) -> dict[str, Any]:
    return {
        "normalized_skill_name": task.normalized_skill,
        "display_name": task.display_name,
        "day_index": task.day_index,
        "scheduled_date": task.scheduled_date.isoformat() if task.scheduled_date else None,
        "allocated_minutes": task.allocated_minutes,
        "allocated_hours": round(task.allocated_minutes / 60.0, 2),
        "priority_score": task.priority_score,
        "task_description": task.description,
        "is_complete": task.is_complete,
    }


def row_to_task(row: dict[str, Any]) -> StudyTask:
    allocated_minutes = row.get("allocated_minutes")
    if allocated_minutes is None:
        allocated_minutes = (row.get("allocated_hours") or 0) * 60.0
    return StudyTask(
        normalized_skill=row["normalized_skill_name"],
        display_name=row.get("display_name") or row["normalized_skill_name"],
        day_index=row.get("day_index") or 0,
        scheduled_date=date.fromisoformat(row["scheduled_date"]) if row.get("scheduled_date") else None,
        allocated_minutes=allocated_minutes,
        priority_score=row.get("priority_score") or 0.0,
        description=row.get("task_description") or "",
        is_complete=bool(row.get("is_complete", False)),
    )


def _row_to_plain_plan(row: dict[str, Any]) -> StudyPlan:
    """
    Reconstruct just enough of a StudyPlan from a persisted row for
    backend.planning.adaptive.revise_study_plan() to revise it -
    that function only ever reads `.interview_date`, `.hours_available_per_day`,
    and each task's `.is_complete` from `previous_plan`, so a plain
    StudyPlan (not the richer AdaptiveStudyPlan shape) is sufficient
    input regardless of how many revisions came before this one.
    """
    return StudyPlan(
        interview_date=date.fromisoformat(row["interview_date"]) if row.get("interview_date") else None,
        current_date=date.today(),
        days_remaining=row.get("days_remaining") or 0,
        scheduling_days=row.get("scheduling_days") or 0,
        hours_available_per_day=row.get("hours_available_per_day") or discovery.DEFAULT_PREP_HOURS_PER_DAY,
        total_available_minutes=row.get("total_available_minutes") or 0.0,
        prep_items=[],
        allocations=[],
        tasks=[row_to_task(t) for t in row.get("tasks", [])],
        notes=[],
    )


def _topic_label(topic: str) -> str:
    return topic.replace("_", " ").title()


def build_readiness_change_message(before: ReadinessResult, after: ReadinessResult) -> str:
    """
    A plain-language summary of what moved, built ONLY from the two
    already-computed ReadinessResult snapshots - never a fabricated
    causal explanation. Reports the single biggest decrease and the
    single biggest increase (if either exists) with their actual point
    deltas, mirroring the style: "Plan updated because Algorithms
    readiness decreased and Probability readiness improved."
    """
    before_by_topic = {t.topic: t.readiness_score for t in before.topic_readiness}
    deltas = [
        (topic.topic, topic.readiness_score - before_by_topic.get(topic.topic, topic.readiness_score))
        for topic in after.topic_readiness
    ]
    improved = sorted((d for d in deltas if d[1] >= _MIN_REPORTABLE_READINESS_DELTA), key=lambda d: d[1], reverse=True)
    decreased = sorted((d for d in deltas if d[1] <= -_MIN_REPORTABLE_READINESS_DELTA), key=lambda d: d[1])

    parts: list[str] = []
    if decreased:
        topic, delta = decreased[0]
        parts.append(f"{_topic_label(topic)} readiness decreased by {abs(delta):.0f} point(s)")
    if improved:
        topic, delta = improved[0]
        parts.append(f"{_topic_label(topic)} readiness improved by {delta:.0f} point(s)")

    if not parts:
        return "Plan updated - no significant readiness change from this diagnostic."
    return "Plan updated because " + " and ".join(parts) + "."


def get_or_create_plan_view(user_id: str, role_id: str) -> dict[str, Any]:
    """
    The application's CURRENT persisted plan (with real task ids, for
    mark_task_complete()), building and persisting a fresh one via
    backend.services.discovery.build_study_plan_for_application() only
    if none exists yet. Raises ValueError if the application has no
    interview_date (nothing to plan against).
    """
    application = tracking.get_application_status(user_id, role_id)
    if application is None or not application.get("interview_date"):
        raise ValueError("This application has no interview date set yet.")
    application_id = application["id"]

    existing = study_plans_db.get_current_plan(application_id)
    if existing is not None:
        return existing

    plan = discovery.build_study_plan_for_application(user_id, role_id)
    study_plans_db.save_plan(
        user_id,
        application_id,
        interview_date=plan.interview_date.isoformat() if plan.interview_date else None,
        hours_available_per_day=plan.hours_available_per_day,
        days_remaining=plan.days_remaining,
        scheduling_days=plan.scheduling_days,
        total_available_minutes=plan.total_available_minutes,
        version=1,
        previous_version=None,
        revision_reason="Initial plan.",
        notes=plan.notes,
        tasks=[task_to_row(t) for t in plan.tasks],
    )
    return study_plans_db.get_current_plan(application_id)


def mark_task_complete(task_id: str, is_complete: bool = True) -> None:
    """Toggle one task's completion flag - never regenerates the plan."""
    study_plans_db.set_task_complete(task_id, is_complete)


def submit_diagnostic(
    user_id: str, role_id: str, topic: str, observed_level: float, confidence: float = 0.9
) -> dict[str, Any]:
    """
    Record one diagnostic result and produce the next plan revision:

    1. Compute readiness BEFORE this diagnostic (from the accumulated
       history so far), for a before/after comparison.
    2. Persist the diagnostic (backend.db.assessment_results) -
       append-only, never overwrites a prior attempt.
    3. Call backend.planning.adaptive.revise_study_plan() with the FULL
       accumulated diagnostic history against the role's persisted
       requirements and the candidate's current skills. Completed tasks
       are carried over unchanged; only remaining time is rescheduled.
    4. Persist the revision as the new CURRENT plan generation.

    Returns {"plan": <persisted plan row, with tasks/ids>,
    "readiness_before", "readiness_after", "message"} - `message` is
    built only from the two readiness snapshots, never a fabricated
    causal explanation (see build_readiness_change_message()).
    """
    application = tracking.get_application_status(user_id, role_id)
    if application is None or not application.get("interview_date"):
        raise ValueError("This application has no interview date set yet.")
    application_id = application["id"]

    role_row = roles_db.get_role_by_id(role_id)
    if role_row is None:
        raise ValueError("This role no longer exists.")
    role_family = role_row.get("role_family")

    analysis = discovery.analyze_role(user_id, role_row)
    profile = discovery.build_candidate_profile(user_id)

    previous_diagnostics = [
        DiagnosticResult(
            topic=row["normalized_skill_name"], observed_level=row["observed_level"], confidence=row.get("confidence") or 0.9
        )
        for row in assessment_results_db.list_assessment_results(user_id)
    ]
    readiness_before = calculate_readiness(profile, role_family, previous_diagnostics)

    assessment_results_db.save_assessment_result(user_id, topic, observed_level, confidence)

    current_plan_row = study_plans_db.get_current_plan(application_id)
    if current_plan_row is None:
        get_or_create_plan_view(user_id, role_id)
        current_plan_row = study_plans_db.get_current_plan(application_id)
    previous_plan = _row_to_plain_plan(current_plan_row)

    favorite = tracking.get_favorite(user_id, role_id)
    priority = favorite["priority"] if favorite else DEFAULT_FAVORITE_PRIORITY
    priority_multiplier = FAVORITE_PRIORITY_MULTIPLIERS.get(priority, 1.0)

    new_diagnostic = DiagnosticResult(topic=topic, observed_level=observed_level, confidence=confidence)
    revised = revise_study_plan(
        previous_plan,
        new_diagnostic,
        profile,
        role_family,
        analysis.requirements,
        current_date=date.today(),
        previous_diagnostics=previous_diagnostics,
        favorite_priority_multiplier=priority_multiplier,
    )
    # revise_study_plan() only knows how to compute a version number from
    # an AdaptiveStudyPlan instance (previous_plan here is a plain
    # StudyPlan reconstruction - see _row_to_plain_plan()'s docstring),
    # so the true version chain is tracked from the persisted row instead.
    true_previous_version = current_plan_row.get("version") or 1
    revised = revised.model_copy(update={"version": true_previous_version + 1, "previous_version": true_previous_version})

    all_tasks = [*revised.completed_tasks, *revised.new_tasks]
    study_plans_db.save_plan(
        user_id,
        application_id,
        interview_date=revised.interview_date.isoformat() if revised.interview_date else None,
        hours_available_per_day=revised.hours_available_per_day,
        days_remaining=revised.days_remaining,
        scheduling_days=revised.scheduling_days,
        total_available_minutes=revised.remaining_available_minutes,
        version=revised.version,
        previous_version=revised.previous_version,
        revision_reason=revised.revision_reason,
        notes=revised.notes,
        tasks=[task_to_row(t) for t in all_tasks],
    )

    readiness_after = revised.readiness
    return {
        "plan": study_plans_db.get_current_plan(application_id),
        "readiness_before": readiness_before,
        "readiness_after": readiness_after,
        "message": build_readiness_change_message(readiness_before, readiness_after),
    }


def get_current_readiness(user_id: str, role_id: str) -> ReadinessResult:
    """
    Live readiness (backend.planning.readiness) for this role family,
    including every diagnostic submitted so far - always recomputed
    fresh, never cached, so it reflects the latest accumulated evidence
    on every call.
    """
    role_row = roles_db.get_role_by_id(role_id)
    if role_row is None:
        raise ValueError("This role no longer exists.")

    profile = discovery.build_candidate_profile(user_id)
    diagnostics = [
        DiagnosticResult(
            topic=row["normalized_skill_name"], observed_level=row["observed_level"], confidence=row.get("confidence") or 0.9
        )
        for row in assessment_results_db.list_assessment_results(user_id)
    ]
    return calculate_readiness(profile, role_row.get("role_family"), diagnostics)
