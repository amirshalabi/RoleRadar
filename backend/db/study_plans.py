"""
Study plan + study task persistence.

One study_plans row is the CURRENT prep plan for one application;
study_tasks rows are its day-by-day schedule, one per
backend.planning.scheduler.StudyTask. Diagnostic-driven adaptive
replanning (backend.planning.adaptive.revise_study_plan, orchestrated by
backend.services.prep) produces a new plan GENERATION rather than
mutating the old one: save_plan() flips any existing is_current row for
the application to false and inserts a fresh plan + task rows - a prior
generation's row is never deleted or mutated, so the full revision
history stays intact in Postgres.

The one exception to "a plan generation's rows are immutable once
written" is set_task_complete(): marking a task done is a real update to
that task's own is_complete flag, not a new plan generation.
"""

from __future__ import annotations

from typing import Any

from backend.db.client import get_client

STUDY_PLANS_TABLE = "study_plans"
STUDY_TASKS_TABLE = "study_tasks"


def save_plan(
    user_id: str,
    application_id: str,
    *,
    interview_date: str | None,
    hours_available_per_day: float,
    days_remaining: int,
    scheduling_days: int,
    total_available_minutes: float,
    version: int,
    previous_version: int | None,
    revision_reason: str | None,
    notes: list[str],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Persist a new plan generation as the CURRENT plan for `application_id`.
    Returns the inserted study_plans row with its inserted `tasks` (each
    a study_tasks row, including its real `id` - needed for
    set_task_complete()) attached.
    """
    client = get_client()
    client.table(STUDY_PLANS_TABLE).update({"is_current": False}).eq("application_id", application_id).eq(
        "is_current", True
    ).execute()

    plan_values = {
        "user_id": user_id,
        "application_id": application_id,
        "interview_date": interview_date,
        "hours_available_per_day": hours_available_per_day,
        "days_remaining": days_remaining,
        "scheduling_days": scheduling_days,
        "total_available_minutes": total_available_minutes,
        "version": version,
        "previous_version": previous_version,
        "revision_reason": revision_reason,
        "notes": notes,
        "is_current": True,
    }
    response = client.table(STUDY_PLANS_TABLE).insert(plan_values).execute()
    if not response.data:
        raise RuntimeError(f"Insert into {STUDY_PLANS_TABLE} returned no data")
    plan_row = response.data[0]

    if tasks:
        task_rows = [{**task, "study_plan_id": plan_row["id"]} for task in tasks]
        task_response = client.table(STUDY_TASKS_TABLE).insert(task_rows).execute()
        if not task_response.data:
            raise RuntimeError(f"Insert into {STUDY_TASKS_TABLE} returned no data")
        plan_row["tasks"] = task_response.data
    else:
        plan_row["tasks"] = []
    return plan_row


def get_current_plan(application_id: str) -> dict[str, Any] | None:
    """The CURRENT study_plans row for an application (with its tasks embedded), or None if none exists yet."""
    client = get_client()
    response = (
        client.table(STUDY_PLANS_TABLE)
        .select("*")
        .eq("application_id", application_id)
        .eq("is_current", True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    plan_row = response.data[0]
    plan_row["tasks"] = list_tasks(plan_row["id"])
    return plan_row


def list_tasks(study_plan_id: str) -> list[dict[str, Any]]:
    """Every study_tasks row belonging to one plan generation."""
    client = get_client()
    response = client.table(STUDY_TASKS_TABLE).select("*").eq("study_plan_id", study_plan_id).execute()
    return response.data or []


def set_task_complete(task_id: str, is_complete: bool = True) -> dict[str, Any]:
    """Toggle one task's completion flag in place - never a new plan generation."""
    client = get_client()
    response = client.table(STUDY_TASKS_TABLE).update({"is_complete": is_complete}).eq("id", task_id).execute()
    if not response.data:
        raise RuntimeError(f"Update on {STUDY_TASKS_TABLE} returned no data")
    return response.data[0]
