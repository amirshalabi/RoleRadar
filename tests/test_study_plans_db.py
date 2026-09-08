"""Tests for backend.db.study_plans. Uses the in-memory FakeSupabaseClient for real read-after-write behavior."""

from __future__ import annotations

import pytest

from backend.db import study_plans as study_plans_db
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    monkeypatch.setattr(study_plans_db, "get_client", lambda: client)
    return client


def _task(skill="python", minutes=60.0):
    return {
        "normalized_skill_name": skill, "display_name": skill.title(), "day_index": 0, "scheduled_date": None,
        "allocated_minutes": minutes, "allocated_hours": minutes / 60.0, "priority_score": 1.0,
        "task_description": f"{skill} practice", "is_complete": False,
    }


def test_save_plan_persists_plan_and_tasks(fake_client: FakeSupabaseClient) -> None:
    saved = study_plans_db.save_plan(
        "u1", "app-1", interview_date="2026-09-20", hours_available_per_day=2.0, days_remaining=5,
        scheduling_days=5, total_available_minutes=600.0, version=1, previous_version=None,
        revision_reason="Initial plan.", notes=[], tasks=[_task("python"), _task("probability")],
    )

    assert saved["is_current"] is True
    assert len(saved["tasks"]) == 2
    assert all("id" in task for task in saved["tasks"])


def test_get_current_plan_returns_none_when_no_plan(fake_client: FakeSupabaseClient) -> None:
    assert study_plans_db.get_current_plan("app-1") is None


def test_get_current_plan_embeds_tasks(fake_client: FakeSupabaseClient) -> None:
    study_plans_db.save_plan(
        "u1", "app-1", interview_date="2026-09-20", hours_available_per_day=2.0, days_remaining=5,
        scheduling_days=5, total_available_minutes=600.0, version=1, previous_version=None,
        revision_reason="Initial plan.", notes=[], tasks=[_task("python")],
    )

    current = study_plans_db.get_current_plan("app-1")

    assert current is not None
    assert len(current["tasks"]) == 1
    assert current["tasks"][0]["normalized_skill_name"] == "python"


def test_save_plan_supersedes_previous_current_plan(fake_client: FakeSupabaseClient) -> None:
    first = study_plans_db.save_plan(
        "u1", "app-1", interview_date="2026-09-20", hours_available_per_day=2.0, days_remaining=5,
        scheduling_days=5, total_available_minutes=600.0, version=1, previous_version=None,
        revision_reason="Initial plan.", notes=[], tasks=[_task("python")],
    )
    second = study_plans_db.save_plan(
        "u1", "app-1", interview_date="2026-09-20", hours_available_per_day=2.0, days_remaining=4,
        scheduling_days=4, total_available_minutes=480.0, version=2, previous_version=1,
        revision_reason="Diagnostic completed.", notes=[], tasks=[_task("probability")],
    )

    current = study_plans_db.get_current_plan("app-1")

    assert current["id"] == second["id"]
    assert current["version"] == 2
    # the superseded generation's row still exists (never deleted), just no longer current
    assert first["id"] != second["id"]


def test_set_task_complete_toggles_flag_in_place(fake_client: FakeSupabaseClient) -> None:
    saved = study_plans_db.save_plan(
        "u1", "app-1", interview_date="2026-09-20", hours_available_per_day=2.0, days_remaining=5,
        scheduling_days=5, total_available_minutes=600.0, version=1, previous_version=None,
        revision_reason="Initial plan.", notes=[], tasks=[_task("python")],
    )
    task_id = saved["tasks"][0]["id"]

    study_plans_db.set_task_complete(task_id, True)

    current = study_plans_db.get_current_plan("app-1")
    assert current["tasks"][0]["is_complete"] is True
