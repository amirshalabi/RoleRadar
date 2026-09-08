"""
Tests for backend.services.prep. LLM extraction is monkeypatched (no
real OpenAI calls); Postgres reads/writes go through the in-memory
FakeSupabaseClient for real read-after-write behavior.
"""

from __future__ import annotations

import pytest

from backend.db import applications as applications_db
from backend.db import assessment_results as assessment_results_db
from backend.db import candidates as candidates_db
from backend.db import favorites as favorites_db
from backend.db import rationales as rationales_db
from backend.db import role_requirements as role_requirements_db
from backend.db import roles as roles_db
from backend.db import study_plans as study_plans_db
from backend.db import upserts as upserts_db
from backend.llm.extract_requirements import RoleRequirement
from backend.services import discovery, prep
from tests._fake_supabase import FakeSupabaseClient

SAMPLE_REQUIREMENTS = [
    RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8.0, importance=9.0, required=True, evidence=["Need probability."]),
    RoleRequirement(skill="Python", normalized_skill="python", target_level=7.0, importance=7.0, required=True, evidence=["Build trading signals."]),
]


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (
        roles_db, candidates_db, favorites_db, applications_db, role_requirements_db,
        rationales_db, study_plans_db, assessment_results_db, upserts_db,
    ):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def _seed_role_and_application(role_family="quant", interview_date="2026-09-20"):
    role_row = roles_db.upsert_role(
        company="Meridian Capital", title="Quant Research Intern", location="NYC",
        description="Build trading signals. Need probability and Python.", role_family=role_family,
    )
    applications_db.upsert_application("u1", role_row["id"], status="interview", interview_date=interview_date)
    return role_row


# ---------------------------------------------------------------------
# get_or_create_plan_view
# ---------------------------------------------------------------------


def test_get_or_create_plan_view_raises_without_interview_date(fake_client: FakeSupabaseClient) -> None:
    role_row = roles_db.upsert_role(company="Acme", title="SWE Intern", description="x", role_family="swe")
    applications_db.upsert_application("u1", role_row["id"], status="applied")

    with pytest.raises(ValueError):
        prep.get_or_create_plan_view("u1", role_row["id"])


def test_get_or_create_plan_view_builds_and_persists_fresh_plan(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=8.0, confidence=0.8, display_name="Python")

    plan = prep.get_or_create_plan_view("u1", role_row["id"])

    assert plan["version"] == 1
    assert all("id" in task for task in plan["tasks"])


def test_get_or_create_plan_view_reuses_persisted_plan_on_second_call(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    first = prep.get_or_create_plan_view("u1", role_row["id"])
    second = prep.get_or_create_plan_view("u1", role_row["id"])

    assert first["id"] == second["id"]


# ---------------------------------------------------------------------
# mark_task_complete
# ---------------------------------------------------------------------


def test_mark_task_complete_is_reflected_on_next_read(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    plan = prep.get_or_create_plan_view("u1", role_row["id"])
    task_id = plan["tasks"][0]["id"]

    prep.mark_task_complete(task_id, True)

    refreshed = study_plans_db.get_current_plan(plan["application_id"])
    completed = [t for t in refreshed["tasks"] if t["id"] == task_id]
    assert completed[0]["is_complete"] is True


# ---------------------------------------------------------------------
# submit_diagnostic
# ---------------------------------------------------------------------


def test_submit_diagnostic_persists_the_diagnostic(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    prep.get_or_create_plan_view("u1", role_row["id"])

    prep.submit_diagnostic("u1", role_row["id"], "probability", observed_level=9.0, confidence=0.95)

    results = assessment_results_db.list_assessment_results("u1")
    assert len(results) == 1
    assert results[0]["normalized_skill_name"] == "probability"


def test_submit_diagnostic_creates_new_plan_generation(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    initial = prep.get_or_create_plan_view("u1", role_row["id"])

    result = prep.submit_diagnostic("u1", role_row["id"], "probability", observed_level=9.0, confidence=0.95)

    assert result["plan"]["version"] == initial["version"] + 1
    assert result["plan"]["id"] != initial["id"]


def test_submit_diagnostic_preserves_completed_task_history(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    initial = prep.get_or_create_plan_view("u1", role_row["id"])
    completed_task = initial["tasks"][0]
    prep.mark_task_complete(completed_task["id"], True)

    result = prep.submit_diagnostic("u1", role_row["id"], "probability", observed_level=9.0, confidence=0.95)

    revised_tasks = result["plan"]["tasks"]
    matching = [
        t for t in revised_tasks
        if t["normalized_skill_name"] == completed_task["normalized_skill_name"] and t["is_complete"]
    ]
    assert matching, "the previously completed task should be carried forward, marked complete"


def test_submit_diagnostic_returns_before_and_after_readiness(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application(role_family="quant")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    candidates_db.upsert_candidate_skill("u1", "probability", estimated_level=2.0, confidence=0.2, display_name="Probability")
    prep.get_or_create_plan_view("u1", role_row["id"])

    result = prep.submit_diagnostic("u1", role_row["id"], "probability", observed_level=9.0, confidence=0.95)

    before_probability = next(t.readiness_score for t in result["readiness_before"].topic_readiness if t.topic == "probability")
    after_probability = next(t.readiness_score for t in result["readiness_after"].topic_readiness if t.topic == "probability")
    assert after_probability > before_probability


def test_submit_diagnostic_message_reports_readiness_improvement(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application(role_family="quant")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    candidates_db.upsert_candidate_skill("u1", "probability", estimated_level=1.0, confidence=0.1, display_name="Probability")
    prep.get_or_create_plan_view("u1", role_row["id"])

    result = prep.submit_diagnostic("u1", role_row["id"], "probability", observed_level=9.5, confidence=0.95)

    assert "improved" in result["message"]
    assert "Plan updated" in result["message"]


def test_get_current_readiness_reflects_prior_diagnostics(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application(role_family="quant")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    before = prep.get_current_readiness("u1", role_row["id"])
    assessment_results_db.save_assessment_result("u1", "probability", observed_level=9.5, confidence=0.95)
    after = prep.get_current_readiness("u1", role_row["id"])

    before_score = next(t.readiness_score for t in before.topic_readiness if t.topic == "probability")
    after_score = next(t.readiness_score for t in after.topic_readiness if t.topic == "probability")
    assert after_score > before_score


def test_submit_diagnostic_creates_plan_when_none_existed_yet(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role_and_application()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    result = prep.submit_diagnostic("u1", role_row["id"], "probability", observed_level=9.0, confidence=0.95)

    assert result["plan"] is not None
    assert result["plan"]["version"] == 2  # initial (version 1, created as a side effect) then revised
