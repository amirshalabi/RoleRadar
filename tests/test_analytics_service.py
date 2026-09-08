"""
Tests for backend.services.analytics. LLM extraction is monkeypatched
(no real OpenAI calls); Postgres reads/writes go through the in-memory
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
from backend.services import analytics, discovery, prep
from tests._fake_supabase import FakeSupabaseClient

SAMPLE_REQUIREMENTS = [
    RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8.0, importance=9.0, required=True, evidence=["x"]),
    RoleRequirement(skill="Python", normalized_skill="python", target_level=7.0, importance=7.0, required=True, evidence=["x"]),
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


def _seed_role(role_family="quant", description="Build trading signals. Need probability and Python."):
    return roles_db.upsert_role(company="Meridian Capital", title="Quant Research Intern", role_family=role_family, description=description)


# ---------------------------------------------------------------------
# get_fit_distribution
# ---------------------------------------------------------------------


def test_get_fit_distribution_empty_when_no_fit_scores(fake_client: FakeSupabaseClient) -> None:
    assert analytics.get_fit_distribution("u1") == []


def test_get_fit_distribution_returns_persisted_scores(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    discovery.analyze_role("u1", role_row)

    scores = analytics.get_fit_distribution("u1")

    assert len(scores) == 1
    assert isinstance(scores[0], float)


# ---------------------------------------------------------------------
# get_readiness_distribution
# ---------------------------------------------------------------------


def test_get_readiness_distribution_empty_without_skills(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family="quant")
    assert analytics.get_readiness_distribution("u1") == []


def test_get_readiness_distribution_includes_live_readiness(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family="quant")
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=7.0, confidence=0.7, display_name="Python")

    scores = analytics.get_readiness_distribution("u1")

    assert len(scores) == 1


# ---------------------------------------------------------------------
# get_top_recurring_skill_gaps
# ---------------------------------------------------------------------


def test_get_top_recurring_skill_gaps_sorted_by_frequency(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_a = _seed_role(description="Need probability.")
    role_b = roles_db.upsert_role(company="Acme", title="SWE Intern", role_family="swe", description="Need python.")
    favorites_db.save_favorite("u1", role_a["id"], priority="dream")
    favorites_db.save_favorite("u1", role_b["id"], priority="backup")
    role_requirements_db.upsert_role_requirements(
        role_a["id"],
        [
            {"normalized_skill_name": "probability", "display_name": "Probability", "target_level": 8.0, "importance": 9.0, "is_required": True, "evidence": []},
            {"normalized_skill_name": "python", "display_name": "Python", "target_level": 7.0, "importance": 7.0, "is_required": True, "evidence": []},
        ],
    )
    role_requirements_db.upsert_role_requirements(
        role_b["id"],
        [{"normalized_skill_name": "python", "display_name": "Python", "target_level": 6.0, "importance": 6.0, "is_required": True, "evidence": []}],
    )

    results = analytics.get_top_recurring_skill_gaps("u1")

    assert results[0].normalized_skill == "python"  # required by 2 roles vs probability's 1
    assert results[0].roles_requiring_it == 2


def test_get_top_recurring_skill_gaps_empty_when_no_favorites(fake_client: FakeSupabaseClient) -> None:
    assert analytics.get_top_recurring_skill_gaps("u1") == []


# ---------------------------------------------------------------------
# get_applications_by_stage
# ---------------------------------------------------------------------


def test_get_applications_by_stage_empty_when_none_tracked(fake_client: FakeSupabaseClient) -> None:
    assert analytics.get_applications_by_stage("u1") == {}


def test_get_applications_by_stage_counts_per_status(fake_client: FakeSupabaseClient) -> None:
    role_a = _seed_role()
    role_b = roles_db.upsert_role(company="Acme", title="SWE Intern", description="x")
    role_c = roles_db.upsert_role(company="Other", title="Other Intern", description="x")
    applications_db.upsert_application("u1", role_a["id"], status="applied")
    applications_db.upsert_application("u1", role_b["id"], status="applied")
    applications_db.upsert_application("u1", role_c["id"], status="interview")

    counts = analytics.get_applications_by_stage("u1")

    assert counts == {"applied": 2, "interview": 1}


# ---------------------------------------------------------------------
# get_completed_study_minutes
# ---------------------------------------------------------------------


def test_get_completed_study_minutes_zero_when_no_plans(fake_client: FakeSupabaseClient) -> None:
    assert analytics.get_completed_study_minutes("u1") == 0.0


def test_get_completed_study_minutes_sums_only_completed_tasks(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    applications_db.upsert_application("u1", role_row["id"], status="interview", interview_date="2026-09-20")
    application = applications_db.get_application("u1", role_row["id"])
    study_plans_db.save_plan(
        "u1", application["id"], interview_date="2026-09-20", hours_available_per_day=2.0, days_remaining=5,
        scheduling_days=5, total_available_minutes=600.0, version=1, previous_version=None,
        revision_reason="Initial plan.", notes=[],
        tasks=[
            {"normalized_skill_name": "python", "display_name": "Python", "day_index": 0, "scheduled_date": None, "allocated_minutes": 60.0, "allocated_hours": 1.0, "priority_score": 1.0, "task_description": "x", "is_complete": True},
            {"normalized_skill_name": "probability", "display_name": "Probability", "day_index": 0, "scheduled_date": None, "allocated_minutes": 90.0, "allocated_hours": 1.5, "priority_score": 1.0, "task_description": "x", "is_complete": False},
        ],
    )

    assert analytics.get_completed_study_minutes("u1") == 60.0


def test_get_completed_study_minutes_never_double_counts_across_generations(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    role_row = _seed_role(role_family="quant")
    applications_db.upsert_application("u1", role_row["id"], status="interview", interview_date="2026-09-20")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    initial = prep.get_or_create_plan_view("u1", role_row["id"])
    prep.mark_task_complete(initial["tasks"][0]["id"], True)
    prep.submit_diagnostic("u1", role_row["id"], "probability", observed_level=9.0, confidence=0.9)

    total = analytics.get_completed_study_minutes("u1")
    completed_minutes = next(t["allocated_minutes"] for t in initial["tasks"] if t["id"] == initial["tasks"][0]["id"])
    assert total == pytest.approx(completed_minutes)  # counted once, from the current generation only


# ---------------------------------------------------------------------
# get_readiness_over_time
# ---------------------------------------------------------------------


def test_get_readiness_over_time_empty_when_no_diagnostics(fake_client: FakeSupabaseClient) -> None:
    assert analytics.get_readiness_over_time("u1") == []


def test_get_readiness_over_time_one_point_per_diagnostic(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family="quant")
    assessment_results_db.save_assessment_result("u1", "probability", observed_level=5.0, confidence=0.8)
    assessment_results_db.save_assessment_result("u1", "probability", observed_level=8.0, confidence=0.9)

    points = analytics.get_readiness_over_time("u1")

    assert len(points) == 2
    assert points[1]["overall_readiness"] >= points[0]["overall_readiness"]
