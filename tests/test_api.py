"""
Tests for the FastAPI layer (backend/api). LLM calls are monkeypatched
(no real OpenAI calls); Postgres reads/writes go through the in-memory
FakeSupabaseClient shared across every backend.db module the API
touches, via httpx's TestClient driving the real ASGI app end to end
(routing, dependency injection, request/response validation, and the
registered exception handlers all execute for real).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.db import applications as applications_db
from backend.db import assessment_results as assessment_results_db
from backend.db import candidates as candidates_db
from backend.db import favorites as favorites_db
from backend.db import metrics as metrics_db
from backend.db import rationales as rationales_db
from backend.db import role_requirements as role_requirements_db
from backend.db import roles as roles_db
from backend.db import study_plans as study_plans_db
from backend.db import upserts as upserts_db
from backend.db import users as users_db
from backend.llm.extract_requirements import RoleRequirement
from backend.main import app
from backend.services import discovery
from backend.utils.metrics import ConcurrencyMetrics, PipelineMetrics
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
        rationales_db, study_plans_db, assessment_results_db, upserts_db, users_db, metrics_db,
    ):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


@pytest.fixture
def api_client(fake_client: FakeSupabaseClient) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------
# health
# ---------------------------------------------------------------------


def test_health_check(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------


def test_create_and_get_role(api_client: TestClient) -> None:
    created = api_client.post("/roles", json={"company": "Meridian Capital", "title": "Quant Intern", "role_family": "quant"})
    assert created.status_code == 200
    role_id = created.json()["id"]

    fetched = api_client.get(f"/roles/{role_id}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "Quant Intern"

    listed = api_client.get("/roles")
    assert listed.status_code == 200
    assert any(r["id"] == role_id for r in listed.json())


def test_get_role_not_found_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/roles/does-not-exist")
    assert response.status_code == 404


def test_score_role(monkeypatch: pytest.MonkeyPatch, api_client: TestClient) -> None:
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    created = api_client.post(
        "/roles", json={"company": "Meridian Capital", "title": "Quant Intern", "role_family": "quant", "description": "Need probability and Python."}
    )
    role_id = created.json()["id"]

    response = api_client.post(f"/roles/{role_id}/score")

    assert response.status_code == 200
    body = response.json()
    assert "fit_result" in body
    assert 0.0 <= body["fit_result"]["overall_score"] <= 100.0


def test_score_role_not_found_returns_404(api_client: TestClient) -> None:
    response = api_client.post("/roles/does-not-exist/score")
    assert response.status_code == 404


# ---------------------------------------------------------------------
# favorites
# ---------------------------------------------------------------------


def test_favorites_full_flow(api_client: TestClient) -> None:
    role_id = api_client.post("/roles", json={"company": "Acme", "title": "SWE Intern"}).json()["id"]

    saved = api_client.post(f"/favorites/{role_id}", json={"priority": "dream"})
    assert saved.status_code == 200
    assert saved.json()["priority"] == "dream"

    listed = api_client.get("/favorites")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["role_id"] == role_id
    assert listed.json()[0]["priority"] == "dream"

    removed = api_client.delete(f"/favorites/{role_id}")
    assert removed.status_code == 204

    listed_after = api_client.get("/favorites")
    assert listed_after.json() == []


def test_save_favorite_defaults_when_no_body_given(api_client: TestClient) -> None:
    role_id = api_client.post("/roles", json={"company": "Acme", "title": "SWE Intern"}).json()["id"]

    response = api_client.post(f"/favorites/{role_id}")

    assert response.status_code == 200
    assert response.json()["priority"] == "interested"


# ---------------------------------------------------------------------
# applications
# ---------------------------------------------------------------------


def test_update_application(api_client: TestClient) -> None:
    role_id = api_client.post("/roles", json={"company": "Acme", "title": "SWE Intern"}).json()["id"]

    response = api_client.patch(f"/applications/{role_id}", json={"status": "applied", "deadline": "2026-09-20"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "applied"
    assert body["deadline"] == "2026-09-20"


# ---------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------


def test_create_plan(monkeypatch: pytest.MonkeyPatch, api_client: TestClient) -> None:
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    role_id = api_client.post(
        "/roles", json={"company": "Meridian Capital", "title": "Quant Intern", "role_family": "quant", "description": "Need probability and Python."}
    ).json()["id"]
    api_client.patch(f"/applications/{role_id}", json={"status": "interview", "interview_date": "2026-09-20"})

    response = api_client.post(f"/plans/{role_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 1
    assert all("id" in task for task in body["tasks"])


def test_create_plan_without_interview_date_returns_400(api_client: TestClient) -> None:
    role_id = api_client.post("/roles", json={"company": "Acme", "title": "SWE Intern"}).json()["id"]

    response = api_client.post(f"/plans/{role_id}")

    assert response.status_code == 400


# ---------------------------------------------------------------------
# assessments
# ---------------------------------------------------------------------


def test_submit_assessment(monkeypatch: pytest.MonkeyPatch, api_client: TestClient) -> None:
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)
    role_id = api_client.post(
        "/roles", json={"company": "Meridian Capital", "title": "Quant Intern", "role_family": "quant", "description": "Need probability and Python."}
    ).json()["id"]
    api_client.patch(f"/applications/{role_id}", json={"status": "interview", "interview_date": "2026-09-20"})
    api_client.post(f"/plans/{role_id}")

    response = api_client.post(
        "/assessments", json={"role_id": role_id, "topic": "probability", "observed_level": 9.0, "confidence": 0.95}
    )

    assert response.status_code == 201
    body = response.json()
    assert "message" in body
    assert body["plan"]["version"] == 2
    assert "overall_readiness" in body["readiness_before"]
    assert "overall_readiness" in body["readiness_after"]


# ---------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------


def test_pipeline_analytics_empty_state(api_client: TestClient) -> None:
    response = api_client.get("/analytics/pipeline")

    assert response.status_code == 200
    assert response.json()["has_data"] is False


def test_pipeline_analytics_with_recorded_run(api_client: TestClient) -> None:
    metrics_db.record_ingestion_run(
        PipelineMetrics(ingested=140, deduplicated=12, hard_filtered=71, keyword_filtered=41, llm_analyzed=16),
        ConcurrencyMetrics(serial_seconds=4.8, concurrent_seconds=1.6),
    )

    response = api_client.get("/analytics/pipeline")

    assert response.status_code == 200
    body = response.json()
    assert body["has_data"] is True
    assert body["ingested"] == 140
    assert body["llm_analyzed"] == 16
    assert body["speedup"] == pytest.approx(3.0)


# ---------------------------------------------------------------------
# candidate
# ---------------------------------------------------------------------


def test_parse_resume(monkeypatch: pytest.MonkeyPatch, api_client: TestClient) -> None:
    from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
    from backend.services import candidate as candidate_service

    fake_profile = CandidateProfile(
        skills=[CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.0, confidence=0.7)]
    )
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: fake_profile)

    response = api_client.post("/candidate/parse", json={"resume_text": "Built Python pipelines."})

    assert response.status_code == 200
    assert response.json()["skills_saved"] == 1


def test_parse_resume_rejects_empty_text(api_client: TestClient) -> None:
    response = api_client.post("/candidate/parse", json={"resume_text": ""})
    assert response.status_code == 422  # Pydantic min_length validation


# ---------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------


def test_supabase_not_configured_returns_503(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    from backend.db.client import SupabaseNotConfiguredError

    def _raise():
        raise SupabaseNotConfiguredError("SUPABASE_URL and SUPABASE_KEY must be set.")

    monkeypatch.setattr(roles_db, "get_client", _raise)
    client = TestClient(app)

    response = client.get("/roles")

    assert response.status_code == 503
