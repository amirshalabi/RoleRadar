"""
Tests for backend.services.discovery. LLM extraction is always
monkeypatched (no real OpenAI calls); Postgres reads/writes go through
the in-memory FakeSupabaseClient for real read-after-write behavior.
"""

from __future__ import annotations

import pytest

from backend.db import applications as applications_db
from backend.db import candidates as candidates_db
from backend.db import favorites as favorites_db
from backend.db import role_requirements as role_requirements_db
from backend.db import roles as roles_db
from backend.db import upserts as upserts_db
from backend.llm.extract_requirements import RoleRequirement
from backend.services import discovery
from tests._fake_supabase import FakeSupabaseClient

SAMPLE_REQUIREMENTS = [
    RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8.0, importance=9.0, required=True, evidence=["Need probability."]),
    RoleRequirement(skill="Python", normalized_skill="python", target_level=7.0, importance=7.0, required=True, evidence=["Build trading signals."]),
]


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (roles_db, candidates_db, favorites_db, applications_db, role_requirements_db, upserts_db):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def _seed_role(role_family="quant", description="Build trading signals. Need probability and Python."):
    return roles_db.upsert_role(
        company="Meridian Capital", title="Quant Research Intern", location="NYC",
        description=description, role_family=role_family,
    )


# ---------------------------------------------------------------------
# list_role_cards
# ---------------------------------------------------------------------


def test_list_role_cards_empty_when_no_roles(fake_client: FakeSupabaseClient) -> None:
    assert discovery.list_role_cards("u1") == []


def test_unanalyzed_role_has_no_fit_score_or_gaps(fake_client: FakeSupabaseClient) -> None:
    _seed_role()

    [card] = discovery.list_role_cards("u1")

    assert card.analyzed is False
    assert card.fit_score is None
    assert card.top_strengths == []
    assert card.top_gap is None


def test_readiness_computed_live_without_analysis(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family="quant")
    candidates_db.upsert_candidate_skill("u1", "probability", estimated_level=5.0, confidence=0.5, display_name="Probability")

    [card] = discovery.list_role_cards("u1")

    assert card.analyzed is False
    assert card.readiness_score is not None  # no LLM/requirements needed for this


def test_readiness_none_without_skills(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family="quant")

    [card] = discovery.list_role_cards("u1")

    assert card.readiness_score is None


def test_readiness_none_without_role_family(fake_client: FakeSupabaseClient) -> None:
    _seed_role(role_family=None)
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=5.0, confidence=0.5, display_name="Python")

    [card] = discovery.list_role_cards("u1")

    assert card.readiness_score is None


def test_card_reflects_favorite_and_application_state(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    favorites_db.save_favorite("u1", role_row["id"], priority="dream")
    applications_db.upsert_application("u1", role_row["id"], status="applied", deadline="2026-09-20")

    [card] = discovery.list_role_cards("u1")

    assert card.is_saved is True
    assert card.priority == "dream"
    assert card.application_status == "applied"
    assert card.deadline == "2026-09-20"


def test_card_defaults_when_not_saved_or_tracked(fake_client: FakeSupabaseClient) -> None:
    _seed_role()

    [card] = discovery.list_role_cards("u1")

    assert card.is_saved is False
    assert card.priority is None
    assert card.application_status is None


# ---------------------------------------------------------------------
# analyze_role
# ---------------------------------------------------------------------


def test_analyze_role_extracts_requirements_when_none_persisted(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    assert {r.normalized_skill for r in analysis.requirements} == {"probability", "python"}
    assert role_requirements_db.list_role_requirements(role_row["id"])  # persisted


def test_analyze_role_reuses_persisted_requirements_without_calling_llm(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    role_requirements_db.upsert_role_requirements(
        role_row["id"],
        [{"normalized_skill_name": "python", "display_name": "Python", "target_level": 6.0, "importance": 7.0, "is_required": True, "evidence": []}],
    )
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("LLM extraction should not be called when requirements are already persisted")

    monkeypatch.setattr(discovery, "extract_role_requirements", fail_if_called)

    analysis = discovery.analyze_role("u1", role_row)

    assert called is False
    assert len(analysis.requirements) == 1


def test_analyze_role_persists_fit_score(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    fit_rows = roles_db.list_fit_scores_for_user("u1")
    assert len(fit_rows) == 1
    assert fit_rows[0]["overall_score"] == pytest.approx(analysis.fit_result.overall_score)


def test_analyze_role_computes_gaps_against_candidate_skills(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=8.0, confidence=0.8, display_name="Python")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    by_skill = {g.normalized_skill: g for g in analysis.gaps}
    assert by_skill["python"].satisfaction_ratio == 1.0
    assert by_skill["probability"].satisfaction_ratio == 0.0


def test_analyze_role_computes_readiness_when_role_family_present(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role(role_family="quant")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    assert analysis.readiness is not None


def test_analyze_role_readiness_none_without_role_family(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role(role_family=None)
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    analysis = discovery.analyze_role("u1", role_row)

    assert analysis.readiness is None


def test_analyze_role_raises_without_description_or_cached_requirements(fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role(description=None)

    with pytest.raises(ValueError):
        discovery.analyze_role("u1", role_row)


def test_card_reflects_analysis_after_it_runs(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    role_row = _seed_role()
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=8.0, confidence=0.8, display_name="Python")
    monkeypatch.setattr(discovery, "extract_role_requirements", lambda role, model=None: SAMPLE_REQUIREMENTS)

    discovery.analyze_role("u1", role_row)
    [card] = discovery.list_role_cards("u1")

    assert card.analyzed is True
    assert card.fit_score is not None
    assert "Python" in card.top_strengths
    assert card.top_gap == "Probability"
