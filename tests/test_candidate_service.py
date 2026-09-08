"""Tests for backend.services.candidate. No real OpenAI calls - extract_candidate_profile is monkeypatched."""

from __future__ import annotations

import pytest

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.db import candidates as candidates_db
from backend.db import upserts as upserts_db
from backend.services import candidate as candidate_service
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (candidates_db, upserts_db):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def _fake_profile() -> CandidateProfile:
    return CandidateProfile(
        skills=[
            CandidateSkillEstimate(normalized_skill_name="Python", display_name="Python", estimated_level=7.0, confidence=0.7, evidence_snippets=["Built Python pipelines"]),
            CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=5.0, confidence=0.4, evidence_snippets=["Skills: Python"]),
        ]
    )


def test_parse_and_save_resume_persists_deduplicated_skills(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: _fake_profile())

    profile = candidate_service.parse_and_save_resume("u1", "resume text")

    assert len(profile.skills) == 1  # "Python" and "python" collapse to one normalized identity
    assert profile.skills[0].confidence == 0.7  # higher-confidence duplicate wins

    stored = candidates_db.list_candidate_skills("u1")
    assert len(stored) == 1
    assert stored[0]["evidence_snippets"] == ["Built Python pipelines"]


def test_parse_and_save_resume_persists_profile_row(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: _fake_profile())

    candidate_service.parse_and_save_resume("u1", "resume text")

    row = candidates_db.get_candidate_profile("u1")
    assert row is not None
    assert row["raw_resume_text"] == "resume text"


def test_parse_and_save_resume_returns_empty_profile_for_no_skills(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: CandidateProfile())

    profile = candidate_service.parse_and_save_resume("u1", "resume text")

    assert profile.skills == []
    assert candidates_db.list_candidate_skills("u1") == []
