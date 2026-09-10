"""Tests for backend.db.candidates. Uses the in-memory FakeSupabaseClient for real read-after-write behavior."""

from __future__ import annotations

import pytest

from backend.db import candidates as candidates_db
from backend.db import upserts as upserts_db
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (candidates_db, upserts_db):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def test_upsert_candidate_skill_defaults_evidence_snippets_to_empty_list(fake_client: FakeSupabaseClient) -> None:
    row = candidates_db.upsert_candidate_skill("u1", "python", estimated_level=7.0, confidence=0.7)
    assert row["evidence_snippets"] == []


def test_upsert_candidate_skill_persists_evidence_snippets(fake_client: FakeSupabaseClient) -> None:
    row = candidates_db.upsert_candidate_skill(
        "u1", "python", estimated_level=7.0, confidence=0.7,
        evidence_snippets=["Built Python ETL pipelines", "Skills: Python, C++"],
    )
    assert row["evidence_snippets"] == ["Built Python ETL pipelines", "Skills: Python, C++"]

    [stored] = candidates_db.list_candidate_skills("u1")
    assert stored["evidence_snippets"] == ["Built Python ETL pipelines", "Skills: Python, C++"]


def test_upsert_candidate_skill_evidence_snippets_update_in_place(fake_client: FakeSupabaseClient) -> None:
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=5.0, confidence=0.5, evidence_snippets=["old"])
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=8.0, confidence=0.8, evidence_snippets=["new"])

    [stored] = candidates_db.list_candidate_skills("u1")
    assert stored["evidence_snippets"] == ["new"]


def test_delete_candidate_skill_removes_only_the_named_skill(fake_client: FakeSupabaseClient) -> None:
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=7.0, confidence=0.7)
    candidates_db.upsert_candidate_skill("u1", "java", estimated_level=5.0, confidence=0.5)

    candidates_db.delete_candidate_skill("u1", "java")

    remaining = {row["normalized_skill_name"] for row in candidates_db.list_candidate_skills("u1")}
    assert remaining == {"python"}


def test_delete_candidate_skill_does_not_affect_other_users(fake_client: FakeSupabaseClient) -> None:
    candidates_db.upsert_candidate_skill("u1", "python", estimated_level=7.0, confidence=0.7)
    candidates_db.upsert_candidate_skill("u2", "python", estimated_level=6.0, confidence=0.6)

    candidates_db.delete_candidate_skill("u1", "python")

    assert candidates_db.list_candidate_skills("u1") == []
    assert len(candidates_db.list_candidate_skills("u2")) == 1


def test_upsert_candidate_profile_omits_resume_metadata_keys_when_not_supplied(fake_client: FakeSupabaseClient) -> None:
    """Only including resume_filename/content_hash/parsed_at in the upsert payload when explicitly given means an unrelated profile update (e.g. no filename supplied) never nulls out a previously-saved one."""
    candidates_db.upsert_candidate_profile("u1", resume_filename="resume_v1.pdf", resume_content_hash="hash1", resume_parsed_at="2026-01-01T00:00:00+00:00")
    candidates_db.upsert_candidate_profile("u1", raw_resume_text="updated text")  # no resume metadata this time

    row = candidates_db.get_candidate_profile("u1")
    assert row["resume_filename"] == "resume_v1.pdf"
    assert row["resume_content_hash"] == "hash1"


def test_upsert_candidate_profile_persists_extended_resume_fields(fake_client: FakeSupabaseClient) -> None:
    candidates_db.upsert_candidate_profile(
        "u1",
        coursework=["Algorithms"],
        programming_languages=["Python"],
        frameworks=["FastAPI"],
        tools=["Docker"],
        research=[{"title": "Thesis", "description": "x"}],
        domain_experience=["Trading"],
    )

    row = candidates_db.get_candidate_profile("u1")
    assert row["coursework"] == ["Algorithms"]
    assert row["frameworks"] == ["FastAPI"]
    assert row["research"] == [{"title": "Thesis", "description": "x"}]
    assert row["domain_experience"] == ["Trading"]
