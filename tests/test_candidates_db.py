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
