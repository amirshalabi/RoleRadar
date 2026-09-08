"""Tests for backend.db.assessment_results. Uses the in-memory FakeSupabaseClient for real read-after-write behavior."""

from __future__ import annotations

import pytest

from backend.db import assessment_results as assessment_results_db
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    monkeypatch.setattr(assessment_results_db, "get_client", lambda: client)
    return client


def test_list_assessment_results_empty_when_none_taken(fake_client: FakeSupabaseClient) -> None:
    assert assessment_results_db.list_assessment_results("u1") == []


def test_save_assessment_result_persists_topic_and_confidence(fake_client: FakeSupabaseClient) -> None:
    row = assessment_results_db.save_assessment_result("u1", "probability", observed_level=7.0, confidence=0.9)

    assert row["normalized_skill_name"] == "probability"
    assert row["observed_level"] == 7.0
    assert row["confidence"] == 0.9


def test_save_assessment_result_never_overwrites_a_prior_attempt(fake_client: FakeSupabaseClient) -> None:
    assessment_results_db.save_assessment_result("u1", "probability", observed_level=5.0, confidence=0.8)
    assessment_results_db.save_assessment_result("u1", "probability", observed_level=7.0, confidence=0.9)

    results = assessment_results_db.list_assessment_results("u1")
    assert len(results) == 2


def test_list_assessment_results_scoped_per_user(fake_client: FakeSupabaseClient) -> None:
    assessment_results_db.save_assessment_result("u1", "probability", observed_level=5.0)
    assessment_results_db.save_assessment_result("u2", "algorithms", observed_level=6.0)

    assert len(assessment_results_db.list_assessment_results("u1")) == 1
    assert len(assessment_results_db.list_assessment_results("u2")) == 1
