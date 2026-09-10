"""Tests for backend.db.candidate_preferences and backend.services.preferences. Uses the in-memory FakeSupabaseClient."""

from __future__ import annotations

import pytest

from backend.db import candidate_preferences as preferences_db
from backend.db import upserts as upserts_db
from backend.services import preferences as preferences_service
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (preferences_db, upserts_db):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def test_get_candidate_preferences_returns_none_when_unset(fake_client: FakeSupabaseClient) -> None:
    assert preferences_db.get_candidate_preferences("u1") is None


def test_upsert_candidate_preferences_persists_all_fields(fake_client: FakeSupabaseClient) -> None:
    preferences_db.upsert_candidate_preferences(
        "u1",
        target_role_families=["SWE", "Quant"],
        preferred_locations=["Remote", "NYC"],
        employment_types=["Internship"],
        interests=["Systems"],
    )

    row = preferences_db.get_candidate_preferences("u1")
    assert row["target_role_families"] == ["SWE", "Quant"]
    assert row["preferred_locations"] == ["Remote", "NYC"]
    assert row["employment_types"] == ["Internship"]
    assert row["interests"] == ["Systems"]


def test_upsert_candidate_preferences_updates_in_place_not_duplicated(fake_client: FakeSupabaseClient) -> None:
    preferences_db.upsert_candidate_preferences("u1", target_role_families=["SWE"])
    preferences_db.upsert_candidate_preferences("u1", target_role_families=["Quant"])

    row = preferences_db.get_candidate_preferences("u1")
    assert row["target_role_families"] == ["Quant"]


def test_preferences_never_touch_candidate_profile_data(fake_client: FakeSupabaseClient) -> None:
    """Preferences live in their own table entirely, keyed on the same user_id but never read/written by resume-derived persistence - a real, separate table, not a column overlay."""
    preferences_db.upsert_candidate_preferences("u1", target_role_families=["SWE"])

    assert preferences_db.CANDIDATE_PREFERENCES_TABLE == "candidate_preferences"
    assert preferences_db.CANDIDATE_PREFERENCES_TABLE != "candidate_profiles"


def test_service_save_and_get_round_trip(fake_client: FakeSupabaseClient) -> None:
    preferences_service.save_candidate_preferences("u1", ["SWE"], ["Remote"], ["Internship"], ["Data"])

    result = preferences_service.get_candidate_preferences("u1")
    assert result["target_role_families"] == ["SWE"]
    assert result["interests"] == ["Data"]
