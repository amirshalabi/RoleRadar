"""
Tests for backend.services.tracking - the cross-table orchestration a
Streamlit page would call. Both backend.db.favorites and
backend.db.applications are pointed at the SAME in-memory
FakeSupabaseClient instance, so this exercises the real interaction
between the two tables (e.g. favoriting bumping application status)
without a live Supabase project.
"""

from __future__ import annotations

import pytest

from backend.db import applications, favorites
from backend.services import tracking
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    monkeypatch.setattr(favorites, "get_client", lambda: client)
    monkeypatch.setattr(applications, "get_client", lambda: client)
    return client


def test_save_role_creates_favorite_and_marks_application_saved(fake_client: FakeSupabaseClient) -> None:
    tracking.save_role(user_id="u1", role_id="r1")

    assert tracking.list_saved_roles("u1")[0]["role_id"] == "r1"
    assert tracking.get_application_status("u1", "r1")["status"] == "saved"


def test_save_role_does_not_create_duplicate_rows_when_called_twice(fake_client: FakeSupabaseClient) -> None:
    tracking.save_role(user_id="u1", role_id="r1")
    tracking.save_role(user_id="u1", role_id="r1")

    assert len(fake_client._tables[favorites.FAVORITES_TABLE].rows) == 1
    assert len(fake_client._tables[applications.APPLICATIONS_TABLE].rows) == 1


def test_save_role_does_not_downgrade_an_application_already_further_along(
    fake_client: FakeSupabaseClient,
) -> None:
    tracking.update_application_stage("u1", "r1", status="applied")

    tracking.save_role(user_id="u1", role_id="r1")

    assert tracking.get_application_status("u1", "r1")["status"] == "applied"


def test_unsave_role_removes_favorite_but_preserves_application(fake_client: FakeSupabaseClient) -> None:
    tracking.save_role(user_id="u1", role_id="r1")
    tracking.update_application_stage("u1", "r1", status="applied")

    tracking.unsave_role(user_id="u1", role_id="r1")

    assert tracking.list_saved_roles("u1") == []
    assert tracking.get_application_status("u1", "r1")["status"] == "applied"


def test_set_role_priority_updates_only_that_favorite(fake_client: FakeSupabaseClient) -> None:
    tracking.save_role(user_id="u1", role_id="r1", priority="interested")

    result = tracking.set_role_priority("u1", "r1", "dream")

    assert result["priority"] == "dream"


def test_set_role_notes_updates_notes(fake_client: FakeSupabaseClient) -> None:
    tracking.save_role(user_id="u1", role_id="r1")

    result = tracking.set_role_notes("u1", "r1", "Recruiter reached out directly")

    assert result["notes"] == "Recruiter reached out directly"


def test_update_application_stage_preserves_interview_date_across_calls(fake_client: FakeSupabaseClient) -> None:
    tracking.update_application_stage("u1", "r1", status="interview", interview_date="2026-05-01")

    tracking.update_application_stage("u1", "r1", notes="Onsite scheduled")

    result = tracking.get_application_status("u1", "r1")
    assert result["interview_date"] == "2026-05-01"
    assert result["notes"] == "Onsite scheduled"
    assert result["status"] == "interview"


def test_list_applications_returns_all_stages_for_user(fake_client: FakeSupabaseClient) -> None:
    tracking.save_role(user_id="u1", role_id="r1")
    tracking.update_application_stage("u1", "r2", status="offer")

    result = tracking.list_applications("u1")

    statuses = {row["role_id"]: row["status"] for row in result}
    assert statuses == {"r1": "saved", "r2": "offer"}
