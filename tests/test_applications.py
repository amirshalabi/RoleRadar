"""
Tests for backend.db.applications. Uses the in-memory FakeSupabaseClient
(tests/_fake_supabase.py) so real create-then-update sequences behave
like a real table would, without a live Supabase project.
"""

from __future__ import annotations

import pytest

from backend.db import applications
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    monkeypatch.setattr(applications, "get_client", lambda: client)
    return client


def test_upsert_application_rejects_invalid_status(fake_client: FakeSupabaseClient) -> None:
    with pytest.raises(ValueError):
        applications.upsert_application(user_id="u1", role_id="r1", status="not-a-real-stage")


def test_upsert_application_creates_row_with_default_status_when_status_omitted(
    fake_client: FakeSupabaseClient,
) -> None:
    result = applications.upsert_application(user_id="u1", role_id="r1")

    assert result["status"] == applications.DEFAULT_STATUS


def test_update_application_stage_advances_status(fake_client: FakeSupabaseClient) -> None:
    applications.upsert_application(user_id="u1", role_id="r1", status="applied")

    result = applications.upsert_application(user_id="u1", role_id="r1", status="interview")

    assert result["status"] == "interview"
    assert len(fake_client._tables[applications.APPLICATIONS_TABLE].rows) == 1  # updated in place, not duplicated


def test_stage_transition_does_not_erase_previously_set_notes(fake_client: FakeSupabaseClient) -> None:
    applications.upsert_application(user_id="u1", role_id="r1", status="applied", notes="Referred by a friend")

    result = applications.upsert_application(user_id="u1", role_id="r1", status="oa")

    assert result["status"] == "oa"
    assert result["notes"] == "Referred by a friend"


def test_interview_date_persists_across_later_unrelated_updates(fake_client: FakeSupabaseClient) -> None:
    applications.upsert_application(user_id="u1", role_id="r1", status="interview", interview_date="2026-03-15")

    # A later call that only touches notes should not disturb interview_date.
    result = applications.upsert_application(user_id="u1", role_id="r1", notes="Prep DS&A and behavioral")

    assert result["interview_date"] == "2026-03-15"
    assert result["notes"] == "Prep DS&A and behavioral"
    assert result["status"] == "interview"  # also untouched


def test_interview_date_persists_after_being_set_before_status_existed(fake_client: FakeSupabaseClient) -> None:
    applications.upsert_application(user_id="u1", role_id="r1", interview_date="2026-04-01")

    result = applications.upsert_application(user_id="u1", role_id="r1", status="interview")

    assert result["interview_date"] == "2026-04-01"
    assert result["status"] == "interview"


def test_application_date_and_deadline_are_independently_settable(fake_client: FakeSupabaseClient) -> None:
    applications.upsert_application(user_id="u1", role_id="r1", application_date="2026-01-10")
    result = applications.upsert_application(user_id="u1", role_id="r1", deadline="2026-01-31")

    assert result["application_date"] == "2026-01-10"
    assert result["deadline"] == "2026-01-31"


def test_get_application_returns_none_when_not_found(fake_client: FakeSupabaseClient) -> None:
    assert applications.get_application(user_id="u1", role_id="never-tracked") is None


def test_list_applications_returns_only_that_users_rows(fake_client: FakeSupabaseClient) -> None:
    applications.upsert_application(user_id="u1", role_id="r1")
    applications.upsert_application(user_id="u2", role_id="r2")

    result = applications.list_applications(user_id="u1")

    assert [a["role_id"] for a in result] == ["r1"]


def test_repeated_upserts_never_create_duplicate_rows(fake_client: FakeSupabaseClient) -> None:
    for status in ("discovered", "saved", "applied", "oa", "interview", "offer"):
        applications.upsert_application(user_id="u1", role_id="r1", status=status)

    assert len(fake_client._tables[applications.APPLICATIONS_TABLE].rows) == 1
