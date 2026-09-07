"""Tests for backend.db.role_requirements. Uses the in-memory FakeSupabaseClient for real read-after-write behavior."""

from __future__ import annotations

import pytest

from backend.db import role_requirements as role_requirements_db
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    monkeypatch.setattr(role_requirements_db, "get_client", lambda: client)
    return client


def _requirement(skill="python", target=7.0, importance=8.0, required=True):
    return {
        "normalized_skill_name": skill,
        "display_name": skill.title(),
        "target_level": target,
        "importance": importance,
        "is_required": required,
        "evidence": ["Proficiency required."],
    }


def test_list_role_requirements_empty_when_none_extracted(fake_client: FakeSupabaseClient) -> None:
    assert role_requirements_db.list_role_requirements("role-1") == []


def test_upsert_role_requirements_persists_all(fake_client: FakeSupabaseClient) -> None:
    rows = role_requirements_db.upsert_role_requirements(
        "role-1", [_requirement("python"), _requirement("c++", target=6.0)]
    )

    assert len(rows) == 2
    assert {row["normalized_skill_name"] for row in rows} == {"python", "c++"}
    stored = role_requirements_db.list_role_requirements("role-1")
    assert len(stored) == 2


def test_upsert_role_requirements_empty_list_is_a_noop(fake_client: FakeSupabaseClient) -> None:
    assert role_requirements_db.upsert_role_requirements("role-1", []) == []
    assert role_requirements_db.list_role_requirements("role-1") == []


def test_upsert_role_requirements_idempotent_on_repeat_extraction(fake_client: FakeSupabaseClient) -> None:
    """Re-extracting the same role's requirements must update in place, never duplicate."""
    role_requirements_db.upsert_role_requirements("role-1", [_requirement("python", target=6.0)])
    role_requirements_db.upsert_role_requirements("role-1", [_requirement("python", target=8.0)])

    stored = role_requirements_db.list_role_requirements("role-1")
    assert len(stored) == 1
    assert stored[0]["target_level"] == 8.0


def test_requirements_scoped_per_role(fake_client: FakeSupabaseClient) -> None:
    role_requirements_db.upsert_role_requirements("role-1", [_requirement("python")])
    role_requirements_db.upsert_role_requirements("role-2", [_requirement("java")])

    assert len(role_requirements_db.list_role_requirements("role-1")) == 1
    assert len(role_requirements_db.list_role_requirements("role-2")) == 1


def test_evidence_round_trips_through_persistence(fake_client: FakeSupabaseClient) -> None:
    [row] = role_requirements_db.upsert_role_requirements("role-1", [_requirement("python")])

    assert row["evidence"] == ["Proficiency required."]
