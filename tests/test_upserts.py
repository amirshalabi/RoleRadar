"""
Tests for idempotent upsert behavior in backend/db.

None of these tests hit a live Supabase project. get_client() is either
exercised directly to confirm it fails gracefully without credentials,
or monkeypatched with a small in-memory fake that records the calls
each module makes, so we can assert *what* would be sent to the
database without a live connection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from backend.db import roles, upserts
from backend.db.client import SupabaseNotConfiguredError, get_client
from backend.utils.config import get_settings


class _RecordingBuilder:
    """Records every chained call and returns canned data on execute()."""

    def __init__(self, table_name: str, calls_log: list[tuple], response_data: list[dict[str, Any]]):
        self.table_name = table_name
        self._calls_log = calls_log
        self._response_data = response_data

    def __getattr__(self, name: str):
        def method(*args: Any, **kwargs: Any) -> "_RecordingBuilder":
            self._calls_log.append((self.table_name, name, args, kwargs))
            return self

        return method

    def execute(self) -> SimpleNamespace:
        self._calls_log.append((self.table_name, "execute", (), {}))
        return SimpleNamespace(data=self._response_data)


class FakeSupabaseClient:
    """Minimal stand-in for the Supabase client, used only to test query
    construction (table/conflict targets/filters) without a live database."""

    def __init__(self, response_data: list[dict[str, Any]] | None = None):
        self.calls: list[tuple] = []
        self._response_data = response_data if response_data is not None else []

    def table(self, name: str) -> _RecordingBuilder:
        return _RecordingBuilder(name, self.calls, self._response_data)


@pytest.fixture(autouse=True)
def _clear_caches():
    get_settings.cache_clear()
    get_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_client.cache_clear()


def test_get_client_raises_clearly_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    with pytest.raises(SupabaseNotConfiguredError):
        get_client()


def test_upsert_row_returns_first_row_and_uses_on_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeSupabaseClient(response_data=[{"id": "1", "external_id": "abc"}])
    monkeypatch.setattr(upserts, "get_client", lambda: fake_client)

    result = upserts.upsert_row("roles", {"external_id": "abc"}, on_conflict="external_id")

    assert result == {"id": "1", "external_id": "abc"}
    assert (
        "roles",
        "upsert",
        ({"external_id": "abc"},),
        {"on_conflict": "external_id"},
    ) in fake_client.calls


def test_upsert_row_raises_when_no_data_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeSupabaseClient(response_data=[])
    monkeypatch.setattr(upserts, "get_client", lambda: fake_client)

    with pytest.raises(RuntimeError):
        upserts.upsert_row("roles", {"external_id": "abc"}, on_conflict="external_id")


def test_upsert_role_repeated_calls_produce_same_external_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core idempotency guarantee: re-ingesting the same logical role
    (even with different casing/whitespace) must upsert onto one row."""
    captured_external_ids: list[str] = []

    def fake_upsert_row(table: str, values: dict, on_conflict: str) -> dict:
        assert table == roles.ROLES_TABLE
        assert on_conflict == "external_id"
        captured_external_ids.append(values["external_id"])
        return {**values, "id": "role-1"}

    monkeypatch.setattr(roles, "upsert_row", fake_upsert_row)

    roles.upsert_role(company="Acme Corp", title="Backend Engineer", location="Remote")
    roles.upsert_role(company="  acme   corp ", title="BACKEND engineer", location="remote")

    assert len(captured_external_ids) == 2
    assert captured_external_ids[0] == captured_external_ids[1]


def test_upsert_role_respects_explicit_external_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_upsert_row(table: str, values: dict, on_conflict: str) -> dict:
        captured.update(values)
        return values

    monkeypatch.setattr(roles, "upsert_row", fake_upsert_row)

    roles.upsert_role(company="Acme", title="SWE Intern", external_id="provider-123")

    assert captured["external_id"] == "provider-123"


def test_upsert_fit_score_keys_on_user_and_role(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_upsert_row(table: str, values: dict, on_conflict: str) -> dict:
        captured["table"] = table
        captured["on_conflict"] = on_conflict
        captured["values"] = values
        return values

    monkeypatch.setattr(roles, "upsert_row", fake_upsert_row)

    roles.upsert_fit_score(user_id="u1", role_id="r1", overall_score=82.5)

    assert captured["table"] == roles.FIT_SCORES_TABLE
    assert captured["on_conflict"] == "user_id,role_id"
    assert captured["values"]["user_id"] == "u1"
    assert captured["values"]["role_id"] == "r1"



# Favorites- and applications-specific behavior now lives in
# tests/test_favorites.py and tests/test_applications.py, since those
# modules moved from a blind Postgres upsert to a check-then-insert /
# update-if-exists pattern (see their module docstrings for why) that
# doesn't fit this file's "generic upsert_row behavior" scope.
