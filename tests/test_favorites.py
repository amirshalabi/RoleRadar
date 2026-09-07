"""
Tests for backend.db.favorites. Uses the in-memory FakeSupabaseClient
(tests/_fake_supabase.py) so real create-then-read-then-update
sequences behave like a real table would, without a live Supabase
project.
"""

from __future__ import annotations

import pytest

from backend.db import favorites
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    monkeypatch.setattr(favorites, "get_client", lambda: client)
    return client


def test_save_favorite_twice_creates_only_one_logical_favorite(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1")
    favorites.save_favorite(user_id="u1", role_id="r1")

    all_favorites = fake_client._tables[favorites.FAVORITES_TABLE].rows
    assert len(all_favorites) == 1


def test_save_favorite_repeat_call_does_not_reset_priority(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1", priority="dream")
    result = favorites.save_favorite(user_id="u1", role_id="r1", priority="interested")

    assert result["priority"] == "dream"  # first save wins; repeat save doesn't clobber it


def test_save_favorite_rejects_invalid_priority(fake_client: FakeSupabaseClient) -> None:
    with pytest.raises(ValueError):
        favorites.save_favorite(user_id="u1", role_id="r1", priority="urgent")


def test_save_favorite_different_roles_creates_separate_rows(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1")
    favorites.save_favorite(user_id="u1", role_id="r2")

    assert len(fake_client._tables[favorites.FAVORITES_TABLE].rows) == 2


def test_remove_favorite_deletes_the_row(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1")

    favorites.remove_favorite(user_id="u1", role_id="r1")

    assert favorites.get_favorite(user_id="u1", role_id="r1") is None
    assert fake_client._tables[favorites.FAVORITES_TABLE].rows == []


def test_remove_favorite_on_nonexistent_favorite_is_a_safe_no_op(fake_client: FakeSupabaseClient) -> None:
    favorites.remove_favorite(user_id="u1", role_id="does-not-exist")  # should not raise


def test_remove_favorite_only_removes_the_targeted_role(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1")
    favorites.save_favorite(user_id="u1", role_id="r2")

    favorites.remove_favorite(user_id="u1", role_id="r1")

    remaining = favorites.list_favorites(user_id="u1")
    assert [f["role_id"] for f in remaining] == ["r2"]


def test_update_favorite_priority_changes_existing_favorite(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1", priority="interested")

    result = favorites.update_favorite_priority(user_id="u1", role_id="r1", priority="dream")

    assert result["priority"] == "dream"
    assert favorites.get_favorite(user_id="u1", role_id="r1")["priority"] == "dream"


def test_update_favorite_priority_rejects_invalid_value(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1")

    with pytest.raises(ValueError):
        favorites.update_favorite_priority(user_id="u1", role_id="r1", priority="not-a-real-priority")


def test_update_favorite_priority_raises_if_not_favorited(fake_client: FakeSupabaseClient) -> None:
    with pytest.raises(ValueError):
        favorites.update_favorite_priority(user_id="u1", role_id="never-saved", priority="dream")


def test_update_favorite_notes_does_not_change_priority(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1", priority="high")

    result = favorites.update_favorite_notes(user_id="u1", role_id="r1", notes="Great culture fit")

    assert result["notes"] == "Great culture fit"
    assert result["priority"] == "high"


def test_list_favorites_returns_only_that_users_rows(fake_client: FakeSupabaseClient) -> None:
    favorites.save_favorite(user_id="u1", role_id="r1")
    favorites.save_favorite(user_id="u2", role_id="r2")

    result = favorites.list_favorites(user_id="u1")

    assert [f["role_id"] for f in result] == ["r1"]
