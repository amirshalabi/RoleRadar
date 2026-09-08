"""
Automated Streamlit page tests (streamlit.testing.v1.AppTest) for the
"nothing set up yet" empty states: no resume uploaded, no jobs ingested,
no favorites saved, no interview scheduled.

Each test runs the page's REAL script end to end (not a mock of the UI)
against the same in-memory FakeSupabaseClient the rest of the suite
uses. Demo mode is off by default (AppTest's session_state starts
empty, so ui_common.is_demo_mode() reads its default of False), so
these exercise the actual "connected but no data yet" branch a brand
new real user would hit - not the demo-mode sample-data branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

from backend.db import applications as applications_db
from backend.db import candidates as candidates_db
from backend.db import favorites as favorites_db
from backend.db import rationales as rationales_db
from backend.db import role_requirements as role_requirements_db
from backend.db import roles as roles_db
from backend.db import upserts as upserts_db
from backend.db import users as users_db
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (
        roles_db, candidates_db, favorites_db, applications_db,
        role_requirements_db, rationales_db, upserts_db, users_db,
    ):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def _markdown_texts(app_test: AppTest) -> list[str]:
    return [element.value for element in app_test.markdown]


def test_discover_page_empty_state_when_no_jobs(fake_client: FakeSupabaseClient) -> None:
    at = AppTest.from_file(str(_PAGES_DIR / "1_Discover.py")).run(timeout=30)

    assert not at.exception
    assert any("No roles ingested yet" in text for text in _markdown_texts(at))


def test_favorites_page_empty_state_when_no_favorites(fake_client: FakeSupabaseClient) -> None:
    at = AppTest.from_file(str(_PAGES_DIR / "2_Favorites.py")).run(timeout=30)

    assert not at.exception
    assert any("No saved roles yet" in text for text in _markdown_texts(at))


def test_skill_gaps_page_empty_state_when_no_resume_uploaded(fake_client: FakeSupabaseClient) -> None:
    at = AppTest.from_file(str(_PAGES_DIR / "4_Skill_Gaps.py")).run(timeout=30)

    assert not at.exception
    assert any("No candidate skills tracked yet" in text for text in _markdown_texts(at))


def test_interview_prep_page_empty_state_when_no_interview_scheduled(fake_client: FakeSupabaseClient) -> None:
    at = AppTest.from_file(str(_PAGES_DIR / "5_Interview_Prep.py")).run(timeout=30)

    assert not at.exception
    assert any("No applications with an interview date yet" in text for text in _markdown_texts(at))


def test_discover_page_shows_database_not_configured_notice_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """No fake client patched in at all - simulates SUPABASE_URL/KEY genuinely unset."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    from backend.utils.config import get_settings

    get_settings.cache_clear()

    at = AppTest.from_file(str(_PAGES_DIR / "1_Discover.py")).run(timeout=30)

    assert not at.exception
    assert any("Database not connected" in w.value for w in at.warning)
    get_settings.cache_clear()
