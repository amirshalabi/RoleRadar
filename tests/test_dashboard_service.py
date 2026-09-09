"""
Tests for backend.services.dashboard. Uses the in-memory
FakeSupabaseClient for real read sequences, and directly exercises the
pure helper functions (_find_upcoming_interview, _find_urgent_applications)
with hand-built data for precise control over edge cases.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.db import applications as applications_db
from backend.db import candidates as candidates_db
from backend.db import favorites as favorites_db
from backend.db import metrics as metrics_db
from backend.db import roles as roles_db
from backend.db.client import get_client
from backend.services import dashboard
from backend.services import tracking
from backend.utils.config import get_settings
from backend.utils.metrics import PipelineMetrics
from tests._fake_supabase import FakeSupabaseClient

TODAY = date(2026, 9, 7)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (candidates_db, roles_db, applications_db, favorites_db, metrics_db):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def _app_row(role_id="r1", status="applied", interview_date=None, deadline=None, title="Role", company="Co", role_family=None):
    return {
        "role_id": role_id,
        "status": status,
        "interview_date": interview_date,
        "deadline": deadline,
        "roles": {"title": title, "company": company, "role_family": role_family},
    }


# ---------------------------------------------------------------------
# _find_upcoming_interview
# ---------------------------------------------------------------------


def test_find_upcoming_interview_picks_nearest_future_date() -> None:
    apps = [
        _app_row(role_id="r1", interview_date="2026-09-20", title="Far"),
        _app_row(role_id="r2", interview_date="2026-09-10", title="Near"),
    ]

    result = dashboard._find_upcoming_interview(apps, TODAY)

    assert result.role_title == "Near"
    assert result.days_away == 3


def test_find_upcoming_interview_ignores_past_dates() -> None:
    apps = [_app_row(interview_date="2026-09-01")]

    assert dashboard._find_upcoming_interview(apps, TODAY) is None


def test_find_upcoming_interview_none_when_no_dates_set() -> None:
    apps = [_app_row(interview_date=None)]

    assert dashboard._find_upcoming_interview(apps, TODAY) is None


def test_find_upcoming_interview_includes_role_family() -> None:
    apps = [_app_row(interview_date="2026-09-10", role_family="quant")]

    result = dashboard._find_upcoming_interview(apps, TODAY)

    assert result.role_family == "quant"


# ---------------------------------------------------------------------
# _find_urgent_applications
# ---------------------------------------------------------------------


def test_urgent_applications_includes_near_deadlines() -> None:
    apps = [_app_row(status="applied", deadline="2026-09-14")]  # 7 days out

    result = dashboard._find_urgent_applications(apps, TODAY)

    assert len(result) == 1
    assert result[0].days_until_deadline == 7


def test_urgent_applications_excludes_far_deadlines() -> None:
    apps = [_app_row(status="applied", deadline="2026-12-01")]

    assert dashboard._find_urgent_applications(apps, TODAY) == []


def test_urgent_applications_excludes_past_deadlines() -> None:
    apps = [_app_row(status="applied", deadline="2026-09-01")]

    assert dashboard._find_urgent_applications(apps, TODAY) == []


def test_urgent_applications_excludes_terminal_statuses() -> None:
    apps = [
        _app_row(status="rejected", deadline="2026-09-10"),
        _app_row(status="withdrawn", deadline="2026-09-10"),
        _app_row(status="offer", deadline="2026-09-10"),
    ]

    assert dashboard._find_urgent_applications(apps, TODAY) == []


def test_urgent_applications_includes_interview_stage_without_deadline() -> None:
    apps = [_app_row(status="interview", deadline=None)]

    result = dashboard._find_urgent_applications(apps, TODAY)

    assert len(result) == 1
    assert result[0].days_until_deadline is None


def test_urgent_applications_excludes_no_deadline_non_interview() -> None:
    apps = [_app_row(status="applied", deadline=None)]

    assert dashboard._find_urgent_applications(apps, TODAY) == []


def test_urgent_applications_sorted_by_days_remaining() -> None:
    apps = [
        _app_row(role_id="r1", status="applied", deadline="2026-09-18", title="Later"),
        _app_row(role_id="r2", status="applied", deadline="2026-09-09", title="Sooner"),
    ]

    result = dashboard._find_urgent_applications(apps, TODAY)

    assert [r.role_title for r in result] == ["Sooner", "Later"]


def test_urgent_applications_capped_at_five() -> None:
    apps = [_app_row(role_id=f"r{i}", status="applied", deadline="2026-09-10") for i in range(10)]

    result = dashboard._find_urgent_applications(apps, TODAY)

    assert len(result) == 5


# ---------------------------------------------------------------------
# _build_pipeline_stats
# ---------------------------------------------------------------------


def test_build_pipeline_stats_none_when_no_run() -> None:
    assert dashboard._build_pipeline_stats(None) is None


def test_build_pipeline_stats_computes_avoidance_rate_via_real_pipeline_metrics() -> None:
    run = {
        "jobs_ingested": 100, "jobs_deduplicated": 10, "jobs_eliminated_hard_filter": 50,
        "jobs_eliminated_keyword_filter": 30, "jobs_eliminated_semantic": 0, "jobs_reaching_llm": 10,
        "finished_at": "2026-09-07T00:00:00Z",
    }

    stats = dashboard._build_pipeline_stats(run)

    expected = PipelineMetrics(ingested=100, deduplicated=10, hard_filtered=50, keyword_filtered=30, llm_analyzed=10)
    assert stats.llm_avoidance_rate == expected.llm_avoidance_rate()
    assert stats.ingested == 100


def test_build_pipeline_stats_handles_missing_fields_as_zero() -> None:
    stats = dashboard._build_pipeline_stats({"finished_at": None})

    assert stats.ingested == 0
    assert stats.llm_avoidance_rate is None


# ---------------------------------------------------------------------
# build_dashboard_data: graceful degradation + real end-to-end assembly
# ---------------------------------------------------------------------


def test_build_dashboard_data_without_credentials_reports_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Simulates missing Supabase credentials by clearing the env vars and
    the get_settings()/get_client() lru_caches - not by relying on the
    ambient environment actually lacking real credentials, since a
    developer's .env may have live ones configured.
    """
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    get_settings.cache_clear()
    get_client.cache_clear()
    try:
        result = dashboard.build_dashboard_data("some-user")

        assert result.database_connected is False
        assert result.profile_status.has_profile is False
        assert result.top_match is None
    finally:
        get_settings.cache_clear()
        get_client.cache_clear()


def test_build_dashboard_data_empty_account_shows_empty_states(fake_client: FakeSupabaseClient) -> None:
    result = dashboard.build_dashboard_data("u1")

    assert result.database_connected is True
    assert result.profile_status.has_profile is False
    assert result.top_match is None
    assert result.upcoming_interview is None
    assert result.overall_readiness is None
    assert result.saved_roles_count == 0
    assert result.urgent_applications == []


def test_build_dashboard_data_reflects_real_favorites_and_applications(fake_client: FakeSupabaseClient) -> None:
    tracking.save_role("u1", "role-1", priority="dream")
    tracking.update_application_stage("u1", "role-1", status="applied", deadline="2026-09-14")

    result = dashboard.build_dashboard_data("u1")

    assert result.saved_roles_count == 1
    assert result.active_applications_count == 1


# ---------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------


def test_demo_dashboard_data_is_flagged_as_demo() -> None:
    result = dashboard.build_demo_dashboard_data()

    assert result.is_demo is True
    assert result.database_connected is True


def test_demo_dashboard_data_populates_every_section() -> None:
    result = dashboard.build_demo_dashboard_data()

    assert result.profile_status.has_profile is True
    assert result.top_match is not None
    assert result.upcoming_interview is not None
    assert result.overall_readiness is not None
    assert result.highest_leverage_skill is not None
    assert result.pipeline_stats is not None


def test_demo_dashboard_data_dates_are_internally_consistent() -> None:
    result = dashboard.build_demo_dashboard_data()

    interview_date = date.fromisoformat(result.upcoming_interview.interview_date)
    assert (interview_date - date.today()).days == result.upcoming_interview.days_away

    urgent = result.urgent_applications[0]
    deadline = date.fromisoformat(urgent.deadline)
    assert (deadline - date.today()).days == urgent.days_until_deadline
