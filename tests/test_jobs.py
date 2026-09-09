"""
Tests for backend.ingestion.jobs (source adapter interface, demo
adapters, per-source fetch+normalize). Uses tiny simulated latencies so
these run fast while still exercising real time.sleep()-based I/O.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.ingestion import jobs as jobs_module
from backend.ingestion.jobs import (
    AdzunaNotConfiguredError,
    AdzunaSourceAdapter,
    DemoAPISourceAdapter,
    DemoBoardSourceAdapter,
    DemoUnreliableSourceAdapter,
    MalformedRecordError,
    SourceAdapter,
    fetch_and_normalize_source,
)
from backend.ingestion.normalize import Role
from backend.utils.config import get_settings


def test_source_adapter_is_abstract() -> None:
    import pytest

    with pytest.raises(TypeError):
        SourceAdapter()  # type: ignore[abstract]


def test_demo_api_adapter_returns_raw_records() -> None:
    adapter = DemoAPISourceAdapter(simulated_latency_seconds=0.01)

    raw_jobs = adapter.fetch_raw_jobs()

    assert len(raw_jobs) == 2
    assert all("jobTitle" in job and "companyName" in job for job in raw_jobs)


def test_demo_board_adapter_includes_one_malformed_record() -> None:
    adapter = DemoBoardSourceAdapter(simulated_latency_seconds=0.01)

    raw_jobs = adapter.fetch_raw_jobs()

    assert len(raw_jobs) == 2
    malformed = [job for job in raw_jobs if "title" not in job and "company" not in job]
    assert len(malformed) == 1


def test_demo_unreliable_adapter_raises_when_configured_to_fail() -> None:
    import pytest

    adapter = DemoUnreliableSourceAdapter(simulated_latency_seconds=0.01, fail=True)

    with pytest.raises(RuntimeError):
        adapter.fetch_raw_jobs()


def test_demo_unreliable_adapter_succeeds_when_not_configured_to_fail() -> None:
    adapter = DemoUnreliableSourceAdapter(simulated_latency_seconds=0.01, fail=False)

    raw_jobs = adapter.fetch_raw_jobs()

    assert len(raw_jobs) == 1


# ---------------------------------------------------------------------
# AdzunaSourceAdapter
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_adzuna_response(results: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"results": results},
    )


def test_adzuna_adapter_raises_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
    adapter = AdzunaSourceAdapter()

    with pytest.raises(AdzunaNotConfiguredError):
        adapter.fetch_raw_jobs()


def test_adzuna_adapter_flattens_nested_company_and_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADZUNA_APP_ID", "test-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test-key")
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        return _fake_adzuna_response(
            [
                {
                    "id": 123,
                    "title": "Software Engineer Intern",
                    "company": {"display_name": "Acme Corp"},
                    "location": {"display_name": "New York, NY"},
                    "description": "Build things.",
                    "redirect_url": "https://adzuna.example/jobs/123",
                }
            ]
        )

    monkeypatch.setattr(jobs_module.httpx, "get", fake_get)

    raw_jobs = AdzunaSourceAdapter().fetch_raw_jobs()

    assert raw_jobs == [
        {
            "external_id": "123",
            "title": "Software Engineer Intern",
            "company": "Acme Corp",
            "location": "New York, NY",
            "description": "Build things.",
            "url": "https://adzuna.example/jobs/123",
        }
    ]
    assert captured["params"]["app_id"] == "test-id"
    assert captured["params"]["app_key"] == "test-key"


def test_adzuna_adapter_normalizes_into_a_valid_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flattened dict this adapter produces must be exactly what normalize_role() already expects - no downstream changes needed."""
    monkeypatch.setenv("ADZUNA_APP_ID", "test-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test-key")
    monkeypatch.setattr(
        jobs_module.httpx, "get",
        lambda url, params, timeout: _fake_adzuna_response(
            [{"id": 1, "title": "Data Intern", "company": {"display_name": "Acme"}, "location": {"display_name": "Remote"}, "description": "x", "redirect_url": "https://x"}]
        ),
    )

    roles, malformed = fetch_and_normalize_source(AdzunaSourceAdapter())

    assert malformed == []
    assert len(roles) == 1
    assert isinstance(roles[0], Role)
    assert roles[0].title == "Data Intern"
    assert roles[0].company == "Acme"


def test_adzuna_adapter_handles_missing_id_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADZUNA_APP_ID", "test-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test-key")
    monkeypatch.setattr(
        jobs_module.httpx, "get",
        lambda url, params, timeout: _fake_adzuna_response(
            [{"title": "Intern", "company": {"display_name": "Acme"}, "location": {}, "description": None, "redirect_url": None}]
        ),
    )

    [raw_job] = AdzunaSourceAdapter().fetch_raw_jobs()

    assert raw_job["external_id"] is None  # falls back to a derived external_id in normalize_role()
    assert raw_job["location"] is None


def test_adzuna_adapter_empty_results_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADZUNA_APP_ID", "test-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test-key")
    monkeypatch.setattr(jobs_module.httpx, "get", lambda url, params, timeout: _fake_adzuna_response([]))

    assert AdzunaSourceAdapter().fetch_raw_jobs() == []


def test_adzuna_adapter_propagates_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADZUNA_APP_ID", "test-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test-key")

    def _raise_for_status():
        raise jobs_module.httpx.HTTPStatusError("boom", request=None, response=None)

    monkeypatch.setattr(
        jobs_module.httpx, "get",
        lambda url, params, timeout: SimpleNamespace(raise_for_status=_raise_for_status, json=lambda: {}),
    )

    with pytest.raises(jobs_module.httpx.HTTPStatusError):
        AdzunaSourceAdapter().fetch_raw_jobs()


# ---------------------------------------------------------------------
# fetch_and_normalize_source
# ---------------------------------------------------------------------


class _FixedAdapter(SourceAdapter):
    """A minimal test-only adapter returning a fixed, pre-set list of raw records."""

    def __init__(self, name: str, jobs: list[dict]):
        self.name = name
        self._jobs = jobs

    def fetch_raw_jobs(self) -> list[dict]:
        return self._jobs


def test_fetch_and_normalize_source_normalizes_valid_records() -> None:
    adapter = _FixedAdapter("test_source", [{"title": "SWE Intern", "company": "Acme"}])

    roles, malformed = fetch_and_normalize_source(adapter)

    assert len(roles) == 1
    assert isinstance(roles[0], Role)
    assert roles[0].title == "SWE Intern"
    assert malformed == []


def test_fetch_and_normalize_source_separates_malformed_records() -> None:
    adapter = _FixedAdapter(
        "test_source",
        [
            {"title": "SWE Intern", "company": "Acme"},
            {"description": "missing title and company"},
        ],
    )

    roles, malformed = fetch_and_normalize_source(adapter)

    assert len(roles) == 1
    assert len(malformed) == 1
    assert isinstance(malformed[0], MalformedRecordError)
    assert malformed[0].source_name == "test_source"
    assert malformed[0].raw_record == {"description": "missing title and company"}


def test_fetch_and_normalize_source_all_malformed_returns_no_roles() -> None:
    adapter = _FixedAdapter("test_source", [{"description": "x"}, {"description": "y"}])

    roles, malformed = fetch_and_normalize_source(adapter)

    assert roles == []
    assert len(malformed) == 2


def test_fetch_and_normalize_source_empty_source_returns_empty() -> None:
    adapter = _FixedAdapter("test_source", [])

    roles, malformed = fetch_and_normalize_source(adapter)

    assert roles == []
    assert malformed == []


def test_fetch_and_normalize_source_propagates_whole_source_failure() -> None:
    """Whole-source failure (the fetch call itself raising) is NOT caught here - see backend.ingestion.concurrent."""
    import pytest

    class _FailingAdapter(SourceAdapter):
        name = "failing"

        def fetch_raw_jobs(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fetch_and_normalize_source(_FailingAdapter())
