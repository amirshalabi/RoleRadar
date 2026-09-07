"""
Tests for backend.ingestion.jobs (source adapter interface, demo
adapters, per-source fetch+normalize). Uses tiny simulated latencies so
these run fast while still exercising real time.sleep()-based I/O.
"""

from __future__ import annotations

from backend.ingestion.jobs import (
    DemoAPISourceAdapter,
    DemoBoardSourceAdapter,
    DemoUnreliableSourceAdapter,
    MalformedRecordError,
    SourceAdapter,
    fetch_and_normalize_source,
)
from backend.ingestion.normalize import Role


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
