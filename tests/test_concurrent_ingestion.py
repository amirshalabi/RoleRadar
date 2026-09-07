"""
Tests for backend.ingestion.concurrent. These measure REAL elapsed
time (small but nonzero time.sleep()-based adapters) rather than
mocking away the concurrency being tested - the whole point of this
module is genuine wall-clock behavior, so faking the clock here would
test nothing.
"""

from __future__ import annotations

import time

import pytest

from backend.db.client import SupabaseNotConfiguredError
from backend.ingestion import concurrent as concurrent_module
from backend.ingestion.concurrent import (
    concurrent_ingest,
    run_ingestion_pipeline,
    serial_ingest,
)
from backend.ingestion.jobs import SourceAdapter


class _FixedAdapter(SourceAdapter):
    """Test-only adapter with a controllable sleep, return value, and failure mode."""

    def __init__(self, name: str, jobs: list[dict] | None = None, sleep: float = 0.0, error: Exception | None = None):
        self.name = name
        self._jobs = jobs if jobs is not None else [{"title": f"{name} role", "company": "TestCo"}]
        self._sleep = sleep
        self._error = error

    def fetch_raw_jobs(self) -> list[dict]:
        if self._sleep:
            time.sleep(self._sleep)
        if self._error:
            raise self._error
        return self._jobs


# ---------------------------------------------------------------------
# serial_ingest / concurrent_ingest: basic correctness
# ---------------------------------------------------------------------


def test_serial_ingest_fetches_all_sources() -> None:
    adapters = [_FixedAdapter("a"), _FixedAdapter("b"), _FixedAdapter("c")]

    result = serial_ingest(adapters, per_source_timeout=2.0)

    assert result.total_records_fetched == 3
    assert {o.source_name for o in result.source_outcomes} == {"a", "b", "c"}


def test_concurrent_ingest_fetches_all_sources() -> None:
    adapters = [_FixedAdapter("a"), _FixedAdapter("b"), _FixedAdapter("c")]

    result = concurrent_ingest(adapters, per_source_timeout=2.0)

    assert result.total_records_fetched == 3
    assert {o.source_name for o in result.source_outcomes} == {"a", "b", "c"}


def test_empty_adapter_list_returns_empty_result() -> None:
    result = concurrent_ingest([], per_source_timeout=2.0)

    assert result.roles == []
    assert result.total_records_fetched == 0
    assert result.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------
# Individual source failure / partial success
# ---------------------------------------------------------------------


def test_individual_source_failure_does_not_stop_other_sources() -> None:
    adapters = [
        _FixedAdapter("good", jobs=[{"title": "A", "company": "X"}]),
        _FixedAdapter("bad", error=RuntimeError("simulated failure")),
    ]

    result = concurrent_ingest(adapters, per_source_timeout=2.0)

    by_name = {o.source_name: o for o in result.source_outcomes}
    assert by_name["good"].status == "success"
    assert by_name["bad"].status == "failed"
    assert "simulated failure" in by_name["bad"].error
    assert result.failed_source_count == 1
    assert result.total_records_fetched == 1  # only the good source's record


# ---------------------------------------------------------------------
# Malformed records
# ---------------------------------------------------------------------


def test_malformed_records_counted_without_failing_the_source() -> None:
    adapters = [
        _FixedAdapter(
            "mixed",
            jobs=[{"title": "Valid", "company": "X"}, {"description": "missing title/company"}],
        )
    ]

    result = concurrent_ingest(adapters, per_source_timeout=2.0)

    assert result.total_records_fetched == 1
    assert result.malformed_record_count == 1
    assert result.source_outcomes[0].status == "success"


# ---------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------


def test_slow_source_is_marked_as_timed_out() -> None:
    adapters = [
        _FixedAdapter("fast", sleep=0.02),
        _FixedAdapter("slow", sleep=2.0),
    ]

    result = concurrent_ingest(adapters, per_source_timeout=0.2)

    by_name = {o.source_name: o for o in result.source_outcomes}
    assert by_name["fast"].status == "success"
    assert by_name["slow"].status == "timeout"
    assert result.timed_out_source_count == 1


def test_no_timeout_when_all_sources_are_fast() -> None:
    adapters = [_FixedAdapter("a", sleep=0.01), _FixedAdapter("b", sleep=0.01)]

    result = concurrent_ingest(adapters, per_source_timeout=2.0)

    assert result.timed_out_source_count == 0


# ---------------------------------------------------------------------
# Real, measured concurrency speedup - the core claim of this module
# ---------------------------------------------------------------------


def test_concurrent_ingest_is_measurably_faster_than_serial_for_io_bound_sources() -> None:
    """
    Three sources that each take ~0.15s: serial should take roughly
    their SUM (~0.45s); concurrent should take roughly the MAX
    (~0.15s). This asserts real measured wall-clock behavior, not a
    hardcoded/invented number.
    """
    def make_adapters() -> list[SourceAdapter]:
        return [_FixedAdapter("a", sleep=0.15), _FixedAdapter("b", sleep=0.15), _FixedAdapter("c", sleep=0.15)]

    serial_result = serial_ingest(make_adapters(), per_source_timeout=5.0)
    concurrent_result = concurrent_ingest(make_adapters(), per_source_timeout=5.0)

    assert concurrent_result.elapsed_seconds < serial_result.elapsed_seconds
    # Loose bound (not asserting a precise theoretical ratio) to avoid
    # flakiness under CI/system load, while still requiring a real,
    # substantial speedup rather than a marginal one.
    assert concurrent_result.elapsed_seconds < serial_result.elapsed_seconds * 0.7


def test_serial_ingest_runtime_scales_with_number_of_sources() -> None:
    two_sources = [_FixedAdapter("a", sleep=0.1), _FixedAdapter("b", sleep=0.1)]
    four_sources = [_FixedAdapter(f"s{i}", sleep=0.1) for i in range(4)]

    two_result = serial_ingest(two_sources, per_source_timeout=5.0)
    four_result = serial_ingest(four_sources, per_source_timeout=5.0)

    assert four_result.elapsed_seconds > two_result.elapsed_seconds


# ---------------------------------------------------------------------
# run_ingestion_pipeline: full metrics + upsert handling
# ---------------------------------------------------------------------


def test_pipeline_reports_measured_speedup_never_hardcoded() -> None:
    adapters = [_FixedAdapter("a", sleep=0.1), _FixedAdapter("b", sleep=0.1), _FixedAdapter("c", sleep=0.1)]

    result = run_ingestion_pipeline(adapters, per_source_timeout=5.0, upsert=False)

    assert result.serial_elapsed_seconds > 0
    assert result.concurrent_elapsed_seconds > 0
    # speedup is intentionally rounded to 3 decimals in production code
    # (see IngestionRunResult), so compare with a tolerance that
    # accounts for that rounding rather than exact full-precision equality.
    assert result.speedup == pytest.approx(
        result.serial_elapsed_seconds / result.concurrent_elapsed_seconds, abs=0.001
    )
    assert result.speedup > 1.0


def test_pipeline_source_counts_and_totals() -> None:
    adapters = [
        _FixedAdapter("a", jobs=[{"title": "A1", "company": "X"}, {"title": "A2", "company": "X"}]),
        _FixedAdapter("b", jobs=[{"title": "B1", "company": "Y"}]),
    ]

    result = run_ingestion_pipeline(adapters, per_source_timeout=5.0, upsert=False)

    assert result.source_counts == {"a": 2, "b": 1}
    assert result.total_records_fetched == 3


def test_pipeline_deduplicates_across_sources() -> None:
    duplicate_job = {"title": "Same Role", "company": "SameCo", "location": "Remote"}
    adapters = [
        _FixedAdapter("a", jobs=[duplicate_job]),
        _FixedAdapter("b", jobs=[duplicate_job]),
    ]

    result = run_ingestion_pipeline(adapters, per_source_timeout=5.0, upsert=False)

    assert result.total_records_fetched == 2
    assert result.duplicates_removed == 1
    assert len(result.unique_roles) == 1


def test_pipeline_records_failed_and_timed_out_sources() -> None:
    adapters = [
        _FixedAdapter("good"),
        _FixedAdapter("bad", error=RuntimeError("down")),
        _FixedAdapter("slow", sleep=2.0),
    ]

    result = run_ingestion_pipeline(adapters, per_source_timeout=0.2, upsert=False)

    assert result.failed_sources == ["bad"]
    assert result.timed_out_sources == ["slow"]


def test_pipeline_upsert_false_skips_persistence_with_reason() -> None:
    adapters = [_FixedAdapter("a")]

    result = run_ingestion_pipeline(adapters, per_source_timeout=5.0, upsert=False)

    assert result.upserted_count == 0
    assert result.upsert_skipped_reason == "upsert=False; persistence skipped."


def test_pipeline_upsert_gracefully_skips_without_credentials() -> None:
    adapters = [_FixedAdapter("a")]

    result = run_ingestion_pipeline(adapters, per_source_timeout=5.0, upsert=True)

    assert result.upserted_count == 0
    assert result.upsert_skipped_reason is not None
    assert "SUPABASE_URL" in result.upsert_skipped_reason


def test_pipeline_calls_upsert_role_once_per_unique_role(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_upsert_role(**kwargs):
        calls.append(kwargs)
        return {**kwargs, "id": str(len(calls))}

    monkeypatch.setattr(concurrent_module, "upsert_role", fake_upsert_role)

    duplicate_job = {"title": "Same Role", "company": "SameCo"}
    adapters = [
        _FixedAdapter("a", jobs=[duplicate_job, {"title": "Unique", "company": "X"}]),
        _FixedAdapter("b", jobs=[duplicate_job]),
    ]

    result = run_ingestion_pipeline(adapters, per_source_timeout=5.0, upsert=True)

    assert len(calls) == 2  # duplicate collapsed before upsert
    assert result.upserted_count == 2
    assert result.upsert_skipped_reason is None


def test_pipeline_upsert_stops_gracefully_if_credentials_missing_mid_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_configured(**kwargs):
        raise SupabaseNotConfiguredError("not configured")

    monkeypatch.setattr(concurrent_module, "upsert_role", raise_not_configured)

    result = run_ingestion_pipeline([_FixedAdapter("a")], per_source_timeout=5.0, upsert=True)

    assert result.upserted_count == 0
    assert result.upsert_skipped_reason == "not configured"
