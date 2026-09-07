"""
Tests for backend.db.metrics. Uses the in-memory FakeSupabaseClient
(tests/_fake_supabase.py) so real insert-then-read sequences behave
like a real table would, without a live Supabase project.
"""

from __future__ import annotations

import pytest

from backend.db import metrics as metrics_db
from backend.utils.metrics import ConcurrencyMetrics, LLMUsageMetrics, PipelineMetrics
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    monkeypatch.setattr(metrics_db, "get_client", lambda: client)
    return client


def _metrics() -> PipelineMetrics:
    return PipelineMetrics(ingested=100, deduplicated=10, hard_filtered=50, keyword_filtered=30, llm_analyzed=10)


def test_record_ingestion_run_inserts_one_row(fake_client: FakeSupabaseClient) -> None:
    run = metrics_db.record_ingestion_run(_metrics())

    assert len(fake_client._tables[metrics_db.INGESTION_RUNS_TABLE].rows) == 1
    assert run["jobs_ingested"] == 100
    assert run["jobs_deduplicated"] == 10
    assert run["jobs_eliminated_hard_filter"] == 50
    assert run["jobs_eliminated_keyword_filter"] == 30
    assert run["jobs_reaching_llm"] == 10
    assert run["estimated_llm_calls_avoided"] == 80


def test_record_ingestion_run_twice_creates_two_rows(fake_client: FakeSupabaseClient) -> None:
    """Not idempotent - each call is a genuinely new run/audit-log entry."""
    metrics_db.record_ingestion_run(_metrics())
    metrics_db.record_ingestion_run(_metrics())

    assert len(fake_client._tables[metrics_db.INGESTION_RUNS_TABLE].rows) == 2


def test_record_ingestion_run_includes_concurrency_metrics(fake_client: FakeSupabaseClient) -> None:
    run = metrics_db.record_ingestion_run(_metrics(), concurrency_metrics=ConcurrencyMetrics(serial_seconds=1.4, concurrent_seconds=0.7))

    assert run["serial_ingestion_seconds"] == 1.4
    assert run["concurrent_ingestion_seconds"] == 0.7


def test_record_ingestion_run_omits_concurrency_fields_when_not_given(fake_client: FakeSupabaseClient) -> None:
    run = metrics_db.record_ingestion_run(_metrics())

    assert run["serial_ingestion_seconds"] is None
    assert run["concurrent_ingestion_seconds"] is None


def test_record_ingestion_run_includes_llm_usage_when_captured(fake_client: FakeSupabaseClient) -> None:
    usage = LLMUsageMetrics(call_count=2, prompt_tokens=300, completion_tokens=60, total_tokens=360, estimated_cost_usd=0.001)

    run = metrics_db.record_ingestion_run(_metrics(), llm_usage=usage)

    assert run["llm_prompt_tokens"] == 300
    assert run["llm_completion_tokens"] == 60
    assert run["llm_estimated_cost_usd"] == 0.001


def test_record_ingestion_run_omits_llm_usage_fields_when_no_calls_captured(fake_client: FakeSupabaseClient) -> None:
    run = metrics_db.record_ingestion_run(_metrics(), llm_usage=LLMUsageMetrics(call_count=0))

    assert "llm_prompt_tokens" not in run or run.get("llm_prompt_tokens") is None


def test_record_ingestion_run_creates_pipeline_metrics_rows(fake_client: FakeSupabaseClient) -> None:
    metrics_db.record_ingestion_run(_metrics(), concurrency_metrics=ConcurrencyMetrics(serial_seconds=1.4, concurrent_seconds=0.7))

    metric_rows = fake_client._tables[metrics_db.PIPELINE_METRICS_TABLE].rows
    metric_names = {row["metric_name"] for row in metric_rows}
    assert "ingested" in metric_names
    assert "llm_avoidance_rate" in metric_names
    assert "speedup" in metric_names


def test_pipeline_metrics_rows_reference_the_created_run(fake_client: FakeSupabaseClient) -> None:
    run = metrics_db.record_ingestion_run(_metrics())

    metric_rows = fake_client._tables[metrics_db.PIPELINE_METRICS_TABLE].rows
    assert all(row["ingestion_run_id"] == run["id"] for row in metric_rows)


def test_list_ingestion_runs_returns_most_recent_first(fake_client: FakeSupabaseClient) -> None:
    metrics_db.record_ingestion_run(PipelineMetrics(ingested=1, deduplicated=0, hard_filtered=0, keyword_filtered=0, llm_analyzed=1))
    metrics_db.record_ingestion_run(_metrics())

    runs = metrics_db.list_ingestion_runs(limit=10)

    assert len(runs) == 2


def test_get_latest_ingestion_run_returns_none_when_empty(fake_client: FakeSupabaseClient) -> None:
    assert metrics_db.get_latest_ingestion_run() is None


def test_get_latest_ingestion_run_returns_a_run_after_recording(fake_client: FakeSupabaseClient) -> None:
    metrics_db.record_ingestion_run(_metrics())

    latest = metrics_db.get_latest_ingestion_run()

    assert latest is not None
    assert latest["jobs_ingested"] == 100


def test_list_pipeline_metrics_scopes_to_one_run(fake_client: FakeSupabaseClient) -> None:
    run_a = metrics_db.record_ingestion_run(_metrics())
    run_b = metrics_db.record_ingestion_run(_metrics())

    metrics_for_a = metrics_db.list_pipeline_metrics(run_a["id"])

    assert all(row["ingestion_run_id"] == run_a["id"] for row in metrics_for_a)
    assert run_a["id"] != run_b["id"]
