"""
Pipeline run persistence: ingestion_runs (one row per pipeline
execution) and pipeline_metrics (named metric values attached to a run).

Unlike roles.py/candidates.py/favorites.py/applications.py, this module
does NOT use idempotent upserts. Every pipeline run is a genuinely new
event - an audit-log entry, not a logical entity that gets repeated
with updates - so record_ingestion_run() is a plain INSERT. There is no
natural key to detect "the same run twice"; calling this twice creates
two rows, exactly as running the pipeline twice should.

The read functions here (list_ingestion_runs, get_latest_ingestion_run,
list_pipeline_metrics) are what an Analytics dashboard page would call.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from backend.db.client import get_client
from backend.utils.metrics import ConcurrencyMetrics, LLMUsageMetrics, PipelineMetrics

logger = logging.getLogger(__name__)

INGESTION_RUNS_TABLE = "ingestion_runs"
PIPELINE_METRICS_TABLE = "pipeline_metrics"


def record_ingestion_run(
    pipeline_metrics: PipelineMetrics,
    concurrency_metrics: ConcurrencyMetrics | None = None,
    llm_usage: LLMUsageMetrics | None = None,
    source: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> dict[str, Any]:
    """
    Insert one ingestion_runs row summarizing a completed pipeline run,
    plus one pipeline_metrics row per named metric it produced (funnel
    counts, derived rates, speedup, and - only if `llm_usage` actually
    captured real data - token usage/cost). Returns the created
    ingestion_runs row.

    `finished_at` defaults to now if not supplied. `started_at` is only
    included in the insert if explicitly supplied - otherwise the
    column's own database default applies, rather than this function
    guessing a start time it doesn't actually know.
    """
    client = get_client()

    values: dict[str, Any] = {
        "finished_at": (finished_at or datetime.now(timezone.utc)).isoformat(),
        "source": source,
        "jobs_ingested": pipeline_metrics.ingested,
        "jobs_deduplicated": pipeline_metrics.deduplicated,
        "jobs_eliminated_hard_filter": pipeline_metrics.hard_filtered,
        "jobs_eliminated_keyword_filter": pipeline_metrics.keyword_filtered,
        "jobs_eliminated_semantic": pipeline_metrics.semantic_filtered,
        "jobs_reaching_llm": pipeline_metrics.llm_analyzed,
        "estimated_llm_calls_avoided": pipeline_metrics.estimated_llm_calls_avoided(),
        "serial_ingestion_seconds": concurrency_metrics.serial_seconds if concurrency_metrics else None,
        "concurrent_ingestion_seconds": concurrency_metrics.concurrent_seconds if concurrency_metrics else None,
    }
    if started_at is not None:
        values["started_at"] = started_at.isoformat()
    if llm_usage is not None and llm_usage.call_count > 0:
        values["llm_prompt_tokens"] = llm_usage.prompt_tokens
        values["llm_completion_tokens"] = llm_usage.completion_tokens
        values["llm_total_tokens"] = llm_usage.total_tokens
        values["llm_estimated_cost_usd"] = llm_usage.estimated_cost_usd

    response = client.table(INGESTION_RUNS_TABLE).insert(values).execute()
    if not response.data:
        raise RuntimeError(f"Insert into {INGESTION_RUNS_TABLE} returned no data")
    run_row = response.data[0]

    metric_rows = _build_metric_rows(run_row["id"], pipeline_metrics, concurrency_metrics, llm_usage)
    if metric_rows:
        client.table(PIPELINE_METRICS_TABLE).insert(metric_rows).execute()

    return run_row


def _build_metric_rows(
    run_id: str,
    pipeline_metrics: PipelineMetrics,
    concurrency_metrics: ConcurrencyMetrics | None,
    llm_usage: LLMUsageMetrics | None,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    def add(name: str, value: float | None) -> None:
        if value is not None:
            rows.append({"ingestion_run_id": run_id, "metric_name": name, "metric_value": value, "recorded_at": now})

    add("ingested", pipeline_metrics.ingested)
    add("deduplicated", pipeline_metrics.deduplicated)
    add("hard_filtered", pipeline_metrics.hard_filtered)
    add("keyword_filtered", pipeline_metrics.keyword_filtered)
    add("semantic_filtered", pipeline_metrics.semantic_filtered)
    add("llm_analyzed", pipeline_metrics.llm_analyzed)
    add("eligible_input_count", pipeline_metrics.eligible_input_count())
    add("llm_avoidance_rate", pipeline_metrics.llm_avoidance_rate())

    if concurrency_metrics is not None:
        add("serial_seconds", concurrency_metrics.serial_seconds)
        add("concurrent_seconds", concurrency_metrics.concurrent_seconds)
        add("speedup", concurrency_metrics.speedup())

    if llm_usage is not None and llm_usage.call_count > 0:
        add("llm_prompt_tokens", llm_usage.prompt_tokens)
        add("llm_completion_tokens", llm_usage.completion_tokens)
        add("llm_total_tokens", llm_usage.total_tokens)
        add("llm_estimated_cost_usd", llm_usage.estimated_cost_usd)

    return rows


def list_ingestion_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Most recent ingestion runs first - for an Analytics dashboard's run history."""
    client = get_client()
    response = (
        client.table(INGESTION_RUNS_TABLE).select("*").order("finished_at", desc=True).limit(limit).execute()
    )
    return response.data or []


def get_latest_ingestion_run() -> dict[str, Any] | None:
    """The most recent ingestion run, or None if none have been recorded yet."""
    runs = list_ingestion_runs(limit=1)
    return runs[0] if runs else None


def list_pipeline_metrics(ingestion_run_id: str) -> list[dict[str, Any]]:
    """Every named metric recorded for one run."""
    client = get_client()
    response = (
        client.table(PIPELINE_METRICS_TABLE).select("*").eq("ingestion_run_id", ingestion_run_id).execute()
    )
    return response.data or []
