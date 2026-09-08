"""Pipeline observability - backend.db.metrics / backend.utils.metrics."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.schemas import PipelineAnalyticsResponse
from backend.db import metrics as metrics_db
from backend.utils.metrics import PipelineMetrics

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/pipeline", response_model=PipelineAnalyticsResponse)
def pipeline_analytics() -> PipelineAnalyticsResponse:
    """The most recent ingestion run's funnel counts, LLM avoidance rate, and concurrency speedup - all measured, never estimated."""
    latest = metrics_db.get_latest_ingestion_run()
    if latest is None:
        return PipelineAnalyticsResponse(has_data=False)

    metrics = PipelineMetrics(
        ingested=latest.get("jobs_ingested") or 0,
        deduplicated=latest.get("jobs_deduplicated") or 0,
        hard_filtered=latest.get("jobs_eliminated_hard_filter") or 0,
        keyword_filtered=latest.get("jobs_eliminated_keyword_filter") or 0,
        semantic_filtered=latest.get("jobs_eliminated_semantic") or 0,
        llm_analyzed=latest.get("jobs_reaching_llm") or 0,
    )
    serial_seconds = latest.get("serial_ingestion_seconds")
    concurrent_seconds = latest.get("concurrent_ingestion_seconds")
    speedup = round(serial_seconds / concurrent_seconds, 3) if serial_seconds and concurrent_seconds else None

    return PipelineAnalyticsResponse(
        has_data=True,
        ingested=metrics.ingested,
        deduplicated=metrics.deduplicated,
        hard_filtered=metrics.hard_filtered,
        keyword_filtered=metrics.keyword_filtered,
        semantic_filtered=metrics.semantic_filtered,
        llm_analyzed=metrics.llm_analyzed,
        llm_avoidance_rate=metrics.llm_avoidance_rate(),
        serial_ingestion_seconds=serial_seconds,
        concurrent_ingestion_seconds=concurrent_seconds,
        speedup=speedup,
        finished_at=latest.get("finished_at"),
    )
