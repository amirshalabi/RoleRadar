"""
Analytics page: pipeline observability (ingestion, filtering,
concurrency speedup, LLM-cost avoidance).

Reads backend.db.metrics - real recorded ingestion_runs rows only.
"""

from __future__ import annotations

import streamlit as st

from backend.db import metrics as metrics_db
from backend.db.client import SupabaseNotConfiguredError
from backend.utils.metrics import PipelineMetrics
from ui_common import configure_page, database_not_configured_notice, empty_state, is_demo_mode

configure_page("Analytics", icon="📈")
st.title("📈 Analytics")
st.caption("Pipeline observability: what was ingested, filtered, and analyzed - measured, never invented.")
st.divider()

_DEMO_RUNS = [
    {
        "jobs_ingested": 140, "jobs_deduplicated": 12, "jobs_eliminated_hard_filter": 71,
        "jobs_eliminated_keyword_filter": 41, "jobs_eliminated_semantic": 0, "jobs_reaching_llm": 16,
        "serial_ingestion_seconds": 4.8, "concurrent_ingestion_seconds": 1.6, "finished_at": "sample run",
    },
]

if is_demo_mode():
    runs = _DEMO_RUNS
else:
    try:
        runs = metrics_db.list_ingestion_runs(limit=20)
    except SupabaseNotConfiguredError:
        database_not_configured_notice()
        st.stop()

if not runs:
    empty_state(
        "No ingestion runs recorded yet",
        detail="Run backend.ingestion.concurrent.run_ingestion_pipeline() to populate pipeline metrics.",
    )
    st.stop()

latest = runs[0]
metrics = PipelineMetrics(
    ingested=latest.get("jobs_ingested") or 0,
    deduplicated=latest.get("jobs_deduplicated") or 0,
    hard_filtered=latest.get("jobs_eliminated_hard_filter") or 0,
    keyword_filtered=latest.get("jobs_eliminated_keyword_filter") or 0,
    semantic_filtered=latest.get("jobs_eliminated_semantic") or 0,
    llm_analyzed=latest.get("jobs_reaching_llm") or 0,
)

st.subheader("Most recent run")
cols = st.columns(5)
cols[0].metric("Ingested", metrics.ingested)
cols[1].metric("Deduplicated", metrics.deduplicated)
cols[2].metric("Hard-filtered", metrics.hard_filtered)
cols[3].metric("Keyword-filtered", metrics.keyword_filtered)
cols[4].metric("Reached LLM", metrics.llm_analyzed)

avoidance = metrics.llm_avoidance_rate()
if avoidance is not None:
    st.success(f"**{avoidance * 100:.0f}%** of candidate postings were eliminated before LLM analysis.")

serial_seconds = latest.get("serial_ingestion_seconds")
concurrent_seconds = latest.get("concurrent_ingestion_seconds")
if serial_seconds and concurrent_seconds:
    speed_cols = st.columns(3)
    speed_cols[0].metric("Serial", f"{serial_seconds:.2f}s")
    speed_cols[1].metric("Concurrent", f"{concurrent_seconds:.2f}s")
    speed_cols[2].metric("Speedup", f"{serial_seconds / concurrent_seconds:.2f}x")

if latest.get("llm_estimated_cost_usd"):
    st.caption(
        f"LLM usage: {latest.get('llm_prompt_tokens', 0)} prompt + {latest.get('llm_completion_tokens', 0)} "
        f"completion tokens · est. ${latest['llm_estimated_cost_usd']:.4f}"
    )

st.divider()
st.subheader("Run history")
for run in runs:
    with st.container(border=True):
        st.caption(run.get("finished_at", "unknown time"))
        run_metrics = PipelineMetrics(
            ingested=run.get("jobs_ingested") or 0,
            deduplicated=run.get("jobs_deduplicated") or 0,
            hard_filtered=run.get("jobs_eliminated_hard_filter") or 0,
            keyword_filtered=run.get("jobs_eliminated_keyword_filter") or 0,
            semantic_filtered=run.get("jobs_eliminated_semantic") or 0,
            llm_analyzed=run.get("jobs_reaching_llm") or 0,
        )
        rate = run_metrics.llm_avoidance_rate()
        st.write(
            f"{run_metrics.ingested} ingested → {run_metrics.llm_analyzed} reached LLM"
            + (f" ({rate * 100:.0f}% avoided)" if rate is not None else "")
        )
