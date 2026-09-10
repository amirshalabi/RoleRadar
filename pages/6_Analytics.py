"""
Analytics page: measured engineering/product metrics only.

Pipeline metrics come from backend.db.metrics (real recorded
ingestion_runs rows). Matching/Applications/Preparation metrics come
from backend.services.analytics, which reads already-persisted or
already-computed data (backend.services.discovery,
backend.services.tracking, backend.db.study_plans,
backend.db.assessment_results) - nothing on this page is invented,
estimated, or hardcoded. Any metric with no underlying data renders an
explicit empty state instead of a chart with nothing in it.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from backend.db import metrics as metrics_db
from backend.db.client import SupabaseNotConfiguredError
from backend.services import analytics
from backend.utils.metrics import PipelineMetrics
from ui import components
from ui.theme import GOLD, TEXT_SECONDARY, style_plotly_fig
from ui_common import configure_page, database_not_configured_notice, get_current_user_id, is_demo_mode

configure_page("Analytics", icon="📈")
components.render_page_header(
    "Quant Desk",
    "Analytics",
    "Measured engineering/product metrics - every number here is either directly recorded or simple arithmetic over recorded numbers. Nothing is invented.",
)

_APPLICATION_STAGE_ORDER = ["discovered", "saved", "applied", "oa", "interview", "offer", "rejected", "withdrawn"]
_STAGE_LABEL = {
    "discovered": "Discovered", "saved": "Saved", "applied": "Applied", "oa": "OA",
    "interview": "Interview", "offer": "Offer", "rejected": "Rejected", "withdrawn": "Withdrawn",
}

demo = is_demo_mode()


# ---------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------

_DEMO_RUNS = [
    {
        "jobs_ingested": 140, "jobs_deduplicated": 12, "jobs_eliminated_hard_filter": 71,
        "jobs_eliminated_keyword_filter": 41, "jobs_eliminated_semantic": 0, "jobs_reaching_llm": 16,
        "serial_ingestion_seconds": 4.8, "concurrent_ingestion_seconds": 1.6, "finished_at": "sample run",
    },
]
_DEMO_FIT_DISTRIBUTION = [57.4, 42.0, 68.5, 35.0, 61.2, 49.8, 72.0]
_DEMO_READINESS_DISTRIBUTION = [31.5, 62.0, 45.0, 55.5, 28.0]
_DEMO_APPLICATIONS_BY_STAGE = {"discovered": 3, "saved": 2, "applied": 2, "interview": 1}
_DEMO_COMPLETED_STUDY_MINUTES = 240.0
_DEMO_READINESS_OVER_TIME = [
    {"taken_at": "2026-08-25T00:00:00", "overall_readiness": 24.0},
    {"taken_at": "2026-08-30T00:00:00", "overall_readiness": 29.5},
    {"taken_at": "2026-09-05T00:00:00", "overall_readiness": 31.5},
]


def _demo_top_recurring_skill_gaps():
    from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
    from backend.ingestion.normalize import normalize_role
    from backend.llm.extract_requirements import RoleRequirement
    from backend.matching.cross_role import FavoriteRoleContext, calculate_skill_roi

    profile = CandidateProfile(
        skills=[
            CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.5, confidence=0.7),
            CandidateSkillEstimate(normalized_skill_name="algorithms", display_name="Algorithms", estimated_level=4.0, confidence=0.4),
            CandidateSkillEstimate(normalized_skill_name="probability", display_name="Probability", estimated_level=3.0, confidence=0.3),
        ]
    )
    contexts = [
        FavoriteRoleContext(
            role=normalize_role({"external_id": "demo-1", "title": "Quantitative Research Intern", "company": "Meridian Capital", "role_family": "quant"}),
            requirements=[
                RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8, importance=9, required=True, evidence=["x"]),
                RoleRequirement(skill="Python", normalized_skill="python", target_level=7, importance=7, required=True, evidence=["x"]),
            ],
            priority="dream",
        ),
        FavoriteRoleContext(
            role=normalize_role({"external_id": "demo-2", "title": "Software Engineering Intern", "company": "Acme Corp", "role_family": "swe"}),
            requirements=[
                RoleRequirement(skill="Algorithms", normalized_skill="algorithms", target_level=7, importance=8, required=True, evidence=["x"]),
                RoleRequirement(skill="Python", normalized_skill="python", target_level=6, importance=6, required=True, evidence=["x"]),
            ],
            priority="backup",
        ),
    ]
    results = calculate_skill_roi(profile, contexts)
    return sorted(results, key=lambda r: (r.roles_requiring_it, r.average_gap), reverse=True)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

if demo:
    runs = _DEMO_RUNS
    fit_distribution = _DEMO_FIT_DISTRIBUTION
    readiness_distribution = _DEMO_READINESS_DISTRIBUTION
    top_skill_gaps = _demo_top_recurring_skill_gaps()
    applications_by_stage = _DEMO_APPLICATIONS_BY_STAGE
    completed_study_minutes = _DEMO_COMPLETED_STUDY_MINUTES
    readiness_over_time = _DEMO_READINESS_OVER_TIME
else:
    user_id = get_current_user_id()
    if user_id is None:
        database_not_configured_notice()
        st.stop()

    try:
        runs = metrics_db.list_ingestion_runs(limit=20)
    except SupabaseNotConfiguredError:
        database_not_configured_notice()
        st.stop()

    fit_distribution = analytics.get_fit_distribution(user_id)
    readiness_distribution = analytics.get_readiness_distribution(user_id)
    top_skill_gaps = analytics.get_top_recurring_skill_gaps(user_id)
    applications_by_stage = analytics.get_applications_by_stage(user_id)
    completed_study_minutes = analytics.get_completed_study_minutes(user_id)
    readiness_over_time = analytics.get_readiness_over_time(user_id)


# ---------------------------------------------------------------------
# 1. Pipeline
# ---------------------------------------------------------------------

components.render_section_header(
    "Pipeline",
    "What was ingested, deduplicated, filtered, and sent to the LLM during job ingestion (backend.ingestion.concurrent.run_ingestion_pipeline).",
)

if not runs:
    components.render_empty_state(
        "No ingestion runs recorded yet",
        detail="Run scripts/run_ingestion.py (or backend.ingestion.concurrent.run_ingestion_pipeline()) to populate pipeline metrics.",
    )
else:
    latest = runs[0]
    metrics = PipelineMetrics(
        ingested=latest.get("jobs_ingested") or 0,
        deduplicated=latest.get("jobs_deduplicated") or 0,
        hard_filtered=latest.get("jobs_eliminated_hard_filter") or 0,
        keyword_filtered=latest.get("jobs_eliminated_keyword_filter") or 0,
        semantic_filtered=latest.get("jobs_eliminated_semantic") or 0,
        llm_analyzed=latest.get("jobs_reaching_llm") or 0,
    )

    components.render_metric_strip(
        [
            {"label": "Jobs Ingested", "value": str(metrics.ingested)},
            {"label": "Duplicates Removed", "value": str(metrics.deduplicated)},
            {"label": "Hard Filtered", "value": str(metrics.hard_filtered)},
            {"label": "Keyword Filtered", "value": str(metrics.keyword_filtered)},
            {"label": "Semantic Filtered", "value": str(metrics.semantic_filtered)},
            {"label": "LLM Analyzed", "value": str(metrics.llm_analyzed), "tone": "gold"},
        ]
    )

    funnel_stages = ["Ingested", "After Dedup", "After Hard Filter", "After Keyword Filter", "Reached LLM"]
    funnel_values = [
        metrics.ingested,
        metrics.eligible_input_count(),
        max(metrics.eligible_input_count() - metrics.hard_filtered, 0),
        max(metrics.eligible_input_count() - metrics.hard_filtered - metrics.keyword_filtered, 0),
        metrics.llm_analyzed,
    ]
    fig = px.funnel(x=funnel_values, y=funnel_stages, title="Postings Remaining After Each Stage")
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

    avoidance = metrics.llm_avoidance_rate()
    if avoidance is not None:
        components.render_status_badge(f"LLM WORKLOAD AVOIDANCE: {avoidance * 100:.0f}%", tone="positive")
        st.caption(
            "LLM workload avoidance = 1 − (postings sent to the LLM ÷ deduplicated eligible postings). "
            "Higher means cheaper, faster deterministic filtering did more of the work."
        )

    if latest.get("llm_estimated_cost_usd"):
        components.render_meta_line(
            [
                f"{latest.get('llm_prompt_tokens', 0)} prompt tokens",
                f"{latest.get('llm_completion_tokens', 0)} completion tokens",
                f"est. ${latest['llm_estimated_cost_usd']:.4f}",
            ]
        )

    components.render_section_header("Concurrency")
    serial_seconds = latest.get("serial_ingestion_seconds")
    concurrent_seconds = latest.get("concurrent_ingestion_seconds")
    if serial_seconds and concurrent_seconds:
        components.render_metric_strip(
            [
                {"label": "Serial Time", "value": f"{serial_seconds:.2f}s"},
                {"label": "Concurrent Time", "value": f"{concurrent_seconds:.2f}s"},
                {"label": "Measured Speedup", "value": f"{serial_seconds / concurrent_seconds:.2f}x", "tone": "gold"},
            ]
        )
        st.caption("Speedup = serial time ÷ concurrent time, both measured running the SAME pipeline logic - never simulated.")
    else:
        components.render_empty_state("No timing data for this run", detail="This run didn't record serial/concurrent timing.")

    with st.expander("Run history"):
        for index, run in enumerate(runs):
            with components.panel(f"run-{index}"):
                components.render_meta_line([run.get("finished_at", "unknown time")])
                run_metrics = PipelineMetrics(
                    ingested=run.get("jobs_ingested") or 0, deduplicated=run.get("jobs_deduplicated") or 0,
                    hard_filtered=run.get("jobs_eliminated_hard_filter") or 0, keyword_filtered=run.get("jobs_eliminated_keyword_filter") or 0,
                    semantic_filtered=run.get("jobs_eliminated_semantic") or 0, llm_analyzed=run.get("jobs_reaching_llm") or 0,
                )
                rate = run_metrics.llm_avoidance_rate()
                st.write(
                    f"{run_metrics.ingested} ingested → {run_metrics.llm_analyzed} reached LLM"
                    + (f" ({rate * 100:.0f}% avoided)" if rate is not None else "")
                )


# ---------------------------------------------------------------------
# 2. Matching
# ---------------------------------------------------------------------

components.divider()
components.render_section_header("Matching", "How fit and readiness scores are distributed across the roles you've analyzed, and which skills recur most often.")

match_cols = st.columns(2)
with match_cols[0]:
    st.markdown('<p class="rr-eyebrow">Fit Distribution</p>', unsafe_allow_html=True)
    st.caption("Overall fit score (backend.matching.scorer) for every role you've analyzed.")
    if fit_distribution:
        fig = px.histogram(x=fit_distribution, nbins=10, range_x=[0, 100], labels={"x": "Fit score"})
        fig.update_layout(yaxis_title="Roles", showlegend=False)
        st.plotly_chart(style_plotly_fig(fig), use_container_width=True)
    else:
        components.render_empty_state("No fit scores yet", detail="Analyze a role from Discover to see its fit score here.")

with match_cols[1]:
    st.markdown('<p class="rr-eyebrow">Readiness Distribution</p>', unsafe_allow_html=True)
    st.caption("Interview readiness (backend.planning.readiness) for every role with tracked skills and a role family.")
    if readiness_distribution:
        fig = px.histogram(x=readiness_distribution, nbins=10, range_x=[0, 100], labels={"x": "Readiness score"})
        fig.update_layout(yaxis_title="Roles", showlegend=False)
        st.plotly_chart(style_plotly_fig(fig, accent=TEXT_SECONDARY), use_container_width=True)
    else:
        components.render_empty_state("No readiness scores yet", detail="Upload a resume and add a role with a role family to see this.")

st.markdown('<p class="rr-eyebrow">Top Recurring Skill Gaps</p>', unsafe_allow_html=True)
st.caption("Skills required by the most of your favorite roles, ranked by how many roles need them (backend.matching.cross_role).")
if top_skill_gaps:
    fig = px.bar(
        x=[r.roles_requiring_it for r in top_skill_gaps],
        y=[r.display_skill for r in top_skill_gaps],
        orientation="h",
        labels={"x": "Roles requiring it", "y": "Skill"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)
else:
    components.render_empty_state("No recurring skill gaps yet", detail="Save and analyze at least one favorite role to see this.")


# ---------------------------------------------------------------------
# 3. Applications
# ---------------------------------------------------------------------

components.divider()
components.render_section_header("Applications", "How many tracked applications sit at each pipeline stage right now.")

if applications_by_stage:
    ordered_stages = [s for s in _APPLICATION_STAGE_ORDER if s in applications_by_stage]
    fig = px.bar(
        x=[_STAGE_LABEL[s] for s in ordered_stages],
        y=[applications_by_stage[s] for s in ordered_stages],
        labels={"x": "Stage", "y": "Applications"},
    )
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)
else:
    components.render_empty_state("No applications tracked yet", detail="Saving a role from Discover automatically starts tracking it here.")


# ---------------------------------------------------------------------
# 4. Preparation
# ---------------------------------------------------------------------

components.divider()
components.render_section_header("Preparation", "How your interview readiness has moved as you've submitted diagnostics, and how much study time you've logged as complete.")

prep_cols = st.columns(2)
with prep_cols[0]:
    st.markdown('<p class="rr-eyebrow">Readiness Over Time</p>', unsafe_allow_html=True)
    st.caption("Overall readiness recomputed as of each diagnostic you submitted, in order (backend.planning.readiness).")
    if readiness_over_time:
        fig = px.line(
            x=[p["taken_at"] for p in readiness_over_time], y=[p["overall_readiness"] for p in readiness_over_time],
            markers=True, labels={"x": "Diagnostic taken at", "y": "Overall readiness"},
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(style_plotly_fig(fig), use_container_width=True)
    else:
        components.render_empty_state("No diagnostics submitted yet", detail="Submit a diagnostic on the Interview Prep page to start tracking this.")

with prep_cols[1]:
    st.markdown('<p class="rr-eyebrow">Completed Study Minutes</p>', unsafe_allow_html=True)
    st.caption("Sum of allocated minutes for tasks you've marked complete, across every application's current study plan.")
    if completed_study_minutes:
        components.render_metric_strip(
            [{"label": "Completed", "value": f"{completed_study_minutes:.0f} min", "sublabel": f"{completed_study_minutes / 60.0:.1f}h", "tone": "gold"}]
        )
    else:
        components.render_empty_state("No completed study tasks yet", detail="Mark a task complete on the Interview Prep page to see this.")
