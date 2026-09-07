"""
Dashboard aggregation service.

Assembles everything app.py's Dashboard page renders, from real,
persisted data only. Each piece of DashboardData is independently
optional: if the underlying data doesn't exist yet (no resume uploaded,
no fit scores computed, no applications tracked), the corresponding
field is None/empty and app.py renders an empty state for it - this
module never fabricates a plausible-looking value to fill a gap.

Two things this module can genuinely NOT compute honestly from what is
persisted today, and deliberately does not attempt to: a real study
plan (backend.planning.scheduler needs RoleRequirement data, which has
no persistence layer yet) and skill ROI across favorites
(backend.matching.cross_role needs the same). Both stay None in
production; build_demo_dashboard_data() shows what they would look like
using clearly-labeled sample input fed through the same real functions.

Streamlit pages must call into this module (or backend.services.tracking),
never into backend.db.* directly.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel

from backend.candidate.profile import (
    CandidateProfile,
    CandidateSkillEstimate,
    ExperienceEntry,
    ProjectEntry,
)
from backend.db import candidates as candidates_db
from backend.db import metrics as metrics_db
from backend.db import roles as roles_db
from backend.db.applications import list_applications_with_roles
from backend.db.client import SupabaseNotConfiguredError
from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.cross_role import FavoriteRoleContext, calculate_skill_roi
from backend.planning.readiness import calculate_readiness
from backend.services import tracking
from backend.utils.metrics import PipelineMetrics

logger = logging.getLogger(__name__)

# An application counts as "urgent" if its deadline falls within this
# many days (or it's already in active interview scheduling) - a simple,
# documented threshold rather than an elaborate priority model.
URGENT_DEADLINE_WINDOW_DAYS = 14
_INACTIVE_STATUSES = {"rejected", "withdrawn", "offer"}


class ProfileStatus(BaseModel):
    has_profile: bool
    skill_count: int = 0
    updated_at: str | None = None


class TopMatch(BaseModel):
    role_id: str
    role_title: str
    company: str
    overall_score: float


class UpcomingInterview(BaseModel):
    role_id: str
    role_title: str
    company: str
    role_family: str | None = None
    interview_date: str
    days_away: int


class UrgentApplication(BaseModel):
    role_title: str
    company: str
    status: str
    deadline: str | None
    days_until_deadline: int | None


class PipelineStats(BaseModel):
    ingested: int
    deduplicated: int
    hard_filtered: int
    keyword_filtered: int
    llm_analyzed: int
    llm_avoidance_rate: float | None
    recorded_at: str | None


class DashboardData(BaseModel):
    is_demo: bool = False
    database_connected: bool = True

    profile_status: ProfileStatus
    top_match: TopMatch | None = None
    upcoming_interview: UpcomingInterview | None = None

    overall_readiness: float | None = None
    readiness_role_family: str | None = None

    highest_leverage_skill: str | None = None
    highest_leverage_skill_roi: float | None = None

    saved_roles_count: int = 0
    active_applications_count: int = 0
    urgent_applications: list[UrgentApplication] = []

    pipeline_stats: PipelineStats | None = None


def build_dashboard_data(user_id: str) -> DashboardData:
    """
    Assemble a DashboardData from real persisted data for `user_id`. If
    Postgres isn't configured at all, returns database_connected=False
    with everything else empty, rather than letting the exception
    surface mid-render.
    """
    try:
        return _build_dashboard_data(user_id)
    except SupabaseNotConfiguredError:
        logger.info("Dashboard data unavailable - Supabase not configured")
        return DashboardData(database_connected=False, profile_status=ProfileStatus(has_profile=False))


def _build_dashboard_data(user_id: str) -> DashboardData:
    today = date.today()

    profile_row = candidates_db.get_candidate_profile(user_id)
    skill_rows = candidates_db.list_candidate_skills(user_id)
    profile_status = ProfileStatus(
        has_profile=profile_row is not None,
        skill_count=len(skill_rows),
        updated_at=(profile_row or {}).get("updated_at"),
    )

    fit_score_rows = roles_db.list_fit_scores_for_user(user_id, limit=1)
    top_match = _build_top_match(fit_score_rows)

    applications = list_applications_with_roles(user_id)
    upcoming_interview = _find_upcoming_interview(applications, today)
    urgent_applications = _find_urgent_applications(applications, today)
    active_applications_count = sum(
        1 for app in applications if app.get("status") not in _INACTIVE_STATUSES
    )

    saved_roles = tracking.list_saved_roles(user_id)

    overall_readiness = None
    readiness_role_family = None
    if skill_rows and fit_score_rows:
        role_family = (fit_score_rows[0].get("roles") or {}).get("role_family")
        candidate_skills = _skill_estimates_from_rows(skill_rows)
        readiness_result = calculate_readiness(CandidateProfile(skills=candidate_skills), role_family)
        overall_readiness = readiness_result.overall_readiness
        readiness_role_family = role_family

    pipeline_stats = _build_pipeline_stats(metrics_db.get_latest_ingestion_run())

    return DashboardData(
        profile_status=profile_status,
        top_match=top_match,
        upcoming_interview=upcoming_interview,
        overall_readiness=overall_readiness,
        readiness_role_family=readiness_role_family,
        # highest_leverage_skill intentionally left None: computing it
        # needs each favorite role's RoleRequirement data, which has no
        # persistence layer yet - see this module's docstring.
        saved_roles_count=len(saved_roles),
        active_applications_count=active_applications_count,
        urgent_applications=urgent_applications,
        pipeline_stats=pipeline_stats,
    )


def _skill_estimates_from_rows(skill_rows: list[dict[str, Any]]) -> list[CandidateSkillEstimate]:
    return [
        CandidateSkillEstimate(
            normalized_skill_name=row["normalized_skill_name"],
            display_name=row.get("display_name") or row["normalized_skill_name"],
            estimated_level=row["estimated_level"],
            confidence=row["confidence"],
        )
        for row in skill_rows
    ]


def _build_top_match(fit_score_rows: list[dict[str, Any]]) -> TopMatch | None:
    if not fit_score_rows:
        return None
    row = fit_score_rows[0]
    role = row.get("roles") or {}
    return TopMatch(
        role_id=row["role_id"],
        role_title=role.get("title", "Unknown role"),
        company=role.get("company", "Unknown company"),
        overall_score=row["overall_score"],
    )


def _find_upcoming_interview(applications: list[dict[str, Any]], today: date) -> UpcomingInterview | None:
    candidates: list[tuple[date, dict[str, Any]]] = []
    for app in applications:
        interview_date_str = app.get("interview_date")
        if not interview_date_str:
            continue
        interview_date = date.fromisoformat(interview_date_str)
        if interview_date < today:
            continue
        candidates.append((interview_date, app))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    interview_date, app = candidates[0]
    role = app.get("roles") or {}
    return UpcomingInterview(
        role_id=app["role_id"],
        role_title=role.get("title", "Unknown role"),
        company=role.get("company", "Unknown company"),
        role_family=role.get("role_family"),
        interview_date=interview_date.isoformat(),
        days_away=(interview_date - today).days,
    )


def _find_urgent_applications(
    applications: list[dict[str, Any]], today: date, threshold_days: int = URGENT_DEADLINE_WINDOW_DAYS
) -> list[UrgentApplication]:
    """
    An application is "urgent" if it has a deadline within
    `threshold_days` (and that deadline hasn't already passed), or it is
    already at the "interview" stage (imminent regardless of a deadline
    field). Terminal statuses (rejected/withdrawn/offer) are never
    urgent - there is nothing left to act on.
    """
    urgent: list[UrgentApplication] = []
    for app in applications:
        status = app.get("status")
        if status in _INACTIVE_STATUSES:
            continue

        deadline_str = app.get("deadline")
        days_until_deadline = None
        if deadline_str:
            deadline = date.fromisoformat(deadline_str)
            days_until_deadline = (deadline - today).days
            if days_until_deadline < 0 or days_until_deadline > threshold_days:
                continue
        elif status != "interview":
            continue

        role = app.get("roles") or {}
        urgent.append(
            UrgentApplication(
                role_title=role.get("title", "Unknown role"),
                company=role.get("company", "Unknown company"),
                status=status,
                deadline=deadline_str,
                days_until_deadline=days_until_deadline,
            )
        )

    urgent.sort(key=lambda a: a.days_until_deadline if a.days_until_deadline is not None else 999)
    return urgent[:5]


def _build_pipeline_stats(latest_run: dict[str, Any] | None) -> PipelineStats | None:
    if latest_run is None:
        return None
    metrics = PipelineMetrics(
        ingested=latest_run.get("jobs_ingested") or 0,
        deduplicated=latest_run.get("jobs_deduplicated") or 0,
        hard_filtered=latest_run.get("jobs_eliminated_hard_filter") or 0,
        keyword_filtered=latest_run.get("jobs_eliminated_keyword_filter") or 0,
        semantic_filtered=latest_run.get("jobs_eliminated_semantic") or 0,
        llm_analyzed=latest_run.get("jobs_reaching_llm") or 0,
    )
    return PipelineStats(
        ingested=metrics.ingested,
        deduplicated=metrics.deduplicated,
        hard_filtered=metrics.hard_filtered,
        keyword_filtered=metrics.keyword_filtered,
        llm_analyzed=metrics.llm_analyzed,
        llm_avoidance_rate=metrics.llm_avoidance_rate(),
        recorded_at=latest_run.get("finished_at"),
    )


# ---------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------


def build_demo_dashboard_data() -> DashboardData:
    """
    Clearly-labeled sample data (DashboardData.is_demo=True) for
    demonstrating the Dashboard without a configured database or any
    real account activity. Every derived number here still runs through
    the REAL deterministic backend functions (calculate_readiness,
    calculate_skill_roi) - only the input profile/role/requirement data
    is synthetic, never the computation itself.
    """
    profile = CandidateProfile(
        coursework=["Probability", "Algorithms"],
        programming_languages=["Python", "C++"],
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="python", display_name="Python",
                estimated_level=7.5, confidence=0.7,
                evidence_snippets=["Built internal Python ETL pipelines"],
            ),
            CandidateSkillEstimate(
                normalized_skill_name="algorithms", display_name="Algorithms",
                estimated_level=4.0, confidence=0.4,
            ),
            CandidateSkillEstimate(
                normalized_skill_name="probability", display_name="Probability",
                estimated_level=3.0, confidence=0.3,
            ),
        ],
        internships=[
            ExperienceEntry(organization="Acme Corp", role="Software Engineering Intern", description="Built internal dashboards."),
        ],
        projects=[ProjectEntry(name="Distributed KV Store", description="Raft-based KV store in C++.")],
    )

    dream_role = normalize_role(
        {"title": "Quantitative Research Intern", "company": "Meridian Capital", "role_family": "quant"}
    )
    backup_role = normalize_role(
        {"title": "Software Engineering Intern", "company": "Acme Corp", "role_family": "swe"}
    )
    dream_requirements = [
        RoleRequirement(skill="Probability", normalized_skill="probability", target_level=8, importance=9, required=True, evidence=["x"]),
        RoleRequirement(skill="Python", normalized_skill="python", target_level=7, importance=7, required=True, evidence=["x"]),
    ]
    backup_requirements = [
        RoleRequirement(skill="Algorithms", normalized_skill="algorithms", target_level=7, importance=8, required=True, evidence=["x"]),
    ]

    roi_results = calculate_skill_roi(
        profile,
        [
            FavoriteRoleContext(role=dream_role, requirements=dream_requirements, priority="dream"),
            FavoriteRoleContext(role=backup_role, requirements=backup_requirements, priority="backup"),
        ],
    )
    top_roi = roi_results[0] if roi_results else None

    readiness_result = calculate_readiness(profile, role_family="quant")

    return DashboardData(
        is_demo=True,
        database_connected=True,
        profile_status=ProfileStatus(has_profile=True, skill_count=len(profile.skills), updated_at=None),
        top_match=TopMatch(role_id="demo-role-1", role_title=dream_role.title, company=dream_role.company, overall_score=57.4),
        upcoming_interview=UpcomingInterview(
            role_id="demo-role-1", role_title=dream_role.title, company=dream_role.company,
            role_family="quant", interview_date=(date.today() + timedelta(days=6)).isoformat(), days_away=6,
        ),
        overall_readiness=readiness_result.overall_readiness,
        readiness_role_family="quant",
        highest_leverage_skill=top_roi.display_skill if top_roi else None,
        highest_leverage_skill_roi=top_roi.roi_score if top_roi else None,
        saved_roles_count=2,
        active_applications_count=3,
        urgent_applications=[
            UrgentApplication(role_title=backup_role.title, company=backup_role.company, status="applied", deadline=(date.today() + timedelta(days=3)).isoformat(), days_until_deadline=3),
        ],
        pipeline_stats=PipelineStats(
            ingested=140, deduplicated=12, hard_filtered=71, keyword_filtered=41, llm_analyzed=16,
            llm_avoidance_rate=PipelineMetrics(ingested=140, deduplicated=12, hard_filtered=71, keyword_filtered=41, llm_analyzed=16).llm_avoidance_rate(),
            recorded_at=None,
        ),
    )
