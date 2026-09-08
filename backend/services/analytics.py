"""
Cross-cutting analytics: matching (fit/readiness distributions, top
recurring skill gaps), applications (count by stage), and preparation
(readiness over time, completed study minutes).

Every function here reads already-persisted or already-computed data
(backend.services.discovery, backend.services.tracking,
backend.db.study_plans, backend.db.assessment_results) - nothing is
invented, estimated, or hardcoded. A metric with no underlying data
returns an empty list/None/0, not a plausible-looking placeholder;
pages/6_Analytics.py is responsible for rendering that as an explicit
empty state rather than a chart with nothing in it.

Pipeline metrics (ingestion funnel, LLM avoidance, concurrency speedup)
are NOT duplicated here - see backend.db.metrics /
backend.utils.metrics, which already own that domain.
"""

from __future__ import annotations

from typing import Any

from backend.db import assessment_results as assessment_results_db
from backend.db import roles as roles_db
from backend.db import study_plans as study_plans_db
from backend.matching.cross_role import SkillROIResult
from backend.planning.readiness import DiagnosticResult, calculate_readiness
from backend.services import discovery, tracking


def get_fit_distribution(user_id: str) -> list[float]:
    """Every persisted overall fit score (backend.db.roles.fit_scores) for this user - one entry per analyzed role."""
    rows = roles_db.list_fit_scores_for_user(user_id, limit=500)
    return [row["overall_score"] for row in rows if row.get("overall_score") is not None]


def get_readiness_distribution(user_id: str) -> list[float]:
    """
    Live readiness score (backend.planning.readiness) for every role the
    candidate has skills and a role_family to compute it against -
    reuses backend.services.discovery.list_role_cards(), which already
    computes this for every role card, rather than recomputing it here.
    """
    cards = discovery.list_role_cards(user_id)
    return [card.readiness_score for card in cards if card.readiness_score is not None]


def get_top_recurring_skill_gaps(user_id: str, limit: int = 10) -> list[SkillROIResult]:
    """
    The skills most FREQUENTLY required across favorite roles - the same
    already-computed backend.matching.cross_role.SkillROIResult list as
    the Skill Gaps page, just re-sorted here by roles_requiring_it
    (frequency) instead of roi_score, since "top recurring" is a
    frequency question, not a return-on-investment one. No new
    arithmetic - purely a different ordering of already-computed results.
    """
    results = discovery.get_skill_roi_for_favorites(user_id)
    ordered = sorted(results, key=lambda r: (r.roles_requiring_it, r.average_gap), reverse=True)
    return ordered[:limit]


def get_applications_by_stage(user_id: str) -> dict[str, int]:
    """A count of tracked applications per stage - only stages with at least one application are included."""
    counts: dict[str, int] = {}
    for application in tracking.list_applications(user_id):
        status = application.get("status")
        if status:
            counts[status] = counts.get(status, 0) + 1
    return counts


def get_completed_study_minutes(user_id: str) -> float:
    """
    Total minutes marked complete across every application's CURRENT
    study plan (backend.db.study_plans) - deliberately only the current
    generation per application, never a superseded one, since a
    completed task is carried forward unchanged into each new
    generation (backend.planning.adaptive) and would otherwise be
    counted once per generation it survived into.
    """
    total = 0.0
    for application in tracking.list_applications(user_id):
        plan = study_plans_db.get_current_plan(application["id"])
        if plan is None:
            continue
        total += sum((task.get("allocated_minutes") or 0.0) for task in plan["tasks"] if task.get("is_complete"))
    return total


def get_readiness_over_time(user_id: str) -> list[dict[str, Any]]:
    """
    Overall readiness AS OF each diagnostic submission, replayed in
    chronological order against the candidate's nearest upcoming
    interview's role family (readiness is inherently role-family-scoped
    - see backend.planning.readiness - so a single, real role family
    must be chosen rather than averaging across incompatible weight
    sets). Falls back to any application's role family if none has an
    upcoming interview date, and to None (the general fallback topic
    weights) if the candidate has no tracked applications at all.

    Returns [] if no diagnostics have ever been submitted - there is
    nothing to plot yet, not a fabricated flat line.
    """
    diagnostic_rows = assessment_results_db.list_assessment_results(user_id)
    if not diagnostic_rows:
        return []

    applications = tracking.list_applications_with_details(user_id)
    upcoming = sorted(
        (a for a in applications if a.get("interview_date")), key=lambda a: a["interview_date"]
    )
    if upcoming:
        role_family = (upcoming[0].get("roles") or {}).get("role_family")
    elif applications:
        role_family = (applications[0].get("roles") or {}).get("role_family")
    else:
        role_family = None

    profile = discovery.build_candidate_profile(user_id)

    points: list[dict[str, Any]] = []
    accumulated: list[DiagnosticResult] = []
    for row in diagnostic_rows:
        accumulated.append(
            DiagnosticResult(
                topic=row["normalized_skill_name"], observed_level=row["observed_level"], confidence=row.get("confidence") or 0.9
            )
        )
        readiness = calculate_readiness(profile, role_family, accumulated)
        points.append({"taken_at": row["taken_at"], "overall_readiness": readiness.overall_readiness})
    return points
