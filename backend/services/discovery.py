"""
Discover service: browsing ingested roles, per-role fit/readiness/gap
summaries, and on-demand deep analysis.

Two very different costs are kept deliberately separate here:

- Readiness (backend.planning.readiness) only needs candidate skills and
  a role's role_family - both already persisted - so it is safe to
  compute live, for every role, on every page load. No LLM call.
- Fit score, skill gaps, and top strengths/gaps additionally need
  RoleRequirement data, which only exists after an LLM extraction call
  (backend.llm.extract_requirements). That must NOT happen implicitly
  for every card on every page load (slow, and spends real money per
  view) - it only happens inside analyze_role(), triggered explicitly
  by the user opening a role's detailed analysis. Once analyzed, the
  requirements and fit score are persisted (backend.db.role_requirements,
  backend.db.roles.upsert_fit_score), so later views of the same role
  are instant reads, not repeat LLM calls.

This module performs the assembly (which role has which data, which
fields need computing); the actual scoring/gap/readiness ARITHMETIC
always comes from backend.matching / backend.planning - never
reimplemented here.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.db import candidates as candidates_db
from backend.db import rationales as rationales_db
from backend.db import role_requirements as role_requirements_db
from backend.db import roles as roles_db
from backend.ingestion.normalize import Role
from backend.llm.extract_requirements import RoleRequirement, extract_role_requirements
from backend.llm.rationale import FitRationale, generate_fit_rationale
from backend.matching.cross_role import (
    DEFAULT_FAVORITE_PRIORITY,
    FAVORITE_PRIORITY_MULTIPLIERS,
    ComparisonSummary,
    FavoriteRoleComparison,
    FavoriteRoleContext,
    SkillROIResult,
    calculate_skill_roi,
    compare_favorite_roles,
    estimate_prep_hours_for_role,
    summarize_comparison,
)
from backend.matching.gaps import SkillGapResult, calculate_skill_gaps
from backend.matching.scorer import FitScoreResult, calculate_fit_score
from backend.planning.readiness import ReadinessResult, calculate_readiness
from backend.planning.scheduler import StudyPlan, generate_study_plan
from backend.services import tracking
from backend.utils.hashing import hash_json

logger = logging.getLogger(__name__)

# Cap on how many roles a "top strengths" list shows per card, per the
# page spec's "top 2-3 matching strengths."
_MAX_CARD_STRENGTHS = 3
# A skill only counts as a "strength" on a card if the candidate meets
# at least this fraction of the target level - a token/near-zero match
# shouldn't be presented as something to be proud of.
_STRENGTH_SATISFACTION_THRESHOLD = 0.5


def list_available_roles(limit: int = 50) -> list[dict[str, Any]]:
    """Most recently ingested roles first."""
    return roles_db.list_roles(limit=limit)


def get_role(role_id: str) -> dict[str, Any] | None:
    """Fetch a single role by its internal id, or None if it doesn't exist."""
    return roles_db.get_role_by_id(role_id)


class RoleCard(BaseModel):
    """
    Everything one Discover role card needs. Fields that require an LLM
    extraction (fit_score, top_strengths, top_gap) are None/empty until
    analyze_role() has been run for this role at least once -
    `analyzed` tells the page which case it's in. readiness_score is
    independently available whenever the candidate has skills and this
    role has a role_family, regardless of analysis status.
    """

    role_id: str
    external_id: str
    title: str
    company: str
    location: str | None = None
    role_family: str | None = None
    description: str | None = None

    is_saved: bool = False
    priority: str | None = None
    notes: str | None = None

    application_status: str | None = None
    deadline: str | None = None
    interview_date: str | None = None

    analyzed: bool = False
    fit_score: float | None = None
    readiness_score: float | None = None
    top_strengths: list[str] = Field(default_factory=list)
    top_gap: str | None = None


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


def build_candidate_profile(user_id: str) -> CandidateProfile:
    """
    The candidate's current skills-only CandidateProfile (no persisted
    coursework/domain_experience/etc. - see this module's docstring for
    that pre-existing data-model limitation), for any caller (this
    module or backend.services.prep) that needs a fresh profile without
    duplicating the skill-row-to-estimate conversion.
    """
    skill_rows = candidates_db.list_candidate_skills(user_id)
    return CandidateProfile(skills=_skill_estimates_from_rows(skill_rows))


def _requirement_from_row(row: dict[str, Any]) -> RoleRequirement:
    return RoleRequirement(
        skill=row.get("display_name") or row["normalized_skill_name"],
        normalized_skill=row["normalized_skill_name"],
        target_level=row["target_level"],
        importance=row["importance"],
        required=row["is_required"],
        evidence=row.get("evidence") or [],
    )


def _role_from_row(row: dict[str, Any]) -> Role:
    return Role(
        external_id=row["external_id"],
        company=row["company"],
        title=row["title"],
        location=row.get("location"),
        url=row.get("url"),
        description=row.get("description"),
        role_family=row.get("role_family"),
        raw_source=row.get("raw_source") or {},
    )


def _top_strengths_and_gap(gaps: list[SkillGapResult]) -> tuple[list[str], str | None]:
    strengths = [gap for gap in gaps if gap.satisfaction_ratio >= _STRENGTH_SATISFACTION_THRESHOLD]
    strengths.sort(key=lambda gap: gap.satisfaction_ratio, reverse=True)
    top_strengths = [gap.display_skill for gap in strengths[:_MAX_CARD_STRENGTHS]]

    gaps_with_room = [gap for gap in gaps if gap.raw_gap > 0]
    gaps_with_room.sort(key=lambda gap: gap.weighted_gap, reverse=True)
    top_gap = gaps_with_room[0].display_skill if gaps_with_room else None

    return top_strengths, top_gap


def list_role_cards(user_id: str) -> list[RoleCard]:
    """
    Build one RoleCard per persisted role. Favorite/application state is
    always real (read from Postgres). Readiness is computed live and
    cheaply. Fit score / strengths / gap are populated only for roles
    that already have persisted requirements (i.e. have been through
    analyze_role() at least once) - otherwise `analyzed=False` and the
    page shows an empty state for those fields.
    """
    role_rows = roles_db.list_roles(limit=200)
    if not role_rows:
        return []

    favorites_by_role = {row["role_id"]: row for row in tracking.list_saved_roles(user_id)}
    applications_by_role = {row["role_id"]: row for row in tracking.list_applications(user_id)}
    fit_scores_by_role = {row["role_id"]: row for row in roles_db.list_fit_scores_for_user(user_id, limit=200)}

    skill_rows = candidates_db.list_candidate_skills(user_id)
    candidate_skills = _skill_estimates_from_rows(skill_rows)

    cards: list[RoleCard] = []
    for role_row in role_rows:
        role_id = role_row["id"]
        favorite = favorites_by_role.get(role_id)
        application = applications_by_role.get(role_id)
        fit_row = fit_scores_by_role.get(role_id)

        readiness_score = None
        if candidate_skills and role_row.get("role_family"):
            readiness_score = calculate_readiness(
                CandidateProfile(skills=candidate_skills), role_row["role_family"]
            ).overall_readiness

        top_strengths: list[str] = []
        top_gap: str | None = None
        analyzed = fit_row is not None
        if analyzed and candidate_skills:
            requirement_rows = role_requirements_db.list_role_requirements(role_id)
            if requirement_rows:
                requirements = [_requirement_from_row(row) for row in requirement_rows]
                gaps = calculate_skill_gaps(candidate_skills, requirements)
                top_strengths, top_gap = _top_strengths_and_gap(gaps)

        cards.append(
            RoleCard(
                role_id=role_id,
                external_id=role_row["external_id"],
                title=role_row["title"],
                company=role_row["company"],
                location=role_row.get("location"),
                role_family=role_row.get("role_family"),
                description=role_row.get("description"),
                is_saved=favorite is not None,
                priority=favorite["priority"] if favorite else None,
                notes=favorite.get("notes") if favorite else None,
                application_status=application["status"] if application else None,
                deadline=application.get("deadline") if application else None,
                interview_date=application.get("interview_date") if application else None,
                analyzed=analyzed,
                fit_score=fit_row["overall_score"] if fit_row else None,
                readiness_score=readiness_score,
                top_strengths=top_strengths,
                top_gap=top_gap,
            )
        )
    return cards


class RoleAnalysis(BaseModel):
    """Full detail for one role, returned by analyze_role()."""

    role: Role
    requirements: list[RoleRequirement]
    fit_result: FitScoreResult
    gaps: list[SkillGapResult]
    readiness: ReadinessResult | None = None


def analyze_role(user_id: str, role_row: dict[str, Any]) -> RoleAnalysis:
    """
    Full analysis for one role. Extracts and persists requirements ONLY
    if none exist yet for this role (the only LLM call in this module);
    always computes and persists a fresh fit score (deterministic,
    reflects the candidate's current skills even if requirements were
    already cached from an earlier candidate/session).

    Raises ValueError if the role has no description to extract
    requirements from and none are cached yet.
    """
    role = _role_from_row(role_row)
    role_id = role_row["id"]

    requirement_rows = role_requirements_db.list_role_requirements(role_id)
    if requirement_rows:
        requirements = [_requirement_from_row(row) for row in requirement_rows]
    else:
        if not role.description:
            raise ValueError(f"Role {role.title!r} has no description to extract requirements from.")
        requirements = extract_role_requirements(role)
        role_requirements_db.upsert_role_requirements(
            role_id,
            [
                {
                    "normalized_skill_name": requirement.normalized_skill,
                    "display_name": requirement.skill,
                    "target_level": requirement.target_level,
                    "importance": requirement.importance,
                    "is_required": requirement.required,
                    "evidence": requirement.evidence,
                }
                for requirement in requirements
            ],
        )

    profile = build_candidate_profile(user_id)

    fit_result = calculate_fit_score(profile, role, requirements)
    roles_db.upsert_fit_score(user_id, role_id, fit_result.overall_score, fit_result.weights_used)

    gaps = calculate_skill_gaps(profile.skills, requirements)
    readiness = calculate_readiness(profile, role.role_family) if role.role_family else None

    return RoleAnalysis(role=role, requirements=requirements, fit_result=fit_result, gaps=gaps, readiness=readiness)


def _rationale_inputs_hash(skill_rows: list[dict[str, Any]], requirements: list[RoleRequirement]) -> str:
    """
    A stable digest of everything a generated rationale actually depends
    on: the candidate's skill estimates and the role's requirements.
    Order-independent (both lists are sorted before hashing) so a
    same-content re-fetch from Postgres never produces a spurious cache
    miss just because row order differs.
    """
    skills_payload = sorted(
        (row["normalized_skill_name"], float(row["estimated_level"]), float(row["confidence"]))
        for row in skill_rows
    )
    requirements_payload = sorted(
        (requirement.normalized_skill, float(requirement.target_level), float(requirement.importance), requirement.required)
        for requirement in requirements
    )
    return hash_json({"skills": skills_payload, "requirements": requirements_payload})


def get_or_generate_rationale(user_id: str, role_row: dict[str, Any], analysis: RoleAnalysis) -> FitRationale:
    """
    Return the evidence-grounded rationale for an already-analyzed role,
    reusing a persisted one (backend.db.rationales) whenever the
    candidate's skills and the role's requirements haven't changed since
    it was generated. generate_fit_rationale() - the most expensive call
    in this module, several Qdrant retrievals plus one LLM call - only
    runs on a genuine cache miss, never on a plain Streamlit rerun of a
    page that's already showing this role.
    """
    role_id = role_row["id"]
    skill_rows = candidates_db.list_candidate_skills(user_id)
    inputs_hash = _rationale_inputs_hash(skill_rows, analysis.requirements)

    cached = rationales_db.get_rationale(user_id, role_id)
    if cached is not None and cached["inputs_hash"] == inputs_hash:
        return FitRationale.model_validate(cached["rationale"])

    profile = CandidateProfile(skills=_skill_estimates_from_rows(skill_rows))
    rationale = generate_fit_rationale(
        profile, analysis.role, analysis.requirements, analysis.fit_result, analysis.gaps, user_id, role_id
    )
    rationales_db.upsert_rationale(user_id, role_id, inputs_hash, rationale.model_dump(mode="json"))
    return rationale


# ---------------------------------------------------------------------
# Cross-favorite comparison and skill ROI (backend.matching.cross_role)
# ---------------------------------------------------------------------


def build_favorite_contexts(user_id: str, role_ids: list[str]) -> list[FavoriteRoleContext]:
    """
    Build one backend.matching.cross_role.FavoriteRoleContext per role
    id, reusing each role's persisted requirements
    (backend.db.role_requirements) if it has already been analyzed -
    this NEVER calls the LLM. A role with no persisted requirements yet
    (never opened via analyze_role()) contributes an empty requirements
    list rather than triggering an extraction as a side effect of
    loading a comparison/ROI view - it simply won't surface in any
    skill-gap-based output for skills it hasn't been analyzed against.
    """
    favorites_by_role = {row["role_id"]: row for row in tracking.list_saved_roles(user_id)}
    contexts: list[FavoriteRoleContext] = []
    for role_id in role_ids:
        role_row = roles_db.get_role_by_id(role_id)
        if role_row is None:
            continue
        requirement_rows = role_requirements_db.list_role_requirements(role_id)
        favorite = favorites_by_role.get(role_id)
        contexts.append(
            FavoriteRoleContext(
                role=_role_from_row(role_row),
                requirements=[_requirement_from_row(row) for row in requirement_rows],
                priority=favorite["priority"] if favorite else DEFAULT_FAVORITE_PRIORITY,
                role_id=role_id,
            )
        )
    return contexts


def compare_roles(user_id: str, role_ids: list[str]) -> tuple[list[FavoriteRoleComparison], ComparisonSummary]:
    """
    Full deterministic side-by-side comparison of 2-4 favorite roles:
    fit (recomputed fresh from each role's persisted requirements
    against the candidate's CURRENT skills - never a stale cached
    fit_scores row), readiness, top skill gaps, and an estimated
    prep-hours burden (backend.matching.cross_role.estimate_prep_hours_for_role),
    plus the three summary rollups
    (backend.matching.cross_role.summarize_comparison). Raises
    ValueError if `role_ids` isn't 2-4 entries (see compare_favorite_roles()).
    """
    profile = build_candidate_profile(user_id)
    contexts = build_favorite_contexts(user_id, role_ids)

    readiness_scores: dict[str, float] = {}
    prep_hours_by_role: dict[str, float] = {}
    for context in contexts:
        gaps = calculate_skill_gaps(profile.skills, context.requirements)
        prep_hours_by_role[context.role.external_id] = estimate_prep_hours_for_role(gaps)
        if context.role.role_family:
            readiness_scores[context.role.external_id] = calculate_readiness(
                profile, context.role.role_family
            ).overall_readiness

    comparisons = compare_favorite_roles(
        profile, contexts, readiness_scores=readiness_scores, prep_hours_by_role=prep_hours_by_role
    )
    return comparisons, summarize_comparison(comparisons)


def get_skill_roi_for_favorites(user_id: str) -> list[SkillROIResult]:
    """
    Cross-role skill ROI (backend.matching.cross_role.calculate_skill_roi)
    across EVERY one of a user's saved roles - not just a compared
    subset - for a "common skill gaps across your favorites" view.
    """
    favorite_role_ids = [row["role_id"] for row in tracking.list_saved_roles(user_id)]
    contexts = build_favorite_contexts(user_id, favorite_role_ids)
    profile = build_candidate_profile(user_id)
    return calculate_skill_roi(profile, contexts)


# ---------------------------------------------------------------------
# Interview prep plan for one tracked application
# ---------------------------------------------------------------------

# Assumed daily study time when an application has no
# hours_available_per_day of its own set yet - a documented default, not
# a measurement (mirrors the HEURISTIC framing in backend.matching.cross_role).
DEFAULT_PREP_HOURS_PER_DAY = 2.0


def build_study_plan_for_application(user_id: str, role_id: str) -> StudyPlan:
    """
    Build a fresh, fully deterministic interview-prep study plan
    (backend.planning.scheduler.generate_study_plan) for one tracked
    application: the role's persisted requirements (extracted via
    analyze_role() only if this role has never been analyzed before),
    the candidate's current skill gaps and readiness against it, the
    application's own interview_date and hours_available_per_day, and a
    favorite-priority multiplier if the role is favorited.

    Always recomputed fresh from current data - nothing about the plan
    is cached/persisted, so calling this again after skills, the role's
    requirements, or the application's interview_date/hours change
    immediately reflects that, with no separate invalidation step needed.

    Raises ValueError if the application has no interview_date set (there
    is nothing to schedule against) or the role has no description/cached
    requirements to analyze.
    """
    application = tracking.get_application_status(user_id, role_id)
    if application is None or not application.get("interview_date"):
        raise ValueError("This application has no interview date set yet.")

    role_row = roles_db.get_role_by_id(role_id)
    if role_row is None:
        raise ValueError("This role no longer exists.")

    analysis = analyze_role(user_id, role_row)

    favorite = tracking.get_favorite(user_id, role_id)
    priority = favorite["priority"] if favorite else DEFAULT_FAVORITE_PRIORITY
    priority_multiplier = FAVORITE_PRIORITY_MULTIPLIERS.get(priority, 1.0)
    hours_per_day = application.get("hours_available_per_day") or DEFAULT_PREP_HOURS_PER_DAY

    return generate_study_plan(
        analysis.gaps,
        interview_date=date.fromisoformat(application["interview_date"]),
        current_date=date.today(),
        hours_available_per_day=hours_per_day,
        readiness=analysis.readiness,
        favorite_priority_multiplier=priority_multiplier,
    )


def get_requirement_evidence_for_skill(role_id: str, normalized_skill: str) -> list[str]:
    """
    The persisted job-posting evidence quotes
    (backend.db.role_requirements.evidence) backing one role's
    requirement for one skill - a direct Postgres read, no LLM/Qdrant
    call. Returns [] if this role has no persisted requirement for that
    skill (never analyzed, or the skill isn't required here).
    """
    for row in role_requirements_db.list_role_requirements(role_id):
        if row["normalized_skill_name"] == normalized_skill:
            return row.get("evidence") or []
    return []
