"""
Cross-role skill-gap analysis, skill ROI, and favorite-role comparison.

Pure, deterministic Python over a candidate's saved/favorite roles - no
LLM calls. Two entry points:

- calculate_skill_roi(): across every favorite role, which skill would
  most improve the candidate's overall position if they invested time
  learning it. See "IMPORTANT: this is a heuristic, not a measurement"
  below before trusting these numbers for anything beyond rough
  prioritization.
- compare_favorite_roles(): a side-by-side comparison of 2-4 favorite
  roles (fit, top gaps, and readiness/prep-hours when available),
  intended to power a future "Compare Favorites" UI.

IMPORTANT: this is a heuristic, not a measurement. skill_roi in
particular multiplies together several numbers of very different
character (a 0-10 gap, an integer role count, a 0-10 importance rating,
a hand-picked priority multiplier) and divides by an estimated learning
cost that is, at best, an educated guess (see DEFAULT_LEARNING_COST_HOURS
below). The resulting score is useful for RANKING skills relative to
each other for THIS candidate's CURRENT favorites - it is not a
calibrated, comparable-across-users, or scientifically validated
estimate of anything. Treat "skill X has the highest ROI" as "skill X
looks like the best use of your time among these options," not as a
number with real-world units.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile
from backend.ingestion.normalize import Role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.gaps import SkillGapResult, calculate_skill_gaps
from backend.matching.scorer import calculate_fit_score

# Multiplier applied to a role's contribution to skill_roi based on how
# much the candidate wants that role. Mirrors backend.db.favorites'
# priority levels (dream/high/interested/backup) without importing that
# module directly - backend/matching stays decoupled from backend/db so
# this module can be tested and reasoned about without a database.
FAVORITE_PRIORITY_MULTIPLIERS: dict[str, float] = {
    "dream": 3.0,
    "high": 2.0,
    "interested": 1.0,
    "backup": 0.5,
}
DEFAULT_FAVORITE_PRIORITY = "interested"

# HEURISTIC, NOT SCIENTIFIC. Rough estimated hours to close a full 0->10
# proficiency gap in a skill, used only so skill_roi's denominator isn't
# pure guesswork times one for every skill alike. These numbers are
# illustrative defaults - "how long a motivated CS student typically
# takes to get comfortable with X" - with no calibration against real
# learning-outcome data. Two ways to make this less made-up over time:
# (1) pass `learning_cost_overrides` to calculate_skill_roi() with
# better numbers for skills you care about, or (2) once
# assessment_results (diagnostics) exist, replace this table with
# hours-to-improvement correlations mined from real user data.
DEFAULT_LEARNING_COST_HOURS: dict[str, float] = {
    "python": 40.0,
    "c++": 80.0,
    "java": 60.0,
    "javascript": 40.0,
    "typescript": 45.0,
    "sql": 25.0,
    "algorithms": 60.0,
    "data structures": 50.0,
    "systems": 70.0,
    "operating systems": 60.0,
    "machine learning": 100.0,
    "deep learning": 110.0,
    "statistics": 50.0,
    "probability": 40.0,
    "probability and statistics": 45.0,
    "linear algebra": 40.0,
    "react": 30.0,
    "docker": 20.0,
    "kubernetes": 40.0,
    "distributed systems": 90.0,
    "market knowledge": 60.0,
}
# Used for any skill not present in DEFAULT_LEARNING_COST_HOURS above -
# a deliberately middle-of-the-road guess, not a measurement.
FALLBACK_LEARNING_COST_HOURS = 50.0


class RoleSummary(BaseModel):
    """A lightweight, UI-friendly reference to one role."""

    external_id: str
    title: str
    company: str


class FavoriteRoleContext(BaseModel):
    """One saved/favorite role's data, as needed for cross-role analysis and comparison."""

    role: Role
    requirements: list[RoleRequirement] = Field(default_factory=list)
    priority: str = DEFAULT_FAVORITE_PRIORITY


def _priority_multiplier(priority: str) -> float:
    return FAVORITE_PRIORITY_MULTIPLIERS.get(priority, FAVORITE_PRIORITY_MULTIPLIERS[DEFAULT_FAVORITE_PRIORITY])


def estimate_learning_cost_hours(normalized_skill: str, overrides: dict[str, float] | None = None) -> float:
    """
    Look up an estimated number of hours to close a full skill-level gap
    (see the HEURISTIC disclaimer at the top of this module). Checks
    `overrides` first (per-call configurability), then the module-level
    DEFAULT_LEARNING_COST_HOURS table, then FALLBACK_LEARNING_COST_HOURS
    for anything unrecognized.
    """
    if overrides and normalized_skill in overrides:
        return overrides[normalized_skill]
    return DEFAULT_LEARNING_COST_HOURS.get(normalized_skill, FALLBACK_LEARNING_COST_HOURS)


class SkillROIResult(BaseModel):
    """
    One skill's cross-favorite-role return-on-investment. `roi_score` is
    `raw_roi` normalized to 0-100 relative to the highest raw_roi among
    the candidate's CURRENT favorites in THIS computation - it is a
    relative ranking aid, not an absolute or portable measurement (see
    this module's docstring).
    """

    normalized_skill: str
    display_skill: str
    roles_requiring_it: int
    average_gap: float
    average_importance: float
    weighted_priority: float
    estimated_learning_cost_hours: float
    raw_roi: float
    roi_score: float = Field(
        ge=0, le=100, description="raw_roi normalized 0-100 relative to the best-ROI skill in this result set."
    )
    affected_roles: list[RoleSummary]


def calculate_skill_roi(
    profile: CandidateProfile,
    favorite_contexts: list[FavoriteRoleContext],
    learning_cost_overrides: dict[str, float] | None = None,
) -> list[SkillROIResult]:
    """
    For every skill required by at least one favorite role, compute:

        raw_roi = average_gap x roles_requiring_it x average_importance
                  x weighted_priority / estimated_learning_cost_hours

    - average_gap: mean raw_gap (0-10, see backend.matching.gaps) for
      this skill across the favorite roles that require it - the
      requirement-importance weighting stays a SEPARATE factor below, so
      it isn't counted twice.
    - roles_requiring_it: how many distinct favorite roles require this
      skill ("frequency_across_saved_roles").
    - average_importance: mean requirement importance (0-10) across
      those roles.
    - weighted_priority: mean FAVORITE_PRIORITY_MULTIPLIER across those
      roles' favorite priorities - a MEAN, not a sum, so this factor
      reflects "how prized are the roles that need this skill"
      independent of roles_requiring_it (which already captures "how
      many roles need it").
    - estimated_learning_cost_hours: see estimate_learning_cost_hours()
      and the HEURISTIC disclaimer at the top of this module.

    Returns results sorted by roi_score descending (most actionable
    first). Empty input returns an empty list.
    """
    if not favorite_contexts:
        return []

    # normalized_skill -> [(SkillGapResult, FavoriteRoleContext), ...],
    # at most one entry per role (defensive dedup - a role's own
    # requirements shouldn't repeat a skill, but don't let it inflate
    # this skill's frequency if it somehow does).
    accumulated: dict[str, list[tuple[SkillGapResult, FavoriteRoleContext]]] = defaultdict(list)
    for context in favorite_contexts:
        seen_in_this_role: set[str] = set()
        for gap in calculate_skill_gaps(profile.skills, context.requirements):
            if gap.normalized_skill in seen_in_this_role:
                continue
            seen_in_this_role.add(gap.normalized_skill)
            accumulated[gap.normalized_skill].append((gap, context))

    raw_by_skill: dict[str, dict[str, Any]] = {}
    for normalized_skill, entries in accumulated.items():
        gaps = [gap for gap, _ in entries]
        contexts = [context for _, context in entries]
        frequency = len(entries)

        average_gap = sum(gap.raw_gap for gap in gaps) / frequency
        average_importance = sum(gap.importance for gap in gaps) / frequency
        weighted_priority = sum(_priority_multiplier(context.priority) for context in contexts) / frequency
        learning_cost = estimate_learning_cost_hours(normalized_skill, learning_cost_overrides)
        raw_roi = (average_gap * frequency * average_importance * weighted_priority) / learning_cost

        raw_by_skill[normalized_skill] = {
            "display_skill": gaps[0].display_skill,
            "frequency": frequency,
            "average_gap": average_gap,
            "average_importance": average_importance,
            "weighted_priority": weighted_priority,
            "learning_cost": learning_cost,
            "raw_roi": raw_roi,
            "affected_roles": [
                RoleSummary(external_id=context.role.external_id, title=context.role.title, company=context.role.company)
                for context in contexts
            ],
        }

    max_raw_roi = max((data["raw_roi"] for data in raw_by_skill.values()), default=0.0)

    results = [
        SkillROIResult(
            normalized_skill=normalized_skill,
            display_skill=data["display_skill"],
            roles_requiring_it=data["frequency"],
            average_gap=round(data["average_gap"], 3),
            average_importance=round(data["average_importance"], 3),
            weighted_priority=round(data["weighted_priority"], 3),
            estimated_learning_cost_hours=data["learning_cost"],
            raw_roi=round(data["raw_roi"], 6),
            roi_score=round(100.0 * data["raw_roi"] / max_raw_roi, 2) if max_raw_roi > 0 else 0.0,
            affected_roles=data["affected_roles"],
        )
        for normalized_skill, data in raw_by_skill.items()
    ]
    results.sort(key=lambda result: result.roi_score, reverse=True)
    return results


class SkillGapSummary(BaseModel):
    """A compact view of one skill gap, for display in a comparison table."""

    skill: str
    gap: float
    importance: float
    required: bool


class FavoriteRoleComparison(BaseModel):
    """
    One favorite role's side-by-side comparison data. `readiness_score`
    and `prep_hours_allocated` are Optional because
    backend.planning.readiness and backend.planning.scheduler are not
    implemented yet - they are None unless the caller supplies
    pre-computed values (see compare_favorite_roles()'s parameters).
    """

    role: RoleSummary
    priority: str
    fit_score: float
    fit_components: dict[str, float]
    readiness_score: float | None = None
    top_gaps: list[SkillGapSummary]
    prep_hours_allocated: float | None = None


def estimate_prep_hours_for_role(
    skill_gaps: list[SkillGapResult], learning_cost_overrides: dict[str, float] | None = None
) -> float:
    """
    Rough total hours to close every one of a single role's skill gaps:
    for each gap, the fraction of a full 0->10 gap still open
    (raw_gap / 10) times estimate_learning_cost_hours() for that skill -
    the same per-skill hour estimates calculate_skill_roi() uses,
    applied here to one role's own gaps rather than aggregated across
    favorites. Same HEURISTIC disclaimer as the rest of this module: a
    ranking aid, not a calibrated time estimate.
    """
    return round(
        sum(
            (gap.raw_gap / 10.0) * estimate_learning_cost_hours(gap.normalized_skill, learning_cost_overrides)
            for gap in skill_gaps
        ),
        1,
    )


def compare_favorite_roles(
    profile: CandidateProfile,
    favorite_contexts: list[FavoriteRoleContext],
    *,
    top_n_gaps: int = 5,
    readiness_scores: dict[str, float] | None = None,
    prep_hours_by_role: dict[str, float] | None = None,
) -> list[FavoriteRoleComparison]:
    """
    Compare 2-4 favorite roles side by side, powering a future "Compare
    Favorites" UI: deterministic fit (backend.matching.scorer, computed
    fresh here), each role's biggest gaps by weighted_gap, and -
    wherever the caller can supply them - readiness and allocated prep
    hours.

    `readiness_scores`/`prep_hours_by_role` are optional dicts keyed by
    role external_id. Neither backend.planning.readiness nor
    backend.planning.scheduler exist yet, so this function cannot
    compute those itself; a role's readiness_score/prep_hours_allocated
    is simply None unless the caller passes a value in for it.

    Raises ValueError if `favorite_contexts` doesn't have between 2 and
    4 entries.
    """
    if not (2 <= len(favorite_contexts) <= 4):
        raise ValueError(f"compare_favorite_roles expects 2-4 favorite roles, got {len(favorite_contexts)}.")

    readiness_scores = readiness_scores or {}
    prep_hours_by_role = prep_hours_by_role or {}

    comparisons: list[FavoriteRoleComparison] = []
    for context in favorite_contexts:
        fit_score = calculate_fit_score(profile, context.role, context.requirements)
        top_gaps = sorted(fit_score.technical_detail.gaps, key=lambda gap: gap.weighted_gap, reverse=True)[:top_n_gaps]

        comparisons.append(
            FavoriteRoleComparison(
                role=RoleSummary(
                    external_id=context.role.external_id, title=context.role.title, company=context.role.company
                ),
                priority=context.priority,
                fit_score=fit_score.overall_score,
                fit_components=fit_score.components.model_dump(),
                readiness_score=readiness_scores.get(context.role.external_id),
                top_gaps=[
                    SkillGapSummary(skill=gap.display_skill, gap=gap.raw_gap, importance=gap.importance, required=gap.required)
                    for gap in top_gaps
                ],
                prep_hours_allocated=prep_hours_by_role.get(context.role.external_id),
            )
        )
    return comparisons


class ComparisonSummary(BaseModel):
    """Three single-winner rollups over an already-computed comparison set. None if there's nothing to summarize."""

    best_current_match: RoleSummary | None = None
    highest_potential_upside: RoleSummary | None = None
    largest_prep_burden: RoleSummary | None = None


def summarize_comparison(comparisons: list[FavoriteRoleComparison]) -> ComparisonSummary:
    """
    Deterministic summaries over compare_favorite_roles()'s output:

    - best_current_match: highest already-computed overall fit_score.
    - highest_potential_upside: largest remaining TECHNICAL fit headroom
      (100 - fit_components["technical"]). Technical fit is the
      dimension most directly addressable through further study/prep -
      unlike constraints/interest/domain, which usually can't be closed
      by studying - so this ranks "which role would benefit most from
      closing skill gaps" rather than raw overall-score headroom, which
      can be dominated by dimensions no amount of prep will move.
    - largest_prep_burden: highest prep_hours_allocated. None if no
      comparison in the set carries a prep_hours_allocated value.

    Returns an all-None ComparisonSummary for an empty input rather than
    raising - callers decide how to render "nothing to compare yet".
    """
    if not comparisons:
        return ComparisonSummary()

    best = max(comparisons, key=lambda c: c.fit_score)
    upside = max(comparisons, key=lambda c: 100.0 - c.fit_components.get("technical", 100.0))

    burdened = [c for c in comparisons if c.prep_hours_allocated is not None]
    heaviest = max(burdened, key=lambda c: c.prep_hours_allocated) if burdened else None

    return ComparisonSummary(
        best_current_match=best.role,
        highest_potential_upside=upside.role,
        largest_prep_burden=heaviest.role if heaviest else None,
    )
