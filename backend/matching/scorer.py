"""
Deterministic fit scoring.

Computes the overall candidate-job fit score as a weighted sum of six
component scores (technical, experience, coursework, domain, interest,
constraints), each independently 0-100. The LLM never sees or chooses
any of these numbers - every score in this module is arithmetic over
structured CandidateProfile / Role / RoleRequirement data. The LLM's
only role anywhere in RoleRadar is to explain a score already computed
here (see backend/llm/rationale.py, not yet implemented).

Some components (experience, domain, interest, constraints) currently
rely on simple heuristics because the data model does not yet capture
everything a full implementation would need (e.g. there is no
"years of experience required" field on RoleRequirement, and
CandidateProfile has no explicit interest/constraint fields yet). Each
heuristic is documented at its function definition, returns a `note`
explaining the limitation, and is designed to degrade to a neutral
score rather than to zero when data is missing - the goal is to avoid
an under-specified input arbitrarily destroying a candidate's score.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.candidate.skills import normalize_skill_name
from backend.ingestion.normalize import Role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.confidence import aggregate_confidence, confidence_label
from backend.matching.gaps import SkillGapResult, calculate_skill_gaps

# A required requirement's importance counts this many times as much as
# the same importance value on a preferred/optional requirement, when
# aggregating gap satisfaction into a 0-100 component score. This only
# affects score *weighting* - the raw_gap/weighted_gap values reported
# by calculate_skill_gaps() are unaffected, so gap-based prep
# prioritization (backend/planning) can apply its own weighting later.
REQUIRED_IMPORTANCE_MULTIPLIER = 1.5

DEFAULT_WEIGHTS: dict[str, float] = {
    "technical": 0.35,
    "experience": 0.20,
    "coursework": 0.10,
    "domain": 0.15,
    "interest": 0.10,
    "constraints": 0.10,
}

# Role-family-specific weight overrides, keyed by normalized role_family.
# Empty for now - falls back to DEFAULT_WEIGHTS for every role family
# until specific overrides are validated (e.g. a "quant" family might
# weight technical higher and coursework lower). Each override must sum
# to 1.0; that invariant is checked in get_weights_for_role_family().
ROLE_FAMILY_WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {}


def _clamp_score(value: float) -> float:
    """Clamp a component or overall score into the valid [0, 100] range."""
    return max(0.0, min(100.0, value))


def _effective_weight(importance: float, required: bool) -> float:
    """
    A requirement's contribution weight when aggregating gap
    satisfaction into a fit score: its importance, boosted for required
    requirements so that missing a required skill hurts the score more
    than missing an equally-important optional one. Shared by
    calculate_technical_fit() (over SkillGapResult) and
    calculate_coursework_fit() (over RoleRequirement) since both carry
    the same importance/required fields.
    """
    multiplier = REQUIRED_IMPORTANCE_MULTIPLIER if required else 1.0
    return importance * multiplier


class ComponentFitResult(BaseModel):
    """A single component's score plus enough structured detail to explain it."""

    score: float
    details: list[dict[str, Any]] = Field(default_factory=list)
    note: str | None = None


class TechnicalFitResult(BaseModel):
    """Technical fit score plus the full per-requirement gap breakdown and its aggregate confidence."""

    score: float
    gaps: list[SkillGapResult] = Field(default_factory=list)
    aggregate_confidence: float
    confidence_label: str
    note: str | None = None


class FitScoreComponents(BaseModel):
    technical: float
    experience: float
    coursework: float
    domain: float
    interest: float
    constraints: float


class FitScoreResult(BaseModel):
    """The complete output of calculate_fit_score(): overall score, components, and full detail for each."""

    overall_score: float
    components: FitScoreComponents
    weights_used: dict[str, float]
    technical_detail: TechnicalFitResult
    experience_detail: ComponentFitResult
    coursework_detail: ComponentFitResult
    domain_detail: ComponentFitResult
    interest_detail: ComponentFitResult
    constraint_detail: ComponentFitResult


def get_weights_for_role_family(role_family: str | None) -> dict[str, float]:
    """
    Return the component weights to use for a given role family,
    falling back to DEFAULT_WEIGHTS when no override is registered.
    This is the extension point for role-family-specific tuning (e.g. a
    quant role weighting technical/domain higher) without changing
    calculate_fit_score() itself.
    """
    if role_family:
        override = ROLE_FAMILY_WEIGHT_OVERRIDES.get(normalize_skill_name(role_family))
        if override is not None:
            return override
    return DEFAULT_WEIGHTS


def calculate_technical_fit(
    candidate_skills: list[CandidateSkillEstimate], requirements: list[RoleRequirement]
) -> TechnicalFitResult:
    """
    Technical fit: for each requirement, how much of the target level
    has the candidate satisfied (capped at 100% credit - exceeding a
    requirement doesn't earn extra points), aggregated with required
    requirements weighted more heavily than optional ones
    (REQUIRED_IMPORTANCE_MULTIPLIER).

    Confidence is deliberately NOT folded into this score - an
    under-evidenced but plausible estimate scores the same as a
    well-evidenced one with the same level. Instead, aggregate_confidence
    is reported alongside the score so callers can display "72% fit,
    moderate confidence" rather than silently discounting the score.

    If there are no requirements to evaluate, this returns a full score
    with a note: a role with zero extracted requirements more likely
    indicates missing/failed extraction than a role with no technical
    bar, so this should be treated as low-signal, not a true perfect
    match.
    """
    gap_results = calculate_skill_gaps(candidate_skills, requirements)

    if not gap_results:
        return TechnicalFitResult(
            score=100.0,
            gaps=[],
            aggregate_confidence=0.0,
            confidence_label=confidence_label(0.0),
            note=(
                "No technical requirements were extracted for this role; defaulting to "
                "a full technical fit score since there is nothing to fail against. "
                "Treat as low-signal rather than a true perfect match."
            ),
        )

    total_weight = 0.0
    weighted_satisfaction = 0.0
    for gap in gap_results:
        requirement_weight = _effective_weight(gap.importance, gap.required)
        total_weight += requirement_weight
        weighted_satisfaction += requirement_weight * gap.satisfaction_ratio

    if total_weight <= 0:
        score = 100.0
        note = "All requirements had zero importance; defaulting to a full technical fit score."
    else:
        score = 100.0 * weighted_satisfaction / total_weight
        note = None

    confidence = aggregate_confidence(gap_results)
    return TechnicalFitResult(
        score=_clamp_score(score),
        gaps=gap_results,
        aggregate_confidence=confidence,
        confidence_label=confidence_label(confidence),
        note=note,
    )


def calculate_experience_fit(profile: CandidateProfile) -> ComponentFitResult:
    """
    HEURISTIC (data model limitation): RoleRequirement does not currently
    capture an experience dimension separate from skills (e.g. no
    "2+ years professional experience" field), so this measures only how
    much internship/project/research evidence the candidate has, without
    comparing it against a role-specific bar. It will need enrichment
    once experience-specific requirements are extracted per role.

    Score = up to 60 points for internships (30 each, max 2 counted) +
    up to 30 points for projects (10 each, max 3 counted) + up to 10
    points for research (10 for having at least 1).
    """
    internship_points = min(len(profile.internships), 2) * 30.0
    project_points = min(len(profile.projects), 3) * 10.0
    research_points = 10.0 if profile.research else 0.0
    score = internship_points + project_points + research_points

    return ComponentFitResult(
        score=_clamp_score(score),
        details=[
            {"signal": "internships", "count": len(profile.internships), "points": internship_points},
            {"signal": "projects", "count": len(profile.projects), "points": project_points},
            {"signal": "research", "count": len(profile.research), "points": research_points},
        ],
        note=(
            "Heuristic: measures volume of internship/project/research evidence only. "
            "Does not yet compare against a role-specific years-of-experience requirement, "
            "since RoleRequirement has no such field today."
        ),
    )


def calculate_coursework_fit(
    profile: CandidateProfile, requirements: list[RoleRequirement]
) -> ComponentFitResult:
    """
    Coursework fit measures formal academic exposure to each
    requirement's knowledge area (e.g. a "Probability" course covering a
    "probability" requirement), as distinct from technical_fit's
    measure of demonstrated applied skill. The same requirement list can
    legitimately score independently on both dimensions.

    HEURISTIC: coverage is a normalized substring match between a
    requirement's normalized_skill and the candidate's coursework
    entries - there's no notion of "level" for a completed course, so
    this is covered/not-covered rather than graded like technical_fit.
    """
    if not requirements:
        return ComponentFitResult(
            score=100.0,
            details=[],
            note="No requirements were extracted for this role; defaulting to full coursework fit.",
        )

    normalized_coursework = [normalize_skill_name(course) for course in profile.coursework]

    total_weight = 0.0
    covered_weight = 0.0
    details: list[dict[str, Any]] = []
    for requirement in requirements:
        weight = _effective_weight(requirement.importance, requirement.required)
        total_weight += weight
        key = normalize_skill_name(requirement.normalized_skill)
        covered = any(key in course or course in key for course in normalized_coursework)
        if covered:
            covered_weight += weight
        details.append({"skill": requirement.skill, "covered": covered, "weight": weight})

    score = 100.0 * covered_weight / total_weight if total_weight > 0 else 100.0
    return ComponentFitResult(score=_clamp_score(score), details=details, note=None)


def calculate_domain_fit(profile: CandidateProfile, role: Role) -> ComponentFitResult:
    """
    HEURISTIC (data model limitation): there is no structured
    "domain requirements" list extracted per role - only
    RoleRequirement's flat skill list. Domain fit is approximated by
    word-overlap between the candidate's self-reported domain_experience
    and the role's title/role_family text.

    Weak or absent evidence resolves to a neutral-to-moderate score
    rather than 0, since a keyword miss is weak evidence of a true
    domain mismatch: with matches, score ranges 30-100 based on overlap
    fraction; with no domain_experience or no role signal to compare
    against, score defaults to a neutral 50.
    """
    if not profile.domain_experience:
        return ComponentFitResult(
            score=50.0,
            details=[],
            note="Candidate profile has no domain_experience entries; defaulting to a neutral score.",
        )

    role_signal = normalize_skill_name(" ".join(filter(None, [role.role_family, role.title])))
    if not role_signal:
        return ComponentFitResult(
            score=50.0,
            details=[],
            note="Role has no title/role_family text to compare against; defaulting to a neutral score.",
        )

    role_words = set(role_signal.split())
    matches = []
    for entry in profile.domain_experience:
        entry_words = set(normalize_skill_name(entry).split())
        if entry_words & role_words:
            matches.append(entry)

    match_fraction = len(matches) / len(profile.domain_experience)
    score = 30.0 + 70.0 * match_fraction
    return ComponentFitResult(
        score=_clamp_score(score),
        details=[{"domain_experience": e, "matched": e in matches} for e in profile.domain_experience],
        note=(
            "Heuristic: word-overlap between candidate domain_experience and role title/family. "
            "Floors at 30 rather than 0 because this is weak evidence of mismatch, not proof."
        ),
    )


def calculate_interest_fit(
    role: Role, declared_interests: list[str] | None = None
) -> ComponentFitResult:
    """
    HEURISTIC (data model limitation): CandidateProfile does not
    currently capture explicit candidate-stated interests (unlike the
    `candidate_profiles.interests` column reserved for this in the
    schema). Until that input exists, this defaults to a neutral score.
    `declared_interests` is accepted as a forward-compatible parameter so
    callers with that data (e.g. a future onboarding survey) can pass it
    in without an API change.
    """
    if not declared_interests:
        return ComponentFitResult(
            score=50.0,
            details=[],
            note=(
                "No declared candidate interests are available (CandidateProfile has no "
                "interest field yet); defaulting to a neutral score."
            ),
        )

    role_signal = normalize_skill_name(" ".join(filter(None, [role.role_family, role.title])))
    role_words = set(role_signal.split())
    normalized_interests = [normalize_skill_name(interest) for interest in declared_interests]
    matches = [i for i in normalized_interests if set(i.split()) & role_words]

    match_fraction = len(matches) / len(normalized_interests) if normalized_interests else 0.0
    score = 30.0 + 70.0 * match_fraction
    return ComponentFitResult(
        score=_clamp_score(score),
        details=[{"interest": i, "matched": i in matches} for i in normalized_interests],
        note="Heuristic: word-overlap between declared interests and role title/family.",
    )


def calculate_constraint_fit(
    role: Role, candidate_constraints: dict[str, Any] | None = None
) -> ComponentFitResult:
    """
    HEURISTIC (data model limitation): candidate location/work-
    authorization preferences are not yet captured anywhere in the data
    model. `candidate_constraints` is accepted as a forward-compatible
    parameter (e.g. {"remote_only": True, "preferred_locations": [...]})
    for once that input exists. With no constraints supplied, this
    defaults to a full score: absence of a stated constraint means
    nothing is known to conflict with the role, not that a comparison
    was skipped.
    """
    if not candidate_constraints:
        return ComponentFitResult(
            score=100.0,
            details=[],
            note=(
                "No candidate constraints are captured in the current data model yet "
                "(e.g. location/work-authorization preferences); defaulting to full "
                "constraint fit since nothing is known to conflict with this role."
            ),
        )

    violations: list[str] = []
    role_location = normalize_skill_name(role.location) if role.location else None

    if candidate_constraints.get("remote_only") and role_location and role_location != "remote":
        violations.append(f"Role location '{role.location}' is not remote, but candidate requires remote-only.")

    preferred_locations = candidate_constraints.get("preferred_locations")
    if preferred_locations and role_location:
        normalized_preferred = {normalize_skill_name(loc) for loc in preferred_locations}
        if not any(p in role_location or role_location in p for p in normalized_preferred):
            violations.append(
                f"Role location '{role.location}' is not among candidate's preferred locations."
            )

    score = 100.0 if not violations else 20.0
    return ComponentFitResult(
        score=_clamp_score(score),
        details=[{"violation": v} for v in violations],
        note=None,
    )


def calculate_fit_score(
    profile: CandidateProfile,
    role: Role,
    requirements: list[RoleRequirement],
    declared_interests: list[str] | None = None,
    candidate_constraints: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> FitScoreResult:
    """
    Compute the complete deterministic fit score: six component scores
    (each 0-100) combined into a single weighted overall score (0-100).

    `weights` defaults to get_weights_for_role_family(role.role_family),
    so a caller never has to think about weight selection unless they
    want to override it explicitly (e.g. for A/B testing a role-family
    override before registering it in ROLE_FAMILY_WEIGHT_OVERRIDES).
    """
    resolved_weights = weights or get_weights_for_role_family(role.role_family)

    technical = calculate_technical_fit(profile.skills, requirements)
    experience = calculate_experience_fit(profile)
    coursework = calculate_coursework_fit(profile, requirements)
    domain = calculate_domain_fit(profile, role)
    interest = calculate_interest_fit(role, declared_interests)
    constraints = calculate_constraint_fit(role, candidate_constraints)

    components = FitScoreComponents(
        technical=technical.score,
        experience=experience.score,
        coursework=coursework.score,
        domain=domain.score,
        interest=interest.score,
        constraints=constraints.score,
    )

    overall = (
        components.technical * resolved_weights["technical"]
        + components.experience * resolved_weights["experience"]
        + components.coursework * resolved_weights["coursework"]
        + components.domain * resolved_weights["domain"]
        + components.interest * resolved_weights["interest"]
        + components.constraints * resolved_weights["constraints"]
    )

    return FitScoreResult(
        overall_score=_clamp_score(overall),
        components=components,
        weights_used=resolved_weights,
        technical_detail=technical,
        experience_detail=experience,
        coursework_detail=coursework,
        domain_detail=domain,
        interest_detail=interest,
        constraint_detail=constraints,
    )
