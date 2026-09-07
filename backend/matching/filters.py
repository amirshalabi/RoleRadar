"""
Deterministic hard filters.

The first, cheapest stage of the matching pipeline: eliminate roles that
are obviously wrong for a candidate using plain Python comparisons,
before any embedding or LLM call is made. No filter in this module ever
calls an LLM.

Two filtering stages run in sequence, each only on the survivors of the
one before it:

  1. Hard constraints (filter_expired_deadline, filter_employment_type,
     filter_role_family, filter_location, filter_seniority,
     filter_graduation_year, filter_major, filter_experience) - clear
     eligibility rules.
  2. Keyword/skill overlap (filter_keyword_skill_overlap) - a coarse,
     high-recall lexical check meant only to catch postings with zero
     relation to anything the candidate knows.

A third stage (semantic/embedding retrieval) is the natural next step
and is intentionally NOT built here - see run_filter_pipeline()'s
docstring for exactly where it plugs in.

Every individual filter returns pass/fail plus a machine-readable reason
code (module-level REASON_* constants) and a human-readable detail
string, and never fails a role for lack of data it needs - see
"Do not force filtering when required data is absent" in each filter's
docstring for what "insufficient data" means for that filter.

Role-specific facts this module needs (deadline, employment type,
seniority, graduation-year restriction, allowed majors, minimum years of
experience) are not part of the canonical Role model - they are
inconsistent across sources in the same way title/company are, so they
are read here directly from Role.raw_source via the same
first-alias-wins pattern backend/ingestion/normalize.py uses for the
fields that *are* canonical.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile
from backend.candidate.skills import normalize_skill_name
from backend.ingestion.normalize import Role

# ---------------------------------------------------------------------
# Machine-readable reason codes
# ---------------------------------------------------------------------
REASON_PASS = "pass"
REASON_SKIPPED_INSUFFICIENT_DATA = "skipped_insufficient_data"
REASON_EXPIRED_DEADLINE = "expired_deadline"
REASON_EMPLOYMENT_TYPE_MISMATCH = "employment_type_mismatch"
REASON_ROLE_FAMILY_MISMATCH = "role_family_mismatch"
REASON_LOCATION_MISMATCH = "location_mismatch"
REASON_SENIORITY_MISMATCH = "seniority_mismatch"
REASON_GRADUATION_YEAR_MISMATCH = "graduation_year_mismatch"
REASON_MAJOR_RESTRICTED = "major_restricted"
REASON_EXPERIENCE_MISMATCH = "experience_mismatch"
REASON_NO_KEYWORD_SKILL_OVERLAP = "no_keyword_skill_overlap"

STAGE_HARD_CONSTRAINTS = "hard_constraints"
STAGE_KEYWORD_SKILL_OVERLAP = "keyword_skill_overlap"


class FilterOutcome(BaseModel):
    """One filter's verdict on one role: pass/fail plus why, machine- and human-readable."""

    filter_name: str
    passed: bool
    reason: str
    detail: str


class CandidateFilterContext(BaseModel):
    """
    Candidate-side inputs the hard filters compare a role against.

    `known_skills` and `graduation_year`/`major` can be derived from a
    CandidateProfile (see build_candidate_filter_context()).
    `target_role_families`, `preferred_locations`, `remote_only`,
    `seniority_level`, `acceptable_employment_types`, and
    `max_experience_years` are not captured anywhere in CandidateProfile
    today, so they must be supplied explicitly (e.g. from onboarding
    preferences) - every field defaults to "no preference declared",
    which is what makes each corresponding filter skip rather than
    reject when the candidate hasn't stated one.
    """

    known_skills: list[str] = Field(default_factory=list)
    graduation_year: int | None = None
    major: str | None = None
    target_role_families: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    seniority_level: str | None = None
    acceptable_employment_types: list[str] = Field(default_factory=list)
    max_experience_years: float | None = None


def build_candidate_filter_context(profile: CandidateProfile, **overrides: Any) -> CandidateFilterContext:
    """
    Build a CandidateFilterContext from a CandidateProfile, filling in
    known_skills/graduation_year/major from it. Preferences the profile
    doesn't capture (target_role_families, preferred_locations,
    seniority_level, acceptable_employment_types, max_experience_years)
    can be passed as keyword overrides; any not supplied are left at
    their "no preference declared" default.
    """
    known_skills = {normalize_skill_name(skill.normalized_skill_name) for skill in profile.skills}
    known_skills.update(normalize_skill_name(lang) for lang in profile.programming_languages)
    known_skills.update(normalize_skill_name(fw) for fw in profile.frameworks)
    known_skills.update(normalize_skill_name(tool) for tool in profile.tools)

    graduation_year = None
    major = None
    if profile.education:
        primary_education = profile.education[0]
        graduation_year = primary_education.graduation_year
        major = primary_education.major

    fields: dict[str, Any] = {
        "known_skills": sorted(known_skills),
        "graduation_year": graduation_year,
        "major": major,
    }
    fields.update(overrides)
    return CandidateFilterContext(**fields)


def _pass(filter_name: str, detail: str) -> FilterOutcome:
    return FilterOutcome(filter_name=filter_name, passed=True, reason=REASON_PASS, detail=detail)


def _skip(filter_name: str, detail: str) -> FilterOutcome:
    return FilterOutcome(
        filter_name=filter_name, passed=True, reason=REASON_SKIPPED_INSUFFICIENT_DATA, detail=detail
    )


def _fail(filter_name: str, reason: str, detail: str) -> FilterOutcome:
    return FilterOutcome(filter_name=filter_name, passed=False, reason=reason, detail=detail)


def _first_present_raw(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty raw value found under any of `keys`, unconverted."""
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return None


def _first_present_text(raw: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty value found under any of `keys`, as a stripped string."""
    value = _first_present_raw(raw, keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------
# Stage 1: hard constraint filters
# ---------------------------------------------------------------------

_DEADLINE_KEYS = (
    "deadline",
    "applicationDeadline",
    "application_deadline",
    "closingDate",
    "closing_date",
    "expiresAt",
    "expires_at",
)


def filter_expired_deadline(
    role: Role, candidate: CandidateFilterContext, *, today: date | None = None
) -> FilterOutcome:
    """Fail if the role's application deadline has already passed. Skips if no deadline is present or parseable."""
    name = "expired_deadline"
    resolved_today = today or date.today()
    deadline = _parse_date(_first_present_raw(role.raw_source, _DEADLINE_KEYS))
    if deadline is None:
        return _skip(name, "No parseable application deadline found in role data.")
    if deadline < resolved_today:
        return _fail(
            name,
            REASON_EXPIRED_DEADLINE,
            f"Application deadline {deadline.isoformat()} has passed (today: {resolved_today.isoformat()}).",
        )
    return _pass(name, f"Deadline {deadline.isoformat()} has not passed.")


_EMPLOYMENT_TYPE_KEYS = ("employmentType", "employment_type", "jobType", "job_type", "type")


def filter_employment_type(role: Role, candidate: CandidateFilterContext) -> FilterOutcome:
    """Fail if the role's employment type isn't among the candidate's acceptable types. Skips if either side is unstated."""
    name = "employment_type"
    if not candidate.acceptable_employment_types:
        return _skip(name, "Candidate declared no employment-type preference.")
    role_value = _first_present_text(role.raw_source, _EMPLOYMENT_TYPE_KEYS)
    if role_value is None:
        return _skip(name, "Role data does not specify an employment type.")

    normalized_role_value = normalize_skill_name(role_value)
    acceptable = {normalize_skill_name(t) for t in candidate.acceptable_employment_types}
    if normalized_role_value in acceptable:
        return _pass(name, f"Employment type '{role_value}' is acceptable.")
    return _fail(
        name,
        REASON_EMPLOYMENT_TYPE_MISMATCH,
        f"Employment type '{role_value}' is not among accepted types {sorted(candidate.acceptable_employment_types)}.",
    )


def filter_role_family(role: Role, candidate: CandidateFilterContext) -> FilterOutcome:
    """Fail if the role's family isn't among the candidate's target families. Skips if either side is unstated."""
    name = "role_family"
    if not candidate.target_role_families:
        return _skip(name, "Candidate declared no target role families.")
    if not role.role_family:
        return _skip(name, "Role has no role_family classification.")

    normalized_role_family = normalize_skill_name(role.role_family)
    acceptable = {normalize_skill_name(f) for f in candidate.target_role_families}
    if normalized_role_family in acceptable:
        return _pass(name, f"Role family '{role.role_family}' matches candidate's target families.")
    return _fail(
        name,
        REASON_ROLE_FAMILY_MISMATCH,
        f"Role family '{role.role_family}' is not among candidate's target families {sorted(candidate.target_role_families)}.",
    )


_WORKPLACE_TYPE_KEYS = ("workplaceType", "workplace_type", "locationType", "location_type")
_LOCATION_REQUIRED_KEYS = ("locationRequired", "location_required", "onSiteRequired", "on_site_required")


def _is_location_explicitly_required(raw: dict[str, Any]) -> bool | None:
    """Return True/False if the posting explicitly states an on-site/remote policy, else None (unknown)."""
    explicit_flag = _first_present_raw(raw, _LOCATION_REQUIRED_KEYS)
    if isinstance(explicit_flag, bool):
        return explicit_flag

    workplace_type = _first_present_text(raw, _WORKPLACE_TYPE_KEYS)
    if workplace_type:
        normalized = normalize_skill_name(workplace_type)
        if "remote" in normalized or "anywhere" in normalized:
            return False
        compact = normalized.replace("-", "").replace(" ", "")
        if "onsite" in compact or "hybrid" in compact or "inperson" in compact:
            return True
    return None


def filter_location(role: Role, candidate: CandidateFilterContext) -> FilterOutcome:
    """
    Fail only when the posting EXPLICITLY requires on-site/hybrid
    presence (via a locationRequired flag or a workplaceType of
    on-site/hybrid) and that location isn't among the candidate's
    preferred locations. Skips for remote-friendly postings, postings
    with no workplace-type signal at all, or when either side lacks
    the data needed to compare - a bare `location` string alone is not
    treated as a hard requirement, since many postings list one purely
    informationally.
    """
    name = "location"
    if _is_location_explicitly_required(role.raw_source) is not True:
        return _skip(name, "Role does not explicitly require an on-site/hybrid location.")
    if not candidate.preferred_locations:
        return _skip(name, "Candidate declared no location preference.")
    if not role.location:
        return _skip(name, "Role does not specify a location despite requiring on-site presence.")

    normalized_role_location = normalize_skill_name(role.location)
    normalized_preferred = {normalize_skill_name(loc) for loc in candidate.preferred_locations}
    if any(pref in normalized_role_location or normalized_role_location in pref for pref in normalized_preferred):
        return _pass(name, f"Role location '{role.location}' matches candidate's preferred locations.")
    return _fail(
        name,
        REASON_LOCATION_MISMATCH,
        f"Role requires on-site presence in '{role.location}', which is not among candidate's "
        f"preferred locations {sorted(candidate.preferred_locations)}.",
    )


_SENIORITY_KEYS = ("seniority", "seniorityLevel", "seniority_level", "experienceLevel", "experience_level", "level")
_SENIOR_TERMS = ("senior", "staff", "principal", "lead", "director", "manager")
_JUNIOR_TERMS = ("intern", "internship", "entry", "entrylevel", "newgrad", "junior", "associate", "student")


def filter_seniority(role: Role, candidate: CandidateFilterContext) -> FilterOutcome:
    """
    Fail only the obvious case: a role's seniority text reads senior+
    (senior/staff/principal/lead/director/manager) while the candidate
    has declared a junior-facing level (intern/entry/new-grad/junior/
    associate/student). Anything else - including seniority terms this
    heuristic doesn't recognize - passes rather than guesses. Skips if
    either side hasn't stated a level.
    """
    name = "seniority"
    if not candidate.seniority_level:
        return _skip(name, "Candidate declared no seniority level.")
    role_value = _first_present_text(role.raw_source, _SENIORITY_KEYS)
    if not role_value:
        return _skip(name, "Role does not specify a seniority level.")

    normalized_role_value = normalize_skill_name(role_value).replace("-", "").replace(" ", "")
    normalized_candidate_value = normalize_skill_name(candidate.seniority_level).replace("-", "").replace(" ", "")

    role_reads_senior = any(term in normalized_role_value for term in _SENIOR_TERMS)
    candidate_reads_junior = any(term in normalized_candidate_value for term in _JUNIOR_TERMS)

    if role_reads_senior and candidate_reads_junior:
        return _fail(
            name,
            REASON_SENIORITY_MISMATCH,
            f"Role seniority '{role_value}' appears to require senior-level experience, above "
            f"candidate's declared '{candidate.seniority_level}' level.",
        )
    return _pass(
        name,
        f"Role seniority '{role_value}' is not an obvious mismatch for candidate's declared "
        f"'{candidate.seniority_level}' level.",
    )


_GRAD_YEAR_MIN_KEYS = ("minGraduationYear", "min_graduation_year", "gradYearMin", "grad_year_min")
_GRAD_YEAR_MAX_KEYS = ("maxGraduationYear", "max_graduation_year", "gradYearMax", "grad_year_max")
_GRAD_YEAR_EXACT_KEYS = ("graduationYear", "graduation_year", "gradYear", "grad_year", "classYear", "class_year")


def filter_graduation_year(role: Role, candidate: CandidateFilterContext) -> FilterOutcome:
    """Fail if the candidate's graduation year falls outside the role's stated range/value. Skips if either side is unstated."""
    name = "graduation_year"
    if candidate.graduation_year is None:
        return _skip(name, "Candidate graduation year is unknown.")

    raw_min = _first_present_raw(role.raw_source, _GRAD_YEAR_MIN_KEYS)
    raw_max = _first_present_raw(role.raw_source, _GRAD_YEAR_MAX_KEYS)
    raw_exact = _first_present_raw(role.raw_source, _GRAD_YEAR_EXACT_KEYS)
    if raw_min is None and raw_max is None and raw_exact is None:
        return _skip(name, "Role does not specify a graduation-year restriction.")

    try:
        min_year = int(raw_min) if raw_min is not None else None
        max_year = int(raw_max) if raw_max is not None else None
        exact_year = int(raw_exact) if raw_exact is not None else None
    except (TypeError, ValueError):
        return _skip(name, "Role's graduation-year restriction could not be parsed.")

    if exact_year is not None and min_year is None and max_year is None:
        min_year = max_year = exact_year

    if min_year is not None and candidate.graduation_year < min_year:
        return _fail(
            name,
            REASON_GRADUATION_YEAR_MISMATCH,
            f"Role requires graduation year >= {min_year}; candidate graduates {candidate.graduation_year}.",
        )
    if max_year is not None and candidate.graduation_year > max_year:
        return _fail(
            name,
            REASON_GRADUATION_YEAR_MISMATCH,
            f"Role requires graduation year <= {max_year}; candidate graduates {candidate.graduation_year}.",
        )
    return _pass(name, f"Candidate graduation year {candidate.graduation_year} satisfies role's restriction.")


_ALLOWED_MAJORS_KEYS = ("allowedMajors", "allowed_majors", "eligibleMajors", "eligible_majors", "majors")


def filter_major(role: Role, candidate: CandidateFilterContext) -> FilterOutcome:
    """Fail if the candidate's major isn't among the role's eligible majors. Skips if either side is unstated/unparseable."""
    name = "major"
    if not candidate.major:
        return _skip(name, "Candidate major is unknown.")

    raw_value = _first_present_raw(role.raw_source, _ALLOWED_MAJORS_KEYS)
    if raw_value is None:
        return _skip(name, "Role does not restrict eligible majors.")

    if isinstance(raw_value, str):
        allowed_majors = [m.strip() for m in raw_value.split(",") if m.strip()]
    elif isinstance(raw_value, (list, tuple)):
        allowed_majors = [str(m).strip() for m in raw_value if str(m).strip()]
    else:
        return _skip(name, "Role's major restriction could not be parsed.")

    if not allowed_majors:
        return _skip(name, "Role's major restriction list is empty.")

    normalized_candidate_major = normalize_skill_name(candidate.major)
    normalized_allowed = {normalize_skill_name(m) for m in allowed_majors}
    if any(normalized_candidate_major in m or m in normalized_candidate_major for m in normalized_allowed):
        return _pass(name, f"Candidate major '{candidate.major}' is eligible.")
    return _fail(
        name,
        REASON_MAJOR_RESTRICTED,
        f"Candidate major '{candidate.major}' is not among role's eligible majors {allowed_majors}.",
    )


_MIN_EXPERIENCE_YEARS_KEYS = (
    "minExperienceYears",
    "min_experience_years",
    "yearsExperience",
    "years_experience",
    "experienceYearsRequired",
    "experience_years_required",
)
OBVIOUS_EXPERIENCE_GAP_YEARS = 2.0


def filter_experience(role: Role, candidate: CandidateFilterContext) -> FilterOutcome:
    """
    Fail only an OBVIOUS mismatch: the role's minimum years of
    experience exceeds the candidate's by more than
    OBVIOUS_EXPERIENCE_GAP_YEARS. This is deliberately conservative -
    fine-grained experience fit is scored later
    (backend/matching/scorer.py's calculate_experience_fit), not
    eliminated here. Skips if either side is unstated/unparseable.
    """
    name = "experience"
    if candidate.max_experience_years is None:
        return _skip(name, "Candidate's years of experience is unknown.")

    raw_value = _first_present_raw(role.raw_source, _MIN_EXPERIENCE_YEARS_KEYS)
    if raw_value is None:
        return _skip(name, "Role does not specify a minimum years-of-experience requirement.")

    try:
        min_years = float(raw_value)
    except (TypeError, ValueError):
        return _skip(name, "Role's minimum years-of-experience requirement could not be parsed.")

    gap = min_years - candidate.max_experience_years
    if gap > OBVIOUS_EXPERIENCE_GAP_YEARS:
        return _fail(
            name,
            REASON_EXPERIENCE_MISMATCH,
            f"Role requires {min_years}+ years of experience, well above candidate's "
            f"{candidate.max_experience_years} years.",
        )
    return _pass(
        name,
        f"Role's experience requirement ({min_years} years) is not an obvious mismatch for "
        f"candidate ({candidate.max_experience_years} years).",
    )


HARD_CONSTRAINT_FILTERS: list[Callable[[Role, CandidateFilterContext], FilterOutcome]] = [
    filter_expired_deadline,
    filter_employment_type,
    filter_role_family,
    filter_location,
    filter_seniority,
    filter_graduation_year,
    filter_major,
    filter_experience,
]


# ---------------------------------------------------------------------
# Stage 2: keyword/skill overlap
# ---------------------------------------------------------------------


def filter_keyword_skill_overlap(role: Role, candidate: CandidateFilterContext) -> FilterOutcome:
    """
    Fail only if NONE of the candidate's known skills appear anywhere in
    the role's title/description text. This is a deliberately coarse,
    high-recall lexical check meant to catch postings with zero
    relation to anything the candidate knows - not to judge true
    relevance, which is scoring's (and later, semantic retrieval's) job.
    Skips if the candidate has no known skills or the role has no
    title/description text to check against.
    """
    name = "keyword_skill_overlap"
    if not candidate.known_skills:
        return _skip(name, "Candidate has no known skills recorded to check overlap against.")

    combined_text = normalize_skill_name(" ".join(filter(None, [role.title, role.description])))
    if not combined_text:
        return _skip(name, "Role has no title/description text to check keyword overlap against.")

    matched_skills = [skill for skill in candidate.known_skills if skill and skill in combined_text]
    if matched_skills:
        return _pass(name, f"Found keyword overlap: {matched_skills}.")
    return _fail(
        name,
        REASON_NO_KEYWORD_SKILL_OVERLAP,
        "No overlap found between candidate's known skills and the role's title/description text.",
    )


# ---------------------------------------------------------------------
# Pipeline orchestration + observability
# ---------------------------------------------------------------------


class RoleFilterResult(BaseModel):
    """Full filtering history for one role: every outcome computed, in order, and the final verdict."""

    role: Role
    stage_reached: str
    passed: bool
    outcomes: list[FilterOutcome] = Field(default_factory=list)
    failure_reason: str | None = None


class FilterPipelineMetrics(BaseModel):
    """
    Pipeline-level counts for observability. Extend here (and in
    run_filter_pipeline()) when a semantic filtering stage is added,
    e.g. a `removed_by_semantic_filtering` field - no other field needs
    to change.
    """

    input_count: int
    removed_by_hard_constraints: int
    removed_by_keyword_filtering: int
    remaining: int


class FilterPipelineResult(BaseModel):
    metrics: FilterPipelineMetrics
    surviving_roles: list[Role]
    role_results: list[RoleFilterResult]


def _evaluate_role_against_stage(
    role: Role,
    candidate: CandidateFilterContext,
    filters: list[Callable[[Role, CandidateFilterContext], FilterOutcome]],
) -> tuple[bool, list[FilterOutcome], str | None]:
    """Run `filters` in order against one role, short-circuiting on the first failure."""
    outcomes: list[FilterOutcome] = []
    for filter_fn in filters:
        outcome = filter_fn(role, candidate)
        outcomes.append(outcome)
        if not outcome.passed:
            return False, outcomes, outcome.reason
    return True, outcomes, None


def run_filter_pipeline(roles: list[Role], candidate: CandidateFilterContext) -> FilterPipelineResult:
    """
    Run every role through stage 1 (hard constraints), then stage 2
    (keyword/skill overlap) for stage-1 survivors only, tracking counts
    at each stage.

    To add stage 3 (semantic/embedding retrieval): after this function's
    loop, take `result.surviving_roles`, run them through the new stage,
    add a `removed_by_semantic_filtering` field to FilterPipelineMetrics,
    and extend RoleFilterResult.stage_reached / outcomes the same way
    stage 2 is threaded through stage 1 below - no changes are needed to
    any existing filter function or to stage 1/2's own logic.
    """
    role_results: list[RoleFilterResult] = []
    surviving_roles: list[Role] = []
    removed_by_hard_constraints = 0
    removed_by_keyword_filtering = 0

    for role in roles:
        stage1_passed, stage1_outcomes, stage1_failure_reason = _evaluate_role_against_stage(
            role, candidate, HARD_CONSTRAINT_FILTERS
        )
        if not stage1_passed:
            removed_by_hard_constraints += 1
            role_results.append(
                RoleFilterResult(
                    role=role,
                    stage_reached=STAGE_HARD_CONSTRAINTS,
                    passed=False,
                    outcomes=stage1_outcomes,
                    failure_reason=stage1_failure_reason,
                )
            )
            continue

        stage2_passed, stage2_outcomes, stage2_failure_reason = _evaluate_role_against_stage(
            role, candidate, [filter_keyword_skill_overlap]
        )
        combined_outcomes = stage1_outcomes + stage2_outcomes
        if not stage2_passed:
            removed_by_keyword_filtering += 1
            role_results.append(
                RoleFilterResult(
                    role=role,
                    stage_reached=STAGE_KEYWORD_SKILL_OVERLAP,
                    passed=False,
                    outcomes=combined_outcomes,
                    failure_reason=stage2_failure_reason,
                )
            )
            continue

        surviving_roles.append(role)
        role_results.append(
            RoleFilterResult(
                role=role,
                stage_reached=STAGE_KEYWORD_SKILL_OVERLAP,
                passed=True,
                outcomes=combined_outcomes,
                failure_reason=None,
            )
        )

    metrics = FilterPipelineMetrics(
        input_count=len(roles),
        removed_by_hard_constraints=removed_by_hard_constraints,
        removed_by_keyword_filtering=removed_by_keyword_filtering,
        remaining=len(surviving_roles),
    )
    return FilterPipelineResult(metrics=metrics, surviving_roles=surviving_roles, role_results=role_results)
