"""
Tests for backend.matching.filters (deterministic staged filtering
pipeline). No LLM calls occur anywhere in this module or its tests.
"""

from __future__ import annotations

from datetime import date

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate, EducationEntry
from backend.ingestion.normalize import normalize_role
from backend.matching.filters import (
    REASON_EMPLOYMENT_TYPE_MISMATCH,
    REASON_EXPERIENCE_MISMATCH,
    REASON_EXPIRED_DEADLINE,
    REASON_GRADUATION_YEAR_MISMATCH,
    REASON_LOCATION_MISMATCH,
    REASON_MAJOR_RESTRICTED,
    REASON_NO_KEYWORD_SKILL_OVERLAP,
    REASON_PASS,
    REASON_ROLE_FAMILY_MISMATCH,
    REASON_SENIORITY_MISMATCH,
    REASON_SKIPPED_INSUFFICIENT_DATA,
    CandidateFilterContext,
    build_candidate_filter_context,
    filter_experience,
    filter_expired_deadline,
    filter_graduation_year,
    filter_keyword_skill_overlap,
    filter_location,
    filter_major,
    filter_role_family,
    filter_seniority,
    run_filter_pipeline,
)


def _role(**overrides):
    raw = {"title": "Software Engineer Intern", "company": "Acme Corp"}
    raw.update(overrides)
    return normalize_role(raw)


def _candidate(**overrides) -> CandidateFilterContext:
    return CandidateFilterContext(**overrides)


# ---------------------------------------------------------------------
# filter_expired_deadline
# ---------------------------------------------------------------------


def test_expired_deadline_fails_when_past() -> None:
    role = _role(deadline="2020-01-01")

    outcome = filter_expired_deadline(role, _candidate(), today=date(2026, 1, 1))

    assert outcome.passed is False
    assert outcome.reason == REASON_EXPIRED_DEADLINE


def test_deadline_in_future_passes() -> None:
    role = _role(deadline="2030-01-01")

    outcome = filter_expired_deadline(role, _candidate(), today=date(2026, 1, 1))

    assert outcome.passed is True
    assert outcome.reason == REASON_PASS


def test_missing_deadline_skips_rather_than_fails() -> None:
    role = _role()

    outcome = filter_expired_deadline(role, _candidate())

    assert outcome.passed is True
    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


def test_unparseable_deadline_skips() -> None:
    role = _role(deadline="whenever")

    outcome = filter_expired_deadline(role, _candidate())

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# filter_role_family
# ---------------------------------------------------------------------


def test_role_family_mismatch_fails() -> None:
    role = _role(role_family="marketing")
    candidate = _candidate(target_role_families=["swe"])

    outcome = filter_role_family(role, candidate)

    assert outcome.passed is False
    assert outcome.reason == REASON_ROLE_FAMILY_MISMATCH


def test_role_family_match_passes() -> None:
    role = _role(role_family="swe")
    candidate = _candidate(target_role_families=["SWE"])

    outcome = filter_role_family(role, candidate)

    assert outcome.passed is True
    assert outcome.reason == REASON_PASS


def test_role_family_skips_when_candidate_has_no_preference() -> None:
    role = _role(role_family="marketing")

    outcome = filter_role_family(role, _candidate())

    assert outcome.passed is True
    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


def test_role_family_skips_when_role_has_no_family() -> None:
    role = _role()
    candidate = _candidate(target_role_families=["swe"])

    outcome = filter_role_family(role, candidate)

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# filter_location
# ---------------------------------------------------------------------


def test_location_mismatch_fails_when_explicitly_onsite() -> None:
    role = _role(location="Austin, TX", workplaceType="On-site")
    candidate = _candidate(preferred_locations=["Remote"])

    outcome = filter_location(role, candidate)

    assert outcome.passed is False
    assert outcome.reason == REASON_LOCATION_MISMATCH


def test_location_skips_when_remote_friendly() -> None:
    role = _role(location="Austin, TX", workplaceType="Remote")
    candidate = _candidate(preferred_locations=["New York"])

    outcome = filter_location(role, candidate)

    assert outcome.passed is True
    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


def test_location_skips_when_no_workplace_type_signal_at_all() -> None:
    role = _role(location="Austin, TX")
    candidate = _candidate(preferred_locations=["New York"])

    outcome = filter_location(role, candidate)

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


def test_location_passes_when_onsite_location_matches_preference() -> None:
    role = _role(location="Remote", workplaceType="On-site")
    candidate = _candidate(preferred_locations=["Remote"])

    outcome = filter_location(role, candidate)

    assert outcome.passed is True
    assert outcome.reason == REASON_PASS


# ---------------------------------------------------------------------
# filter_seniority
# ---------------------------------------------------------------------


def test_seniority_mismatch_fails_for_senior_role_and_junior_candidate() -> None:
    role = _role(seniority="Senior")
    candidate = _candidate(seniority_level="intern")

    outcome = filter_seniority(role, candidate)

    assert outcome.passed is False
    assert outcome.reason == REASON_SENIORITY_MISMATCH


def test_seniority_passes_for_intern_role_and_intern_candidate() -> None:
    role = _role(seniority="Internship")
    candidate = _candidate(seniority_level="intern")

    outcome = filter_seniority(role, candidate)

    assert outcome.passed is True
    assert outcome.reason == REASON_PASS


def test_seniority_skips_when_role_does_not_specify_level() -> None:
    role = _role()
    candidate = _candidate(seniority_level="intern")

    outcome = filter_seniority(role, candidate)

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# filter_graduation_year
# ---------------------------------------------------------------------


def test_graduation_year_mismatch_fails_below_minimum() -> None:
    role = _role(minGraduationYear=2027)
    candidate = _candidate(graduation_year=2026)

    outcome = filter_graduation_year(role, candidate)

    assert outcome.passed is False
    assert outcome.reason == REASON_GRADUATION_YEAR_MISMATCH


def test_graduation_year_exact_match_passes() -> None:
    role = _role(graduationYear=2026)
    candidate = _candidate(graduation_year=2026)

    outcome = filter_graduation_year(role, candidate)

    assert outcome.passed is True
    assert outcome.reason == REASON_PASS


def test_graduation_year_skips_when_role_has_no_restriction() -> None:
    role = _role()
    candidate = _candidate(graduation_year=2026)

    outcome = filter_graduation_year(role, candidate)

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


def test_graduation_year_skips_when_candidate_year_unknown() -> None:
    role = _role(graduationYear=2026)

    outcome = filter_graduation_year(role, _candidate())

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# filter_major
# ---------------------------------------------------------------------


def test_major_restriction_fails_when_not_eligible() -> None:
    role = _role(allowedMajors=["Electrical Engineering", "Physics"])
    candidate = _candidate(major="Computer Science")

    outcome = filter_major(role, candidate)

    assert outcome.passed is False
    assert outcome.reason == REASON_MAJOR_RESTRICTED


def test_major_restriction_passes_when_eligible() -> None:
    role = _role(allowedMajors=["Computer Science", "Math"])
    candidate = _candidate(major="Computer Science")

    outcome = filter_major(role, candidate)

    assert outcome.passed is True
    assert outcome.reason == REASON_PASS


def test_major_skips_when_role_has_no_restriction() -> None:
    role = _role()
    candidate = _candidate(major="Computer Science")

    outcome = filter_major(role, candidate)

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# filter_experience
# ---------------------------------------------------------------------


def test_experience_obvious_mismatch_fails() -> None:
    role = _role(minExperienceYears=5)
    candidate = _candidate(max_experience_years=0)

    outcome = filter_experience(role, candidate)

    assert outcome.passed is False
    assert outcome.reason == REASON_EXPERIENCE_MISMATCH


def test_experience_within_gap_threshold_passes() -> None:
    role = _role(minExperienceYears=1)
    candidate = _candidate(max_experience_years=0)

    outcome = filter_experience(role, candidate)

    assert outcome.passed is True
    assert outcome.reason == REASON_PASS


def test_experience_skips_when_candidate_years_unknown() -> None:
    role = _role(minExperienceYears=5)

    outcome = filter_experience(role, _candidate())

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# filter_keyword_skill_overlap
# ---------------------------------------------------------------------


def test_keyword_overlap_fails_when_no_overlap() -> None:
    role = _role(description="Write blog posts and manage social media campaigns.")
    candidate = _candidate(known_skills=["python", "react"])

    outcome = filter_keyword_skill_overlap(role, candidate)

    assert outcome.passed is False
    assert outcome.reason == REASON_NO_KEYWORD_SKILL_OVERLAP


def test_keyword_overlap_passes_with_any_match() -> None:
    role = _role(description="Build features in Python and React.")
    candidate = _candidate(known_skills=["python", "react"])

    outcome = filter_keyword_skill_overlap(role, candidate)

    assert outcome.passed is True
    assert outcome.reason == REASON_PASS


def test_keyword_overlap_skips_when_candidate_has_no_known_skills() -> None:
    role = _role(description="Write blog posts.")

    outcome = filter_keyword_skill_overlap(role, _candidate())

    assert outcome.reason == REASON_SKIPPED_INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# build_candidate_filter_context
# ---------------------------------------------------------------------


def test_build_candidate_filter_context_derives_skills_and_education() -> None:
    profile = CandidateProfile(
        programming_languages=["Python"],
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="react", display_name="React", estimated_level=5, confidence=0.5
            )
        ],
        education=[EducationEntry(institution="State U", major="Computer Science", graduation_year=2026)],
    )

    context = build_candidate_filter_context(profile, target_role_families=["swe"])

    assert "python" in context.known_skills
    assert "react" in context.known_skills
    assert context.major == "Computer Science"
    assert context.graduation_year == 2026
    assert context.target_role_families == ["swe"]


# ---------------------------------------------------------------------
# run_filter_pipeline: observability + the requested end-to-end scenarios
# ---------------------------------------------------------------------


def test_pipeline_removes_expired_role() -> None:
    candidate = _candidate()
    expired_role = _role(deadline="2000-01-01")

    result = run_filter_pipeline([expired_role], candidate)

    assert result.metrics.input_count == 1
    assert result.metrics.removed_by_hard_constraints == 1
    assert result.metrics.remaining == 0
    assert result.role_results[0].failure_reason == REASON_EXPIRED_DEADLINE


def test_pipeline_matching_role_survives() -> None:
    candidate = _candidate(
        known_skills=["python"],
        target_role_families=["swe"],
        seniority_level="intern",
        acceptable_employment_types=["internship"],
    )
    role = _role(
        role_family="swe",
        employmentType="Internship",
        seniority="Internship",
        description="Build backend services in Python.",
    )

    result = run_filter_pipeline([role], candidate)

    assert result.metrics.remaining == 1
    assert result.surviving_roles == [role]
    assert result.role_results[0].passed is True


def test_pipeline_missing_optional_metadata_does_not_cause_accidental_rejection() -> None:
    """A role with almost no optional metadata should never be rejected just because that data is absent."""
    candidate = _candidate(
        known_skills=["python"],
        target_role_families=["swe"],
        preferred_locations=["Remote"],
        seniority_level="intern",
        acceptable_employment_types=["internship"],
        graduation_year=2026,
        major="Computer Science",
        max_experience_years=0,
    )
    bare_role = _role(description="Some Python work.")  # no deadline, employment type, family, location type, etc.

    result = run_filter_pipeline([bare_role], candidate)

    assert result.metrics.remaining == 1
    skipped_reasons = {o.reason for o in result.role_results[0].outcomes}
    assert REASON_SKIPPED_INSUFFICIENT_DATA in skipped_reasons


def test_pipeline_filtering_reasons_are_visible_for_every_role() -> None:
    candidate = _candidate(target_role_families=["swe"])
    roles = [_role(role_family="marketing"), _role(role_family="swe")]

    result = run_filter_pipeline(roles, candidate)

    assert len(result.role_results) == 2
    for role_result in result.role_results:
        assert role_result.outcomes  # every role has a visible trail of outcomes
        for outcome in role_result.outcomes:
            assert outcome.reason
            assert outcome.detail
    assert result.role_results[0].failure_reason == REASON_ROLE_FAMILY_MISMATCH
    assert result.role_results[1].failure_reason is None


def test_pipeline_distinguishes_hard_constraint_and_keyword_removals() -> None:
    candidate = _candidate(target_role_families=["swe"], known_skills=["python"])
    hard_constraint_reject = _role(role_family="marketing")
    keyword_reject = _role(role_family="swe", description="Write blog posts about company culture.")
    survivor = _role(role_family="swe", description="Backend work in Python.")

    result = run_filter_pipeline([hard_constraint_reject, keyword_reject, survivor], candidate)

    assert result.metrics.input_count == 3
    assert result.metrics.removed_by_hard_constraints == 1
    assert result.metrics.removed_by_keyword_filtering == 1
    assert result.metrics.remaining == 1
    assert result.surviving_roles == [survivor]


def test_pipeline_with_no_candidate_preferences_only_applies_keyword_stage() -> None:
    """With zero declared preferences, every hard filter should skip - only keyword overlap can still reject."""
    candidate = _candidate(known_skills=["python"])
    roles = [
        _role(description="Backend work in Python."),
        _role(description="Marketing copywriting."),
    ]

    result = run_filter_pipeline(roles, candidate)

    assert result.metrics.removed_by_hard_constraints == 0
    assert result.metrics.removed_by_keyword_filtering == 1
    assert result.metrics.remaining == 1
