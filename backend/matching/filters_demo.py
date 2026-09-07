"""
Command-line demo for the staged filtering pipeline.

Builds several mock roles and one mock candidate directly (no LLM calls,
no OPENAI_API_KEY needed - this whole module is offline) and prints the
per-role verdicts plus the pipeline-level counts after each stage.

Usage:
    python -m backend.matching.filters_demo
"""

from __future__ import annotations

from backend.ingestion.normalize import normalize_role
from backend.matching.filters import CandidateFilterContext, run_filter_pipeline


def build_sample_candidate() -> CandidateFilterContext:
    return CandidateFilterContext(
        known_skills=["python", "react", "sql"],
        graduation_year=2026,
        major="Computer Science",
        target_role_families=["swe"],
        preferred_locations=["Remote"],
        seniority_level="intern",
        acceptable_employment_types=["internship"],
        max_experience_years=0,
    )


def build_sample_roles() -> list:
    raw_jobs = [
        {
            # Survives every stage.
            "title": "Software Engineering Intern",
            "companyName": "GoodCo",
            "role_family": "swe",
            "employmentType": "Internship",
            "seniority": "Internship",
            "description": "Build backend services in Python and a React frontend.",
        },
        {
            # Removed at stage 1: expired deadline.
            "title": "Backend Intern",
            "companyName": "StaleCo",
            "role_family": "swe",
            "employmentType": "Internship",
            "description": "Python backend work.",
            "deadline": "2020-01-01",
        },
        {
            # Removed at stage 1: seniority mismatch.
            "title": "Senior Backend Engineer",
            "companyName": "BigCorp",
            "role_family": "swe",
            "employmentType": "Full-time",
            "seniority": "Senior",
            "description": "Own our Python backend at scale.",
        },
        {
            # Removed at stage 1: role family mismatch.
            "title": "Marketing Intern",
            "companyName": "BrandCo",
            "role_family": "marketing",
            "employmentType": "Internship",
            "description": "Write blog posts and manage social media.",
        },
        {
            # Removed at stage 1: on-site location mismatch (candidate wants remote).
            "title": "SWE Intern",
            "companyName": "OfficeOnly",
            "role_family": "swe",
            "employmentType": "Internship",
            "location": "Austin, TX",
            "workplaceType": "On-site",
            "description": "In-office Python development.",
        },
        {
            # Survives stage 1, removed at stage 2: no keyword/skill overlap.
            "title": "SWE Intern",
            "companyName": "QuietCo",
            "role_family": "swe",
            "employmentType": "Internship",
            "description": "You will collaborate cross-functionally to drive outcomes.",
        },
    ]
    return [normalize_role(raw) for raw in raw_jobs]


def main() -> None:
    candidate = build_sample_candidate()
    roles = build_sample_roles()

    result = run_filter_pipeline(roles, candidate)

    print(f"Input roles: {result.metrics.input_count}\n")
    for role_result in result.role_results:
        verdict = "PASSED" if role_result.passed else "REJECTED"
        print(f"[{verdict}] {role_result.role.title} ({role_result.role.company})")
        print(f"    stage reached: {role_result.stage_reached}")
        if role_result.failure_reason:
            print(f"    failure reason: {role_result.failure_reason}")
        for outcome in role_result.outcomes:
            marker = "PASS" if outcome.passed else "FAIL"
            print(f"    - [{marker}] {outcome.filter_name}: {outcome.reason} - {outcome.detail}")
        print()

    print("--- Pipeline metrics ---")
    print(f"input_count:                {result.metrics.input_count}")
    print(f"removed_by_hard_constraints: {result.metrics.removed_by_hard_constraints}")
    print(f"removed_by_keyword_filtering: {result.metrics.removed_by_keyword_filtering}")
    print(f"remaining:                   {result.metrics.remaining}")


if __name__ == "__main__":
    main()
