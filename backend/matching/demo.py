"""
Command-line demo for the deterministic fit scorer.

Builds a hand-crafted CandidateProfile, Role, and RoleRequirement list
directly (no LLM calls, no OPENAI_API_KEY needed - this whole module is
offline) and prints the complete FitScoreResult as JSON.

Usage:
    python -m backend.matching.demo
"""

from __future__ import annotations

from backend.candidate.profile import (
    CandidateProfile,
    CandidateSkillEstimate,
    ExperienceEntry,
    ProjectEntry,
)
from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import RoleRequirement
from backend.matching.scorer import calculate_fit_score


def build_sample_profile() -> CandidateProfile:
    return CandidateProfile(
        coursework=["Probability", "Algorithms", "Operating Systems"],
        programming_languages=["Python", "C++"],
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="python",
                display_name="Python",
                estimated_level=8.0,
                confidence=0.8,
                evidence_snippets=["Built internal Python ETL pipelines"],
            ),
            CandidateSkillEstimate(
                normalized_skill_name="c++",
                display_name="C++",
                estimated_level=3.0,
                confidence=0.3,
                evidence_snippets=["Skills: Python, C++, React, Docker"],
            ),
        ],
        internships=[
            ExperienceEntry(
                organization="Acme Corp",
                role="Software Engineering Intern",
                description="Built internal React dashboards; wrote Python ETL scripts.",
            )
        ],
        projects=[
            ProjectEntry(
                name="Distributed Key-Value Store",
                description="Raft-based distributed KV store in C++.",
                technologies=["C++"],
            )
        ],
        domain_experience=["trading systems"],
    )


def build_sample_role_and_requirements() -> tuple:
    role = normalize_role(
        {
            "jobTitle": "Quantitative Research Intern",
            "companyName": "Meridian Capital",
            "location": "New York, NY",
            "role_family": "quant",
        }
    )
    requirements = [
        RoleRequirement(
            skill="Python",
            normalized_skill="python",
            target_level=7.0,
            importance=9.0,
            required=True,
            evidence=["Proficiency in Python"],
        ),
        RoleRequirement(
            skill="C++",
            normalized_skill="c++",
            target_level=6.0,
            importance=6.0,
            required=False,
            evidence=["C++ experience is a plus"],
        ),
        RoleRequirement(
            skill="Probability & Statistics",
            normalized_skill="probability and statistics",
            target_level=6.0,
            importance=8.0,
            required=True,
            evidence=["Strong foundation in probability and statistics"],
        ),
    ]
    return role, requirements


def main() -> None:
    profile = build_sample_profile()
    role, requirements = build_sample_role_and_requirements()

    result = calculate_fit_score(profile, role, requirements)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
