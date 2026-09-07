"""Tests for backend.rag.chunking (pure text/structured chunking, no embedding or LLM calls)."""

from __future__ import annotations

from backend.candidate.profile import (
    CandidateProfile,
    CandidateSkillEstimate,
    ExperienceEntry,
    ProjectEntry,
    ResearchEntry,
)
from backend.ingestion.normalize import normalize_role
from backend.llm.extract_requirements import RoleRequirement
from backend.rag.chunking import chunk_candidate_evidence, chunk_role_content, chunk_text


def test_chunk_text_returns_empty_list_for_blank_text() -> None:
    assert chunk_text("   ") == []
    assert chunk_text("") == []


def test_chunk_text_returns_single_chunk_when_short() -> None:
    result = chunk_text("A short sentence.", max_chars=100)

    assert result == ["A short sentence."]


def test_chunk_text_splits_long_text_on_word_boundaries() -> None:
    text = " ".join(f"word{i}" for i in range(200))

    chunks = chunk_text(text, max_chars=50, overlap_chars=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50
        assert not chunk.startswith(" ")
        assert not chunk.endswith(" ")


def test_chunk_text_reconstructs_all_words() -> None:
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    chunks = chunk_text(text, max_chars=60, overlap_chars=15)
    seen_words: set[str] = set()
    for chunk in chunks:
        seen_words.update(chunk.split())

    assert set(words) <= seen_words


def test_chunk_candidate_evidence_labels_skills_with_normalized_name() -> None:
    profile = CandidateProfile(
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="python",
                display_name="Python",
                estimated_level=8.0,
                confidence=0.8,
                evidence_snippets=["Built internal Python ETL pipelines"],
            )
        ]
    )

    chunks = chunk_candidate_evidence(profile)

    skill_chunks = [c for c in chunks if c.source_type == "skill"]
    assert len(skill_chunks) == 1
    assert skill_chunks[0].skill == "python"
    assert "ETL pipelines" in skill_chunks[0].text
    assert skill_chunks[0].metadata["display_name"] == "Python"


def test_chunk_candidate_evidence_covers_projects_experience_research_coursework() -> None:
    profile = CandidateProfile(
        coursework=["Algorithms"],
        projects=[ProjectEntry(name="KV Store", description="Distributed KV store in C++.")],
        internships=[
            ExperienceEntry(organization="Acme", role="SWE Intern", description="Built dashboards.")
        ],
        research=[ResearchEntry(title="ML Robustness", description="Studied adversarial examples.")],
    )

    chunks = chunk_candidate_evidence(profile)
    source_types = {c.source_type for c in chunks}

    assert source_types == {"coursework", "project", "experience", "research"}
    coursework_chunk = next(c for c in chunks if c.source_type == "coursework")
    assert coursework_chunk.text == "Algorithms"


def test_chunk_candidate_evidence_empty_profile_returns_no_chunks() -> None:
    assert chunk_candidate_evidence(CandidateProfile()) == []


def test_chunk_role_content_includes_company_role_text() -> None:
    role = normalize_role(
        {"title": "SWE Intern", "company": "Acme Corp", "description": "Build backend services."}
    )

    chunks = chunk_role_content(role)

    assert len(chunks) == 1
    assert chunks[0].source_type == "company_role_text"
    assert "Acme Corp" in chunks[0].text
    assert "Build backend services." in chunks[0].text


def test_chunk_role_content_includes_requirements_labeled_with_skill() -> None:
    role = normalize_role({"title": "SWE Intern", "company": "Acme Corp"})
    requirements = [
        RoleRequirement(
            skill="Python",
            normalized_skill="python",
            target_level=6.0,
            importance=8.0,
            required=True,
            evidence=["Proficiency in Python required."],
        )
    ]

    chunks = chunk_role_content(role, requirements)

    requirement_chunks = [c for c in chunks if c.source_type == "requirement"]
    assert len(requirement_chunks) == 1
    assert requirement_chunks[0].skill == "python"
    assert "Proficiency in Python required." in requirement_chunks[0].text
    assert requirement_chunks[0].metadata["required"] is True


def test_chunk_role_content_with_no_requirements_only_yields_company_role_text() -> None:
    role = normalize_role({"title": "SWE Intern", "company": "Acme Corp"})

    chunks = chunk_role_content(role, requirements=None)

    assert all(c.source_type == "company_role_text" for c in chunks)
