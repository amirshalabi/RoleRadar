"""
Candidate profile construction.

Defines the structured CandidateProfile schema and
extract_candidate_profile(), which uses the LLM to fill that schema in
from unstructured resume text. The LLM performs extraction only; the
result is validated by Pydantic here, and that validated structure -
not the raw resume text or the raw model output - is what all
downstream deterministic matching/scoring/planning code consumes.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from backend.llm.client import parse_structured
from backend.llm.prompts import (
    CANDIDATE_EXTRACTION_SYSTEM_PROMPT,
    build_candidate_extraction_user_prompt,
)

logger = logging.getLogger(__name__)


class EducationEntry(BaseModel):
    institution: str
    degree: str | None = None
    major: str | None = None
    graduation_year: int | None = None
    gpa: float | None = None


class ProjectEntry(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)


class ExperienceEntry(BaseModel):
    """An internship or other work experience entry."""

    organization: str
    role: str
    description: str
    start_date: str | None = None
    end_date: str | None = None


class ResearchEntry(BaseModel):
    title: str
    description: str
    publication: str | None = None


class CandidateSkillEstimate(BaseModel):
    """
    One skill estimate extracted from the resume. `estimated_level` and
    `confidence` are the LLM's best judgment given the evidence text -
    not ground truth - and are treated as such downstream (see
    backend/matching/confidence.py).
    """

    normalized_skill_name: str
    display_name: str
    estimated_level: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)
    evidence_snippets: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    education: list[EducationEntry] = Field(default_factory=list)
    coursework: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    internships: list[ExperienceEntry] = Field(default_factory=list)
    research: list[ResearchEntry] = Field(default_factory=list)
    domain_experience: list[str] = Field(default_factory=list)
    skills: list[CandidateSkillEstimate] = Field(default_factory=list)


def extract_candidate_profile(resume_text: str, model: str | None = None) -> CandidateProfile:
    """
    Use the LLM to extract a structured CandidateProfile from raw resume
    text (already extracted from a PDF via backend.candidate.parser, or
    pasted directly). Raises ValueError for empty input and
    backend.llm.client.LLMExtractionError if the model refuses or
    returns something that doesn't validate against the schema.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text must not be empty")

    logger.info("Extracting candidate profile from resume text (%d chars)", len(resume_text))
    return parse_structured(
        system_prompt=CANDIDATE_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=build_candidate_extraction_user_prompt(resume_text),
        response_model=CandidateProfile,
        model=model,
    )
