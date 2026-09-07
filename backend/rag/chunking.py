"""
Text chunking.

Two ways to produce chunks:

1. chunk_text() - a generic, word-boundary-aware sliding window over
   arbitrary text, used as a fallback for free text (e.g. a role's
   description) and internally whenever a structured field's text is
   longer than one chunk should be.

2. chunk_candidate_evidence() / chunk_role_content() - build chunks
   directly from the already-structured CandidateProfile / Role /
   RoleRequirement models (from backend.candidate.profile and
   backend.llm.extract_requirements), rather than re-splitting a raw
   text blob. Since those models were already extracted and validated,
   each chunk can carry precise, correct metadata (which skill, which
   project) for free - something a naive fixed-size window over raw
   resume/job text cannot do.

No LLM calls and no embedding calls happen here - this module only
produces EvidenceChunk objects (text + labels); backend.rag.embeddings
turns them into vectors and backend.rag.vector_store stores them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile
from backend.ingestion.normalize import Role
from backend.llm.extract_requirements import RoleRequirement

DEFAULT_CHUNK_SIZE_CHARS = 800
DEFAULT_CHUNK_OVERLAP_CHARS = 100


class EvidenceChunk(BaseModel):
    """
    One chunk of text ready to be embedded and stored. Deliberately
    ownership-agnostic (no user_id/role_id) - chunking describes
    *content*, not who it belongs to. backend.rag.vector_store.upsert_chunks()
    attaches user_id/role_id when indexing.
    """

    text: str
    source_type: str
    skill: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """
    Split `text` into overlapping chunks of at most `max_chars`,
    breaking on the nearest preceding whitespace so words aren't cut in
    half. Returns [] for empty/whitespace-only text and a single-element
    list if `text` already fits in one chunk.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]

    chunks: list[str] = []
    start = 0
    text_length = len(stripped)
    while start < text_length:
        end = min(start + max_chars, text_length)
        if end < text_length:
            last_space = stripped.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        piece = stripped[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def chunk_candidate_evidence(profile: CandidateProfile) -> list[EvidenceChunk]:
    """
    Build evidence chunks from a candidate's structured profile:
    - one chunk per skill's evidence snippets, labeled with that skill
    - one chunk per project description
    - one chunk per internship/work experience description
    - one chunk per research entry
    - one chunk per coursework entry (short enough to never need splitting)
    """
    chunks: list[EvidenceChunk] = []

    for skill in profile.skills:
        evidence_text = " ".join(skill.evidence_snippets) if skill.evidence_snippets else skill.display_name
        for piece in chunk_text(evidence_text):
            chunks.append(
                EvidenceChunk(
                    text=piece,
                    source_type="skill",
                    skill=skill.normalized_skill_name,
                    metadata={
                        "display_name": skill.display_name,
                        "estimated_level": skill.estimated_level,
                        "confidence": skill.confidence,
                    },
                )
            )

    for project in profile.projects:
        text = f"{project.name}: {project.description}"
        for piece in chunk_text(text):
            chunks.append(
                EvidenceChunk(
                    text=piece,
                    source_type="project",
                    metadata={"project_name": project.name, "technologies": project.technologies},
                )
            )

    for experience in profile.internships:
        text = f"{experience.role} at {experience.organization}: {experience.description}"
        for piece in chunk_text(text):
            chunks.append(
                EvidenceChunk(
                    text=piece,
                    source_type="experience",
                    metadata={"organization": experience.organization, "role": experience.role},
                )
            )

    for research in profile.research:
        text = f"{research.title}: {research.description}"
        for piece in chunk_text(text):
            chunks.append(EvidenceChunk(text=piece, source_type="research", metadata={"title": research.title}))

    for course in profile.coursework:
        chunks.append(EvidenceChunk(text=course, source_type="coursework"))

    return chunks


def chunk_role_content(role: Role, requirements: list[RoleRequirement] | None = None) -> list[EvidenceChunk]:
    """
    Build content chunks from a role and its (optional) extracted
    requirements:
    - one or more chunks of company/role text (title, company, and the
      free-text description - responsibilities aren't extracted as a
      separate structured field yet, so they're only represented here,
      inside the description text)
    - one chunk per requirement's supporting evidence, labeled with that
      requirement's normalized skill
    """
    chunks: list[EvidenceChunk] = []

    company_role_text = f"{role.title} at {role.company}."
    if role.description:
        company_role_text += f" {role.description}"
    for piece in chunk_text(company_role_text):
        chunks.append(
            EvidenceChunk(
                text=piece,
                source_type="company_role_text",
                metadata={"company": role.company, "title": role.title},
            )
        )

    for requirement in requirements or []:
        evidence_text = " ".join(requirement.evidence) if requirement.evidence else requirement.skill
        text = f"{requirement.skill} requirement: {evidence_text}"
        for piece in chunk_text(text):
            chunks.append(
                EvidenceChunk(
                    text=piece,
                    source_type="requirement",
                    skill=requirement.normalized_skill,
                    metadata={
                        "target_level": requirement.target_level,
                        "importance": requirement.importance,
                        "required": requirement.required,
                    },
                )
            )

    return chunks
