"""
Evidence retrieval.

Domain-specific retrieval functions built on backend.rag.vector_store's
generic top-k/filtered primitives, so callers don't need to know
collection names or payload filter shapes to retrieve supporting
resume/job evidence. This is the retrieval half of RAG; the generation
half (LLM-written, evidence-backed rationales) is backend.llm.rationale,
which calls retrieve_skill_evidence() and retrieve_candidate_evidence()
here BEFORE making any LLM call - retrieval always happens first.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.rag.embeddings import EmbeddingProvider, get_embedding_provider
from backend.rag.vector_store import (
    CANDIDATE_EVIDENCE_COLLECTION,
    ROLE_CONTENT_COLLECTION,
    RetrievedChunk,
    retrieve_filtered,
)


def retrieve_candidate_evidence(
    query_text: str,
    user_id: str,
    *,
    role_id: str | None = None,
    skill: str | None = None,
    top_k: int = 5,
    provider: EmbeddingProvider | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve a candidate's own stored evidence chunks (skills, projects,
    experience, coursework) most similar to `query_text`. Always scoped
    to `user_id` so one candidate's evidence never leaks into another's
    retrieval results; optionally further scoped to a specific role or
    skill.
    """
    provider = provider or get_embedding_provider()
    metadata_filter: dict[str, str] = {"user_id": user_id}
    if role_id is not None:
        metadata_filter["role_id"] = role_id
    if skill is not None:
        metadata_filter["skill"] = skill
    return retrieve_filtered(
        CANDIDATE_EVIDENCE_COLLECTION, query_text, provider, metadata_filter=metadata_filter, top_k=top_k
    )


def retrieve_role_content(
    query_text: str,
    role_id: str,
    *,
    skill: str | None = None,
    top_k: int = 5,
    provider: EmbeddingProvider | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve a role's own stored content chunks (requirements,
    company/role text) most similar to `query_text`. Always scoped to
    `role_id`; optionally further scoped to a specific requirement's
    skill.
    """
    provider = provider or get_embedding_provider()
    metadata_filter: dict[str, str] = {"role_id": role_id}
    if skill is not None:
        metadata_filter["skill"] = skill
    return retrieve_filtered(ROLE_CONTENT_COLLECTION, query_text, provider, metadata_filter=metadata_filter, top_k=top_k)


class SkillEvidenceBundle(BaseModel):
    """Both sides of the evidence for one skill: what the candidate claims, and what the role requires."""

    skill: str
    candidate_evidence: list[RetrievedChunk] = Field(default_factory=list)
    role_evidence: list[RetrievedChunk] = Field(default_factory=list)


def retrieve_skill_evidence(
    user_id: str,
    role_id: str,
    skill: str,
    *,
    top_k: int = 3,
    provider: EmbeddingProvider | None = None,
) -> SkillEvidenceBundle:
    """
    Retrieve both the candidate's evidence and the role's requirement
    evidence for one normalized skill, in a single call. This is the
    pairing backend.llm.rationale needs for its per-skill rationale
    (candidate evidence + role requirement evidence side by side) and is
    retrieved before any LLM call is made.
    """
    provider = provider or get_embedding_provider()
    candidate_evidence = retrieve_candidate_evidence(skill, user_id, role_id=role_id, skill=skill, top_k=top_k, provider=provider)
    role_evidence = retrieve_role_content(skill, role_id, skill=skill, top_k=top_k, provider=provider)
    return SkillEvidenceBundle(skill=skill, candidate_evidence=candidate_evidence, role_evidence=role_evidence)
