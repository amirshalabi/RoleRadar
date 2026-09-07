"""
Command-line demo for the RAG pipeline: candidate profile -> chunk
evidence -> embed -> upsert into Qdrant, then a role-driven query ->
retrieve relevant candidate evidence.

Unlike backend/matching/demo.py, this one is NOT offline: embedding
requires OPENAI_API_KEY and storing/querying vectors requires
QDRANT_URL + QDRANT_API_KEY. If either is missing, this prints a clear
message (via the OpenAINotConfiguredError / QdrantNotConfiguredError
that get_embedding_provider()/upsert_chunks() raise) and stops, rather
than crashing with a raw traceback.

Usage:
    python -m backend.rag.demo
"""

from __future__ import annotations

from backend.candidate.profile import (
    CandidateProfile,
    CandidateSkillEstimate,
    ExperienceEntry,
    ProjectEntry,
)
from backend.ingestion.normalize import normalize_role
from backend.llm.client import OpenAINotConfiguredError
from backend.rag.chunking import chunk_candidate_evidence
from backend.rag.embeddings import get_embedding_provider
from backend.rag.retrieval import retrieve_candidate_evidence
from backend.rag.vector_store import CANDIDATE_EVIDENCE_COLLECTION, QdrantNotConfiguredError, upsert_chunks

DEMO_USER_ID = "demo-user"


def build_sample_profile() -> CandidateProfile:
    return CandidateProfile(
        coursework=["Probability", "Algorithms"],
        programming_languages=["Python", "C++"],
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name="python",
                display_name="Python",
                estimated_level=8.0,
                confidence=0.8,
                evidence_snippets=["Built internal Python ETL pipelines"],
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
                description="Raft-based distributed KV store in C++ with leader election.",
                technologies=["C++"],
            )
        ],
    )


def main() -> None:
    profile = build_sample_profile()
    chunks = chunk_candidate_evidence(profile)

    print(f"Built {len(chunks)} candidate evidence chunks:")
    for chunk in chunks:
        label = f"{chunk.source_type}/{chunk.skill}" if chunk.skill else chunk.source_type
        print(f"  [{label}] {chunk.text[:80]}")

    try:
        provider = get_embedding_provider()
        upserted = upsert_chunks(CANDIDATE_EVIDENCE_COLLECTION, chunks, provider, user_id=DEMO_USER_ID)
    except (OpenAINotConfiguredError, QdrantNotConfiguredError) as exc:
        print(f"\nStopping before embed/upsert - {exc}")
        return

    print(f"\nUpserted {upserted} vectors into '{CANDIDATE_EVIDENCE_COLLECTION}'.")

    role = normalize_role(
        {
            "title": "Software Engineering Intern",
            "company": "Acme Corp",
            "description": "Looking for a backend intern comfortable with Python and distributed systems.",
        }
    )
    query_text = f"{role.title}: {role.description}"
    results = retrieve_candidate_evidence(query_text, DEMO_USER_ID, top_k=3, provider=provider)

    print(f"\nTop {len(results)} candidate evidence chunks for role query '{role.title}':")
    for result in results:
        text_preview = str(result.payload.get("text", ""))[:80]
        print(f"  score={result.score:.3f} source_type={result.payload.get('source_type')} text={text_preview!r}")


if __name__ == "__main__":
    main()
