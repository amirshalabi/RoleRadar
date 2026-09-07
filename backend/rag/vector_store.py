"""
Qdrant vector store wrapper.

Owns three collections:

- candidate_evidence: chunks of a candidate's resume evidence (skills,
  project descriptions, work experience, coursework).
- role_content: chunks of a role's content (requirements, company/role
  text).
- prep_resources: reserved for study/resource material indexed for
  retrieval during interview prep planning (backend/planning, not yet
  implemented) - ensure_collection() works for it today, nothing
  populates it yet.

Qdrant upsert semantics: every write in this module goes through
Qdrant's native `client.upsert()`, which inserts a point if its ID is
new and REPLACES the existing point in place if that ID already exists
- there is no separate "insert" path anywhere in this module. Point IDs
are derived deterministically (generate_point_id(), a UUID5 hash of the
chunk's identity: user_id + role_id + source_type + skill + text) so
re-indexing the same logical chunk always lands on the same point ID
and overwrites it, rather than accumulating duplicates on every
re-index. This is the same idempotency principle backend/db/upserts.py
applies to Postgres rows, applied here to vector points.

Credentials (QDRANT_URL, QDRANT_API_KEY) are read from
backend.utils.config; get_qdrant_client() raises
QdrantNotConfiguredError - not a crash - when they're missing, so
callers (and tests) can handle a not-yet-configured vector store
explicitly, the same way backend.db.client.get_client() and
backend.llm.client.get_openai_client() do for their services.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from backend.rag.chunking import EvidenceChunk
from backend.rag.embeddings import EmbeddingProvider
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)

CANDIDATE_EVIDENCE_COLLECTION = "candidate_evidence"
ROLE_CONTENT_COLLECTION = "role_content"
PREP_RESOURCES_COLLECTION = "prep_resources"

# Namespace for deterministic point IDs - fixed so the same chunk always
# hashes to the same UUID across process restarts.
_POINT_ID_NAMESPACE = uuid.UUID("a7f8c9d2-1b3e-4f5a-9c8d-2e1f0a9b8c7d")


class QdrantNotConfiguredError(RuntimeError):
    """Raised when QDRANT_URL / QDRANT_API_KEY are not set."""


class RetrievedChunk(BaseModel):
    """One retrieval result: the stored point's ID, similarity score, and full payload (including its text)."""

    id: str
    score: float
    payload: dict[str, Any]


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client, or raise if not configured."""
    settings = get_settings()
    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise QdrantNotConfiguredError(
            "QDRANT_URL and QDRANT_API_KEY must be set to access the vector store. See .env.example."
        )
    logger.info("Initializing Qdrant client")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection(collection_name: str, vector_size: int) -> None:
    """Create `collection_name` with cosine-distance vectors of `vector_size` if it doesn't already exist."""
    client = get_qdrant_client()
    if client.collection_exists(collection_name):
        return
    logger.info("Creating Qdrant collection '%s' (size=%d)", collection_name, vector_size)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def generate_point_id(*parts: str) -> str:
    """
    Derive a deterministic Qdrant point ID (a valid UUID string) from
    `parts`. The same parts always produce the same ID, so re-indexing
    the same logical chunk (see upsert_chunks()) overwrites its existing
    point instead of creating a duplicate.
    """
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, "|".join(parts)))


def upsert_chunks(
    collection_name: str,
    chunks: list[EvidenceChunk],
    provider: EmbeddingProvider,
    *,
    user_id: str,
    role_id: str | None = None,
) -> int:
    """
    Embed and upsert `chunks` into `collection_name`. Every point's
    payload always includes user_id, role_id, source_type, skill, and
    text, plus any extra metadata the chunk carries. Returns the number
    of points upserted (0 if `chunks` is empty - no collection is
    created and no API calls are made).

    Calling this again with logically identical chunks (same user_id,
    role_id, source_type, skill, and text) overwrites the same points
    rather than duplicating them - see this module's docstring for why.
    """
    if not chunks:
        return 0

    ensure_collection(collection_name, provider.dimensions)
    vectors = provider.embed([chunk.text for chunk in chunks])

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        point_id = generate_point_id(user_id, role_id or "", chunk.source_type, chunk.skill or "", chunk.text)
        payload = {
            "user_id": user_id,
            "role_id": role_id,
            "source_type": chunk.source_type,
            "skill": chunk.skill,
            "text": chunk.text,
            **chunk.metadata,
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    client = get_qdrant_client()
    logger.info("Upserting %d point(s) into '%s'", len(points), collection_name)
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def _build_filter(metadata_filter: dict[str, Any]) -> Filter:
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in metadata_filter.items()
        if value is not None
    ]
    return Filter(must=conditions)


def _query(
    collection_name: str,
    query_text: str,
    provider: EmbeddingProvider,
    *,
    top_k: int,
    metadata_filter: dict[str, Any] | None,
) -> list[RetrievedChunk]:
    client = get_qdrant_client()
    query_vector = provider.embed_one(query_text)
    query_filter = _build_filter(metadata_filter) if metadata_filter else None
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return [
        RetrievedChunk(id=str(point.id), score=point.score, payload=point.payload or {})
        for point in response.points
    ]


def retrieve_top_k(
    collection_name: str,
    query_text: str,
    provider: EmbeddingProvider,
    *,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Retrieve the top-k chunks most similar to `query_text` in `collection_name`, unfiltered."""
    return _query(collection_name, query_text, provider, top_k=top_k, metadata_filter=None)


def retrieve_filtered(
    collection_name: str,
    query_text: str,
    provider: EmbeddingProvider,
    *,
    metadata_filter: dict[str, Any],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k chunks most similar to `query_text` in
    `collection_name`, restricted to points whose payload matches every
    key/value in `metadata_filter` (e.g. {"user_id": ..., "skill": ...}).
    """
    return _query(collection_name, query_text, provider, top_k=top_k, metadata_filter=metadata_filter)
