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

Querying a collection that has never been written to (e.g. retrieval
for a brand-new user with no resume evidence indexed yet) is treated as
"no results," not an error: retrieve_top_k()/retrieve_filtered() catch
Qdrant's 404 for a missing collection and return [] - a caller should
never need to call ensure_collection() defensively before a read.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams

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


# Payload fields every retrieval call in backend/rag/retrieval.py filters
# on (user_id/role_id/skill - see retrieve_candidate_evidence(),
# retrieve_role_content(), retrieve_skill_evidence()). Qdrant requires an
# explicit index on a field before it can be used in a query_filter -
# without one, a filtered query_points() call raises a 400 "Index
# required but not found" error, not just "no results."
_FILTERED_PAYLOAD_FIELDS = ("user_id", "role_id", "skill")


def ensure_collection(collection_name: str, vector_size: int) -> None:
    """
    Create `collection_name` with cosine-distance vectors of
    `vector_size` if it doesn't already exist, and ensure it has payload
    indexes on every field this module's callers ever filter by.

    The index step runs even for an already-existing collection (not
    just on first creation) so a collection created before these indexes
    existed self-heals the next time anything upserts into it, rather
    than staying permanently unfilterable - create_payload_index() is
    idempotent, safe to call again for a field that's already indexed.
    """
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        logger.info("Creating Qdrant collection '%s' (size=%d)", collection_name, vector_size)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    for field_name in _FILTERED_PAYLOAD_FIELDS:
        client.create_payload_index(collection_name, field_name, field_schema=PayloadSchemaType.KEYWORD)


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


def delete_by_metadata(collection_name: str, metadata_filter: dict[str, Any]) -> None:
    """
    Delete every point in `collection_name` whose payload matches every
    key/value in `metadata_filter` (e.g. {"user_id": ...} to clear one
    candidate's evidence before re-indexing a replaced resume, so a
    changed/removed skill's old evidence text - which would otherwise
    hash to a different, never-cleaned-up point ID, see this module's
    docstring - doesn't linger as stale retrieval results forever).

    A no-op, not an error, if `collection_name` doesn't exist yet (e.g.
    the very first resume upload for a user, before anything has ever
    been indexed).
    """
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        return
    client.delete(collection_name=collection_name, points_selector=_build_filter(metadata_filter))


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
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            # Nothing has ever been indexed into this collection yet
            # (e.g. a brand-new user with no resume evidence indexed, or
            # a role that has never been through rationale generation) -
            # equivalent to "no results found," not an error worth
            # surfacing to the caller. upsert_chunks() creates the
            # collection on first write, so this is expected pre-write.
            logger.info("Qdrant collection '%s' does not exist yet - returning no results", collection_name)
            return []
        raise
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
