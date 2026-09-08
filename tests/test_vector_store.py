"""
Tests for backend.rag.vector_store. No real Qdrant instance is used:
get_qdrant_client is monkeypatched with a small in-memory fake client
that records calls, so we can assert exactly what would be sent to
Qdrant (collection names, vector sizes, point IDs, payloads, filters)
without a live connection.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.rag.chunking import EvidenceChunk
from backend.rag.embeddings import EmbeddingProvider
from backend.rag.vector_store import (
    CANDIDATE_EVIDENCE_COLLECTION,
    QdrantNotConfiguredError,
    ensure_collection,
    generate_point_id,
    get_qdrant_client,
    retrieve_filtered,
    retrieve_top_k,
    upsert_chunks,
)
from backend.utils.config import get_settings


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake: each text's vector is [len(text), 0.0, 0.0]."""

    @property
    def dimensions(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 0.0] for t in texts]


class FakeQdrantClient:
    def __init__(self, existing_collections: set[str] | None = None, query_points: list | None = None):
        self.existing_collections = existing_collections or set()
        self.created_collections: list[tuple[str, int]] = []
        self.upserted: list[tuple[str, list]] = []
        self.queries: list[dict] = []
        self._query_points = query_points or []

    def collection_exists(self, name: str) -> bool:
        return name in self.existing_collections

    def create_collection(self, collection_name: str, vectors_config, **kwargs):
        self.created_collections.append((collection_name, vectors_config.size))
        self.existing_collections.add(collection_name)
        return True

    def upsert(self, collection_name: str, points, **kwargs):
        self.upserted.append((collection_name, list(points)))
        return SimpleNamespace(status="completed")

    def query_points(self, collection_name, query, query_filter, limit, with_payload, **kwargs):
        self.queries.append(
            {
                "collection_name": collection_name,
                "query": query,
                "query_filter": query_filter,
                "limit": limit,
            }
        )
        return SimpleNamespace(points=self._query_points)


@pytest.fixture(autouse=True)
def _clear_caches():
    get_settings.cache_clear()
    get_qdrant_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_qdrant_client.cache_clear()


def test_get_qdrant_client_raises_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with pytest.raises(QdrantNotConfiguredError):
        get_qdrant_client()


def test_ensure_collection_creates_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeQdrantClient(existing_collections=set())
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)

    ensure_collection("my_collection", 1536)

    assert fake_client.created_collections == [("my_collection", 1536)]


def test_ensure_collection_skips_when_already_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeQdrantClient(existing_collections={"my_collection"})
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)

    ensure_collection("my_collection", 1536)

    assert fake_client.created_collections == []


def test_generate_point_id_is_deterministic() -> None:
    first = generate_point_id("u1", "r1", "skill", "python", "some text")
    second = generate_point_id("u1", "r1", "skill", "python", "some text")

    assert first == second


def test_generate_point_id_differs_for_different_content() -> None:
    first = generate_point_id("u1", "r1", "skill", "python", "some text")
    second = generate_point_id("u1", "r1", "skill", "python", "different text")

    assert first != second


def test_generate_point_id_is_valid_uuid_format() -> None:
    import uuid

    point_id = generate_point_id("u1", "", "skill", "python", "text")

    assert uuid.UUID(point_id)  # does not raise


def test_upsert_chunks_returns_zero_for_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeQdrantClient()
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)

    result = upsert_chunks(CANDIDATE_EVIDENCE_COLLECTION, [], FakeEmbeddingProvider(), user_id="u1")

    assert result == 0
    assert fake_client.upserted == []
    assert fake_client.created_collections == []


def test_upsert_chunks_builds_full_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeQdrantClient(existing_collections={CANDIDATE_EVIDENCE_COLLECTION})
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)
    chunk = EvidenceChunk(text="Built Python ETL pipelines", source_type="skill", skill="python", metadata={"confidence": 0.8})

    count = upsert_chunks(CANDIDATE_EVIDENCE_COLLECTION, [chunk], FakeEmbeddingProvider(), user_id="u1", role_id="r1")

    assert count == 1
    [(collection_name, points)] = fake_client.upserted
    assert collection_name == CANDIDATE_EVIDENCE_COLLECTION
    [point] = points
    assert point.payload["user_id"] == "u1"
    assert point.payload["role_id"] == "r1"
    assert point.payload["source_type"] == "skill"
    assert point.payload["skill"] == "python"
    assert point.payload["text"] == "Built Python ETL pipelines"
    assert point.payload["confidence"] == 0.8


def test_upsert_chunks_same_chunk_reuses_same_point_id_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idempotency guarantee: re-indexing the same logical chunk must overwrite, not duplicate."""
    fake_client = FakeQdrantClient(existing_collections={CANDIDATE_EVIDENCE_COLLECTION})
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)
    chunk = EvidenceChunk(text="Built Python ETL pipelines", source_type="skill", skill="python")

    upsert_chunks(CANDIDATE_EVIDENCE_COLLECTION, [chunk], FakeEmbeddingProvider(), user_id="u1")
    upsert_chunks(CANDIDATE_EVIDENCE_COLLECTION, [chunk], FakeEmbeddingProvider(), user_id="u1")

    first_id = fake_client.upserted[0][1][0].id
    second_id = fake_client.upserted[1][1][0].id
    assert first_id == second_id


def test_upsert_chunks_ensures_collection_with_provider_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeQdrantClient(existing_collections=set())
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)
    chunk = EvidenceChunk(text="text", source_type="coursework")

    upsert_chunks(CANDIDATE_EVIDENCE_COLLECTION, [chunk], FakeEmbeddingProvider(), user_id="u1")

    assert fake_client.created_collections == [(CANDIDATE_EVIDENCE_COLLECTION, 3)]


def test_retrieve_top_k_returns_scored_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_points = [
        SimpleNamespace(id="p1", score=0.9, payload={"text": "chunk one"}),
        SimpleNamespace(id="p2", score=0.5, payload={"text": "chunk two"}),
    ]
    fake_client = FakeQdrantClient(query_points=fake_points)
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)

    results = retrieve_top_k(CANDIDATE_EVIDENCE_COLLECTION, "query", FakeEmbeddingProvider(), top_k=2)

    assert [r.id for r in results] == ["p1", "p2"]
    assert results[0].score == 0.9
    assert fake_client.queries[0]["query_filter"] is None
    assert fake_client.queries[0]["limit"] == 2


def test_retrieve_filtered_builds_metadata_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeQdrantClient(query_points=[])
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)

    retrieve_filtered(
        CANDIDATE_EVIDENCE_COLLECTION,
        "query",
        FakeEmbeddingProvider(),
        metadata_filter={"user_id": "u1", "skill": "python"},
        top_k=3,
    )

    query_filter = fake_client.queries[0]["query_filter"]
    assert query_filter is not None
    condition_keys = {c.key for c in query_filter.must}
    assert condition_keys == {"user_id", "skill"}


def test_retrieve_top_k_returns_empty_list_when_no_points_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeQdrantClient(query_points=[])
    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: fake_client)

    results = retrieve_top_k(CANDIDATE_EVIDENCE_COLLECTION, "query", FakeEmbeddingProvider())

    assert results == []


def test_retrieve_top_k_returns_empty_list_when_collection_does_not_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Querying a collection that was never created (e.g. a brand-new user
    with no resume evidence indexed yet) must degrade to "no results,"
    not propagate Qdrant's 404 as an unhandled exception.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    class _NotFoundQdrantClient(FakeQdrantClient):
        def query_points(self, *args, **kwargs):
            raise UnexpectedResponse(status_code=404, reason_phrase="Not Found", content=b"Collection not found", headers={})

    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: _NotFoundQdrantClient())

    results = retrieve_top_k(CANDIDATE_EVIDENCE_COLLECTION, "query", FakeEmbeddingProvider())

    assert results == []


def test_retrieve_top_k_reraises_non_404_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 (missing collection) is swallowed into an empty result; any other Qdrant error must NOT be silently hidden."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    class _ServerErrorQdrantClient(FakeQdrantClient):
        def query_points(self, *args, **kwargs):
            raise UnexpectedResponse(status_code=500, reason_phrase="Internal Server Error", content=b"boom", headers={})

    monkeypatch.setattr("backend.rag.vector_store.get_qdrant_client", lambda: _ServerErrorQdrantClient())

    with pytest.raises(UnexpectedResponse):
        retrieve_top_k(CANDIDATE_EVIDENCE_COLLECTION, "query", FakeEmbeddingProvider())


def test_retrieve_filtered_omits_none_values() -> None:
    from backend.rag.vector_store import _build_filter

    result = _build_filter({"user_id": "u1", "role_id": None})

    assert [c.key for c in result.must] == ["user_id"]
