"""
Tests for backend.rag.retrieval. retrieve_filtered is monkeypatched so
no real Qdrant/OpenAI calls happen - these tests only verify that the
domain-specific functions build the correct collection name and
metadata filter.
"""

from __future__ import annotations

import pytest

from backend.rag import retrieval as retrieval_module
from backend.rag.embeddings import EmbeddingProvider
from backend.rag.retrieval import retrieve_candidate_evidence, retrieve_role_content
from backend.rag.vector_store import CANDIDATE_EVIDENCE_COLLECTION, ROLE_CONTENT_COLLECTION


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_retrieve_candidate_evidence_scopes_to_user(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_retrieve_filtered(collection_name, query_text, provider, *, metadata_filter, top_k):
        captured["collection_name"] = collection_name
        captured["metadata_filter"] = metadata_filter
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(retrieval_module, "retrieve_filtered", fake_retrieve_filtered)

    retrieve_candidate_evidence("query text", "u1", provider=FakeEmbeddingProvider())

    assert captured["collection_name"] == CANDIDATE_EVIDENCE_COLLECTION
    assert captured["metadata_filter"] == {"user_id": "u1"}


def test_retrieve_candidate_evidence_adds_optional_role_and_skill_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_retrieve_filtered(collection_name, query_text, provider, *, metadata_filter, top_k):
        captured["metadata_filter"] = metadata_filter
        return []

    monkeypatch.setattr(retrieval_module, "retrieve_filtered", fake_retrieve_filtered)

    retrieve_candidate_evidence(
        "query text", "u1", role_id="r1", skill="python", top_k=10, provider=FakeEmbeddingProvider()
    )

    assert captured["metadata_filter"] == {"user_id": "u1", "role_id": "r1", "skill": "python"}


def test_retrieve_role_content_scopes_to_role(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_retrieve_filtered(collection_name, query_text, provider, *, metadata_filter, top_k):
        captured["collection_name"] = collection_name
        captured["metadata_filter"] = metadata_filter
        return []

    monkeypatch.setattr(retrieval_module, "retrieve_filtered", fake_retrieve_filtered)

    retrieve_role_content("query text", "r1", provider=FakeEmbeddingProvider())

    assert captured["collection_name"] == ROLE_CONTENT_COLLECTION
    assert captured["metadata_filter"] == {"role_id": "r1"}
