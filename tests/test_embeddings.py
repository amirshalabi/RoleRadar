"""
Tests for backend.rag.embeddings. No real OpenAI calls are made:
get_openai_client is monkeypatched with a fake client shaped like the
OpenAI SDK's embeddings endpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.llm import client as llm_client
from backend.rag import embeddings as embeddings_module
from backend.rag.embeddings import OpenAIEmbeddingProvider, get_embedding_provider
from backend.utils.config import get_settings


@pytest.fixture(autouse=True)
def _clear_caches():
    get_settings.cache_clear()
    llm_client.get_openai_client.cache_clear()
    yield
    get_settings.cache_clear()
    llm_client.get_openai_client.cache_clear()


def _fake_client_returning(vectors: list[list[float]]):
    def create(*, input, model):
        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])

    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def test_dimensions_known_model() -> None:
    provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
    assert provider.dimensions == 1536


def test_dimensions_falls_back_for_unknown_model() -> None:
    provider = OpenAIEmbeddingProvider(model="some-future-model")
    assert provider.dimensions == 1536


def test_embed_empty_list_makes_no_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_get_openai_client():
        nonlocal called
        called = True
        raise AssertionError("should not be called for an empty batch")

    monkeypatch.setattr(embeddings_module, "get_openai_client", fake_get_openai_client)
    provider = OpenAIEmbeddingProvider()

    result = provider.embed([])

    assert result == []
    assert called is False


def test_embed_returns_one_vector_per_text_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _fake_client_returning([[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(embeddings_module, "get_openai_client", lambda: fake_client)
    provider = OpenAIEmbeddingProvider()

    result = provider.embed(["first", "second"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_one_wraps_single_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _fake_client_returning([[0.5, 0.6]])
    monkeypatch.setattr(embeddings_module, "get_openai_client", lambda: fake_client)
    provider = OpenAIEmbeddingProvider()

    result = provider.embed_one("only text")

    assert result == [0.5, 0.6]


def test_embed_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIEmbeddingProvider()

    with pytest.raises(llm_client.OpenAINotConfiguredError):
        provider.embed(["text"])


def test_get_embedding_provider_returns_openai_provider() -> None:
    provider = get_embedding_provider()

    assert isinstance(provider, OpenAIEmbeddingProvider)
