"""
Embedding provider abstraction.

Defines the EmbeddingProvider interface every other RAG module
(chunking is provider-agnostic; vector_store.py and retrieval.py depend
only on this interface) uses to turn text into vectors. The only
implementation today is OpenAIEmbeddingProvider, but nothing outside
this file knows that: get_embedding_provider() is the single seam where
a SentenceTransformers (or any other) backend would be swapped in later
- e.g. by branching on an EMBEDDING_PROVIDER env var - without touching
chunking.py, vector_store.py, or retrieval.py at all.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from backend.llm.client import get_openai_client
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Common interface every embedding backend must implement."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The fixed length of every vector this provider returns. Qdrant collections are sized from this."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text, in the same order."""

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single text."""
        return self.embed([text])[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeds text using the OpenAI embeddings API. Raises OpenAINotConfiguredError if OPENAI_API_KEY is unset."""

    # Native output size per model, used to size Qdrant collections
    # without making a network call just to ask. Falls back to 1536
    # (text-embedding-3-small's size) for an unrecognized model name.
    _KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str | None = None):
        self._model = model or get_settings().openai_embedding_model

    @property
    def dimensions(self) -> int:
        return self._KNOWN_DIMENSIONS.get(self._model, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = get_openai_client()
        logger.info("Embedding %d text(s) with model=%s", len(texts), self._model)
        response = client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


def get_embedding_provider() -> EmbeddingProvider:
    """
    Return the configured embedding provider. Always OpenAI today; this
    factory is the intended extension point for a future
    SentenceTransformers implementation.
    """
    return OpenAIEmbeddingProvider()
