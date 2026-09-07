"""
Embedding provider abstraction.

Will define a common interface for generating text embeddings, backed
initially by the OpenAI embeddings API, designed so a SentenceTransformers
implementation could be swapped in later without changing callers.
"""
