"""
Centralized application configuration loaded from environment variables.

python-dotenv loads a local .env file (if present) and this module exposes
a single typed Settings object via get_settings(). Other modules should
read configuration through get_settings() rather than calling os.getenv
directly, so required variables are defined and validated in one place.

No external services (OpenAI, Supabase, Qdrant) are called from here -
this module only loads and types configuration values.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    """Typed application settings sourced from environment variables."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    supabase_url: str | None = None
    supabase_key: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings loaded from the environment."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_KEY"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
