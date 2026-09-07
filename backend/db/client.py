"""
Supabase PostgreSQL client.

Initializes and exposes a lazily-created, cached Supabase client using
SUPABASE_URL and SUPABASE_KEY read from backend.utils.config (which in
turn reads from environment variables). Credentials are never
hardcoded.

When credentials are missing - e.g. in local scaffolding or unit tests
that run without live infrastructure - get_client() raises
SupabaseNotConfiguredError instead of silently returning a broken
client, so callers (and tests) can handle the missing-credentials case
explicitly rather than crashing deep inside a query.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from supabase import Client, create_client

from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


class SupabaseNotConfiguredError(RuntimeError):
    """Raised when SUPABASE_URL / SUPABASE_KEY are not set."""


@lru_cache
def get_client() -> Client:
    """Return a cached Supabase client, or raise if not configured."""
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        raise SupabaseNotConfiguredError(
            "SUPABASE_URL and SUPABASE_KEY must be set to access the database. "
            "See .env.example."
        )
    logger.info("Initializing Supabase client")
    return create_client(settings.supabase_url, settings.supabase_key)
