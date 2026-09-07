"""
Shared idempotent upsert helper.

Wraps the Supabase/PostgREST upsert call with one consistent interface
used by roles.py, candidates.py, favorites.py, and applications.py, so
conflict-target and update-on-conflict behavior - the mechanism that
makes repeated ingestion idempotent - stays identical across every
table instead of being reimplemented per module.
"""

from __future__ import annotations

import logging
from typing import Any

from postgrest.exceptions import APIError

from backend.db.client import get_client

logger = logging.getLogger(__name__)


def upsert_row(
    table: str,
    values: dict[str, Any],
    on_conflict: str,
) -> dict[str, Any]:
    """
    Insert `values` into `table`, updating the existing row in place if
    a row matching the `on_conflict` unique constraint already exists.

    `on_conflict` must name the exact unique constraint's column(s) as a
    comma-separated string (e.g. "external_id" or "user_id,role_id").

    Returns the resulting row. Calling this repeatedly with the same
    logical object (same on_conflict column values) never creates a
    duplicate row.
    """
    client = get_client()
    try:
        response = (
            client.table(table).upsert(values, on_conflict=on_conflict).execute()
        )
    except APIError:
        logger.exception("Upsert into %s failed", table)
        raise
    if not response.data:
        raise RuntimeError(f"Upsert into {table} returned no data")
    return response.data[0]
