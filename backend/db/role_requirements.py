"""
Role requirements persistence.

Idempotent upsert and query operations for the role_requirements table,
keyed on UNIQUE(role_id, normalized_skill_name), so re-extracting a
role's requirements (backend.llm.extract_requirements - an LLM call)
never creates duplicate rows for the same skill, and only needs to
happen once per role: callers should check list_role_requirements()
first and only call extract_role_requirements() when it comes back
empty (see backend.services.discovery.analyze_role).
"""

from __future__ import annotations

import logging
from typing import Any

from backend.db.client import get_client

logger = logging.getLogger(__name__)

ROLE_REQUIREMENTS_TABLE = "role_requirements"


def list_role_requirements(role_id: str) -> list[dict[str, Any]]:
    """Return every persisted requirement for a role, or [] if none have been extracted yet."""
    client = get_client()
    response = client.table(ROLE_REQUIREMENTS_TABLE).select("*").eq("role_id", role_id).execute()
    return response.data or []


def upsert_role_requirements(role_id: str, requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Upsert several requirements for one role in a single batch call,
    keyed on UNIQUE(role_id, normalized_skill_name). Each dict in
    `requirements` must have normalized_skill_name, display_name,
    target_level, importance, is_required, and evidence.
    """
    if not requirements:
        return []
    client = get_client()
    rows = [{"role_id": role_id, **requirement} for requirement in requirements]
    response = (
        client.table(ROLE_REQUIREMENTS_TABLE).upsert(rows, on_conflict="role_id,normalized_skill_name").execute()
    )
    if not response.data:
        raise RuntimeError(f"Upsert into {ROLE_REQUIREMENTS_TABLE} returned no data")
    return response.data
