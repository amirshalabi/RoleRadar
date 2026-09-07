"""
Role rationale persistence.

Stores the full generated FitRationale (backend.llm.rationale) as JSON,
one row per (user, role), alongside a hash of the inputs it was
generated from (backend.utils.hashing.hash_json over candidate skills +
role requirements). backend.services.discovery reads that hash to
decide whether a cached rationale can be reused as-is or whether inputs
have changed enough to warrant a fresh (LLM + Qdrant) generation - this
module itself makes no such decision, it only stores/retrieves rows.
"""

from __future__ import annotations

from typing import Any

from backend.db.client import get_client
from backend.db.upserts import upsert_row

RATIONALES_TABLE = "role_rationales"


def get_rationale(user_id: str, role_id: str) -> dict[str, Any] | None:
    """Fetch the persisted rationale row for a (user, role) pair, or None if none exists yet."""
    client = get_client()
    response = (
        client.table(RATIONALES_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("role_id", role_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def upsert_rationale(user_id: str, role_id: str, inputs_hash: str, rationale: dict[str, Any]) -> dict[str, Any]:
    """
    Insert or overwrite the persisted rationale for a (user, role) pair,
    keyed on UNIQUE(user_id, role_id). A blind upsert - unlike
    favorites, there is no user-editable state on this row to preserve,
    so a freshly generated rationale always fully replaces whatever was
    stored before.
    """
    values = {
        "user_id": user_id,
        "role_id": role_id,
        "inputs_hash": inputs_hash,
        "rationale": rationale,
    }
    return upsert_row(RATIONALES_TABLE, values, on_conflict="user_id,role_id")
