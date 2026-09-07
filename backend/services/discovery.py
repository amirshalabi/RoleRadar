"""
Discover service: browsing ingested roles.

Thin wrapper over backend.db.roles so the Discover page never imports
backend.db directly, consistent with every other page.
"""

from __future__ import annotations

from typing import Any

from backend.db import roles as roles_db


def list_available_roles(limit: int = 50) -> list[dict[str, Any]]:
    """Most recently ingested roles first."""
    return roles_db.list_roles(limit=limit)
