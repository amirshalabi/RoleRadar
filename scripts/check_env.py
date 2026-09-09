#!/usr/bin/env python3
"""
Verify that .env is correctly wired up for Supabase, OpenAI, and Qdrant -
one cheap, real connectivity check per service. Never prints a secret
value, only pass/fail and (on failure) the error the service itself
returned.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.config import get_settings  # noqa: E402


def check_supabase() -> tuple[bool, str]:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        return False, "SUPABASE_URL / SUPABASE_KEY not set in .env"
    try:
        from backend.db.client import get_client

        client = get_client()
        client.table("users").select("id").limit(1).execute()
        return True, "connected, and the 'users' table exists (schema.sql has been applied)"
    except Exception as exc:  # noqa: BLE001 - reporting the real error is the point
        message = str(exc)
        if "does not exist" in message.lower() or "42P01" in message:
            return False, "connected, but the schema hasn't been applied yet - run sql/schema.sql in the Supabase SQL Editor"
        return False, f"connection failed: {message}"


def check_openai() -> tuple[bool, str]:
    settings = get_settings()
    if not settings.openai_api_key:
        return False, "OPENAI_API_KEY not set in .env"
    try:
        from backend.llm.client import get_openai_client

        client = get_openai_client()
        client.models.list()  # cheap, no completion tokens billed
        return True, "connected"
    except Exception as exc:  # noqa: BLE001
        return False, f"connection failed: {exc}"


def check_qdrant() -> tuple[bool, str]:
    settings = get_settings()
    if not settings.qdrant_url or not settings.qdrant_api_key:
        return False, "QDRANT_URL / QDRANT_API_KEY not set in .env"
    try:
        from backend.rag.vector_store import get_qdrant_client

        client = get_qdrant_client()
        client.get_collections()
        return True, "connected"
    except Exception as exc:  # noqa: BLE001
        return False, f"connection failed: {exc}"


def main() -> int:
    checks = [("Supabase", check_supabase), ("OpenAI", check_openai), ("Qdrant", check_qdrant)]
    all_ok = True
    for name, check in checks:
        ok, detail = check()
        all_ok &= ok
        print(f"{'✅' if ok else '❌'} {name}: {detail}")
    if not all_ok:
        print("\nFill in the missing values in .env (see .env.example for the exact variable names), then re-run this script.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
