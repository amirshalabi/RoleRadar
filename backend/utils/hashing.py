"""
Deterministic hashing helpers.

Derives a stable external_id for a role from normalized company, title,
location, and URL when a source API does not supply its own stable
identifier, so repeated ingestion of the same logical role is idempotent
(same inputs always produce the same external_id, regardless of case or
whitespace differences between ingestion runs).

Also provides hash_json(), a generic stable-payload hash used to detect
when the inputs behind a cached LLM result have changed (see
backend.services.discovery's rationale caching) - unrelated to role
identity, but the same "same logical inputs -> same digest" idea.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    """Lowercase, strip, and collapse internal whitespace for stable hashing."""
    if not value:
        return ""
    return _WHITESPACE_RE.sub(" ", value.strip().lower())


def generate_role_external_id(
    company: str | None,
    title: str | None,
    location: str | None = None,
    url: str | None = None,
) -> str:
    """
    Derive a deterministic external_id from normalized company, title,
    location, and url. The same logical role (even with different
    casing/whitespace in the source data) always produces the same id.
    """
    payload = "|".join(
        normalize_text(part) for part in (company, title, location, url)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_json(payload: Any) -> str:
    """
    Deterministic sha256 hex digest of a JSON-serializable payload -
    stable across process restarts and regardless of dict key order
    (sort_keys=True). Callers needing order-independence for a list
    (e.g. a set of skills) must sort it themselves before passing it in.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
