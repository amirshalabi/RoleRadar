"""
Job normalization.

Different sources (APIs, manual entry, scraped listings) name the same
job fields differently - jobTitle vs title, companyName vs company, and
so on. normalize_role() maps any of a known set of key aliases into one
canonical Role model, so every module downstream of ingestion (dedup,
requirement extraction, scoring, persistence) always works with the
same shape regardless of where a posting came from.

This module performs no LLM calls - normalization is pure, deterministic
key-mapping and string cleanup.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.utils.hashing import generate_role_external_id


class Role(BaseModel):
    """
    Canonical, normalized representation of a job posting. This is what
    every downstream module (requirement extraction, filtering, scoring,
    persistence) consumes - never the raw source dict directly.
    """

    external_id: str
    company: str
    title: str
    location: str | None = None
    url: str | None = None
    description: str | None = None
    role_family: str | None = None
    raw_source: dict[str, Any] = Field(default_factory=dict)


# Known key aliases per canonical field, in priority order. Extend these
# tuples as new sources are integrated rather than special-casing sources
# elsewhere in the codebase.
_TITLE_KEYS = ("title", "jobTitle", "job_title", "position", "roleTitle", "role_title")
_COMPANY_KEYS = ("company", "companyName", "company_name", "employer", "organization")
_DESCRIPTION_KEYS = ("description", "jobDescription", "job_description", "desc", "summary")
_URL_KEYS = ("url", "jobUrl", "job_url", "link", "applyUrl", "apply_url")
_LOCATION_KEYS = ("location", "jobLocation", "job_location", "city")
_ROLE_FAMILY_KEYS = ("role_family", "roleFamily", "category", "jobFamily", "job_family")
_EXTERNAL_ID_KEYS = ("external_id", "id", "jobId", "job_id", "postingId", "posting_id")


def _first_present(raw: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty value found under any of `keys`, as a stripped string."""
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def normalize_role(raw: dict[str, Any]) -> Role:
    """
    Normalize a raw job dict with inconsistent keys into a canonical
    Role. Raises ValueError if no known alias supplies a title or
    company, since a role can't be meaningfully identified without them.

    If the source doesn't supply a stable id of its own, one is derived
    deterministically from normalized company + title + location + url
    (backend.utils.hashing.generate_role_external_id), matching the same
    derivation backend.db.roles.upsert_role() falls back to - so a role
    normalized here and later re-ingested through any other path
    resolves to the same external_id.
    """
    title = _first_present(raw, _TITLE_KEYS)
    company = _first_present(raw, _COMPANY_KEYS)
    if not title or not company:
        raise ValueError(
            "Raw job data must include a title and company under one of the "
            f"known key aliases (title: {_TITLE_KEYS}, company: {_COMPANY_KEYS})"
        )

    location = _first_present(raw, _LOCATION_KEYS)
    url = _first_present(raw, _URL_KEYS)
    description = _first_present(raw, _DESCRIPTION_KEYS)
    role_family = _first_present(raw, _ROLE_FAMILY_KEYS)
    external_id = _first_present(raw, _EXTERNAL_ID_KEYS)
    if not external_id:
        external_id = generate_role_external_id(company, title, location, url)

    return Role(
        external_id=external_id,
        company=company,
        title=title,
        location=location,
        url=url,
        description=description,
        role_family=role_family,
        raw_source=raw,
    )
