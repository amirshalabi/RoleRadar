"""
Job deduplication.

Deduplicates a merged list of already-normalized Role objects by
external_id - the same stable identity backend.ingestion.normalize's
normalize_role() and backend.utils.hashing.generate_role_external_id()
already guarantee is deterministic (same company+title+location+url
always produces the same id, regardless of casing/whitespace or which
source it came from). This means two sources describing the same
logical job - or the same source appearing twice across a retry - both
collapse to a single entry here, before any database write happens.

Pure Python, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.ingestion.normalize import Role


class DeduplicationResult(BaseModel):
    """Deduplicated roles plus enough detail to explain what was removed."""

    unique_roles: list[Role]
    duplicate_count: int
    duplicates_by_external_id: dict[str, int] = Field(
        description="external_id -> how many EXTRA copies beyond the first were removed."
    )


def deduplicate_roles(roles: list[Role]) -> DeduplicationResult:
    """
    Deduplicate `roles` by external_id, keeping the FIRST occurrence in
    list order (so if `roles` is ordered by source priority, the
    highest-priority source's version of a role wins).
    """
    seen: dict[str, Role] = {}
    duplicate_counts: dict[str, int] = {}

    for role in roles:
        if role.external_id in seen:
            duplicate_counts[role.external_id] = duplicate_counts.get(role.external_id, 0) + 1
            continue
        seen[role.external_id] = role

    return DeduplicationResult(
        unique_roles=list(seen.values()),
        duplicate_count=sum(duplicate_counts.values()),
        duplicates_by_external_id=duplicate_counts,
    )
