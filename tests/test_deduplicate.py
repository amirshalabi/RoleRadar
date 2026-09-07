"""Tests for backend.ingestion.deduplicate. Pure Python, no I/O."""

from __future__ import annotations

from backend.ingestion.deduplicate import deduplicate_roles
from backend.ingestion.normalize import normalize_role


def _role(title="SWE Intern", company="Acme", **overrides):
    raw = {"title": title, "company": company}
    raw.update(overrides)
    return normalize_role(raw)


def test_deduplicate_no_duplicates_returns_all_roles_unchanged() -> None:
    roles = [_role("A"), _role("B"), _role("C")]

    result = deduplicate_roles(roles)

    assert len(result.unique_roles) == 3
    assert result.duplicate_count == 0
    assert result.duplicates_by_external_id == {}


def test_deduplicate_removes_exact_duplicates() -> None:
    role = _role("SWE Intern", "Acme", location="Remote")
    roles = [role, role, role]

    result = deduplicate_roles(roles)

    assert len(result.unique_roles) == 1
    assert result.duplicate_count == 2


def test_deduplicate_same_role_from_two_sources_collapses_to_one() -> None:
    """Same logical job normalized from two different sources' key styles still shares one external_id."""
    from_source_a = normalize_role({"jobTitle": "SWE Intern", "companyName": "Acme Corp", "location": "Remote"})
    from_source_b = normalize_role({"title": "SWE Intern", "company": "Acme Corp", "location": "Remote"})

    result = deduplicate_roles([from_source_a, from_source_b])

    assert len(result.unique_roles) == 1
    assert result.duplicate_count == 1


def test_deduplicate_keeps_first_occurrence() -> None:
    first = normalize_role({"title": "SWE Intern", "company": "Acme", "description": "first version"})
    second = normalize_role({"title": "SWE Intern", "company": "Acme", "description": "second version"})
    # both normalize to the same external_id (same company+title+location+url)
    assert first.external_id == second.external_id

    result = deduplicate_roles([first, second])

    assert result.unique_roles[0].description == "first version"


def test_deduplicate_tracks_count_per_external_id() -> None:
    role_a = _role("A")
    role_b = _role("B")
    roles = [role_a, role_a, role_a, role_b, role_b]

    result = deduplicate_roles(roles)

    assert result.duplicates_by_external_id[role_a.external_id] == 2
    assert result.duplicates_by_external_id[role_b.external_id] == 1
    assert result.duplicate_count == 3


def test_deduplicate_empty_input_returns_empty_result() -> None:
    result = deduplicate_roles([])

    assert result.unique_roles == []
    assert result.duplicate_count == 0
