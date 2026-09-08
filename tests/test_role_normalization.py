"""Tests for backend.ingestion.normalize (raw job dict -> canonical Role)."""

from __future__ import annotations

import pytest

from backend.ingestion.normalize import Role, normalize_role
from backend.utils.hashing import generate_role_external_id


def test_normalize_role_maps_alternate_keys() -> None:
    raw = {
        "jobTitle": "Backend Engineer",
        "companyName": "Acme Corp",
        "jobDescription": "Build backend services.",
        "jobUrl": "https://acme.example/jobs/1",
        "location": "Remote",
    }

    role = normalize_role(raw)

    assert isinstance(role, Role)
    assert role.title == "Backend Engineer"
    assert role.company == "Acme Corp"
    assert role.description == "Build backend services."
    assert role.url == "https://acme.example/jobs/1"
    assert role.location == "Remote"


def test_normalize_role_maps_canonical_keys_directly() -> None:
    raw = {
        "title": "Frontend Engineer",
        "company": "Globex",
        "description": "Build UI.",
        "url": "https://globex.example/jobs/2",
    }

    role = normalize_role(raw)

    assert role.title == "Frontend Engineer"
    assert role.company == "Globex"


def test_normalize_role_prefers_supplied_external_id() -> None:
    raw = {
        "jobId": "provider-abc-123",
        "title": "SWE",
        "company": "Acme",
    }

    role = normalize_role(raw)

    assert role.external_id == "provider-abc-123"


def test_normalize_role_derives_external_id_when_missing() -> None:
    raw = {"title": "SWE Intern", "company": "Acme Corp", "location": "Remote"}

    role = normalize_role(raw)

    expected = generate_role_external_id("Acme Corp", "SWE Intern", "Remote", None)
    assert role.external_id == expected


def test_normalize_role_derived_external_id_is_stable_across_key_styles() -> None:
    raw_camel = {
        "jobTitle": "SWE Intern",
        "companyName": "Acme Corp",
        "location": "Remote",
    }
    raw_snake = {
        "job_title": "SWE Intern",
        "company_name": "Acme Corp",
        "location": "Remote",
    }

    role_a = normalize_role(raw_camel)
    role_b = normalize_role(raw_snake)

    assert role_a.external_id == role_b.external_id


def test_normalize_role_preserves_raw_source() -> None:
    raw = {"title": "SWE", "company": "Acme", "extraField": "keep me"}

    role = normalize_role(raw)

    assert role.raw_source == raw


def test_normalize_role_raises_without_title_or_company() -> None:
    with pytest.raises(ValueError):
        normalize_role({"description": "no title or company here"})


def test_normalize_role_raises_when_title_missing() -> None:
    with pytest.raises(ValueError):
        normalize_role({"company": "Acme"})


def test_normalize_role_raises_when_company_missing() -> None:
    with pytest.raises(ValueError):
        normalize_role({"title": "SWE"})


def test_normalize_role_treats_blank_strings_as_absent() -> None:
    raw = {"title": "SWE", "company": "Acme", "location": "   "}

    role = normalize_role(raw)

    assert role.location is None


def test_normalize_role_missing_location_key_defaults_to_none() -> None:
    role = normalize_role({"title": "SWE", "company": "Acme"})

    assert role.location is None
    # A missing location must not block persistence downstream - the
    # role should still normalize fully and get a stable external_id.
    assert role.external_id == generate_role_external_id("Acme", "SWE", None, None)


def test_normalize_role_missing_url_key_defaults_to_none() -> None:
    role = normalize_role({"title": "SWE", "company": "Acme", "location": "Remote"})

    assert role.url is None


def test_normalize_role_missing_description_key_defaults_to_none() -> None:
    role = normalize_role({"title": "SWE", "company": "Acme"})

    assert role.description is None


def test_normalize_role_empty_description_string_treated_as_absent() -> None:
    role = normalize_role({"title": "SWE", "company": "Acme", "description": ""})

    assert role.description is None


def test_normalize_role_missing_location_and_url_and_description_together() -> None:
    """The realistic worst case: a posting that supplies only title/company - everything else optional is absent."""
    role = normalize_role({"title": "SWE Intern", "company": "Acme Corp"})

    assert role.location is None
    assert role.url is None
    assert role.description is None
    assert role.role_family is None
    assert role.external_id  # still deterministically derived, never empty/None
