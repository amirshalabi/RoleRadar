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
