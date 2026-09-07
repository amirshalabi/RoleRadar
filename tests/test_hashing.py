"""Tests for backend.utils.hashing (deterministic role external_id generation)."""

from backend.utils.hashing import generate_role_external_id, normalize_text


def test_normalize_text_lowercases_and_strips() -> None:
    assert normalize_text("  Acme Corp  ") == "acme corp"


def test_normalize_text_collapses_internal_whitespace() -> None:
    assert normalize_text("Software   Engineer\tIntern") == "software engineer intern"


def test_normalize_text_handles_none_and_empty() -> None:
    assert normalize_text(None) == ""
    assert normalize_text("") == ""


def test_generate_role_external_id_is_deterministic() -> None:
    first = generate_role_external_id("Acme Corp", "Backend Engineer", "Remote", "https://acme.example/jobs/1")
    second = generate_role_external_id("Acme Corp", "Backend Engineer", "Remote", "https://acme.example/jobs/1")
    assert first == second


def test_generate_role_external_id_is_case_and_whitespace_insensitive() -> None:
    first = generate_role_external_id("Acme Corp", "Backend Engineer", "Remote", "https://acme.example/jobs/1")
    second = generate_role_external_id("  acme   corp ", "BACKEND engineer", "remote", "https://acme.example/jobs/1")
    assert first == second


def test_generate_role_external_id_differs_for_different_roles() -> None:
    first = generate_role_external_id("Acme Corp", "Backend Engineer", "Remote")
    second = generate_role_external_id("Acme Corp", "Frontend Engineer", "Remote")
    assert first != second


def test_generate_role_external_id_differs_for_different_companies() -> None:
    first = generate_role_external_id("Acme Corp", "Backend Engineer", "Remote")
    second = generate_role_external_id("Globex Corp", "Backend Engineer", "Remote")
    assert first != second


def test_generate_role_external_id_handles_missing_optional_fields() -> None:
    result = generate_role_external_id("Acme Corp", "Backend Engineer")
    assert isinstance(result, str)
    assert len(result) == 64  # sha256 hex digest length
