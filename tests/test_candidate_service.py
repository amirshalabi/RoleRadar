"""Tests for backend.services.candidate. No real OpenAI calls - extract_candidate_profile is monkeypatched."""

from __future__ import annotations

import pytest

from backend.candidate.profile import CandidateProfile, CandidateSkillEstimate
from backend.db import candidates as candidates_db
from backend.db import upserts as upserts_db
from backend.services import candidate as candidate_service
from tests._fake_supabase import FakeSupabaseClient


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    client = FakeSupabaseClient()
    for module in (candidates_db, upserts_db):
        monkeypatch.setattr(module, "get_client", lambda c=client: c)
    return client


def _fake_profile() -> CandidateProfile:
    return CandidateProfile(
        skills=[
            CandidateSkillEstimate(normalized_skill_name="Python", display_name="Python", estimated_level=7.0, confidence=0.7, evidence_snippets=["Built Python pipelines"]),
            CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=5.0, confidence=0.4, evidence_snippets=["Skills: Python"]),
        ]
    )


def test_parse_and_save_resume_persists_deduplicated_skills(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: _fake_profile())

    profile = candidate_service.parse_and_save_resume("u1", "resume text")

    assert len(profile.skills) == 1  # "Python" and "python" collapse to one normalized identity
    assert profile.skills[0].confidence == 0.7  # higher-confidence duplicate wins

    stored = candidates_db.list_candidate_skills("u1")
    assert len(stored) == 1
    assert stored[0]["evidence_snippets"] == ["Built Python pipelines"]


def test_parse_and_save_resume_persists_profile_row(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: _fake_profile())

    candidate_service.parse_and_save_resume("u1", "resume text")

    row = candidates_db.get_candidate_profile("u1")
    assert row is not None
    assert row["raw_resume_text"] == "resume text"


def test_parse_and_save_resume_returns_empty_profile_for_no_skills(monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient) -> None:
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: CandidateProfile())

    profile = candidate_service.parse_and_save_resume("u1", "resume text")

    assert profile.skills == []
    assert candidates_db.list_candidate_skills("u1") == []


# ---------------------------------------------------------------------
# save_candidate_profile - replacing a resume removes stale skills
# ---------------------------------------------------------------------


def test_save_candidate_profile_persists_extended_fields(fake_client: FakeSupabaseClient) -> None:
    profile = CandidateProfile(coursework=["Algorithms"], programming_languages=["Python"], domain_experience=["Trading"])

    candidate_service.save_candidate_profile("u1", profile, resume_text="text")

    row = candidates_db.get_candidate_profile("u1")
    assert row["coursework"] == ["Algorithms"]
    assert row["programming_languages"] == ["Python"]
    assert row["domain_experience"] == ["Trading"]


def test_save_candidate_profile_removes_skill_dropped_from_new_resume(fake_client: FakeSupabaseClient) -> None:
    first = CandidateProfile(skills=[CandidateSkillEstimate(normalized_skill_name="java", display_name="Java", estimated_level=6.0, confidence=0.6)])
    candidate_service.save_candidate_profile("u1", first)
    assert {row["normalized_skill_name"] for row in candidates_db.list_candidate_skills("u1")} == {"java"}

    second = CandidateProfile(skills=[CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.0, confidence=0.7)])
    candidate_service.save_candidate_profile("u1", second)

    remaining = candidates_db.list_candidate_skills("u1")
    assert {row["normalized_skill_name"] for row in remaining} == {"python"}


def test_save_candidate_profile_keeps_skill_present_in_both_saves(fake_client: FakeSupabaseClient) -> None:
    profile = CandidateProfile(skills=[CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=5.0, confidence=0.5)])
    candidate_service.save_candidate_profile("u1", profile)
    candidate_service.save_candidate_profile("u1", profile)

    assert len(candidates_db.list_candidate_skills("u1")) == 1


# ---------------------------------------------------------------------
# load_candidate_profile / get_resume_metadata
# ---------------------------------------------------------------------


def test_load_candidate_profile_returns_none_without_a_saved_profile(fake_client: FakeSupabaseClient) -> None:
    assert candidate_service.load_candidate_profile("u1") is None


def test_get_resume_metadata_returns_none_without_a_saved_profile(fake_client: FakeSupabaseClient) -> None:
    assert candidate_service.get_resume_metadata("u1") is None


def test_load_candidate_profile_reconstructs_full_profile(fake_client: FakeSupabaseClient) -> None:
    profile = CandidateProfile(
        coursework=["Algorithms"],
        skills=[CandidateSkillEstimate(normalized_skill_name="python", display_name="Python", estimated_level=7.0, confidence=0.7, evidence_snippets=["Built Python pipelines"])],
    )
    candidate_service.save_candidate_profile("u1", profile)

    loaded = candidate_service.load_candidate_profile("u1")

    assert loaded is not None
    assert loaded.coursework == ["Algorithms"]
    assert loaded.skills[0].normalized_skill_name == "python"
    assert loaded.skills[0].evidence_snippets == ["Built Python pipelines"]


# ---------------------------------------------------------------------
# process_resume_upload - hashing, LLM extraction, Qdrant indexing
# ---------------------------------------------------------------------


@pytest.fixture
def stubbed_indexing(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Records calls into chunk_candidate_evidence/upsert_chunks/delete_by_metadata without touching real Qdrant/OpenAI."""
    calls = {"deleted": [], "indexed": []}
    monkeypatch.setattr(candidate_service, "delete_by_metadata", lambda collection, metadata_filter: calls["deleted"].append((collection, metadata_filter)))
    monkeypatch.setattr(candidate_service, "chunk_candidate_evidence", lambda profile: ["chunk"])
    monkeypatch.setattr(candidate_service, "upsert_chunks", lambda collection, chunks, provider, *, user_id: calls["indexed"].append((collection, chunks, user_id)))
    monkeypatch.setattr(candidate_service, "get_embedding_provider", lambda: object())
    return calls


def test_process_resume_upload_rejects_blank_text(fake_client: FakeSupabaseClient) -> None:
    with pytest.raises(candidate_service.ResumeProcessingError):
        candidate_service.process_resume_upload("u1", "   ")


def test_process_resume_upload_indexes_evidence_after_successful_parse(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient, stubbed_indexing: dict
) -> None:
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: _fake_profile())

    result = candidate_service.process_resume_upload("u1", "resume text", filename="resume.pdf")

    assert result.is_new_resume is True
    assert result.evidence_indexed is True
    assert len(stubbed_indexing["deleted"]) == 1
    assert len(stubbed_indexing["indexed"]) == 1


def test_process_resume_upload_degrades_gracefully_when_indexing_fails(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient
) -> None:
    monkeypatch.setattr(candidate_service, "extract_candidate_profile", lambda text, model=None: _fake_profile())
    monkeypatch.setattr(candidate_service, "delete_by_metadata", lambda collection, metadata_filter: None)
    monkeypatch.setattr(candidate_service, "chunk_candidate_evidence", lambda profile: ["chunk"])

    def _boom(*args, **kwargs):
        raise RuntimeError("qdrant is down")

    monkeypatch.setattr(candidate_service, "upsert_chunks", _boom)
    monkeypatch.setattr(candidate_service, "get_embedding_provider", lambda: object())

    result = candidate_service.process_resume_upload("u1", "resume text")

    # The upload itself still succeeds - skills are saved even though evidence indexing failed.
    assert result.skills_saved == 1
    assert result.evidence_indexed is False
    assert "qdrant is down" in result.evidence_index_error
    assert candidates_db.list_candidate_skills("u1") != []


def test_process_resume_upload_skips_reparse_for_identical_resume(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient, stubbed_indexing: dict
) -> None:
    call_count = {"n": 0}

    def _extract(text, model=None):
        call_count["n"] += 1
        return _fake_profile()

    monkeypatch.setattr(candidate_service, "extract_candidate_profile", _extract)

    first = candidate_service.process_resume_upload("u1", "resume text", filename="resume.pdf")
    second = candidate_service.process_resume_upload("u1", "resume text", filename="resume.pdf")

    assert call_count["n"] == 1  # the second call never re-invoked the LLM
    assert first.is_new_resume is True
    assert second.is_new_resume is False
    assert second.skills_saved == first.skills_saved
    # No duplicate rows were created by processing the same resume twice.
    assert len(candidates_db.list_candidate_skills("u1")) == first.skills_saved


def test_process_resume_upload_reparses_when_resume_text_changes(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeSupabaseClient, stubbed_indexing: dict
) -> None:
    call_count = {"n": 0}

    def _extract(text, model=None):
        call_count["n"] += 1
        return _fake_profile()

    monkeypatch.setattr(candidate_service, "extract_candidate_profile", _extract)

    candidate_service.process_resume_upload("u1", "resume text v1")
    result = candidate_service.process_resume_upload("u1", "resume text v2")

    assert call_count["n"] == 2
    assert result.is_new_resume is True
