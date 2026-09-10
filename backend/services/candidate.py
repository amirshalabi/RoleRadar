"""
Candidate onboarding: raw resume text -> structured profile
(backend.candidate.profile.extract_candidate_profile, the only LLM call
in this module besides the Qdrant embedding call in
_index_resume_evidence) -> Python-side dedup
(backend.candidate.skills.skills_from_profile) -> persisted profile +
skills -> best-effort resume-evidence indexing into Qdrant.

process_resume_upload() is the entry point pages/0_Profile.py calls -
the PDF-to-text step (backend.candidate.parser.extract_text_from_pdf)
stays in the page, since it's a stateless utility with no side effects,
not "the extraction pipeline"; everything with a side effect (the LLM
call, Postgres writes, Qdrant writes) lives here.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from pydantic import BaseModel

from backend.candidate.profile import (
    CandidateProfile,
    CandidateSkillEstimate,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    ResearchEntry,
    extract_candidate_profile,
)
from backend.candidate.skills import skills_from_profile
from backend.db import candidates as candidates_db
from backend.rag.chunking import chunk_candidate_evidence
from backend.rag.embeddings import get_embedding_provider
from backend.rag.vector_store import CANDIDATE_EVIDENCE_COLLECTION, delete_by_metadata, upsert_chunks

logger = logging.getLogger(__name__)


class ResumeProcessingError(RuntimeError):
    """
    Raised for a resume-specific problem the Profile page should show as
    a plain, friendly message (no readable text extracted, blank paste).
    Deliberately distinct from OpenAINotConfiguredError/
    SupabaseNotConfiguredError/QdrantNotConfiguredError, which mean "fix
    your .env," not "try a different file" - callers can catch each
    separately to show the right message.
    """


class ResumeUploadResult(BaseModel):
    """What process_resume_upload() reports back to the Profile page."""

    profile: CandidateProfile
    skills_saved: int
    is_new_resume: bool
    evidence_indexed: bool
    evidence_index_error: str | None = None


def hash_resume_text(resume_text: str) -> str:
    """A stable content hash for "is this the same resume I already have on file" detection - hashes the extracted TEXT, not raw file bytes, so two different PDF exports of the identical resume still match."""
    return hashlib.sha256(resume_text.strip().encode("utf-8")).hexdigest()


def load_candidate_profile(user_id: str) -> CandidateProfile | None:
    """
    Reconstruct the FULL persisted CandidateProfile (education,
    coursework, projects, research, domain experience, skills - not
    just skills) for a user, or None if they've never saved one. For
    the Profile page's "what RoleRadar knows about you" view after a
    page reload.

    Deliberately separate from backend.services.discovery.
    build_candidate_profile(), which intentionally stays a lighter
    skills-only reconstruction for the hot path (fit scoring, readiness)
    that runs on every Discover page load - this one does one extra
    Postgres read (the profile row) that only the Profile page needs.
    """
    profile_row = candidates_db.get_candidate_profile(user_id)
    if profile_row is None:
        return None
    skill_rows = candidates_db.list_candidate_skills(user_id)
    return CandidateProfile(
        education=[EducationEntry(**entry) for entry in profile_row.get("education") or []],
        coursework=profile_row.get("coursework") or [],
        programming_languages=profile_row.get("programming_languages") or [],
        frameworks=profile_row.get("frameworks") or [],
        tools=profile_row.get("tools") or [],
        projects=[ProjectEntry(**entry) for entry in profile_row.get("projects") or []],
        internships=[ExperienceEntry(**entry) for entry in profile_row.get("experience") or []],
        research=[ResearchEntry(**entry) for entry in profile_row.get("research") or []],
        domain_experience=profile_row.get("domain_experience") or [],
        skills=[
            CandidateSkillEstimate(
                normalized_skill_name=row["normalized_skill_name"],
                display_name=row.get("display_name") or row["normalized_skill_name"],
                estimated_level=row["estimated_level"],
                confidence=row["confidence"],
                evidence_snippets=row.get("evidence_snippets") or [],
            )
            for row in skill_rows
        ],
    )


def get_resume_metadata(user_id: str) -> dict[str, str | None] | None:
    """filename/content hash/parsed-at for the currently-saved resume, or None if no resume has been saved yet."""
    profile_row = candidates_db.get_candidate_profile(user_id)
    if profile_row is None:
        return None
    return {
        "filename": profile_row.get("resume_filename"),
        "content_hash": profile_row.get("resume_content_hash"),
        "parsed_at": profile_row.get("resume_parsed_at"),
    }


def save_candidate_profile(
    user_id: str,
    profile: CandidateProfile,
    resume_text: str | None = None,
    *,
    resume_filename: str | None = None,
    resume_content_hash: str | None = None,
    resume_parsed_at: str | None = None,
) -> CandidateProfile:
    """
    Persist an already-extracted CandidateProfile - the persistence half
    of parse_and_save_resume()/process_resume_upload(), split out so a
    caller that already has a profile in hand (e.g.
    scripts/demo_end_to_end.py building one without an LLM call) doesn't
    need to duplicate this logic:

    - One candidate_profiles row: education/experience(=internships)/
      projects/coursework/programming_languages/frameworks/tools/
      research/domain_experience, plus resume metadata when supplied.
      candidate_profiles' interests/constraints columns still have no
      corresponding CandidateProfile field (a pre-existing data-model
      gap - see backend/db/candidates.py - candidate PREFERENCES are a
      separate, user-stated table, not this resume-derived one).
    - One candidate_skills row per DEDUPLICATED skill estimate
      (skills_from_profile - never a possibly-repeated raw list),
      including its real evidence_snippets.
    - Any candidate_skills row for a skill that existed before this call
      but is NOT in the new deduplicated set is deleted - a resume
      re-upload replaces the candidate's skill set rather than only ever
      adding to it, so a skill dropped from a newer resume doesn't stay
      stuck on the profile forever.

    Returns the profile actually persisted (post-dedup), so a caller
    doesn't need a second read to see what was saved.
    """
    deduped_skills = skills_from_profile(profile)
    profile = profile.model_copy(update={"skills": deduped_skills})

    previously_saved_names = {row["normalized_skill_name"] for row in candidates_db.list_candidate_skills(user_id)}

    candidates_db.upsert_candidate_profile(
        user_id,
        raw_resume_text=resume_text,
        education=[entry.model_dump() for entry in profile.education],
        experience=[entry.model_dump() for entry in profile.internships],
        projects=[entry.model_dump() for entry in profile.projects],
        coursework=profile.coursework,
        programming_languages=profile.programming_languages,
        frameworks=profile.frameworks,
        tools=profile.tools,
        research=[entry.model_dump() for entry in profile.research],
        domain_experience=profile.domain_experience,
        resume_filename=resume_filename,
        resume_content_hash=resume_content_hash,
        resume_parsed_at=resume_parsed_at,
    )

    new_names: set[str] = set()
    for skill in deduped_skills:
        candidates_db.upsert_candidate_skill(
            user_id,
            skill.normalized_skill_name,
            estimated_level=skill.estimated_level,
            confidence=skill.confidence,
            display_name=skill.display_name,
            evidence_snippets=skill.evidence_snippets,
        )
        new_names.add(skill.normalized_skill_name)

    for stale_name in previously_saved_names - new_names:
        candidates_db.delete_candidate_skill(user_id, stale_name)

    return profile


def parse_and_save_resume(user_id: str, resume_text: str, model: str | None = None) -> CandidateProfile:
    """Extract a CandidateProfile from `resume_text` and persist it - see save_candidate_profile(). Kept as the plain text-in/profile-out entry point used by the FastAPI route; process_resume_upload() below wraps this with hashing, metadata, and Qdrant indexing for the Streamlit upload flow."""
    profile = extract_candidate_profile(resume_text, model=model)
    return save_candidate_profile(user_id, profile, resume_text=resume_text)


def _index_resume_evidence(user_id: str, profile: CandidateProfile) -> None:
    """
    Best-effort: clear this user's previously-indexed resume evidence
    (a replaced resume's dropped skill/project would otherwise leave its
    old evidence text as a stale, still-retrievable Qdrant point - see
    backend.rag.vector_store's docstring on why point IDs alone can't
    detect that), then re-chunk and re-index the current profile. Raises
    on failure - process_resume_upload() decides how to degrade
    gracefully; this function's job is only to do the indexing, not to
    decide whether a failure should block the rest of the upload.
    """
    delete_by_metadata(CANDIDATE_EVIDENCE_COLLECTION, {"user_id": user_id})
    chunks = chunk_candidate_evidence(profile)
    upsert_chunks(CANDIDATE_EVIDENCE_COLLECTION, chunks, get_embedding_provider(), user_id=user_id)


def process_resume_upload(
    user_id: str,
    resume_text: str,
    *,
    filename: str | None = None,
    model: str | None = None,
) -> ResumeUploadResult:
    """
    The full resume-upload pipeline behind pages/0_Profile.py: hash
    check (skip re-parsing an identical resume) -> LLM extraction ->
    Postgres persistence -> best-effort Qdrant evidence indexing.

    Raises ResumeProcessingError if `resume_text` has no real content.
    Lets OpenAINotConfiguredError/SupabaseNotConfiguredError propagate
    uncaught (the page shows its own "not configured" messaging for
    those, the same pattern every other page already uses) - only
    Qdrant indexing degrades gracefully (evidence_indexed=False,
    evidence_index_error set) rather than failing the whole upload,
    since fit/gap/readiness scoring doesn't depend on it, only
    evidence-grounded rationale text does.
    """
    if not resume_text or not resume_text.strip():
        raise ResumeProcessingError("No readable text was found in that resume. Try a different file, or paste your resume text directly.")

    content_hash = hash_resume_text(resume_text)
    existing_metadata = get_resume_metadata(user_id)
    if existing_metadata is not None and existing_metadata["content_hash"] == content_hash:
        logger.info("Resume for user %s matches the already-saved content hash - skipping re-parse", user_id)
        existing_profile = load_candidate_profile(user_id)
        assert existing_profile is not None  # existing_metadata came from the same row
        return ResumeUploadResult(
            profile=existing_profile,
            skills_saved=len(existing_profile.skills),
            is_new_resume=False,
            evidence_indexed=True,
        )

    profile = extract_candidate_profile(resume_text, model=model)
    profile = save_candidate_profile(
        user_id,
        profile,
        resume_text=resume_text,
        resume_filename=filename,
        resume_content_hash=content_hash,
        resume_parsed_at=datetime.now(timezone.utc).isoformat(),
    )

    evidence_indexed = True
    evidence_index_error: str | None = None
    try:
        _index_resume_evidence(user_id, profile)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: Qdrant/embedding failure must not fail the whole upload
        logger.exception("Resume evidence indexing failed for user %s", user_id)
        evidence_indexed = False
        evidence_index_error = str(exc)

    return ResumeUploadResult(
        profile=profile,
        skills_saved=len(profile.skills),
        is_new_resume=True,
        evidence_indexed=evidence_indexed,
        evidence_index_error=evidence_index_error,
    )
