"""Candidate resume parsing - backend.services.candidate."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_user_id
from backend.api.schemas import CandidateParseRequest, CandidateParseResponse
from backend.services.candidate import parse_and_save_resume

router = APIRouter(prefix="/candidate", tags=["candidate"])


@router.post("/parse", response_model=CandidateParseResponse)
def parse_resume(payload: CandidateParseRequest, user_id: str = Depends(get_user_id)) -> CandidateParseResponse:
    """Extract a structured profile from resume text and persist it (upsert - safe to call again after a re-upload)."""
    profile = parse_and_save_resume(user_id, payload.resume_text)
    return CandidateParseResponse(profile=profile, skills_saved=len(profile.skills))
