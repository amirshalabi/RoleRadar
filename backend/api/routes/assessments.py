"""Diagnostic submission + adaptive replanning - backend.services.prep."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_user_id
from backend.api.schemas import AssessmentCreateRequest, AssessmentResultResponse, StudyPlanResponse
from backend.services import prep

router = APIRouter(prefix="/assessments", tags=["assessments"])


@router.post("", response_model=AssessmentResultResponse, status_code=201)
def submit_assessment(payload: AssessmentCreateRequest, user_id: str = Depends(get_user_id)) -> AssessmentResultResponse:
    """
    Record a diagnostic result and produce the next plan revision
    (backend.services.prep.submit_diagnostic): persists the diagnostic,
    replans remaining time while preserving completed tasks, and reports
    before/after readiness. Always a genuinely new event (201).
    """
    result = prep.submit_diagnostic(user_id, payload.role_id, payload.topic, payload.observed_level, payload.confidence)
    return AssessmentResultResponse(
        plan=StudyPlanResponse.model_validate(result["plan"]),
        readiness_before=result["readiness_before"],
        readiness_after=result["readiness_after"],
        message=result["message"],
    )
