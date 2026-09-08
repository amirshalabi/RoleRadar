"""Application tracking - backend.services.tracking."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_user_id
from backend.api.schemas import ApplicationResponse, ApplicationUpdateRequest
from backend.services import tracking

router = APIRouter(prefix="/applications", tags=["applications"])


@router.patch("/{role_id}", response_model=ApplicationResponse)
def update_application(
    role_id: str, payload: ApplicationUpdateRequest, user_id: str = Depends(get_user_id)
) -> ApplicationResponse:
    """
    Advance/update an application's stage, dates, notes, or prep hours.
    Every field is optional - anything left unset is preserved from the
    existing row rather than cleared (backend.services.tracking.update_application_stage).
    """
    row = tracking.update_application_stage(
        user_id,
        role_id,
        status=payload.status,
        application_date=payload.application_date,
        deadline=payload.deadline,
        interview_date=payload.interview_date,
        notes=payload.notes,
        hours_available_per_day=payload.hours_available_per_day,
    )
    return ApplicationResponse.model_validate(row)
