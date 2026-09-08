"""Interview prep study plans - backend.services.prep."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_user_id
from backend.api.schemas import StudyPlanResponse
from backend.services import prep

router = APIRouter(prefix="/plans", tags=["plans"])


@router.post("/{role_id}", response_model=StudyPlanResponse)
def create_or_get_plan(role_id: str, user_id: str = Depends(get_user_id)) -> StudyPlanResponse:
    """
    The application's current study plan, building and persisting a
    fresh one (backend.planning.scheduler) only if none exists yet.
    Raises 400 (via the ValueError handler) if the application has no
    interview_date set.
    """
    plan = prep.get_or_create_plan_view(user_id, role_id)
    return StudyPlanResponse.model_validate(plan)
