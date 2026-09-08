"""
Role browsing and scoring - backend.services.discovery, plus
backend.db.roles.upsert_role() directly for creation (an idempotent
upsert with no further business logic to wrap - the same directness
several Streamlit pages already use for simple persistence).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.dependencies import get_user_id
from backend.api.schemas import RoleCreateRequest, RoleResponse
from backend.db import roles as roles_db
from backend.services import discovery
from backend.services.discovery import RoleAnalysis

router = APIRouter(prefix="/roles", tags=["roles"])


@router.post("", response_model=RoleResponse)
def create_role(payload: RoleCreateRequest) -> RoleResponse:
    """Insert or update a role, keyed on external_id (or one derived from company+title+location+url)."""
    row = roles_db.upsert_role(
        company=payload.company,
        title=payload.title,
        location=payload.location,
        url=payload.url,
        description=payload.description,
        role_family=payload.role_family,
        external_id=payload.external_id,
    )
    return RoleResponse.model_validate(row)


@router.get("", response_model=list[RoleResponse])
def list_roles(limit: int = 50) -> list[RoleResponse]:
    """Most recently seen roles first."""
    rows = discovery.list_available_roles(limit=limit)
    return [RoleResponse.model_validate(row) for row in rows]


@router.get("/{role_id}", response_model=RoleResponse)
def get_role(role_id: str) -> RoleResponse:
    row = discovery.get_role(role_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Role {role_id!r} not found.")
    return RoleResponse.model_validate(row)


@router.post("/{role_id}/score", response_model=RoleAnalysis)
def score_role(role_id: str, user_id: str = Depends(get_user_id)) -> RoleAnalysis:
    """
    Full deterministic fit/gap/readiness analysis for one role against
    the current candidate (backend.services.discovery.analyze_role) -
    extracts and persists requirements via the LLM only if this role has
    never been analyzed before.
    """
    role_row = discovery.get_role(role_id)
    if role_row is None:
        raise HTTPException(status_code=404, detail=f"Role {role_id!r} not found.")
    return discovery.analyze_role(user_id, role_row)
