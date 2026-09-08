"""Favorites - backend.services.tracking."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends

from backend.api.dependencies import get_user_id
from backend.api.schemas import FavoriteCreateRequest, FavoriteResponse, FavoriteWithRoleResponse
from backend.services import tracking

router = APIRouter(prefix="/favorites", tags=["favorites"])


@router.post("/{role_id}", response_model=FavoriteResponse)
def save_favorite(
    role_id: str,
    payload: FavoriteCreateRequest = Body(default_factory=FavoriteCreateRequest),
    user_id: str = Depends(get_user_id),
) -> FavoriteResponse:
    """Flag a role as a favorite. Idempotent - re-saving an already-favorited role leaves its priority/notes untouched."""
    row = tracking.save_role(user_id, role_id, priority=payload.priority, notes=payload.notes)
    return FavoriteResponse.model_validate(row)


@router.delete("/{role_id}", status_code=204)
def remove_favorite(role_id: str, user_id: str = Depends(get_user_id)) -> None:
    """Un-flag a role as a favorite. A no-op (still 204) if it wasn't favorited."""
    tracking.unsave_role(user_id, role_id)


@router.get("", response_model=list[FavoriteWithRoleResponse])
def list_favorites(user_id: str = Depends(get_user_id)) -> list[FavoriteWithRoleResponse]:
    """Every favorited role, most recently saved first, with role details embedded."""
    rows = tracking.list_saved_roles_with_details(user_id)
    return [FavoriteWithRoleResponse.model_validate(row) for row in rows]
