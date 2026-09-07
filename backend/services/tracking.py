"""
Favorites + application tracking service layer.

These are the functions a Streamlit page should call - never
backend.db.favorites / backend.db.applications directly - so the one
piece of cross-table business logic in this project (favoriting a role
also marks it "saved" in the application pipeline) lives in exactly one
place instead of being re-implemented in every page that lets a user
save a role.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.db import applications as applications_db
from backend.db import favorites as favorites_db

logger = logging.getLogger(__name__)

# An application is bumped to 'saved' when a role is favorited only if
# it has no row yet, or is still sitting at the default 'discovered'
# stage. Any other existing status (already 'saved' or further along,
# e.g. 'applied') is left untouched - favoriting never regresses
# progress. This is the one deliberately simple stage rule in the
# project; see backend.db.applications for why nothing more elaborate
# than this exists yet.
_STATUSES_UPGRADED_TO_SAVED = {None, applications_db.DEFAULT_STATUS}


def save_role(
    user_id: str,
    role_id: str,
    priority: str = favorites_db.DEFAULT_PRIORITY,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Flag a role as a favorite and ensure it has an application-tracking
    row of at least 'saved' status - from the UI's perspective these are
    one action ("save this role"). Idempotent: calling this again for an
    already-saved role never creates a duplicate favorite or application
    row, and never resets priority/notes or regresses an application
    that has already moved past 'saved'.
    """
    favorite = favorites_db.save_favorite(user_id, role_id, priority=priority, notes=notes)

    existing_application = applications_db.get_application(user_id, role_id)
    current_status = existing_application["status"] if existing_application else None
    if current_status in _STATUSES_UPGRADED_TO_SAVED:
        applications_db.upsert_application(user_id, role_id, status="saved")

    return favorite


def unsave_role(user_id: str, role_id: str) -> None:
    """
    Remove a role's favorite flag. Deliberately does NOT touch the
    application row - un-favoriting shouldn't erase application history
    (e.g. a user who already applied and then just wants to declutter
    their favorites list should keep their 'applied' status).
    """
    favorites_db.remove_favorite(user_id, role_id)


def list_saved_roles(user_id: str) -> list[dict[str, Any]]:
    """Return all of a user's favorited roles, most recently saved first."""
    return favorites_db.list_favorites(user_id)


def list_saved_roles_with_details(user_id: str) -> list[dict[str, Any]]:
    """Same as list_saved_roles(), with each role's title/company/etc embedded - for display pages."""
    return favorites_db.list_favorites_with_roles(user_id)


def set_role_priority(user_id: str, role_id: str, priority: str) -> dict[str, Any]:
    """Update the priority (dream/high/interested/backup) of an already-saved role."""
    return favorites_db.update_favorite_priority(user_id, role_id, priority)


def set_role_notes(user_id: str, role_id: str, notes: str | None) -> dict[str, Any]:
    """Update the free-text notes on an already-saved role."""
    return favorites_db.update_favorite_notes(user_id, role_id, notes)


def get_application_status(user_id: str, role_id: str) -> dict[str, Any] | None:
    """Fetch a user's application-tracking row for a role, or None if none exists yet."""
    return applications_db.get_application(user_id, role_id)


def list_applications(user_id: str) -> list[dict[str, Any]]:
    """Return every application-tracking row for a user, across every stage."""
    return applications_db.list_applications(user_id)


def list_applications_with_details(user_id: str) -> list[dict[str, Any]]:
    """Same as list_applications(), with each row's role title/company/etc embedded - for display pages."""
    return applications_db.list_applications_with_roles(user_id)


def update_application_stage(
    user_id: str,
    role_id: str,
    status: str | None = None,
    *,
    application_date: str | None = None,
    deadline: str | None = None,
    interview_date: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Advance/update a role's application-tracking row. Any field left
    None is preserved from the existing row rather than cleared - e.g.
    setting only `interview_date` here does not erase a previously-set
    `deadline` or `notes`.
    """
    return applications_db.upsert_application(
        user_id,
        role_id,
        status=status,
        application_date=application_date,
        deadline=deadline,
        interview_date=interview_date,
        notes=notes,
    )
