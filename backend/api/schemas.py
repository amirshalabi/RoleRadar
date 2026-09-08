"""
API request/response Pydantic models.

Where a domain object is already a Pydantic model built by
backend.services/backend.matching/backend.planning (CandidateProfile,
RoleAnalysis, ReadinessResult, ...), routes return it DIRECTLY rather
than redefining an equivalent schema here - these models exist only for
the parts of the API surface (raw Supabase rows, request bodies) that
don't already have one. `model_config` is left at Pydantic's default
(extra fields ignored), so a *Response model built via .model_validate()
on a raw db row tolerates columns the schema doesn't list.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile
from backend.planning.readiness import ReadinessResult

# ---------------------------------------------------------------------
# candidate
# ---------------------------------------------------------------------


class CandidateParseRequest(BaseModel):
    resume_text: str = Field(min_length=1)


class CandidateParseResponse(BaseModel):
    profile: CandidateProfile
    skills_saved: int


# ---------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------


class RoleCreateRequest(BaseModel):
    company: str
    title: str
    location: str | None = None
    url: str | None = None
    description: str | None = None
    role_family: str | None = None
    external_id: str | None = None


class RoleResponse(BaseModel):
    id: str
    external_id: str
    company: str
    title: str
    location: str | None = None
    url: str | None = None
    description: str | None = None
    role_family: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------
# favorites
# ---------------------------------------------------------------------


class FavoriteCreateRequest(BaseModel):
    priority: str = "interested"
    notes: str | None = None


class FavoriteResponse(BaseModel):
    id: str
    user_id: str
    role_id: str
    priority: str
    notes: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class _EmbeddedRole(BaseModel):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    role_family: str | None = None


class FavoriteWithRoleResponse(FavoriteResponse):
    roles: _EmbeddedRole | None = None


# ---------------------------------------------------------------------
# applications
# ---------------------------------------------------------------------


class ApplicationUpdateRequest(BaseModel):
    status: str | None = None
    application_date: str | None = None
    deadline: str | None = None
    interview_date: str | None = None
    notes: str | None = None
    hours_available_per_day: float | None = None


class ApplicationResponse(BaseModel):
    id: str
    user_id: str
    role_id: str
    status: str
    application_date: str | None = None
    deadline: str | None = None
    interview_date: str | None = None
    notes: str | None = None
    hours_available_per_day: float | None = None
    created_at: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------
# study plans
# ---------------------------------------------------------------------


class StudyTaskResponse(BaseModel):
    id: str
    normalized_skill_name: str
    display_name: str | None = None
    day_index: int | None = None
    scheduled_date: str | None = None
    allocated_minutes: float | None = None
    priority_score: float | None = None
    task_description: str | None = None
    is_complete: bool = False


class StudyPlanResponse(BaseModel):
    id: str
    application_id: str
    interview_date: str | None = None
    hours_available_per_day: float | None = None
    days_remaining: int | None = None
    scheduling_days: int | None = None
    total_available_minutes: float | None = None
    version: int
    previous_version: int | None = None
    revision_reason: str | None = None
    tasks: list[StudyTaskResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------
# assessments
# ---------------------------------------------------------------------


class AssessmentCreateRequest(BaseModel):
    role_id: str
    topic: str
    observed_level: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1, default=0.9)


class AssessmentResultResponse(BaseModel):
    """backend.services.prep.submit_diagnostic()'s return value, shaped for the API - see that function's docstring."""

    plan: StudyPlanResponse
    readiness_before: ReadinessResult
    readiness_after: ReadinessResult
    message: str


# ---------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------


class PipelineAnalyticsResponse(BaseModel):
    has_data: bool
    ingested: int = 0
    deduplicated: int = 0
    hard_filtered: int = 0
    keyword_filtered: int = 0
    semantic_filtered: int = 0
    llm_analyzed: int = 0
    llm_avoidance_rate: float | None = None
    serial_ingestion_seconds: float | None = None
    concurrent_ingestion_seconds: float | None = None
    speedup: float | None = None
    finished_at: str | None = None
