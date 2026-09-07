"""
Structured job requirement extraction.

Uses the LLM to convert an unstructured job description into structured
RoleRequirement objects (skills, target level, importance, required
flag, supporting evidence). The LLM performs extraction only; the result
is validated by Pydantic here, and that validated list - not the raw
description or raw model output - is what the deterministic matching
pipeline (filters, scorer, gaps) consumes.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from backend.ingestion.normalize import Role
from backend.llm.client import parse_structured
from backend.llm.prompts import (
    ROLE_REQUIREMENT_EXTRACTION_SYSTEM_PROMPT,
    build_role_requirement_extraction_user_prompt,
)

logger = logging.getLogger(__name__)


class RoleRequirement(BaseModel):
    """
    One requirement extracted from a job description. `target_level` and
    `importance` are the LLM's best judgment given the posting text - not
    ground truth - and are treated as such by downstream gap/fit
    calculations (see backend/matching/gaps.py, backend/matching/scorer.py).
    """

    skill: str
    normalized_skill: str
    target_level: float = Field(ge=0, le=10)
    importance: float = Field(ge=0, le=10)
    required: bool
    evidence: list[str] = Field(default_factory=list)


class _RoleRequirementExtractionResult(BaseModel):
    """Internal wrapper: OpenAI structured outputs require a top-level object, not a bare list."""

    requirements: list[RoleRequirement] = Field(default_factory=list)


def extract_role_requirements(role: Role, model: str | None = None) -> list[RoleRequirement]:
    """
    Use the LLM to extract structured requirements from `role.description`.

    Raises ValueError if the role has no description to extract from, and
    backend.llm.client.LLMExtractionError if the model refuses or returns
    something that doesn't validate against the schema.
    """
    if not role.description or not role.description.strip():
        raise ValueError("role.description must not be empty")

    logger.info(
        "Extracting requirements for role external_id=%s (%d chars of description)",
        role.external_id,
        len(role.description),
    )
    result = parse_structured(
        system_prompt=ROLE_REQUIREMENT_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=build_role_requirement_extraction_user_prompt(
            title=role.title, company=role.company, description=role.description
        ),
        response_model=_RoleRequirementExtractionResult,
        model=model,
    )
    return result.requirements
