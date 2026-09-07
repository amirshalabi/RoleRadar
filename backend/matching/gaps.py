"""
Skill gap calculation.

Computes, per role requirement, the candidate's raw and
importance-weighted gap against that requirement. This is pure,
deterministic Python - no LLM calls - and produces the structured
per-skill facts that backend/matching/scorer.py (fit scoring),
backend/matching/confidence.py (certainty aggregation), and later
backend/planning (prep prioritization) all build on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateSkillEstimate
from backend.candidate.skills import normalize_skill_name
from backend.llm.extract_requirements import RoleRequirement


class SkillGapResult(BaseModel):
    """
    The gap between a candidate's estimated level and one role
    requirement, plus enough context to explain the number later
    (matched skill, its confidence, whether the requirement is
    required).
    """

    normalized_skill: str
    display_skill: str
    target_level: float
    candidate_level: float
    importance: float
    required: bool
    matched: bool
    confidence: float
    raw_gap: float
    weighted_gap: float
    satisfaction_ratio: float = Field(
        description="min(candidate_level / target_level, 1.0), or 1.0 if target_level is 0."
    )


def calculate_skill_gaps(
    candidate_skills: list[CandidateSkillEstimate],
    requirements: list[RoleRequirement],
) -> list[SkillGapResult]:
    """
    For every requirement, find the candidate's estimate for that skill
    (matched by normalized name) and compute:

        raw_gap = max(target_level - candidate_level, 0)
        weighted_gap = raw_gap * (importance / 10)

    A requirement with no matching candidate skill is treated as
    candidate_level=0, confidence=0.0 - the candidate has made no claim
    to that skill, so there is no evidence to draw on. This is a fact
    about missing evidence, not a penalty layered on top of an existing
    estimate (see backend/matching/confidence.py for how that
    distinction is preserved downstream).
    """
    skills_by_name = {
        normalize_skill_name(skill.normalized_skill_name): skill for skill in candidate_skills
    }

    results: list[SkillGapResult] = []
    for requirement in requirements:
        key = normalize_skill_name(requirement.normalized_skill)
        matched_skill = skills_by_name.get(key)

        candidate_level = matched_skill.estimated_level if matched_skill else 0.0
        confidence = matched_skill.confidence if matched_skill else 0.0

        raw_gap = max(requirement.target_level - candidate_level, 0.0)
        weighted_gap = raw_gap * (requirement.importance / 10.0)
        satisfaction_ratio = (
            min(candidate_level / requirement.target_level, 1.0)
            if requirement.target_level > 0
            else 1.0
        )

        results.append(
            SkillGapResult(
                normalized_skill=key,
                display_skill=requirement.skill,
                target_level=requirement.target_level,
                candidate_level=candidate_level,
                importance=requirement.importance,
                required=requirement.required,
                matched=matched_skill is not None,
                confidence=confidence,
                raw_gap=raw_gap,
                weighted_gap=weighted_gap,
                satisfaction_ratio=satisfaction_ratio,
            )
        )
    return results


# Below this confidence, a "matched" skill is treated as uncertain rather
# than a trustworthy partial/full match for display purposes - the
# candidate claimed it, but the evidence behind that claim is too weak
# to rely on. Purely a UI-classification threshold (see
# classify_gap_status()); it does not affect scoring anywhere upstream.
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.4


def classify_gap_status(gap: SkillGapResult) -> str:
    """
    Deterministic display status for one skill gap:

    - "missing": no matching candidate skill estimate at all.
    - "uncertain": matched, but confidence is below
      UNCERTAIN_CONFIDENCE_THRESHOLD - too weakly evidenced to trust.
    - "met": candidate fully satisfies the target level.
    - "partial": matched with reasonable confidence, but below target.

    Pure classification of already-computed SkillGapResult fields - adds
    no new scoring logic of its own.
    """
    if not gap.matched:
        return "missing"
    if gap.confidence < UNCERTAIN_CONFIDENCE_THRESHOLD:
        return "uncertain"
    if gap.satisfaction_ratio >= 1.0:
        return "met"
    return "partial"
