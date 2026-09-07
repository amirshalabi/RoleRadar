"""
Prep priority calculation.

Merges two possibly-overlapping sources of "what needs work" into one
deterministic, ranked list of PrepItems - no LLM involvement anywhere
in this arithmetic:

  1. Role-specific requirement gaps (backend.matching.gaps.SkillGapResult)
     - how far the candidate is from THIS job posting's stated targets.
  2. Role-family interview-topic readiness gaps
     (backend.planning.readiness.ReadinessResult) - how far the
     candidate is from being interview-ready on THIS role family's
     general topic taxonomy, which may already incorporate diagnostic
     evidence (stronger than resume-only signal).

A skill/topic present in both sources (e.g. "algorithms" is both a job
requirement AND feeds the "dsa" interview topic) is merged into ONE
PrepItem rather than counted twice - see calculate_prep_priorities()'s
docstring for exactly which source wins for which field.

    priority_score = gap x requirement_importance x interview_topic_weight
                      x confidence_adjustment x favorite_priority_multiplier

All four base factors are on bounded, documented scales (gap and
requirement_importance are 0-10; interview_topic_weight and confidence
are 0-1); favorite_priority_multiplier is an optional >=1-or-<1 scalar
(see backend.matching.cross_role.FAVORITE_PRIORITY_MULTIPLIERS) so
prep for a "dream" role's interview can be prioritized over a "backup"
role's without this module needing to import the favorites system.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.candidate.skills import normalize_skill_name
from backend.matching.gaps import SkillGapResult
from backend.planning.readiness import TOPIC_SKILL_ALIASES, ReadinessResult

# Confidence dampens priority but never zeroes it: a skill with NO
# resume evidence at all (confidence=0) usually ALSO has the largest
# possible gap (candidate_level defaults to 0) - dropping its priority
# to zero because we're "not confident" would be backwards, since
# missing evidence is itself often a reason to study something, not a
# reason to skip it. MIN_CONFIDENCE_ADJUSTMENT is the floor; full
# confidence (e.g. a diagnostic result) reaches the full 1.0 multiplier.
MIN_CONFIDENCE_ADJUSTMENT = 0.5

# Neutral importance (0-10 scale) assigned to a readiness topic that has
# no corresponding explicit role requirement (e.g. "behavioral" rarely
# appears as a literal job-posting requirement, but still needs prep).
DEFAULT_REQUIREMENT_IMPORTANCE = 5.0

# Flat baseline interview-topic weight (0-1 scale) assigned to a role
# requirement that doesn't map to any recognized interview topic for
# this role family (e.g. a job-specific tool the curated topic taxonomy
# doesn't cover) - low, but never zero, so it isn't invisible to prep.
UNMAPPED_TOPIC_WEIGHT = 0.1


def confidence_adjustment(confidence: float) -> float:
    """See MIN_CONFIDENCE_ADJUSTMENT above for why this floors rather than zeroes."""
    return MIN_CONFIDENCE_ADJUSTMENT + (1.0 - MIN_CONFIDENCE_ADJUSTMENT) * confidence


class PrepItem(BaseModel):
    """One skill/interview-topic's deterministic prep priority, before any time allocation."""

    normalized_skill: str
    display_name: str
    gap: float = Field(ge=0, le=10)
    requirement_importance: float = Field(ge=0, le=10)
    interview_topic_weight: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    confidence_adjustment: float = Field(ge=0, le=1)
    priority_score: float = Field(ge=0)
    source: str = Field(
        description="One of: role_requirement, readiness_topic, role_requirement_and_readiness_topic."
    )


def _build_alias_to_topic_index(topic_weights: dict[str, float]) -> dict[str, str]:
    """
    normalized alias skill name -> topic name, for every topic this role
    family weights.

    Built in two passes so a topic's OWN name always wins as its own
    match, even though another topic may list it as a weaker borrowed
    alias (e.g. readiness.TOPIC_SKILL_ALIASES lists "probability" under
    BOTH the "probability" topic itself AND "game_ev_reasoning", since
    probability skill evidence is weak-but-real signal for EV reasoning
    too). Without this two-pass ordering, whichever topic happened to
    be processed last in `topic_weights` would silently steal an exact
    self-name match from its rightful topic.
    """
    index: dict[str, str] = {}
    for topic in topic_weights:
        index[normalize_skill_name(topic)] = topic
    for topic in topic_weights:
        for alias in TOPIC_SKILL_ALIASES.get(topic, ()):
            normalized_alias = normalize_skill_name(alias)
            if normalized_alias not in index:
                index[normalized_alias] = topic
    return index


def calculate_prep_priorities(
    skill_gaps: list[SkillGapResult],
    readiness: ReadinessResult | None = None,
    favorite_priority_multiplier: float = 1.0,
) -> list[PrepItem]:
    """
    Build one PrepItem per skill/topic with priority_score > 0 (a
    zero-gap item has nothing to prepare - it's excluded rather than
    returned with a zero allocation later), sorted by priority_score
    descending.

    Merge rule for a skill/topic that appears in both `skill_gaps` and
    `readiness.topic_readiness` (matched via readiness.TOPIC_SKILL_ALIASES):
    - gap and confidence come from the READINESS topic, since it may
      incorporate diagnostic evidence - stronger than the resume-only
      signal behind SkillGapResult.
    - requirement_importance comes from the SkillGapResult, since it is
      specific to the actual job posting, unlike a role-family-wide
      interview topic weight.
    - interview_topic_weight comes from the readiness topic's weight.

    A skill/topic present in only one source uses that source's values
    and a neutral default for the other (DEFAULT_REQUIREMENT_IMPORTANCE
    for a topic-only item, UNMAPPED_TOPIC_WEIGHT for a requirement-only
    item) - see those constants' docstrings above.
    """
    topic_weights = readiness.weights_used if readiness else {}
    topic_by_name = {topic.topic: topic for topic in (readiness.topic_readiness if readiness else [])}
    alias_to_topic = _build_alias_to_topic_index(topic_weights)

    handled_topics: set[str] = set()
    items: list[PrepItem] = []

    for gap in skill_gaps:
        matched_topic_name = alias_to_topic.get(gap.normalized_skill)
        matched_topic = topic_by_name.get(matched_topic_name) if matched_topic_name else None

        if matched_topic is not None:
            handled_topics.add(matched_topic.topic)
            effective_gap = 10.0 - matched_topic.combined_score
            effective_confidence = matched_topic.combined_confidence
            topic_weight = matched_topic.weight
            source = "role_requirement_and_readiness_topic"
        else:
            effective_gap = gap.raw_gap
            effective_confidence = gap.confidence
            topic_weight = UNMAPPED_TOPIC_WEIGHT
            source = "role_requirement"

        adjustment = confidence_adjustment(effective_confidence)
        priority_score = effective_gap * gap.importance * topic_weight * adjustment * favorite_priority_multiplier

        items.append(
            PrepItem(
                normalized_skill=gap.normalized_skill,
                display_name=gap.display_skill,
                gap=round(effective_gap, 3),
                requirement_importance=gap.importance,
                interview_topic_weight=topic_weight,
                confidence=round(effective_confidence, 3),
                confidence_adjustment=round(adjustment, 3),
                priority_score=round(priority_score, 4),
                source=source,
            )
        )

    for topic_name, topic in topic_by_name.items():
        if topic_name in handled_topics:
            continue
        effective_gap = 10.0 - topic.combined_score
        adjustment = confidence_adjustment(topic.combined_confidence)
        priority_score = (
            effective_gap * DEFAULT_REQUIREMENT_IMPORTANCE * topic.weight * adjustment * favorite_priority_multiplier
        )

        items.append(
            PrepItem(
                normalized_skill=topic_name,
                display_name=topic_name.replace("_", " ").title(),
                gap=round(effective_gap, 3),
                requirement_importance=DEFAULT_REQUIREMENT_IMPORTANCE,
                interview_topic_weight=topic.weight,
                confidence=round(topic.combined_confidence, 3),
                confidence_adjustment=round(adjustment, 3),
                priority_score=round(priority_score, 4),
                source="readiness_topic",
            )
        )

    items = [item for item in items if item.priority_score > 0]
    items.sort(key=lambda item: item.priority_score, reverse=True)
    return items
