"""
Interview readiness - deliberately separate from fit.

WHY FIT AND READINESS ARE DIFFERENT (and both matter):

  FIT (backend.matching.scorer.calculate_fit_score) answers: "Is this
  candidate a good match for this ROLE?" It's a comparison of the
  candidate's overall profile - skills, experience, coursework, domain
  background, interests, constraints - against what the ROLE asks for.
  A candidate can have excellent fit (the right background, the right
  interests, no location conflicts) while being completely unprepared
  to sit down and pass an interview tomorrow - fit says nothing about
  interview performance.

  READINESS (this module) answers a narrower, time-sensitive question:
  "Is this candidate prepared to perform well in an INTERVIEW right
  now?" It's scored against a curated set of INTERVIEW TOPICS specific
  to a role family (e.g. quant interviews probe probability, mental
  math, and market knowledge; SWE interviews probe DSA and systems
  design) - not against one particular job posting's requirements.
  Two candidates with identical fit for the same role can have very
  different readiness: one drilled LeetCode and mock interviews all
  month, the other didn't touch algorithms since a class two years ago.

  Conflating the two would be misleading: a high-fit, low-readiness
  candidate needs INTERVIEW PREP, not a different job search; a
  low-fit, high-readiness candidate needs a different ROLE, not more
  practice. Keeping them as separate scores (and separate modules) lets
  backend.planning.scheduler later allocate prep time based on
  readiness gaps specifically, without that logic being entangled with
  fit's six unrelated components.

This module performs NO LLM calls anywhere - readiness, like fit, is
computed by deterministic Python arithmetic over structured data
(CandidateProfile skills + optional diagnostic results). The LLM's only
role anywhere in RoleRadar is explaining an already-computed number, and
readiness is not exempt from that rule.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from backend.candidate.profile import CandidateProfile
from backend.candidate.skills import normalize_skill_name

# ---------------------------------------------------------------------
# Role-family readiness weights (interview TOPICS, not job requirements)
# ---------------------------------------------------------------------

QUANT_READINESS_WEIGHTS: dict[str, float] = {
    "probability": 0.25,
    "coding": 0.20,
    "algorithms": 0.15,
    "mental_math": 0.15,
    "market_knowledge": 0.10,
    "game_ev_reasoning": 0.10,
    "behavioral": 0.05,
}

SWE_READINESS_WEIGHTS: dict[str, float] = {
    "dsa": 0.30,
    "coding": 0.25,
    "language_knowledge": 0.15,
    "systems": 0.15,
    "projects": 0.10,
    "behavioral": 0.05,
}

# Fallback for any role family with no dedicated topic set registered
# below. Deliberately generic - "coding interview basics" - rather than
# a guess at a specific unrecognized family's real interview format.
GENERAL_TECHNICAL_READINESS_WEIGHTS: dict[str, float] = {
    "coding": 0.30,
    "algorithms": 0.20,
    "systems": 0.15,
    "domain_knowledge": 0.15,
    "projects": 0.10,
    "behavioral": 0.10,
}

# Registry of role-family -> topic weights. Each entry's weights must
# sum to 1.0 (checked by tests, not enforced at import time - keep this
# consistent by hand when adding a new family, the same convention
# backend.matching.scorer.ROLE_FAMILY_WEIGHT_OVERRIDES uses).
ROLE_FAMILY_READINESS_WEIGHTS: dict[str, dict[str, float]] = {
    "quant": QUANT_READINESS_WEIGHTS,
    "swe": SWE_READINESS_WEIGHTS,
}


def get_readiness_weights(role_family: str | None) -> dict[str, float]:
    """Return the topic weights for a role family, or GENERAL_TECHNICAL_READINESS_WEIGHTS if unrecognized/None."""
    if role_family:
        weights = ROLE_FAMILY_READINESS_WEIGHTS.get(normalize_skill_name(role_family))
        if weights is not None:
            return weights
    return GENERAL_TECHNICAL_READINESS_WEIGHTS


# ---------------------------------------------------------------------
# HEURISTIC: mapping interview topics to resume-derived evidence
# ---------------------------------------------------------------------
#
# Interview topics are a curated, coarser taxonomy than the free-form
# skill names a resume gets extracted into (backend.candidate.profile).
# "dsa" isn't a skill anyone lists - it's an aggregate of "algorithms"
# and "data structures"; "behavioral" and "market_knowledge" usually
# have no resume skill equivalent at all. This table is a best-effort,
# hand-picked mapping, NOT a validated taxonomy - extend or edit it as
# real usage shows gaps. Topics mapped to () deliberately have no
# resume-derived signal (see _resume_derived_topic_score()).
TOPIC_SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "probability": ("probability", "probability and statistics", "statistics"),
    "coding": ("coding", "programming", "python", "java", "c++", "c", "javascript", "typescript", "go", "rust"),
    "algorithms": ("algorithms", "algorithms and data structures", "data structures"),
    "mental_math": ("mental math", "quantitative reasoning", "arithmetic", "mental arithmetic"),
    "market_knowledge": ("market knowledge", "market microstructure", "trading", "finance"),
    "game_ev_reasoning": ("game theory", "expected value", "probability"),
    "behavioral": (),  # no reliable resume-derived signal for this - see docstring below
    "dsa": ("algorithms", "data structures", "algorithms and data structures"),
    "language_knowledge": ("python", "java", "c++", "c", "javascript", "typescript", "go", "rust"),
    "systems": ("systems", "systems design", "operating systems", "distributed systems", "networking"),
    "domain_knowledge": (),  # too role-family-specific for a generic alias list
}

# "projects" has its own scoring path below (project COUNT, not a skill
# match) since "having built things" isn't a claimed skill.
PROJECT_COUNT_FOR_FULL_CREDIT = 3
# Deliberately lower than a typical direct skill claim - project count
# is a weaker, indirect proxy for interview-topic readiness than a
# skill the candidate explicitly claimed with its own confidence.
PROJECT_EVIDENCE_CONFIDENCE = 0.3


def _resume_derived_topic_score(topic: str, profile: CandidateProfile) -> tuple[float, float]:
    """
    HEURISTIC resume-derived (score 0-10, confidence 0-1) signal for one
    interview topic - see the module and TOPIC_SKILL_ALIASES docstrings
    for why this is an approximation, not a measurement.

    Most topics: match TOPIC_SKILL_ALIASES against the candidate's
    skills and take the HIGHEST estimated_level among any match (their
    strongest evidence for this topic - not diluted by an absent second
    data point), with that skill's own confidence.

    "projects": scored from profile.projects's COUNT instead (see
    PROJECT_EVIDENCE_CONFIDENCE) since it isn't a claimed skill.

    Topics with no alias mapping and no resume evidence (e.g.
    "behavioral" for most candidates) correctly resolve to (0.0, 0.0) -
    no fabricated signal - relying entirely on a diagnostic if one
    exists.
    """
    if topic == "projects":
        if not profile.projects:
            return 0.0, 0.0
        score = min(len(profile.projects) / PROJECT_COUNT_FOR_FULL_CREDIT, 1.0) * 10.0
        return score, PROJECT_EVIDENCE_CONFIDENCE

    aliases = {normalize_skill_name(alias) for alias in TOPIC_SKILL_ALIASES.get(topic, ())}
    if not aliases:
        return 0.0, 0.0

    matched_skills = [
        skill for skill in profile.skills if normalize_skill_name(skill.normalized_skill_name) in aliases
    ]
    if not matched_skills:
        return 0.0, 0.0

    best = max(matched_skills, key=lambda skill: skill.estimated_level)
    return best.estimated_level, best.confidence


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


class DiagnosticResult(BaseModel):
    """
    One diagnostic/assessment attempt for an interview topic. Diagnostics
    are administered per readiness TOPIC (e.g. "take the DSA diagnostic"),
    not per raw skill, so `topic` should match a key in this module's
    weight tables (e.g. "dsa", "probability").

    `confidence` here is an ABSOLUTE trust level in this result (default
    0.9 - a completed diagnostic is strong evidence), unlike the
    `confidence_delta` column on the assessment_results table
    (backend/db, not yet wired to this module) which represents an
    incremental adjustment; whoever eventually loads DB rows into this
    model is responsible for that translation.
    """

    topic: str
    observed_level: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1, default=0.9)
    taken_at: datetime | None = None


# A diagnostic's weighting in the score combination is its own
# confidence multiplied by this constant. This is what makes "diagnostic
# scores should be stronger evidence than resume inference" a guarantee
# rather than a coincidence of typical confidence values: even if a
# diagnostic and a resume skill somehow reported the SAME confidence,
# the diagnostic would still outweigh the resume estimate 3-to-1 in the
# combined score.
DIAGNOSTIC_EVIDENCE_MULTIPLIER = 3.0


def _combine_topic_evidence(
    resume_score: float, resume_confidence: float, diagnostic: DiagnosticResult | None
) -> tuple[float, float, str]:
    """
    Deterministic combination of resume-derived and diagnostic-derived
    evidence for one topic.

    With no diagnostic: the resume-derived estimate is used as-is.

    With a diagnostic: the combined score is a confidence-weighted
    average of the two observed levels, where the diagnostic's weight is
    `diagnostic.confidence * DIAGNOSTIC_EVIDENCE_MULTIPLIER` - so it
    dominates the resume-derived estimate both because diagnostics
    conventionally report higher confidence AND because of the explicit
    multiplier. Combined confidence is the MAX of the two sources' own
    confidence (we are at least as sure as our best single piece of
    evidence, never less sure for having more of it).

    Returns (combined_score [0-10], combined_confidence [0-1],
    evidence_source label).
    """
    if diagnostic is None:
        source = "resume_only" if resume_confidence > 0 else "no_evidence"
        return resume_score, resume_confidence, source

    resume_weight = resume_confidence
    diagnostic_weight = diagnostic.confidence * DIAGNOSTIC_EVIDENCE_MULTIPLIER
    total_weight = resume_weight + diagnostic_weight

    if total_weight <= 0:
        # Neither source carries any confidence - fall back to a plain
        # average of the raw scores rather than dividing by zero.
        combined_score = (resume_score + diagnostic.observed_level) / 2.0
    else:
        combined_score = (
            resume_score * resume_weight + diagnostic.observed_level * diagnostic_weight
        ) / total_weight

    combined_confidence = max(resume_confidence, diagnostic.confidence)
    source = "diagnostic_and_resume" if resume_confidence > 0 else "diagnostic_only"
    return combined_score, combined_confidence, source


# ---------------------------------------------------------------------
# Readiness result
# ---------------------------------------------------------------------

# A topic scoring below this (0-100 scale) after combining all evidence
# is flagged as a "major gap" - a deliberately simple, single threshold
# rather than a more elaborate rule, consistent with this project's
# general preference for the simplest rule that's still defensible.
MAJOR_GAP_READINESS_THRESHOLD = 50.0


def _clamp_score(value: float) -> float:
    """Clamp a readiness score into the valid [0, 100] range."""
    return max(0.0, min(100.0, value))


class TopicReadiness(BaseModel):
    """One interview topic's full readiness breakdown - both raw evidence and the combined result."""

    topic: str
    weight: float
    resume_score: float = Field(ge=0, le=10)
    resume_confidence: float = Field(ge=0, le=1)
    diagnostic_score: float | None = Field(default=None, ge=0, le=10)
    diagnostic_confidence: float | None = Field(default=None, ge=0, le=1)
    combined_score: float = Field(ge=0, le=10)
    combined_confidence: float = Field(ge=0, le=1)
    readiness_score: float = Field(ge=0, le=100, description="combined_score scaled to 0-100.")
    evidence_source: str = Field(
        description="One of: resume_only, diagnostic_only, diagnostic_and_resume, no_evidence."
    )


class ReadinessGap(BaseModel):
    """One topic flagged as a major readiness gap, ranked by how much it drags down overall readiness."""

    topic: str
    readiness_score: float
    weight: float
    weighted_deficit: float = Field(description="weight * (100 - readiness_score); higher = bigger drag on overall readiness.")


class ReadinessResult(BaseModel):
    """Complete output of calculate_readiness(): overall score, full per-topic breakdown, and flagged gaps."""

    overall_readiness: float = Field(ge=0, le=100)
    role_family: str | None
    weights_used: dict[str, float]
    topic_readiness: list[TopicReadiness]
    major_gaps: list[ReadinessGap]


def calculate_readiness(
    profile: CandidateProfile,
    role_family: str | None,
    diagnostics: list[DiagnosticResult] | None = None,
    weights: dict[str, float] | None = None,
) -> ReadinessResult:
    """
    Compute interview readiness (0-100) for a candidate against a role
    family's interview topics - see this module's docstring for why
    this is a separate concept from fit.

    `weights` defaults to get_readiness_weights(role_family). For each
    topic: derive a resume-based estimate (_resume_derived_topic_score),
    combine it with a matching DiagnosticResult if one was supplied
    (_combine_topic_evidence - diagnostics dominate by design), scale to
    0-100, and weight into the overall score. Topics scoring below
    MAJOR_GAP_READINESS_THRESHOLD are returned as major_gaps, ranked by
    how much they drag down the overall score.
    """
    resolved_weights = weights or get_readiness_weights(role_family)
    diagnostics_by_topic = {normalize_skill_name(d.topic): d for d in (diagnostics or [])}

    topic_results: list[TopicReadiness] = []
    weighted_sum = 0.0

    for topic, weight in resolved_weights.items():
        resume_score, resume_confidence = _resume_derived_topic_score(topic, profile)
        diagnostic = diagnostics_by_topic.get(normalize_skill_name(topic))
        combined_score, combined_confidence, source = _combine_topic_evidence(
            resume_score, resume_confidence, diagnostic
        )
        readiness_score = _clamp_score(combined_score * 10.0)

        topic_results.append(
            TopicReadiness(
                topic=topic,
                weight=weight,
                resume_score=round(resume_score, 3),
                resume_confidence=round(resume_confidence, 3),
                diagnostic_score=diagnostic.observed_level if diagnostic else None,
                diagnostic_confidence=diagnostic.confidence if diagnostic else None,
                # combined_score is a weighted average of two values already
                # within [0, 10] (resume_score, diagnostic.observed_level),
                # so it is provably within [0, 10] too - no clamping needed.
                combined_score=round(combined_score, 3),
                combined_confidence=round(combined_confidence, 3),
                readiness_score=round(readiness_score, 2),
                evidence_source=source,
            )
        )
        weighted_sum += weight * readiness_score

    overall_readiness = _clamp_score(weighted_sum)

    major_gaps = sorted(
        (
            ReadinessGap(
                topic=result.topic,
                readiness_score=result.readiness_score,
                weight=result.weight,
                weighted_deficit=round(result.weight * (100.0 - result.readiness_score), 3),
            )
            for result in topic_results
            if result.readiness_score < MAJOR_GAP_READINESS_THRESHOLD
        ),
        key=lambda gap: gap.weighted_deficit,
        reverse=True,
    )

    return ReadinessResult(
        overall_readiness=round(overall_readiness, 2),
        role_family=role_family,
        weights_used=resolved_weights,
        topic_readiness=topic_results,
        major_gaps=major_gaps,
    )
