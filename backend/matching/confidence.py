"""
Skill confidence aggregation.

Fit scores are point estimates computed from candidate skill levels
that are themselves uncertain (resume evidence yields moderate
confidence at best; see backend/candidate/profile.py). Rather than
folding that uncertainty into the fit score itself - which would let a
merely under-evidenced candidate look arbitrarily worse than a
well-evidenced one with the same claimed level - this module aggregates
confidence into its own separate number that callers can surface
alongside the score ("we estimate 72% fit, moderate confidence") rather
than baked invisibly into it.
"""

from __future__ import annotations

from backend.matching.gaps import SkillGapResult


def aggregate_confidence(gap_results: list[SkillGapResult]) -> float:
    """
    Importance-weighted average confidence across a set of skill gap
    results, in [0, 1]. Requirements with no matching candidate skill
    contribute confidence 0.0 (no evidence either way). Returns 0.0 if
    there is nothing to aggregate or every requirement has zero
    importance.
    """
    total_weight = sum(gap.importance for gap in gap_results)
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(gap.importance * gap.confidence for gap in gap_results)
    return round(weighted_sum / total_weight, 4)


def confidence_label(confidence: float) -> str:
    """Map an aggregate confidence value in [0, 1] to a human-readable label for display."""
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.4:
        return "moderate"
    if confidence > 0.0:
        return "low"
    return "unknown"
