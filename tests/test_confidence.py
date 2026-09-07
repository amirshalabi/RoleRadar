"""Tests for backend.matching.confidence (aggregate confidence, kept separate from fit scores)."""

from __future__ import annotations

from backend.matching.confidence import aggregate_confidence, confidence_label
from backend.matching.gaps import SkillGapResult


def _gap(importance=5.0, confidence=0.5) -> SkillGapResult:
    return SkillGapResult(
        normalized_skill="python",
        display_skill="Python",
        target_level=5.0,
        candidate_level=5.0,
        importance=importance,
        required=True,
        matched=True,
        confidence=confidence,
        raw_gap=0.0,
        weighted_gap=0.0,
        satisfaction_ratio=1.0,
    )


def test_aggregate_confidence_is_importance_weighted_average() -> None:
    gaps = [_gap(importance=10.0, confidence=0.8), _gap(importance=10.0, confidence=0.2)]

    result = aggregate_confidence(gaps)

    assert result == 0.5


def test_aggregate_confidence_weights_higher_importance_more() -> None:
    gaps = [_gap(importance=9.0, confidence=1.0), _gap(importance=1.0, confidence=0.0)]

    result = aggregate_confidence(gaps)

    assert result == 0.9


def test_aggregate_confidence_returns_zero_for_empty_input() -> None:
    assert aggregate_confidence([]) == 0.0


def test_aggregate_confidence_returns_zero_when_all_importance_is_zero() -> None:
    gaps = [_gap(importance=0.0, confidence=0.9)]

    assert aggregate_confidence(gaps) == 0.0


def test_confidence_label_thresholds() -> None:
    assert confidence_label(0.9) == "high"
    assert confidence_label(0.7) == "high"
    assert confidence_label(0.5) == "moderate"
    assert confidence_label(0.4) == "moderate"
    assert confidence_label(0.1) == "low"
    assert confidence_label(0.0) == "unknown"
