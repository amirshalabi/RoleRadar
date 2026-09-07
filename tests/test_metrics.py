"""
Tests for backend.utils.metrics. Pure Python, no I/O - every number
here is either a directly-constructed input or a documented arithmetic
function of one.
"""

from __future__ import annotations

import pytest

from backend.utils.metrics import (
    OPENAI_PRICING_USD_PER_MILLION_TOKENS,
    ConcurrencyMetrics,
    LLMUsageMetrics,
    PipelineMetrics,
    TokenUsage,
    aggregate_llm_usage,
    estimate_cost_usd,
)


def _metrics(ingested=100, deduplicated=10, hard_filtered=50, keyword_filtered=30, semantic_filtered=0, llm_analyzed=10):
    return PipelineMetrics(
        ingested=ingested, deduplicated=deduplicated, hard_filtered=hard_filtered,
        keyword_filtered=keyword_filtered, semantic_filtered=semantic_filtered, llm_analyzed=llm_analyzed,
    )


# ---------------------------------------------------------------------
# eligible_input_count / llm_avoidance_rate: the denominator precision requirement
# ---------------------------------------------------------------------


def test_eligible_input_count_excludes_deduplicated_records() -> None:
    metrics = _metrics(ingested=100, deduplicated=10)

    assert metrics.eligible_input_count() == 90


def test_eligible_input_count_never_negative() -> None:
    """Defensive: deduplicated should never exceed ingested in real data, but this must not crash or go negative."""
    metrics = _metrics(ingested=5, deduplicated=10)

    assert metrics.eligible_input_count() == 0


def test_llm_avoidance_rate_matches_the_worked_example() -> None:
    # 100 ingested, 10 dupes -> 90 eligible; 50 hard-filtered, 30
    # keyword-filtered, 10 reach the LLM -> 1 - 10/90 = 0.8889
    metrics = _metrics(ingested=100, deduplicated=10, hard_filtered=50, keyword_filtered=30, llm_analyzed=10)

    assert metrics.llm_avoidance_rate() == pytest.approx(0.8889, abs=0.0001)


def test_llm_avoidance_rate_100_percent_when_nothing_reaches_llm() -> None:
    metrics = _metrics(ingested=100, deduplicated=0, hard_filtered=60, keyword_filtered=40, llm_analyzed=0)

    assert metrics.llm_avoidance_rate() == 1.0


def test_llm_avoidance_rate_0_percent_when_everything_reaches_llm() -> None:
    metrics = _metrics(ingested=50, deduplicated=0, hard_filtered=0, keyword_filtered=0, llm_analyzed=50)

    assert metrics.llm_avoidance_rate() == 0.0


def test_llm_avoidance_rate_is_none_for_empty_eligible_pool() -> None:
    """No division by zero, no fabricated rate, when there's nothing to compute over."""
    metrics = _metrics(ingested=0, deduplicated=0, hard_filtered=0, keyword_filtered=0, llm_analyzed=0)

    assert metrics.eligible_input_count() == 0
    assert metrics.llm_avoidance_rate() is None


def test_llm_avoidance_rate_is_none_when_all_ingested_were_duplicates() -> None:
    metrics = _metrics(ingested=20, deduplicated=20, hard_filtered=0, keyword_filtered=0, llm_analyzed=0)

    assert metrics.llm_avoidance_rate() is None


def test_deduplicated_records_do_not_inflate_avoidance_rate() -> None:
    """
    A run with lots of noisy duplicates but weak actual filtering should
    NOT report a high avoidance rate just because of dedup noise - the
    denominator is deliberately the deduplicated pool, not raw ingested.
    """
    heavy_dupes = _metrics(ingested=1000, deduplicated=900, hard_filtered=0, keyword_filtered=0, llm_analyzed=100)
    # eligible = 100, all 100 reach the LLM -> 0% avoidance, despite 900 "eliminated" raw records being dupes, not filtering
    assert heavy_dupes.eligible_input_count() == 100
    assert heavy_dupes.llm_avoidance_rate() == 0.0


# ---------------------------------------------------------------------
# estimated_llm_calls_avoided / funnel_is_consistent
# ---------------------------------------------------------------------


def test_estimated_llm_calls_avoided_sums_all_filter_stages() -> None:
    metrics = _metrics(hard_filtered=50, keyword_filtered=30, semantic_filtered=5, llm_analyzed=5)

    assert metrics.estimated_llm_calls_avoided() == 85


def test_funnel_is_consistent_true_when_stages_sum_to_eligible() -> None:
    metrics = _metrics(ingested=100, deduplicated=10, hard_filtered=50, keyword_filtered=30, llm_analyzed=10)

    assert metrics.funnel_is_consistent() is True


def test_funnel_is_consistent_false_when_stages_do_not_sum_to_eligible() -> None:
    metrics = _metrics(ingested=100, deduplicated=10, hard_filtered=50, keyword_filtered=30, llm_analyzed=5)  # should be 10

    assert metrics.funnel_is_consistent() is False


def test_pipeline_metrics_rejects_negative_counts() -> None:
    with pytest.raises(Exception):
        PipelineMetrics(ingested=-1, deduplicated=0, hard_filtered=0, keyword_filtered=0, llm_analyzed=0)


# ---------------------------------------------------------------------
# ConcurrencyMetrics.speedup
# ---------------------------------------------------------------------


def test_speedup_basic_ratio() -> None:
    metrics = ConcurrencyMetrics(serial_seconds=1.4, concurrent_seconds=0.7)

    assert metrics.speedup() == 2.0


def test_speedup_is_none_when_concurrent_seconds_is_zero() -> None:
    metrics = ConcurrencyMetrics(serial_seconds=1.4, concurrent_seconds=0.0)

    assert metrics.speedup() is None


def test_speedup_never_invented_matches_real_ratio_exactly() -> None:
    metrics = ConcurrencyMetrics(serial_seconds=3.333, concurrent_seconds=1.111)

    assert metrics.speedup() == pytest.approx(3.333 / 1.111, abs=0.001)


# ---------------------------------------------------------------------
# TokenUsage / estimate_cost_usd / aggregate_llm_usage
# ---------------------------------------------------------------------


def test_estimate_cost_usd_known_model() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
    pricing = OPENAI_PRICING_USD_PER_MILLION_TOKENS["gpt-4o-mini"]

    cost = estimate_cost_usd(usage, "gpt-4o-mini")

    assert cost == pytest.approx(pricing["prompt"] + pricing["completion"])


def test_estimate_cost_usd_unknown_model_returns_none_not_a_guess() -> None:
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)

    assert estimate_cost_usd(usage, "some-future-model-not-in-table") is None


def test_aggregate_llm_usage_sums_captured_calls() -> None:
    usages = [
        TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        TokenUsage(prompt_tokens=200, completion_tokens=40, total_tokens=240),
    ]

    result = aggregate_llm_usage(usages)

    assert result.call_count == 2
    assert result.prompt_tokens == 300
    assert result.completion_tokens == 60
    assert result.total_tokens == 360


def test_aggregate_llm_usage_skips_none_entries_without_treating_as_zero() -> None:
    usages = [TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120), None]

    result = aggregate_llm_usage(usages)

    assert result.call_count == 1  # not 2 - the None call didn't "contribute a zero"
    assert result.prompt_tokens == 100


def test_aggregate_llm_usage_all_none_returns_empty_metrics_not_invented_zeros() -> None:
    result = aggregate_llm_usage([None, None])

    assert result.call_count == 0
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None
    assert result.estimated_cost_usd is None


def test_aggregate_llm_usage_empty_list_returns_empty_metrics() -> None:
    result = aggregate_llm_usage([])

    assert result == LLMUsageMetrics(call_count=0)


def test_aggregate_llm_usage_computes_cost_when_model_given() -> None:
    usages = [TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000)]

    result = aggregate_llm_usage(usages, model="gpt-4o-mini")

    assert result.estimated_cost_usd == pytest.approx(OPENAI_PRICING_USD_PER_MILLION_TOKENS["gpt-4o-mini"]["prompt"])


def test_aggregate_llm_usage_no_cost_when_model_not_given() -> None:
    usages = [TokenUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)]

    result = aggregate_llm_usage(usages)  # no model supplied

    assert result.estimated_cost_usd is None
