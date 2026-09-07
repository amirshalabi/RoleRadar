"""
Pipeline observability and LLM-cost-avoidance metrics.

Every number this module produces is either a directly-supplied
measurement (a caller reports how many postings were ingested,
deduplicated, filtered, etc. - this module never re-derives those
counts by guessing) or a simple, documented arithmetic function of
measured numbers (llm_avoidance_rate, speedup, estimated cost from real
token counts). Nothing here invents a number: a rate/cost that can't be
computed from real data returns None, not a plausible-looking guess.

This module is intentionally foundational - it does not import from
backend.ingestion or backend.matching, even though PipelineMetrics is
built from their result objects in practice (see backend/db/metrics.py
and backend/ingestion/demo.py for that wiring). Keeping the dependency
one-directional (higher-level packages depend on utils, not the other
way around) avoids import cycles as the pipeline grows.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineMetrics(BaseModel):
    """
    Funnel counts for one ingestion+matching run, from raw ingestion
    through to LLM analysis. Every field is a real measured count from
    an actual run.

    Funnel semantics - read this before using eligible_input_count() or
    llm_avoidance_rate():

        ingested                                   (raw records fetched, pre-dedup)
          - deduplicated                           (duplicate raw records collapsed)
          = eligible_input_count()                 (unique candidate postings pool)
              - hard_filtered                      (removed by matching.filters hard constraints)
              - keyword_filtered                   (removed by keyword/skill overlap)
              - semantic_filtered                  (removed by semantic/embedding retrieval;
                                                     always 0 today - that stage does not exist
                                                     yet, but the field is here now so wiring it
                                                     in later needs no schema/model change)
          = llm_analyzed                           (postings actually sent to the LLM)

    `deduplicated`, `hard_filtered`, `keyword_filtered`, and
    `semantic_filtered` are each a COUNT REMOVED at that stage - NOT a
    count remaining. `ingested` is the funnel's starting total;
    `llm_analyzed` is the final survivor count.
    """

    ingested: int = Field(ge=0)
    deduplicated: int = Field(ge=0, description="Duplicate raw records removed - a count removed, not a count remaining.")
    hard_filtered: int = Field(ge=0, description="Removed by deterministic hard-constraint filters.")
    keyword_filtered: int = Field(ge=0, description="Removed by keyword/skill overlap filtering.")
    semantic_filtered: int = Field(
        ge=0, default=0, description="Removed by semantic/embedding retrieval. Always 0 until that stage exists."
    )
    llm_analyzed: int = Field(ge=0, description="Postings actually sent to the LLM for analysis.")

    def eligible_input_count(self) -> int:
        """
        The denominator for llm_avoidance_rate(): unique candidate
        postings entering the filter funnel, i.e. `ingested` minus
        `deduplicated`.

        Deliberately the DEDUPLICATED count, not the raw `ingested`
        count: a duplicate record was never a distinct "candidate
        posting" to begin with, so counting it in the denominator would
        dilute the avoidance rate with noise that had nothing to do
        with any filtering decision.
        """
        return max(self.ingested - self.deduplicated, 0)

    def llm_avoidance_rate(self) -> float | None:
        """
        1 - llm_analyzed / eligible_input_count(): the fraction of
        eligible (deduplicated) candidate postings that were NEVER sent
        to the LLM, because a cheap deterministic stage already
        eliminated them first. This is the number behind a claim like
        "93% of candidate postings were eliminated before LLM
        analysis" - and it is only ever as true as the counts fed into
        this model.

        Returns None if eligible_input_count() is 0 - nothing to
        compute a rate over - rather than dividing by zero or reporting
        a meaningless 0%/100%.
        """
        eligible = self.eligible_input_count()
        if eligible == 0:
            return None
        return round(1.0 - (self.llm_analyzed / eligible), 4)

    def estimated_llm_calls_avoided(self) -> int:
        """hard_filtered + keyword_filtered + semantic_filtered - how many eligible postings never needed an LLM call."""
        return self.hard_filtered + self.keyword_filtered + self.semantic_filtered

    def funnel_is_consistent(self) -> bool:
        """
        True if hard_filtered + keyword_filtered + semantic_filtered +
        llm_analyzed == eligible_input_count() - every eligible posting
        accounted for by exactly one funnel stage. Informational only
        (not enforced as a validation error): a caller reporting a
        partial/in-progress run may legitimately have an inconsistent
        funnel, and this lets that be detected rather than hidden.
        """
        return self.estimated_llm_calls_avoided() + self.llm_analyzed == self.eligible_input_count()


class ConcurrencyMetrics(BaseModel):
    """Real, measured serial-vs-concurrent ingestion timing (see backend.ingestion.concurrent.IngestionRunResult)."""

    serial_seconds: float = Field(ge=0)
    concurrent_seconds: float = Field(ge=0)

    def speedup(self) -> float | None:
        """serial_seconds / concurrent_seconds, rounded. None if concurrent_seconds is 0 (nothing measured)."""
        if self.concurrent_seconds <= 0:
            return None
        return round(self.serial_seconds / self.concurrent_seconds, 3)


class TokenUsage(BaseModel):
    """Token usage from a single OpenAI call's real `usage` field - see backend.llm.client.parse_structured_with_usage()."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


# USD per 1,000,000 tokens, as published by OpenAI at the time this
# table was written. THIS WILL DRIFT - update it when pricing changes.
# Only models listed here get a cost estimate; see estimate_cost_usd().
OPENAI_PRICING_USD_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
}


def estimate_cost_usd(usage: TokenUsage, model: str) -> float | None:
    """
    Estimate USD cost from REAL measured token counts using
    OPENAI_PRICING_USD_PER_MILLION_TOKENS. Returns None - never a
    guessed number - if `model` isn't in that table, since we have no
    real pricing to apply.
    """
    pricing = OPENAI_PRICING_USD_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        return None
    cost = (usage.prompt_tokens / 1_000_000) * pricing["prompt"] + (
        usage.completion_tokens / 1_000_000
    ) * pricing["completion"]
    return round(cost, 6)


class LLMUsageMetrics(BaseModel):
    """
    Aggregate OpenAI token usage across every LLM call made during one
    pipeline run. Every field stays None/0 unless real usage data was
    actually captured for at least one call - see aggregate_llm_usage().
    """

    call_count: int = Field(ge=0, default=0, description="Number of LLM calls whose usage was actually captured.")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


def aggregate_llm_usage(usages: list[TokenUsage | None], model: str | None = None) -> LLMUsageMetrics:
    """
    Sum token usage across every LLM call in a run. Calls with no usage
    data (None - e.g. the API didn't return it, or the call was mocked
    in a test) are skipped entirely, NOT treated as zero - so
    `call_count` always reflects how many calls actually contributed,
    rather than a run with partially-missing usage data silently
    understating its own totals.

    `model` is used only to look up pricing for estimated_cost_usd(); if
    omitted (or pricing for that model is unknown), estimated_cost_usd
    stays None rather than guessing.
    """
    captured = [usage for usage in usages if usage is not None]
    if not captured:
        return LLMUsageMetrics(call_count=0)

    prompt_total = sum(usage.prompt_tokens for usage in captured)
    completion_total = sum(usage.completion_tokens for usage in captured)
    total = sum(usage.total_tokens for usage in captured)

    estimated_cost = None
    if model is not None:
        estimated_cost = estimate_cost_usd(
            TokenUsage(prompt_tokens=prompt_total, completion_tokens=completion_total, total_tokens=total), model
        )

    return LLMUsageMetrics(
        call_count=len(captured),
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        total_tokens=total,
        estimated_cost_usd=estimated_cost,
    )
