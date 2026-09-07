"""
OpenAI API client wrapper.

Provides a thin, cached wrapper around the OpenAI API plus a single
parse_structured() helper used by every structured-extraction call
(candidate profile extraction now; job requirement extraction later).
The API key and model are read from backend.utils.config, never
hardcoded, so the model can be swapped via the OPENAI_MODEL environment
variable without touching code.

This module raises clear, typed errors on failure (missing credentials,
a model refusal, an unparseable response) rather than swallowing them,
so callers can decide how to handle a failed extraction.

parse_structured_with_usage() additionally returns real token usage
from the OpenAI response's own `usage` field (backend.utils.metrics.TokenUsage)
when the API provides it - None otherwise, never an invented count.
parse_structured() is unchanged and calls it internally, discarding the
usage value, so every existing caller's behavior is identical.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from backend.utils.config import get_settings
from backend.utils.metrics import TokenUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAINotConfiguredError(RuntimeError):
    """Raised when OPENAI_API_KEY is not set."""


class LLMExtractionError(RuntimeError):
    """Raised when the model refuses a request or returns an unparseable response."""


@lru_cache
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client, or raise if not configured."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise OpenAINotConfiguredError(
            "OPENAI_API_KEY must be set to call the OpenAI API. See .env.example."
        )
    return OpenAI(api_key=settings.openai_api_key)


def get_model_name() -> str:
    """Return the configured chat model (OPENAI_MODEL env var, default gpt-4o-mini)."""
    return get_settings().openai_model


def parse_structured_with_usage(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str | None = None,
) -> tuple[T, TokenUsage | None]:
    """
    Call the OpenAI chat completions API and parse the response directly
    into `response_model`, also returning real token usage from the
    response's own `usage` field.

    The LLM's only job is to fill in this schema from unstructured text;
    Pydantic validates the result before it is returned, and downstream
    deterministic code (scoring, gaps, planning) consumes only this
    validated structure - never the raw model output.

    Returns (parsed, usage). `usage` is None if the response has no
    usage data (e.g. a test double, or a future API variant that omits
    it) - never a fabricated estimate; see backend.utils.metrics for
    what callers should do with a None usage value.
    """
    client = get_openai_client()
    resolved_model = model or get_model_name()
    logger.info("Calling OpenAI model=%s for structured extraction", resolved_model)

    completion = client.chat.completions.parse(
        model=resolved_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_model,
    )

    message = completion.choices[0].message
    if message.refusal:
        raise LLMExtractionError(f"OpenAI refused the request: {message.refusal}")
    if message.parsed is None:
        raise LLMExtractionError("OpenAI response could not be parsed into the expected schema")

    raw_usage = getattr(completion, "usage", None)
    usage = (
        TokenUsage(
            prompt_tokens=raw_usage.prompt_tokens,
            completion_tokens=raw_usage.completion_tokens,
            total_tokens=raw_usage.total_tokens,
        )
        if raw_usage is not None
        else None
    )
    return message.parsed, usage


def parse_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str | None = None,
) -> T:
    """
    Call the OpenAI chat completions API and parse the response directly
    into `response_model`. See parse_structured_with_usage() for a
    variant that also returns real token usage.
    """
    parsed, _usage = parse_structured_with_usage(system_prompt, user_prompt, response_model, model)
    return parsed
