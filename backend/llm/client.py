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
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from backend.utils.config import get_settings

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


def parse_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str | None = None,
) -> T:
    """
    Call the OpenAI chat completions API and parse the response directly
    into `response_model`.

    The LLM's only job is to fill in this schema from unstructured text;
    Pydantic validates the result before it is returned, and downstream
    deterministic code (scoring, gaps, planning) consumes only this
    validated structure - never the raw model output.
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
    return message.parsed
