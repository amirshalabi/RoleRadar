"""
Tests for backend.llm.client (OpenAI wrapper).

No real API calls are made: get_openai_client is monkeypatched with a
small fake object shaped like the OpenAI SDK client so parse_structured
can be exercised end-to-end (happy path, refusal, unparseable response).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.llm import client as llm_client
from backend.utils.config import get_settings


class _DummyModel(BaseModel):
    value: str


@pytest.fixture(autouse=True)
def _clear_caches():
    get_settings.cache_clear()
    llm_client.get_openai_client.cache_clear()
    yield
    get_settings.cache_clear()
    llm_client.get_openai_client.cache_clear()


def _fake_client_returning(completion) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs: completion))
    )


def test_get_openai_client_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(llm_client.OpenAINotConfiguredError):
        llm_client.get_openai_client()


def test_get_model_name_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert llm_client.get_model_name() == "gpt-4o-mini"


def test_get_model_name_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    assert llm_client.get_model_name() == "gpt-4o"


def test_parse_structured_returns_parsed_value(monkeypatch: pytest.MonkeyPatch) -> None:
    message = SimpleNamespace(parsed=_DummyModel(value="ok"), refusal=None)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: _fake_client_returning(completion))

    result = llm_client.parse_structured("system", "user", _DummyModel)

    assert result.value == "ok"


def test_parse_structured_raises_on_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    message = SimpleNamespace(parsed=None, refusal="cannot assist with that")
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: _fake_client_returning(completion))

    with pytest.raises(llm_client.LLMExtractionError):
        llm_client.parse_structured("system", "user", _DummyModel)


def test_parse_structured_raises_when_parsed_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    message = SimpleNamespace(parsed=None, refusal=None)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: _fake_client_returning(completion))

    with pytest.raises(llm_client.LLMExtractionError):
        llm_client.parse_structured("system", "user", _DummyModel)


def test_parse_structured_with_usage_returns_real_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    message = SimpleNamespace(parsed=_DummyModel(value="ok"), refusal=None)
    usage = SimpleNamespace(prompt_tokens=42, completion_tokens=7, total_tokens=49)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: _fake_client_returning(completion))

    parsed, token_usage = llm_client.parse_structured_with_usage("system", "user", _DummyModel)

    assert parsed.value == "ok"
    assert token_usage.prompt_tokens == 42
    assert token_usage.completion_tokens == 7
    assert token_usage.total_tokens == 49


def test_parse_structured_with_usage_returns_none_when_response_has_no_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """No usage attribute at all on the completion (e.g. an older API shape or a minimal test double) - never invented."""
    message = SimpleNamespace(parsed=_DummyModel(value="ok"), refusal=None)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])  # no .usage attribute
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: _fake_client_returning(completion))

    parsed, token_usage = llm_client.parse_structured_with_usage("system", "user", _DummyModel)

    assert parsed.value == "ok"
    assert token_usage is None


def test_parse_structured_discards_usage_but_behaves_identically(monkeypatch: pytest.MonkeyPatch) -> None:
    """parse_structured() is a thin wrapper - existing callers see no behavior change."""
    message = SimpleNamespace(parsed=_DummyModel(value="ok"), refusal=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: _fake_client_returning(completion))

    result = llm_client.parse_structured("system", "user", _DummyModel)

    assert result.value == "ok"  # plain value returned, no tuple
