"""
OpenAI API client wrapper.

Will provide a thin wrapper around the OpenAI API with error handling
and retry behavior for structured extraction and rationale-generation
calls. Reads its API key from backend.utils.config, never hardcoded.
"""
